"""Photometry of imaging time series (e.g. MIRI F1500W eclipses).

Two classes:

:class:`Photometry`
    Works on a whole exposure cleaned by
    :class:`~jstark.reduction.ExposureCleaner`, one segment at a time (the
    exposure is never stitched). For every integration it measures the
    centroid, the sky background (clipped median of an annulus, or a plane)
    and two fluxes:

    * aperture photometry with exact (fractional) pixel apertures;
    * PSF-weighted ("optimal") photometry,
      ``f = sum(e P D / V) / sum(e P^2 / V)``, with an empirical PSF ``P``:
      the time-averaged normalised image built from *all* integrations,
      shifted to every integration's centroid with a bicubic spline. When the
      background dominates the noise (as at 15 micron) PSF weighting gives
      lower noise than a hard aperture (Naylor 1998, MNRAS 296, 339).

    It also stores the light curves of the aperture pixels for pixel-level
    decorrelation (:meth:`PhotometryResult.pld_basis`); the PLD regressors
    should be fitted together with the transit / eclipse model.

:class:`ApPhoto`
    The original in-memory aperture photometry + PLD helper (moved here from
    ``jstark.jstark``), with a clipped-median sky that ignores bad pixels,
    robust symmetric cosmic-ray flagging, point-to-point noise metrics and an
    optional fit mask for the PLD regression.
"""
import json
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import mad_std
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from ..reduction.io import StepLog, output_filename
from ..reduction.robust import MAD_TO_SIGMA
from .metrics import p2p_sigma, p2p_noise
from .source import ExposureSource
from .trace import fill_bad

try:
    from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats
    from photutils.aperture import aperture_photometry as aphot
except ImportError:  # optional dependency
    CircularAnnulus = CircularAperture = ApertureStats = aphot = None

__all__ = ['Photometry', 'PhotometryResult', 'ApPhoto', 'aperture_weights']


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def pipe_mad(x, axis=0):
    """Median absolute point-to-point difference (as ``jstark.utils.pipe_mad``)."""
    return np.nanmedian(np.abs(np.diff(x, axis=axis)), axis=axis)


def rms(x):
    return np.sqrt(np.nanmean(np.asarray(x) ** 2))


def classic_PCA(Input_Data, standarize=True):
    """Classic PCA (N. Espinoza's TESS implementation, as in ``jstark.utils``)."""
    Data = np.copy(Input_Data)
    if standarize:
        av = np.nanmedian(Input_Data, axis=1)
        for i in range(len(av)):
            Data[i, :] = (Data[i, :] - av[i]) / mad_std(Data[i, :])
    v_cols, ev, v_rows = np.linalg.svd(np.cov(Data))
    idx = ev.argsort()[::-1]
    return v_rows[idx, :], ev[idx], np.dot(v_rows[idx, :], Data)


def aperture_weights(shape, yc, xc, r, subsample=11):
    """Fraction of every pixel inside a circle of radius ``r`` centred at ``(yc, xc)``.

    Uses ``photutils.geometry.circular_overlap_grid`` (exact) when available,
    otherwise sub-pixel sampling.
    """
    ny, nx = shape
    try:
        from photutils.geometry import circular_overlap_grid
        return circular_overlap_grid(-xc - 0.5, nx - xc - 0.5, -yc - 0.5, ny - yc - 0.5,
                                     nx, ny, r, 1, 1)
    except ImportError:
        s = (np.arange(subsample) + 0.5) / subsample - 0.5
        yy = np.arange(ny)[:, None, None, None] + s[None, None, :, None] - yc
        xx = np.arange(nx)[None, :, None, None] + s[None, None, None, :] - xc
        return np.mean((yy ** 2 + xx ** 2) <= r * r, axis=(2, 3))


def _clipped_median(vals, nsigma=3., maxiters=3, axis=1):
    v = np.array(vals, dtype=float, copy=True)
    for _ in range(maxiters):
        med = np.nanmedian(v, axis=axis, keepdims=True)
        sig = MAD_TO_SIGMA * np.nanmedian(np.abs(v - med), axis=axis, keepdims=True)
        bad = np.abs(v - med) > nsigma * sig
        if not bad.any():
            break
        v[bad] = np.nan
    return np.nanmedian(v, axis=axis), np.sum(np.isfinite(v), axis=axis), v


# ----------------------------------------------------------------------
class Photometry:
    """Aperture and PSF-weighted photometry of an exposure, segment by segment.

    Typical use::

        ph = Photometry('/data/F1500W/reduction/<exposure>_cleaninfo.fits')
        ph.show()
        ph.set('background', r_in=15, r_out=25)
        ph.fit_psf()                               # one pass over all integrations
        ph.optimise(radii=[3, 4, 5, 6], psf_radii=[6, 8, 10])
        res = ph.run()                             # all segments -> <exposure>_photometry.fits
        lc, err = res.flux['psf'], res.err['psf']

    Parameters
    ----------
    info : str
        ``<exposure>_cleaninfo.fits`` of a cleaned imaging exposure.
    outdir : str or None
    max_mem_gb : float
    verbose : bool
    """

    DEFAULTS = {
        'centroid': dict(halfwidth=6., niter=3),
        'background': dict(method='annulus', r_in=15., r_out=25., nsigma=3.),
        'aperture': dict(radius=4.),
        'psf': dict(halfwidth=12., ints='all'),
        'extract': dict(psf_radius=8., fit_background=False, variance='mean', pld_radius=None,
                        refine_center=2),
    }

    def __init__(self, info, outdir=None, max_mem_gb=2., verbose=True):
        self.src = ExposureSource(info, max_mem_gb=max_mem_gb)
        import copy
        self.params = copy.deepcopy(self.DEFAULTS)
        self.outdir = outdir or self.src.outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.verbose = verbose
        self.log = StepLog(os.path.join(self.outdir, f'{self.src.exposure}_photometry_log.json'))
        self.center = None
        self.psf_image = None
        self.var_mean = None
        self.outputs = {}
        self.optimisation = None
        if verbose:
            print(f'Photometry: {self.src.mode} | {self.src.exposure} | {self.src.nseg} segment(s), '
                  f'{self.src.nint} integrations')

    def set(self, step, **kwargs):
        unknown = set(kwargs) - set(self.params[step])
        if unknown:
            raise KeyError(f'Unknown parameter(s) {sorted(unknown)} for {step!r}')
        self.params[step].update(kwargs)
        if step in ('centroid', 'psf', 'background'):
            self.psf_image = None
        return self

    def show(self):
        for step, kw in self.params.items():
            print(f'  {step:<10s} ' + ', '.join(f'{k}={v}' for k, v in kw.items()))

    # ------------------------------------------------------------------
    def find_center(self):
        """Star position on the exposure median frame (centre of flux)."""
        img = fill_bad(self.src.median, self.src.badpix)
        from scipy.ndimage import uniform_filter
        sm = uniform_filter(img, 3)
        yc, xc = np.unravel_index(np.argmax(sm), sm.shape)
        yc, xc = float(yc), float(xc)
        bkg = np.nanmedian(img)
        hw = self.params['centroid']['halfwidth']
        yy, xx = np.indices(img.shape)
        for _ in range(10):
            w = aperture_weights(img.shape, yc, xc, hw) * np.clip(img - bkg, 0, None)
            yc, xc = np.sum(w * yy) / np.sum(w), np.sum(w * xx) / np.sum(w)
        self.center = (yc, xc)
        if self.verbose:
            print(f'  star at (row, col) = ({yc:.2f}, {xc:.2f})')
        return self.center

    def _box(self, h):
        yc, xc = self.center
        ny, nx = self.src.shape
        y0, y1 = max(0, int(np.floor(yc - h))), min(ny, int(np.ceil(yc + h)) + 1)
        x0, x1 = max(0, int(np.floor(xc - h))), min(nx, int(np.ceil(xc + h)) + 1)
        return slice(y0, y1), slice(x0, x1)

    def _background(self, data, err, good):
        """Per-frame sky level and its variance (``(n,)`` or ``(n, ny, nx)`` for a plane)."""
        pb = self.params['background']
        yc, xc = self.center
        yy, xx = np.indices(self.src.shape)
        r = np.hypot(yy - yc, xx - xc)
        ann = (r >= pb['r_in']) & (r <= pb['r_out'])
        if pb['method'] in (None, 'none'):
            n = len(data)
            return np.zeros(n), np.zeros(n)
        vals = np.where(good[:, ann], data[:, ann], np.nan)
        med, npix, clipped = _clipped_median(vals, pb['nsigma'])
        e2 = np.where(np.isfinite(clipped), err[:, ann].astype(float) ** 2, np.nan)
        var = (np.pi / 2) * np.nanmean(e2, axis=1) / np.maximum(npix, 1)
        if pb['method'] == 'annulus':
            return med, var
        if pb['method'] == 'plane':
            A = np.stack([np.ones(ann.sum()), (yy[ann] - yc) / 10., (xx[ann] - xc) / 10.], 1)
            out = np.zeros(data.shape, np.float32)
            for i in range(len(data)):
                ok = np.isfinite(clipped[i])
                c = np.linalg.lstsq(A[ok], clipped[i, ok], rcond=None)[0]
                out[i] = c[0] + c[1] * (yy - yc) / 10. + c[2] * (xx - xc) / 10.
            return out, var
        raise ValueError(f'Unknown background method {pb["method"]!r}')

    def _prepare(self, data, err, good):
        """Subtract the sky (in place) and measure per-frame centroids."""
        b, vb = self._background(data, err, good)
        data -= (b[:, None, None] if np.ndim(b) == 1 else b).astype(data.dtype)
        pc = self.params['centroid']
        yc0, xc0 = self.center
        sy, sx = self._box(pc['halfwidth'] + 2)
        sub = data[:, sy, sx].astype(float)
        yy, xx = np.indices(sub.shape[1:])
        yy, xx = yy + sy.start, xx + sx.start
        yc = np.full(len(data), yc0)
        xc = np.full(len(data), xc0)
        for _ in range(pc['niter']):
            w = np.stack([aperture_weights(sub.shape[1:], yc[i] - sy.start, xc[i] - sx.start,
                                           pc['halfwidth']) for i in range(len(data))])
            w = w * np.clip(sub, 0, None)
            s = np.sum(w, axis=(1, 2))
            yc = np.sum(w * yy, axis=(1, 2)) / s
            xc = np.sum(w * xx, axis=(1, 2)) / s
        bmean = b if np.ndim(b) == 1 else b[:, int(round(yc0)), int(round(xc0))]
        return dict(bkg=bmean, var_bkg=vb, yc=yc, xc=xc)

    # ------------------------------------------------------------------
    def fit_psf(self):
        """Empirical PSF: mean of the sky-subtracted, flux-normalised images (all integrations).

        Also accumulates the time-averaged variance of every pixel.
        """
        if self.center is None:
            self.find_center()
        t0 = time.time()
        src = self.src
        h = self.params['psf']['halfwidth']
        R = self.params['extract']['psf_radius']
        sy, sx = self._box(h)
        num = np.zeros((sy.stop - sy.start, sx.stop - sx.start))
        cnt = np.zeros_like(num)
        vsum = np.zeros(src.shape)
        vcnt = np.zeros(src.shape)
        ints = self.params['psf']['ints']
        if isinstance(ints, str) and ints == 'all':
            it = ((d, e, g, gi) for _, d, e, g, gi in src.chunks())
        else:
            sel = src.sample_ints(int(ints))
            it = (src.read_ints(sel[a:a + 200]) for a in range(0, len(sel), 200))
        nint = 0
        for d, e, g, gi in it:
            d = d.astype(np.float32, copy=True)
            prep = self._prepare(d, e, g)
            vsum += np.where(g, e.astype(float) ** 2, 0.).sum(0)
            vcnt += g.sum(0)
            for i in range(len(d)):
                w = aperture_weights(src.shape, prep['yc'][i], prep['xc'][i], R)
                F = np.sum(w * d[i])
                ok = g[i, sy, sx]
                num += np.where(ok, d[i, sy, sx] / F, 0.)
                cnt += ok
            nint += len(d)
        self.psf_image = num / np.maximum(cnt, 1)
        self.psf_box = (sy, sx)
        self.psf_center = self.center
        ys = np.arange(sy.start, sy.stop, dtype=float)
        xs = np.arange(sx.start, sx.stop, dtype=float)
        self._psf_spline = RectBivariateSpline(ys, xs, self.psf_image, kx=3, ky=3)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.var_mean = np.where(vcnt > 0, vsum / vcnt, np.inf)
        self.log.add('fit_psf', nint=nint, halfwidth=h, psf_radius=R)
        if self.verbose:
            print(f'  empirical PSF from {nint} integrations ({time.time() - t0:.0f} s)')
        return self.psf_image

    def _psf_at(self, yc, xc, deriv=False):
        """PSF image shifted to centroid ``(yc, xc)`` on the PSF box (and its derivatives)."""
        sy, sx = self.psf_box
        y0, x0 = self.psf_center
        ys = np.arange(sy.start, sy.stop, dtype=float) - (yc - y0)
        xs = np.arange(sx.start, sx.stop, dtype=float) - (xc - x0)
        P = np.clip(self._psf_spline(ys, xs), 0, None)
        if not deriv:
            return P
        # derivatives with respect to the star position (= minus the image gradient)
        return P, -self._psf_spline(ys, xs, dx=1), -self._psf_spline(ys, xs, dy=1)

    def _refine_center(self, D, V, G, yc, xc, R, niter):
        """Gauss-Newton fit of flux and position with the empirical PSF (one frame)."""
        sy, sx = self.psf_box
        for _ in range(niter):
            e = aperture_weights(self.src.shape, yc, xc, R)[sy, sx]
            P, Py, Px = self._psf_at(yc, xc, deriv=True)
            norm = max(np.sum(P), 1e-30)
            P, Py, Px = P / norm, Py / norm, Px / norm
            w = np.where(G & (e > 0), e / V, 0.)
            f = np.sum(w * P * D) / max(np.sum(w * P * P), 1e-30)
            J = np.stack([P, f * Py, f * Px]).reshape(3, -1)
            r = (D - f * P).ravel()
            ww = w.ravel()
            A = (J * ww) @ J.T
            b = (J * ww) @ r
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            dy, dx = np.clip(step[1:], -0.5, 0.5)
            yc, xc = yc + dy, xc + dx
            if abs(dy) < 1e-5 and abs(dx) < 1e-5:
                break
        return yc, xc

    # ------------------------------------------------------------------
    def extract_chunk(self, data, err, good, radius=None, psf_radius=None):
        """Photometry of a chunk (in memory). Returns a dict of per-integration arrays."""
        if self.psf_image is None:
            self.fit_psf()
        pe = self.params['extract']
        r_ap = self.params['aperture']['radius'] if radius is None else radius
        R = pe['psf_radius'] if psf_radius is None else psf_radius
        d = np.array(data, np.float32, copy=True)
        prep = self._prepare(d, err, good)
        n = len(d)
        sy, sx = self.psf_box
        V = (self.var_mean[None] if pe['variance'] == 'mean' else err.astype(float) ** 2)[:, sy, sx]
        V = np.broadcast_to(V, (n,) + V.shape[1:])
        out = {k: np.zeros(n) for k in ('ap', 'ap_var', 'psf', 'psf_var', 'maskfrac', 'width')}
        rp = pe['pld_radius'] or r_ap
        wpld = aperture_weights(self.src.shape, *self.center, rp) > 0.5
        out['pix'] = d[:, wpld]
        out['pix_index'] = np.array(np.where(wpld)).T
        yy, xx = np.indices(self.src.shape)
        for i in range(n):
            yc, xc = prep['yc'][i], prep['xc'][i]
            if pe['refine_center']:
                yc, xc = self._refine_center(d[i, sy, sx].astype(float), V[i], good[i, sy, sx],
                                             yc, xc, R, pe['refine_center'])
                prep['yc'][i], prep['xc'][i] = yc, xc
            wa = aperture_weights(self.src.shape, yc, xc, r_ap)
            out['ap'][i] = np.sum(wa * d[i])
            out['ap_var'][i] = np.sum(wa ** 2 * err[i].astype(float) ** 2) + \
                np.sum(wa) ** 2 * prep['var_bkg'][i]
            e = aperture_weights(self.src.shape, yc, xc, R)[sy, sx]
            P = self._psf_at(yc, xc)
            P = P / max(np.sum(e * P), 1e-30)
            G = good[i, sy, sx] & (e > 0) & np.isfinite(V[i]) & (V[i] > 0)
            D = d[i, sy, sx].astype(float)
            Vi = np.where(G, V[i], 1.)
            w = np.where(G, e / Vi, 0.)
            if pe['fit_background']:
                A = np.array([[np.sum(w * P * P), np.sum(w * P)], [np.sum(w * P), np.sum(w)]])
                rhs = np.array([np.sum(w * P * D), np.sum(w * D)])
                cov = np.linalg.inv(A)
                f = (cov @ rhs)[0]
                # variance of f for weights e/V (= cov[0,0] for whole pixels)
                g = cov[0, 0] * P + cov[0, 1]
                fv = np.sum(np.where(G, (e / Vi) ** 2 * Vi, 0.) * g * g)
            else:
                den = np.sum(w * P * P)
                f = np.sum(w * P * D) / den
                fv = np.sum(np.where(G, (e / Vi) ** 2 * Vi, 0.) * P * P) / den ** 2 + \
                    (np.sum(w * P) / den) ** 2 * prep['var_bkg'][i]
            out['psf'][i], out['psf_var'][i] = f, fv
            out['maskfrac'][i] = np.sum(np.where(G, 0., e * P)) / max(np.sum(e * P), 1e-30)
            wr = aperture_weights(self.src.shape, yc, xc, R) * np.clip(d[i], 0, None)
            out['width'][i] = np.sqrt(np.sum(wr * ((yy - yc) ** 2 + (xx - xc) ** 2)) / np.sum(wr))
        out.update(bkg=prep['bkg'], yc=prep['yc'], xc=prep['xc'])
        return out

    def extract_segment(self, k, save=True):
        parts, gidx = [], []
        for _, d, e, g, gi in self.src.chunks(segments=[k]):
            parts.append(self.extract_chunk(d, e, g))
            gidx.append(gi)
        gidx = np.concatenate(gidx)
        out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != 'pix_index'}
        out['pix_index'] = parts[0]['pix_index']
        if save:
            fn = output_filename(self.src.files[k], self.outdir, 'phot')
            cols = [fits.Column(name='gidx', format='K', array=gidx)]
            if self.src.times is not None:
                cols.append(fits.Column(name='int_mid_BJD_TDB', format='D', array=self.src.times[gidx]))
            for key in ('ap', 'ap_var', 'psf', 'psf_var', 'maskfrac', 'width', 'bkg', 'yc', 'xc'):
                cols.append(fits.Column(name=key, format='D', array=out[key]))
            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name='PHOT'),
                          fits.ImageHDU(out['pix'].astype(np.float32), name='PIXELS'),
                          fits.ImageHDU(out['pix_index'].astype(np.int32), name='PIXINDEX')]
                         ).writeto(fn, overwrite=True)
            self.outputs[k] = fn
        if self.verbose:
            print(f'  segment {k + 1}/{self.src.nseg}: {len(gidx)} integrations')
        return gidx, out

    def run(self, segments=None, keep_segments=True):
        """Photometry of all (or the given) segments; returns the combined :class:`PhotometryResult`.

        ``keep_segments=False`` deletes the per-segment files (``*_phot.fits``)
        after combining (the combined file holds the same data).
        """
        t0 = time.time()
        if self.psf_image is None:
            self.fit_psf()
        segs = range(self.src.nseg) if segments is None else segments
        for k in segs:
            self.extract_segment(k)
        self.log.add('run', segments=list(segs), params=self.params, seconds=time.time() - t0)
        out = self.combine()
        if not keep_segments:
            from .extractor import remove_files
            remove_files(self.outputs, self.verbose)
        return out

    def combine(self, fname=None):
        fname = fname or os.path.join(self.outdir, f'{self.src.exposure}_photometry.fits')
        tabs, pix = [], []
        for k in sorted(self.outputs):
            with fits.open(self.outputs[k]) as h:
                tabs.append({n: np.asarray(h['PHOT'].data[n]) for n in h['PHOT'].data.names})
                pix.append(np.asarray(h['PIXELS'].data))
                pidx = np.asarray(h['PIXINDEX'].data)
        cat = {n: np.concatenate([t[n] for t in tabs]) for n in tabs[0]}
        cols = [fits.Column(name=n, format='K' if v.dtype.kind in 'iu' else 'D', array=v)
                for n, v in cat.items()]
        hdr = fits.Header()
        hdr['JSMODE'] = self.src.mode
        hdr['EXPOSURE'] = self.src.exposure
        par = json.dumps({'params': self.params, 'center': list(self.center),
                          'optimisation': self.optimisation})
        fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name='PHOT'),
                      fits.ImageHDU(np.concatenate(pix), name='PIXELS'),
                      fits.ImageHDU(pidx.astype(np.int32), name='PIXINDEX'),
                      fits.ImageHDU(self.psf_image.astype(np.float32), name='PSF'),
                      fits.BinTableHDU.from_columns(
                          [fits.Column(name='params', format=f'{len(par)}A', array=np.array([par]))],
                          name='PARAMS')]).writeto(fname, overwrite=True)
        if self.verbose:
            print(f'Saved photometry to {fname}')
        return PhotometryResult(fname)

    # ------------------------------------------------------------------
    def optimise(self, radii=None, psf_radii=None, nblocks=3, block_len=300, segments=None,
                 apply=True):
        """Choose the aperture and PSF-weighting radii by robust point-to-point scatter.

        Uses blocks of consecutive integrations spread over the exposure
        (``segments='all'`` for everything), each normalised by its median.
        """
        if self.psf_image is None:
            self.fit_psf()
        radii = list(radii if radii is not None else [3, 4, 5, 6, 7, 8])
        psf_radii = list(psf_radii if psf_radii is not None else [5, 6, 8, 10, 12])
        src = self.src
        if isinstance(segments, str) and segments == 'all':
            blocks = [(k, slice(0, src.seg_nint[k])) for k in range(src.nseg)]
        else:
            blocks = []
            for i in range(nblocks):
                c = int((i + 0.5) / nblocks * src.nint)
                k = src.segment_of(c)
                n = src.seg_nint[k]
                a = int(np.clip(c - src.seg_start[k] - block_len // 2, 0, max(0, n - block_len)))
                if (k, slice(a, min(n, a + block_len))) not in blocks:
                    blocks.append((k, slice(a, min(n, a + block_len))))
        ap = {r: ([], []) for r in radii}
        ps = {r: ([], []) for r in psf_radii}
        for k, sl in blocks:
            d, e, g, gi = src.read(k, sl)
            for r in radii:
                o = self.extract_chunk(d, e, g, radius=r)
                ap[r][0].append(o['ap'])
                ap[r][1].append(np.sqrt(o['ap_var']))
            for r in psf_radii:
                o = self.extract_chunk(d, e, g, psf_radius=r)
                ps[r][0].append(o['psf'])
                ps[r][1].append(np.sqrt(o['psf_var']))
        res = {'radii': radii, 'psf_radii': psf_radii,
               'aperture': [p2p_noise(*ap[r]) for r in radii],
               'psf': [p2p_noise(*ps[r]) for r in psf_radii]}
        best_ap = radii[int(np.argmin([x[0] for x in res['aperture']]))]
        best_ps = psf_radii[int(np.argmin([x[0] for x in res['psf']]))]
        res['best'] = {'radius': best_ap, 'psf_radius': best_ps}
        self.optimisation = res
        if self.verbose:
            print('  aperture radius : ' + ', '.join(f'{r}: {1e6 * s:.0f} ppm ({q:.2f})'
                                                    for r, (s, q) in zip(radii, res['aperture'])))
            print('  PSF radius      : ' + ', '.join(f'{r}: {1e6 * s:.0f} ppm ({q:.2f})'
                                                    for r, (s, q) in zip(psf_radii, res['psf'])))
        if apply:
            self.params['aperture']['radius'] = best_ap
            self.params['extract']['psf_radius'] = best_ps
        self.log.add('optimise', **{k: v for k, v in res.items()})
        return res


class PhotometryResult:
    """Combined photometry (``<exposure>_photometry.fits``)."""

    def __init__(self, fname):
        self.fname = fname
        with fits.open(fname) as h:
            t = h['PHOT'].data
            self.table = {n: np.asarray(t[n]) for n in t.names}
            self.pixels = np.asarray(h['PIXELS'].data, float)
            self.pixel_index = np.asarray(h['PIXINDEX'].data)
            self.psf = np.asarray(h['PSF'].data)
            self.params = json.loads(h['PARAMS'].data['params'][0])
        self.times = self.table.get('int_mid_BJD_TDB')
        self.flux = {'aperture': self.table['ap'], 'psf': self.table['psf']}
        self.err = {'aperture': np.sqrt(self.table['ap_var']), 'psf': np.sqrt(self.table['psf_var'])}

    def regressors(self):
        """Centroid, width and background time series (for detrending)."""
        return {k: self.table[k] for k in ('xc', 'yc', 'width', 'bkg')}

    def pld_basis(self, npc=None):
        """Pixel-level decorrelation regressors.

        Returns the normalised pixel fractions ``P_hat`` (``nint x npix``) and,
        with ``npc``, the first ``npc`` principal components of them. Fit
        these together with the astrophysical model.
        """
        pix = self.pixels
        phat = pix / np.sum(pix, axis=1, keepdims=True)
        if not npc:
            return phat
        V, ev, pcs = classic_PCA(phat.T)
        return phat, pcs[:npc]


class ApPhoto(object):
    """Helper for aperture photometry on image time-series.

    This class encapsulates common operations required for aperture
    photometry on a data cube of frames (shape ``(N_frames, Ny, Nx)``),
    including centroid estimation, aperture and sky-mask construction,
    simple aperture flux extraction, and pixel-level decorrelation
    (PLD) basis construction.

    Parameters
    ----------
    times : array_like
        1D array of time stamps corresponding to each frame.
    frames : ndarray
        3D data cube with shape ``(N_frames, Ny, Nx)`` containing image
        pixel values for each frame.
    errors : ndarray
        Per-pixel uncertainties with the same shape as ``frames``.
    badpix : ndarray
        Bad-pixel mask, either 2D (shape ``(Ny, Nx)``, the same for all
        frames) or 3D (shape ``(N_frames, Ny, Nx)``, e.g.
        ``CubeCleaner.good_mask``). A value of zero indicates a bad/ignored
        pixel; non-zero indicates a usable pixel. It is stored as a 3D array.
    aprad : float or None, optional
        Aperture radius in pixels for circular aperture photometry.
        Used by :meth:`aperture_mask` and :meth:`simple_aperture_photometry`.
        Ignored when ``brightpix`` is True. Default is ``None``.
    sky_rad1 : float or None, optional
        Inner radius (pixels) of the sky annulus. Used by :meth:`sky_mask`
        and :meth:`simple_aperture_photometry` for background estimation.
        If omitted and ``brightpix`` is False, an empty (zero) mask is returned.
        Default is ``None``.
    sky_rad2 : float or None, optional
        Outer radius (pixels) of the sky annulus. Used by :meth:`sky_mask`
        and :meth:`simple_aperture_photometry` for background estimation.
        Default is ``None``.
    brightpix : bool, optional
        If ``True``, select ``nos_brightest`` brightest pixels to form an aperture 
        instead of circular aperture. Used by :meth:`aperture_mask` and :meth:`sky_mask`. 
        Default ``False``.
    nos_brightest : int, optional
        Number of brightest pixels to include in aperture when ``brightpix=True``.
        Default ``12``.
    nos_faintest : int, optional
        Number of faintest pixels to include in background aperture when it is not None and ``brightpix=True``.
        When None, all pixels that are not in aperture are included. Default is ``None``.
    minmax : dict or None, optional
        Optional spatial bounds for brightest pixel selection with keys
        'rmin', 'rmax', 'cmin', 'cmax'. Default is ``None``.

    Attributes
    ----------
    times, frames, errors, badpix : as above
    aprad, sky_rad1, sky_rad2, brightpix, nos_brightest, minmax : as above
    cen_r, cen_c : ndarray
        Per-frame row/column centroid positions (populated by
        :meth:`find_center`).
    ap_bool_mask_pld : ndarray
        2D aperture boolean mask used by PLD routines.
    Psum, Phat, V, eigenvalues, PCA : ndarray
        Intermediate PLD quantities (pixel-sum, normalized fractions,
        PCA eigenvectors/values and projected time-series) populated by
        :meth:`pixel_level_decorrelation`.

    Main methods
    ------------
    identify_crays(clip, niters)
        Flag cosmic-ray affected pixels and update ``badpix``.
    replace_nan(max_iter)
        Iteratively fill NaNs in ``frames`` from neighboring pixels.
    find_center(rmin,rmax,cmin,cmax)
        Compute center-of-flux centroids per frame.
    simple_aperture_photometry(method, plot, ...)
        Extract aperture photometry using manual, photutils, or median methods.
    pixel_level_decorrelation(removeNan)
        Build normalized pixel level light curves and compute PCA basis for PLD.

    Notes
    -----
    The class stores both raw inputs and intermediate results used by
    subsequent calls; many methods modify the instance in-place and
    also return results for convenience.
    """
    def __init__(self, times, frames, errors, badpix, aprad=None, sky_rad1=None, sky_rad2=None, brightpix=False, nos_brightest=12, nos_faintest=None, minmax=None):
        # The data
        self.times = times
        self.frames = frames
        self.errors = errors
        # 2D maps are broadcast to every frame so that per-frame flags (e.g. cosmic rays) can be added
        self.badpix = np.array(np.broadcast_to(np.asarray(badpix, dtype=float), np.shape(frames)))

        # The aperture information
        self.aprad = aprad
        self.sky_rad1, self.sky_rad2 = sky_rad1, sky_rad2
        self.brightpix = brightpix
        self.nos_brightest = nos_brightest
        self.nos_faintest = nos_faintest
        self.minmax = minmax

        if ( self.aprad == None ) and ( not self.brightpix ):
            raise Exception('You need to define an aperture by providing either aperture radius (aprad)\nOr setting brightpix=True, which will form the aperture from bightest pixels.')
    
    def identify_crays(self, clip=5, niters=5):
        """Identify cosmic-ray affected pixels and update the bad-pixel map
           by comparing each data frame with the median frame.

        Parameters
        ----------
        clip : float, optional
            Sigma threshold used to flag cosmic-ray candidates (default
            ``5``).
        niters : int, optional
            Number of iterative passes to update ``self.badpix`` (default
            ``5``).

        Returns
        -------
        badpix : ndarray
            Updated 2D bad-pixel mask (shape ``(Ny, Nx)``) where flagged
            pixels are set to zero and good pixels are one.
        """
        # Masking bad pixels as NaN

        for _ in range(niters):
            # Flagging bad data as Nan
            frame_new = np.copy(self.frames)
            frame_new[self.badpix == 0.] = np.nan
            
            # Median frame
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median_frame = np.nanmedian(frame_new, axis=0)  # 2D frame
                
                # Creating residuals
                resids = frame_new - median_frame[None,:,:]
                
                # Median and robust (MAD) scatter of the residuals (both are 2D frames)
                med_resid = np.nanmedian(resids, axis=0)
                std_resid = 1.4826 * np.nanmedian(np.abs(resids - med_resid[None]), axis=0)

            # symmetric threshold on the deviation from the median residual
            mask_cr1 = np.abs(resids - med_resid[None]) < clip * std_resid[None]
            self.badpix = mask_cr1*self.badpix

        return self.badpix
    
    def replace_nan(self, max_iter = 50):
        """Fill NaN entries in the data array by the mean of neighbouring pixels.

        On each iteration the array is rolled along each axis to build a
        collection of nearest-neighbour shifts; the mean of these shifted
        arrays is used to replace NaNs. Iteration stops when there are no
        remaining NaNs or when ``max_iter`` is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of fill iterations to perform (default ``50``).

        Returns
        -------
        frames : ndarray
            The data array with NaNs replaced (also modified
            in-place and returned).
        """
        shape = np.append([2*self.frames.ndim], self.frames.shape)
        interp_cube = np.zeros(shape)
        axis = tuple(range(self.frames.ndim))
        shift0 = np.zeros(self.frames.ndim, int)
        shift0[0] = 1
        shift = []     # Shift list will be [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for n in range(self.frames.ndim):
            shift.append(tuple(np.roll(-shift0, n)))
            shift.append(tuple(np.roll(shift0, n)))
        
        for _ in range(max_iter):
            for n in range(2*self.frames.ndim):
                interp_cube[n] = np.roll(self.frames, shift[n], axis = axis)   # interp_cube would be (4, data.shape[0], data.shape[1]) sized array
            with warnings.catch_warnings():                                 # with shifted position in each element (so that we can take its mean)
                warnings.simplefilter('ignore', RuntimeWarning)
                mean_data = np.nanmean(interp_cube, axis=0)
            self.frames[np.isnan(self.frames)] = mean_data[np.isnan(self.frames)]
            if np.sum(np.isnan(self.frames)) == 0:
                break
        
        return self.frames
    
    def plot_median_image(self):
        """This function plots the median image frame
        
        Returns
        -------
        fig : matplotlib.figure.Figure or None
            Figure of median image.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot.
        axs : ndarray or None
            Axes of the figure.
        """
        fig, axs = plt.subplots( figsize=(5,5) )
        im = axs.imshow( np.nanmedian(self.frames, axis=0), interpolation='none' )
        axs.set_title('Median image')

        return fig, im, axs
    
    def find_center(self, rmin=None, rmax=None, cmin=None, cmax=None, plot=False):
        """Compute center-of-flux centroids for each frame in the 3D data cube.

        The centroid for each frame is computed as the center of flux
        (i.e. sum(x*I)/sum(I)) over either the full image or an optional
        subimage defined by the row/column limits ``rmin,rmax,cmin,cmax``.
        Results are stored on the instance as ``self.cen_r`` and
        ``self.cen_c`` and also returned.

        Parameters
        ----------
        rmin, rmax, cmin, cmax : int or None, optional
            Optional integer bounds (rows and columns) defining a subimage
            to use for the centroid calculation. If omitted, the full
            frame is used.
        plot : bool, optional
            If ``True``, produce a plot of centroids as a function of
            time (default ``False``).

        Returns
        -------
        cen_r, cen_c : ndarray
            1D arrays (length ``N_frames``) containing the row and column
            centroid positions (in pixel coordinates) for each frame.
        fig : matplotlib.figure.Figure or None
            Figure of centroids vs time, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.
        """
        self.cen_r, self.cen_c = np.zeros(self.frames.shape[0]), np.zeros(self.frames.shape[0])

        for i in range(len(self.cen_r)):
            # Row is the first index, column is the second index
            # First find the subimage if min & max row, cols are provided
            if (rmin != None)&(rmax != None)&(cmin == None)&(cmax == None):
                subimg = self.frames[i, rmin:rmax, :]
            elif (rmin == None)&(rmax == None)&(cmin != None)&(cmax != None):
                subimg = self.frames[i, :, cmin:cmax]
            elif (rmin != None)&(rmax != None)&(cmin != None)&(cmax != None):
                subimg = self.frames[i, rmin:rmax, cmin:cmax]
            else:
                subimg = np.copy(self.frames[i,:,:])
            
            # Row and column indices
            row_idx, col_idx = np.arange(subimg.shape[0]), np.arange(subimg.shape[1])

            # And now the center of *subimage*
            cen_r_sub = np.nansum(row_idx * np.nansum(subimg, axis=1)) / np.nansum(subimg.flatten())
            cen_c_sub = np.nansum(col_idx * np.nansum(subimg, axis=0)) / np.nansum(subimg.flatten())

            # This was the center of subimage, let's transform that to image coordinates
            if (rmin != None)&(cmin == None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub + rmin, cen_c_sub
            elif (rmin == None)&(cmin != None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub, cen_c_sub + cmin
            elif (rmin != None)&(cmin != None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub + rmin, cen_c_sub + cmin
            else:
                self.cen_r[i], self.cen_c[i] = cen_r_sub, cen_c_sub
        
        if plot:
            fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15,7), sharex=True)

            axs[0].plot(self.times - self.times[0], self.cen_r, 'k-', lw=1.)
            axs[0].set_ylabel('Row')

            axs[1].plot(self.times - self.times[0], self.cen_c, 'k-', lw=1.)
            axs[1].set_xlabel('Time (BJD)')
            axs[1].set_ylabel('Column')

            axs[1].set_xlim([ 0., np.max( self.times - self.times[0] ) ])
        else:
            fig, axs = None, None

        return self.cen_r, self.cen_c, fig, axs
    
    def aperture_mask(self, plot=False):
        """Build a binary aperture mask (and masked frame/error arrays).

        By default a circular aperture is created on the median image using
        the median centroid position computed from ``self.cen_r`` and
        ``self.cen_c``. Alternatively, when ``self.brightpix=True``, the mask is
        formed by selecting the ``self.nos_brightest`` brightest pixels from the
        median image (optionally limited by ``self.minmax`` bounds).

        Parameters
        ----------
        plot : bool, optional
            If ``True``, produce a plot of median image with aperture
            plotted on the top (default ``False``).

        Returns
        -------
        aper_fl_mask : ndarray
            ``self.frames`` multiplied by the 2D aperture boolean mask
            (shape ``(N_frames, Ny, Nx)``).
        aper_err_mask : ndarray
            ``self.errors`` multiplied by the 2D aperture boolean mask.
        ap_bool_msk : ndarray
            2D binary (0/1) mask describing the aperture on a single frame
            (shape ``(Ny, Nx)``).
        fig : matplotlib.figure.Figure or None
            Figure of median image and aperture mask, if ``plot=True``, otherwise ``None``.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        Uses instance attributes ``self.aprad``, ``self.brightpix``,
        ``self.nos_brightest``, and ``self.minmax`` to control aperture
        construction.
        """

        # Median image
        med_img = np.nanmedian(self.frames, axis=0)
        # Now, creating mask
        ap_bool_msk = np.zeros(med_img.shape)
        
        if not self.brightpix:
            # Let's first find a distance array which will contain the distance of each pixels from the center
            idx_arr_r, idx_arr_c = np.meshgrid(np.arange(med_img.shape[0]), np.arange(med_img.shape[1]))
            idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
            
            # Both of above array would be of the dimension of the image array, and we can get row and column index by doing,
            # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

            # Distance array would give distace of each pixel from the center
            distance = np.sqrt(((idx_arr_r - np.nanmedian(self.cen_r))**2) + ((idx_arr_c - np.nanmedian(self.cen_c))**2))

            ap_bool_msk[distance < self.aprad] = 1.

        else:
            # Sorting all pixels values; so that we can select first N bright pixels in the aperture
            med_img_flat = med_img.flatten()
            med_img_sorted = np.sort(med_img_flat)

            # minmax != None
            if self.minmax is not None:
                rmin, rmax = self.minmax['rmin'], self.minmax['rmax']
                cmin, cmax = self.minmax['cmin'], self.minmax['cmax']
            else:
                rmin, rmax = 0, 10000
                cmin, cmax = 0, 10000

            tot_num = True
            ap_pix_nos, i = 0, 0
            while tot_num:
                if np.isnan(med_img_sorted[-1-i]):
                    i = i + 1
                    continue
                else:
                    ind = np.where(med_img == med_img_sorted[-1-i])
                    # To prevent aperture being far from the center
                    if (ind[0][0] > rmax) or (ind[0][0] < rmin) or (ind[1][0] < cmin) or (ind[1][0] > cmax):
                        i = i + 1
                        continue
                    else:
                        ap_bool_msk[ind[0][0], ind[1][0]] = 1.

                        ap_pix_nos = ap_pix_nos + 1
                        if ap_pix_nos >= self.nos_brightest:
                            tot_num = False
                        i = i + 1

        # And, aperture mask
        aper_fl_mask, aper_err_mask = self.frames * ap_bool_msk[None, :, :], self.errors * ap_bool_msk[None, :, :]

        if plot:
            # Figure code
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            im = axs[0].imshow(med_img, interpolation='none')
            axs[0].imshow(ap_bool_msk, alpha=0.5)

            axs[1].imshow(ap_bool_msk, interpolation='none')

            axs[0].set_title('Median image + Aperture mask')
            axs[1].set_title('Aperture mask')
        else:
            fig, im, axs = None, None, None


        return aper_fl_mask, aper_err_mask, ap_bool_msk, fig, im, axs
    
    
    def sky_mask(self, plot=False):
        """Create a sky-annulus mask (and masked frame/error arrays).

        When ``self.brightpix`` is False this builds a circular annulus defined by
        ``self.sky_rad1`` (inner radius) and ``self.sky_rad2`` (outer radius) around 
        the median centroid position. When ``self.brightpix`` is True the function instead
        selects the ``self.nos_brightest`` brightest pixels (optionally limited
        by ``self.minmax``) to define the aperture, and rest of the pixels as background.

        Parameters
        ----------
        plot : bool, optional
            If ``True``, produce a plot of median image with sky mask
            plotted on the top (default ``False``).

        Returns
        -------
        sky_fl_mask : ndarray
            ``self.frames`` multiplied by the 2D sky boolean mask
            (shape ``(N_frames, Ny, Nx)``).
        sky_err_mask : ndarray
            ``self.errors`` multiplied by the 2D sky boolean mask.
        sky_bool_msk : ndarray
            2D binary (0/1) mask describing the sky annulus on a single
            frame (shape ``(Ny, Nx)``).
        fig : matplotlib.figure.Figure or None
            Figure of median image and sky mask, if ``plot=True``, otherwise ``None``.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        Uses instance attributes ``self.sky_rad1``, ``self.sky_rad2``,
        ``self.brightpix``, ``self.nos_brightest``, and ``self.minmax`` to
        control sky mask construction.
        """

        # Median image
        med_img = np.nanmedian(self.frames, axis=0)

        # Now, creating mask
        sky_bool_msk = np.zeros(med_img.shape)

        if ( self.sky_rad1 == None ) and ( self.sky_rad2 == None ) and ( not self.brightpix ):
            pass

        else:
            if ( not self.brightpix ) or ( ( self.sky_rad1 != None ) and ( self.sky_rad2 != None ) ):
                # Let's first find a distance array which will contain the distance of each pixels from the center
                idx_arr_r, idx_arr_c = np.meshgrid(np.arange(med_img.shape[0]), np.arange(med_img.shape[1]))
                idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
                
                # Both of above array would be of the dimension of the image array, i.e, we can get the row and column index by doing.
                # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

                # Distance array would give distace of each pixel from the center
                distance = np.sqrt(((idx_arr_r - np.nanmedian(self.cen_r))**2) + ((idx_arr_c - np.nanmedian(self.cen_c))**2))

                sky_bool_msk[(distance > self.sky_rad1)&(distance < self.sky_rad2)] = 1.
            
            else:
                # Sorting all pixels values; so that we can select first N bright pixels in the aperture
                med_img_flat = med_img.flatten()
                med_img_sorted = np.sort(med_img_flat)

                # minmax != None
                if self.minmax is not None:
                    rmin, rmax = self.minmax['rmin'], self.minmax['rmax']
                    cmin, cmax = self.minmax['cmin'], self.minmax['cmax']
                else:
                    rmin, rmax = 0, 10000
                    cmin, cmax = 0, 10000

                if self.nos_faintest == None:
                    sky_bool_msk = np.ones(med_img.shape)
                    tot_num = True
                    ap_pix_nos, i = 0, 0
                    while tot_num:
                        if np.isnan(med_img_sorted[-1-i]):
                            i = i + 1
                            continue
                        else:
                            ind = np.where(med_img == med_img_sorted[-1-i])
                            # To prevent aperture being far from the center
                            if (ind[0][0] > rmax) or (ind[0][0] < rmin) or (ind[1][0] < cmin) or (ind[1][0] > cmax):
                                i = i + 1
                                continue
                            else:
                                sky_bool_msk[ind[0][0], ind[1][0]] = 0.

                                ap_pix_nos = ap_pix_nos + 1
                                if ap_pix_nos >= self.nos_brightest:
                                    tot_num = False
                                i = i + 1
                else:
                    sky_bool_msk = np.zeros(med_img.shape)
                    tot_num = True
                    ap_pix_nos, i = 0, 0
                    while tot_num:
                        if np.isnan(med_img_sorted[i]):
                            i = i + 1
                            continue
                        else:
                            ind = np.where(med_img == med_img_sorted[i])
                            # To prevent aperture being far from the center
                            if (ind[0][0] < rmax) and (ind[0][0] > rmin) and (ind[1][0] > cmin) and (ind[1][0] < cmax):
                                i = i + 1
                                continue
                            else:
                                sky_bool_msk[ind[0][0], ind[1][0]] = 1.

                                ap_pix_nos = ap_pix_nos + 1
                                if ap_pix_nos >= self.nos_faintest:
                                    tot_num = False
                                i = i + 1

        # And, sky mask
        sky_fl_mask, sky_err_mask = self.frames * sky_bool_msk[None, :, :], self.errors * sky_bool_msk[None, :, :]

        if plot:
            # Figure code
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            im = axs[0].imshow(med_img, interpolation='none')
            axs[0].imshow(sky_bool_msk, alpha=0.5)

            axs[1].imshow(sky_bool_msk, interpolation='none')

            axs[0].set_title('Median image + Bkg mask')
            axs[1].set_title('Bkg mask')
        else:
            fig, im, axs = None, None, None

        return sky_fl_mask, sky_err_mask, sky_bool_msk, fig, im, axs
    
    def simple_aperture_photometry(self, method='manual', plot=False, bkg_corr='median', robust_err=False, **kwargs):
        """Compute aperture photometry for each frame in the data cube.

        The function supports three methods:
        - ``manual``: sums pixels inside a binary aperture mask and subtracts
          the estimated sky background per pixel computed from a sky annulus.
        - ``photutils``: uses ``photutils`` aperture classes to compute
          aperture sums and background statistics per frame (requires
          ``photutils`` to be installed). Additional keyword arguments are
          forwarded to the photutils helpers.
        - ``median``: returns the per-frame median of the aperture pixels
          (useful as a robust estimator or for quick diagnostics).

        Parameters
        ----------
        method : {'manual','photutils','median'}, optional
            Photometry method to use. Default is ``'manual'``.
        plot : bool, optional
            If ``True``, produce a plot of aperture photometry, flux vs time (default ``False``).
        bkg_corr : str, optional
            The method to compute sky background flux: ``'median'`` (default;
            3-sigma clipped median of the good sky pixels) or ``'mean'``.
        robust_err : bool, optional
            Computes robust errors on aperture photometry by including scatter in sky background flux.
            Default is ``False``.
        **kwargs : dict
            Additional keyword arguments forwarded to ``photutils``
            routines when ``method='photutils'`` (for example, custom
            aperture or statistic options).

        Returns
        -------
        ap_flux : ndarray
            1D array (length N_frames) with the background-subtracted
            aperture flux for each frame (same units as ``frames``).
        ap_flux_err : ndarray
            1D array (length N_frames) with the estimated uncertainty on
            the aperture flux for each frame.
        tot_sky_bkg : ndarray
            1D array (length N_frames) with the total sky background
            contribution (summed over aperture pixels) subtracted from
            each frame. For methods that do not estimate a sky background
            this will be zeros.
        fig : matplotlib.figure.Figure or None
            Figure of aperture photometry vs time, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        - The function expects centroids to be available as
          ``self.cen_r`` and ``self.cen_c`` (one per frame) before being
          called.
        - Uses instance attributes ``self.aprad``, ``self.sky_rad1``,
          ``self.sky_rad2``, ``self.brightpix``, ``self.nos_brightest``,
          and ``self.minmax`` to control aperture and background extraction.
        - When using ``photutils``, this routine constructs
          ``CircularAperture``/``CircularAnnulus`` objects per frame and
          uses ``ApertureStats`` and ``aperture_photometry``; errors are
          estimated combining aperture sum errors and annulus statistics
          following standard propagation.
        """
        
        if method == 'manual':
            # Let's first obtain sky background flux per pixel (If sky radii are not None)
            sky_fl_mask, sky_err_mask, sky_bool_mask, _, _, _ = self.sky_mask()
            # Now, the aperture flux
            aper_fl_mask, aper_err_mask, ap_bool_mask, _, _, _ = self.aperture_mask()

            ## Expression of Sky flux per pixel and sky error per pixel
            if np.nansum(sky_bool_mask) != 0:
                # good sky pixels of every frame (bad pixels / cosmic rays excluded)
                sky_px = (sky_bool_mask[None] == 1) & (self.badpix != 0)
                skyvals = np.where(sky_px, self.frames, np.nan)
                skyerr2 = np.where(sky_px, self.errors ** 2, np.nan)
                # 3-sigma clipping (MAD) of every frame's sky pixels
                for _ in range(3):
                    med = np.nanmedian(skyvals, axis=(1, 2))
                    sig = 1.4826 * np.nanmedian(np.abs(skyvals - med[:, None, None]), axis=(1, 2))
                    out = np.abs(skyvals - med[:, None, None]) > 3 * sig[:, None, None]
                    if not out.any():
                        break
                    skyvals[out] = np.nan
                    skyerr2[out] = np.nan
                nsky = np.sum(np.isfinite(skyvals), axis=(1, 2))
                # --------------------- First, calculate the sky flux per pix
                if bkg_corr == 'mean':
                    sky_flx_per_pix = np.nanmean(skyvals, axis=(1, 2))
                    fac = 1.
                elif bkg_corr == 'median':
                    sky_flx_per_pix = np.nanmedian(skyvals, axis=(1, 2))
                    fac = np.sqrt(np.pi / 2)        # efficiency of a median
                else:
                    raise Exception('The background can be computed either using mean or median.')

                # --------------------- And, now, errors on the sky flux per pix
                sky_flx_err_per_pix = fac * np.sqrt(np.nansum(skyerr2, axis=(1, 2))) / np.maximum(nsky, 1)
                if robust_err:
                    sky_flx_err_per_pix = np.sqrt(sky_flx_err_per_pix ** 2 +
                                                  (np.nanstd(skyvals, axis=(1, 2)) / np.sqrt(np.maximum(nsky, 1))) ** 2)
            else:
                sky_flx_per_pix, sky_flx_err_per_pix = 0., 0.

            tot_sky_bkg = np.nansum(ap_bool_mask)*sky_flx_per_pix
            ap_flux = np.nansum(aper_fl_mask, axis=(1,2)) - tot_sky_bkg
            ap_flux_err = np.sqrt( np.nansum( aper_err_mask**2, axis=(1,2) ) + ( (np.nansum(ap_bool_mask) * sky_flx_err_per_pix)**2 ) )

        elif method == 'photutils':

            ap_flux, ap_flux_err, tot_sky_bkg = np.zeros( self.frames.shape[0] ), np.zeros( self.frames.shape[0] ), np.zeros( self.frames.shape[0] )

            for t in range(len(ap_flux)):
                # First let's perform the background subtraction
                if (self.sky_rad1 == None) and (self.sky_rad2 == None):
                    bkg_mean, bkg_std = 0., 0.
                else:
                    sky_aper = CircularAnnulus((int(self.cen_c[t]), int(self.cen_r[t])), r_in=self.sky_rad1, r_out=self.sky_rad2)
                    sky_aperstats = ApertureStats(self.frames[t,:,:], sky_aper, **kwargs)
                    if bkg_corr == 'mean':
                        bkg_mean, bkg_std = sky_aperstats.mean, sky_aperstats.std   # Mean sky background per pixel
                    elif bkg_corr == 'median':
                        bkg_mean, bkg_std = sky_aperstats.median, sky_aperstats.std   # Mean sky background per pixel
                    else:
                        raise Exception('The background can be computed either using mean or median.')
                
                # Now computing the aperture flux
                circ_aper = CircularAperture((self.cen_c[t], self.cen_r[t]), r=self.aprad)
                ap_phot = aphot(data=self.frames[t,:,:], apertures=circ_aper, error=self.errors[t,:,:], **kwargs)
                total_sky_bkg = bkg_mean * circ_aper.area_overlap(self.frames[t,:,:], **kwargs)
                phot_bkgsub = ap_phot['aperture_sum'] - total_sky_bkg  # Background subtraction

                ## Error estimation in background subtracted photometry
                phot_bkgsub_err = np.sqrt( (ap_phot['aperture_sum_err']**2) + ( (circ_aper.area_overlap(self.frames[t,:,:], **kwargs) * bkg_std)**2 ) )

                ## Results
                ap_flux[t], ap_flux_err[t] = phot_bkgsub[0], phot_bkgsub_err[0]
                tot_sky_bkg[t] = total_sky_bkg

        elif method == 'median':
            aper_fl_mask, aper_err_mask, ap_bool_mask, _, _, _ = self.aperture_mask()
            tot_sky_bkg = np.zeros(aper_fl_mask.shape[0])
            aper_fl_mask[aper_fl_mask == 0.] = np.nan
            ap_flux, ap_flux_err = np.nanmedian(aper_fl_mask, axis=(1,2)), np.sqrt( np.nanmedian( aper_fl_mask, axis=(1,2) ) )
        
        else:
            raise Exception('Please enter correct method...\nMethod can either be manual, photutils, or median.')
        
        if plot:
            fig, axs = plt.subplots()
            axs.errorbar(self.times - self.times[0], ap_flux/np.nanmedian(ap_flux), yerr=ap_flux_err/np.nanmedian(ap_flux), fmt='.', color='dodgerblue')
            
            axs.set_xlim( [0., np.max(self.times - self.times[0]) ] )
            axs.set_xlabel( 'Time [BJD - {:.2f}]'.format(self.times[0]) )
            axs.set_ylabel('Relative flux')
        else:
            fig, axs = None, None
            
        return ap_flux, ap_flux_err, tot_sky_bkg, fig, axs
    
    def growth_function(self, rmin, rmax, noise='p2p', plot=False, **kwargs):
        """Compute the growth curve (flux vs aperture radius) and noise.

        Parameters
        ----------
        rmin : int
            Minimum aperture radius (inclusive) in pixels.
        rmax : int
            Maximum aperture radius (exclusive) in pixels.
        noise : {'p2p','pipe','std','rms','astropy'}, optional
            Method used to estimate scatter for each aperture radius.
            Default is 'p2p' (robust point-to-point scatter, insensitive to
            transits / eclipses).
        plot : bool, optional
            If ``True``, produce a plot of the growth function (default ``False``).
        **kwargs : dict
            Additional keyword arguments forwarded to
            ``simple_aperture_photometry`` (for example sky annulus
            parameters).

        Returns
        -------
        ap_flux_radii : ndarray
            Median aperture flux for each radius in the tested range.
        noise_radii : ndarray
            Estimated noise metric (in ppm) for each radius.
        fig : matplotlib.figure.Figure or None
            Figure of the growth function, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.
        """
        radii = np.arange(rmin, rmax, 1)
        ap_flux_radii, noise_radii = np.zeros(len(radii)), np.zeros(len(radii))

        # Choosing the method to compute noise
        if noise == 'p2p':
            noise_func = p2p_sigma
        elif noise == 'pipe':
            noise_func = pipe_mad
        elif noise == 'std':
            noise_func = np.nanstd
        elif noise == 'rms':
            noise_func = rms
        elif noise == 'astropy':
            noise_func = lambda x: mad_std(x, ignore_nan=True)
        else:
            raise ValueError("Method should be one of 'p2p', 'pipe', 'std', 'rms', or 'astropy'.")
        
        for r in tqdm(range(len(radii))):
            if not self.brightpix:
                self.aprad = radii[r]
            else:
                self.nos_brightest = radii[r]
            ap_fl, _, _, _, _ = self.simple_aperture_photometry(**kwargs)
            ap_flux_radii[r] = np.nanmedian( ap_fl )
            noise_radii[r] = noise_func( ap_fl / np.nanmedian( ap_fl ) ) * 1e6

        if plot:
            fig, axs1 = plt.subplots()

            color1 = 'orangered'
            axs1.plot(radii, ap_flux_radii, color=color1)
            axs1.set_xlabel('Radius')
            axs1.set_ylabel('Growth function', color=color1)
            axs1.tick_params(axis='y', which='both', color=color1,  labelcolor=color1)
            
            color2 = 'royalblue'
            axs2 = axs1.twinx()
            axs2.plot(radii, noise_radii, color=color2)
            axs2.axvline(radii[np.argmin(noise_radii)], color='dimgrey', ls='--', lw=1.)
            axs2.set_ylabel('Median Absolute Deviation [ppm]', color=color2, rotation=270, labelpad=25)
            
            axs2.tick_params(axis='y', which='both', color=color2, labelcolor=color2)
            axs2.spines['right'].set_color(color2)
            axs2.spines['left'].set_color(color1)

            axs1.set_xlim([ np.min(radii), np.max(radii) ])

            fig.tight_layout()
        else:
            fig, axs1, axs2 = None, None, None

        return ap_flux_radii, noise_radii, fig, [axs1, axs2]
    
    def pixel_level_decorrelation(self, removeNan=False):
        """Prepare pixel-level decorrelation (PLD) basis functions.

        The method builds the normalized pixel-level light curves (Phat) inside
        the chosen aperture and performs a PCA (principal componene analysis) 
        on them to produce a set of orthogonal basis vectors (``self.PCA``) with 
        singular values and eigenvectors stored on the instance.

        Parameters
        ----------
        removeNan : bool, optional
            If ``True``, remove frames that contain NaNs in the basis
            before performing PCA.

        Returns
        -------
        V : ndarray
            Matrix of eigenvectors from the PCA (shape: n_components, n_pixels).
        eigenvalues : ndarray
            Eigenvalues corresponding to each PCA component.
        PCA : ndarray
            Projected PCA time-series (shape: n_components, n_frames).
        """
        _, _, self.ap_bool_mask_pld, _, _, _ = self.aperture_mask()
        sky_fl, _, sky_bool, _, _, _ = self.sky_mask()

        if np.nansum(sky_bool) == 0.:
            sky_bkg_per_pix = 0.
        else:
            sky_bkg_per_pix = np.nansum(sky_fl, axis=(1,2)) / np.nansum(sky_bool)

        self.idxr, self.idxc = np.where(self.ap_bool_mask_pld == 1)

        pixel_fluxes = np.zeros( ( self.frames.shape[0], int(np.sum(self.ap_bool_mask_pld)) ) )
        for r in range(len(self.idxr)):
            pixel_fluxes[:,r] = self.frames[:, int(self.idxr[r]), int(self.idxc[r])] - sky_bkg_per_pix
        
        # Calculating Phat
        self.Psum = np.nansum( pixel_fluxes, axis=1 )
        self.Phat = pixel_fluxes / self.Psum[:, None]

        if removeNan:
            id0, _ = np.where(np.isnan(self.Phat))
            self.idx_nan = np.ones(self.Phat.shape[0], dtype=bool)
            self.idx_nan[id0] = False
            
            self.Phat = self.Phat[self.idx_nan, :]
            self.times = self.times[self.idx_nan]
            self.Psum = self.Psum[self.idx_nan]

            self.idx_nan = np.ones(self.Phat.shape[0], dtype=bool)
            id0 = np.where( self.Psum == 0. )
            self.idx_nan[id0] = False
            self.Phat, self.Psum = self.Phat[self.idx_nan, :], self.Psum[self.idx_nan]
            self.times = self.times[self.idx_nan]

        else:
            self.idx_nan = np.ones(self.Phat.shape[0], dtype=bool)

        self.V, self.eigenvalues, self.PCA = classic_PCA(self.Phat.T)

        return self.V, self.eigenvalues, self.PCA
    
    def plot_correlation_matrices(self):
        """Plot correlation matrices before and after PCA on pixel-level light curves.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the two correlation matrix subplots.
        axs : ndarray
            Array of two Axes objects for the before/after PCA plots.
        im_list : list
            List with the two AxesImage objects (useful for colorbars).
        """
        # Calculating PCA correlation matrix
        CorrelationMatrix = np.abs(np.corrcoef(self.Phat.T))
        PCACorrelationMatrix = np.abs(np.corrcoef(self.PCA))

        from matplotlib import rcParams
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        
        # Actual figure code
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
        
        im1 = axs[0].imshow(CorrelationMatrix, cmap='magma', vmin=0, vmax=1)
        im2 = axs[1].imshow(PCACorrelationMatrix, cmap='magma', vmin=0, vmax=1)
        
        axs[0].set_xlabel('Element $i$')
        axs[1].set_xlabel('Element $i$')
        
        axs[0].set_ylabel('Element $j$')

        axs[0].set_title('Before PCA')
        axs[1].set_title('After PCA')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.015, 0.75])
        fig.colorbar(im2, cax=cbar_ax)

        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'

        return fig, axs, [im1, im2]
    
    def plot_eigenvectors(self):
        """Visualize the first 10 PCA eigenvectors as spatial maps.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the eigenvector images.
        axs : ndarray
            Array of Axes for each eigenvector subplot.
        """
        comps = 0

        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 5), sharex=True, sharey=True)
        for r in range(axs.shape[0]):
            for c in range(axs.shape[1]):
                eg_vec = np.zeros(self.ap_bool_mask_pld.shape)
                for k in range(len(self.idxr)):
                    eg_vec[self.idxr[k], self.idxc[k]] = self.V[comps, k]
                im = axs[r,c].imshow(eg_vec, interpolation='none', cmap='plasma')
                im.set_clim([-0.22, 0.22])
                axs[r,c].set_xlim([np.min(self.idxc)-3, np.max(self.idxc)+3])
                axs[r,c].set_ylim([np.min(self.idxr)-3, np.max(self.idxr)+3])
                axs[r,c].text(np.max(self.idxc)+1.5, np.max(self.idxr)+2, comps+1, fontweight='bold')#, backgroundcolor='white')
                comps = comps + 1
        fig.suptitle('First 10 eigenvectors from PCA analysis', fontsize=16)

        return fig, axs
    
    def plot_pcs(self, nmax=10):
        """Plot the PC time-series (principal components) up to `nmax`.

        Parameters
        ----------
        nmax : int, optional
            Number of principal components to plot (default 10).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the stacked PC plots.
        axs : ndarray
            Array of Axes objects, one per plotted PC.
        """
        fig, axs = plt.subplots(nrows=int(nmax), ncols=1, figsize=(15,10), sharex=True)
        for i in range(int(nmax)):
            axs[i].plot(self.times - self.times[0], self.PCA[i,:], 'k-', lw=1.)
            axs[i].set_ylabel('PC ' + str(i+1))
            med, std = np.nanmedian(self.PCA[i,:]), mad_std(self.PCA[i,:])
            axs[i].set_ylim([med-4*std, med+4*std])
        plt.xlabel('Time (BJD - {:.4f})'.format(self.times[0]))

        return fig, axs
    
    def pld_correction(self, pc_max=5, extra_vectors=None, fit_mask=None, plot=False):
        """Apply PLD correction using the first `pc_max` principal components.

        Parameters
        ----------
        pc_max : int, optional
            Number of PCA components to include in the PLD fit (default 5).
        extra_vectors : ndarray or None, optional
            Additional vectors to include in the PLD fit (shape: either 1D array 
            of length n_frames or 2D array of shape (n_extra, n_frames)).
            If provided, these will be appended to the PCA basis before fitting.
            Default is ``None`` (no extra vectors).
        fit_mask : ndarray of bool or None, optional
            Frames used to fit the coefficients (e.g. out of transit /
            eclipse). Without it the regression can absorb part of the
            astrophysical signal; for science, fit the PLD regressors jointly
            with the transit / eclipse model instead. Default ``None`` (all).
        plot : bool, optional
            If ``True``, produce a diagnostic plot comparing the raw and
            PLD-predicted flux (default ``False``).

        Returns
        -------
        Psum : ndarray
            Sum of pixel fluxes (SAP flux) before correction.
        prediction : ndarray
            PLD model prediction for the flux (same length as ``Psum``).
        fig : matplotlib.figure.Figure or None
            Diagnostic figure if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the diagnostic figure if produced, otherwise ``None``.
        """
        if extra_vectors is not None:
            # Ensure extra_vectors has the expected dimensionality (2D: n_extra, n_frames)
            extra_vectors = np.asarray(extra_vectors)
            if extra_vectors.ndim == 1:
                # Interpret a 1D array as a single extra vector
                extra_vectors = extra_vectors.reshape(1, -1)
            elif extra_vectors.ndim != 2:
                raise ValueError("extra_vectors must be a 1D or 2D array-like object.")
            
            if extra_vectors.shape[1] != self.PCA.shape[1]:
                raise ValueError("extra_vectors should have the same number of columns (frames) as PCA.")
            X = np.vstack([ np.ones(len(self.Psum)), self.PCA[0:pc_max,:], extra_vectors ])
        else:
            X = np.vstack([ np.ones(len(self.Psum)), self.PCA[0:pc_max,:] ])
        
        # Fit:
        fm = np.ones(len(self.Psum), bool) if fit_mask is None else np.asarray(fit_mask, bool)
        result = np.linalg.lstsq(X.T[fm], self.Psum[fm], rcond=None)
        coeffs = result[0]
        prediction = np.dot(coeffs, X)

        print('>>> --- p2p scatter of uncorrected light curve: {:.2f} ppm'.format(p2p_sigma(self.Psum/np.median(self.Psum))*1e6))
        print('>>> --- p2p scatter of PLD corrected light curve: {:.2f} ppm'.format(p2p_sigma(self.Psum/prediction)*1e6))

        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(15/1.5,10/1.5), sharex=True, sharey=True)

            axs[0].errorbar(self.times-self.times[0], self.Psum/np.median(self.Psum), fmt='.', c='orangered', label='SAP Flux')
            axs[0].plot(self.times-self.times[0], prediction/np.median(self.Psum), c='darkgreen', label='PLD prediction',alpha=0.7, zorder=10)
            axs[0].set_ylabel('Relative Flux')
            axs[0].legend(loc='best')
            axs[0].grid()

            axs[1].errorbar(self.times-self.times[0], self.Psum/prediction, fmt='.', c='cornflowerblue')
            axs[1].set_xlabel('Time [BJD - {:.2f}]'.format(self.times[0]))
            axs[1].set_ylabel('Relative Flux')
            axs[1].grid()

            axs[1].set_xlim([ 0., np.max(self.times-self.times[0]) ])

        else:
            fig, axs = None, None

        return self.Psum, prediction, fig, axs
    
    def noise_with_pcs(self, pc_max, noise='p2p', plot=False):
        """Estimate noise (MAD or other) as a function of included PCs.

        Parameters
        ----------
        pc_max : int
            Maximum number of principal components to test (the method will
            evaluate values from 1 to ``pc_max-1``).
        noise : {'p2p','pipe','std','rms','astropy'}, optional
            Noise estimator to use (default 'p2p', robust point-to-point
            scatter).
        plot : bool, optional
            If ``True``, display a plot of noise metric vs number of PCs.

        Returns
        -------
        pcs_to_include : ndarray
            Array of numbers of PCs tested (1..pc_max-1).
        mad_with_pcs : ndarray
            Noise metric computed for each tested number of PCs.
        fig : matplotlib.figure.Figure or None
            Figure object if ``plot=True``, else ``None``.
        axs : matplotlib.axes.Axes or None
            Axis object for the plot if produced, else ``None``.
        """
        # Choosing the method to compute noise
        if noise == 'p2p':
            noise_func = p2p_sigma
        elif noise == 'pipe':
            noise_func = pipe_mad
        elif noise == 'std':
            noise_func = np.nanstd
        elif noise == 'rms':
            noise_func = rms
        elif noise == 'astropy':
            noise_func = lambda x: mad_std(x, ignore_nan=True)
        else:
            raise ValueError("Method should be one of 'p2p', 'pipe', 'std', 'rms', or 'astropy'.")
        
        pcs_to_include = np.arange(1, pc_max)
        mad_with_pcs = np.zeros(len(pcs_to_include))

        for p in tqdm(range(len(pcs_to_include))):
            # First compute the prediction
            x_pc = np.vstack(( np.ones(len(self.Psum)), self.PCA[0:int(pcs_to_include[p]),:] ))
            res_pc = np.linalg.lstsq(x_pc.T, self.Psum, rcond=None)
            co_pc = res_pc[0]
            pred_pc = np.dot(co_pc, x_pc)

            # Computing the MAD
            mad_with_pcs[p] = noise_func( self.Psum / pred_pc ) * 1e6

        # Number of PCs with minimum noise
        n_pc_of_min_scat = pcs_to_include[np.argmin(mad_with_pcs)]

        if plot:
            fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
            axs.plot(pcs_to_include, mad_with_pcs, c='orangered')
            axs.axvline(n_pc_of_min_scat, color='dimgrey', ls='--', lw=1.)
            
            axs.set_xlabel('Number of PCs included')
            axs.set_ylabel('Median Absolute Deviation [ppm]')

            axs.set_xlim([ np.min(pcs_to_include), np.max(pcs_to_include) ])
            plt.grid()
        else:
            fig, axs = None, None

        return pcs_to_include, mad_with_pcs, fig, axs
