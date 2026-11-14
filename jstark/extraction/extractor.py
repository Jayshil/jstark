"""Spectral extraction of a whole exposure, one segment at a time.

Typical use::

    from jstark.extraction import SpectralExtractor

    ex = SpectralExtractor('/data/NRS1/reduction/<exposure>_cleaninfo.fits')
    ex.show()                                   # parameters of every step
    ex.set('psf', oversample=3, knot_col=30)    # change any of them
    ex.find_trace()                             # static trace + column range
    ex.fit_psf()                                # one streaming pass over all integrations
    ex.optimise(apertures=[3, 4, 5, 6], bkg_inner=[7, 9])   # optional
    spec = ex.run()                             # extract every segment, combine

The exposure is never stitched. :meth:`fit_psf` accumulates the PSF
least-squares problem chunk by chunk (memory independent of the number of
integrations), and :meth:`run` extracts each segment separately, writing
``<root>-segNNN_<det>_spec.fits``; only the small spectra are combined into
``<exposure>_spectra.fits``.
"""
import copy
import json
import os
import time
import warnings

import numpy as np
from astropy.io import fits

from ..reduction.io import StepLog, output_filename
from ..reduction.robust import mad_sigma
from .background import background_region, estimate_background
from .bspline import NormalSums
from .config import get_defaults
from .contamination import load_detection_image, find_sources
from .extract import optimal, box, refine_shift
from .metrics import p2p_noise, white_light
from .psf import SplinePSF
from .source import ExposureSource
from .trace import (Trace, find_trace, trace_from_reference, fill_bad, window, edge_weights,
                    cof_offsets, reference_cof, _gather, _sample_spatial)

__all__ = ['SpectralExtractor', 'Spectra']


class SpectralExtractor:
    """Extract the spectral time series of an exposure cleaned by jstark.

    Parameters
    ----------
    info : str or ExposureCleaner
        ``<exposure>_cleaninfo.fits`` written by
        :class:`~jstark.reduction.ExposureCleaner`.
    outdir : str or None
        Output directory (default: the directory of the cleaned files).
    orders : list of int or None
        Spectral orders (default: ``[1]``, or ``[1, 2]`` for NIRISS/SOSS).
    max_mem_gb : float
        Memory budget for one chunk of integrations.
    verbose : bool
    """

    def __init__(self, info, outdir=None, orders=None, max_mem_gb=2., verbose=True):
        self.src = ExposureSource(info, max_mem_gb=max_mem_gb)
        if not self.src.config.spectroscopic:
            raise ValueError(f'{self.src.mode} is an imaging mode; use jstark.extraction.Photometry.')
        self.mode = self.src.mode
        self.params, default_orders = get_defaults(self.mode)
        self.orders = list(orders or default_orders)
        self.order_params = {o: {} for o in self.orders}
        self.outdir = outdir or self.src.outdir
        os.makedirs(self.outdir, exist_ok=True)
        wf = os.path.join(self.src.outdir, f'{self.src.exposure}_wavelength.fits')
        self.wavelength_file = wf if os.path.isfile(wf) else None
        self.verbose = verbose
        self.log = StepLog(os.path.join(self.outdir, f'{self.src.exposure}_extract_log.json'))
        self.traces, self.psfs = {}, {}
        self.var_mean = None
        self.qmap = None
        self.cov = {}
        self.bkg_mask = self.bkg_ok = None
        self.shat, self.ref = {}, {}
        # field-star contamination (see find_contaminants / add_field_trace)
        self._badpix0 = self.src.badpix.copy()
        self.contam_mask = None
        self.contaminants = []
        self.field_traces = []
        self.field_model = None
        self.outputs = {}
        self.optimisation = None
        self.diagnostics = {}
        if verbose:
            print(f'SpectralExtractor: {self.mode} | {self.src.exposure} | {self.src.nseg} segment(s), '
                  f'{self.src.nint} integrations | orders {self.orders}')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def p(self, step, order=None):
        """Parameters of ``step`` (with the overrides of ``order``)."""
        out = dict(self.params[step])
        if order is not None:
            out.update(self.order_params.get(order, {}).get(step, {}))
        return out

    def set(self, step, order=None, **kwargs):
        """Change parameters of a step (for one ``order`` or all).

        Changing ``columns`` or ``trace`` resets the trace and the PSF;
        changing ``psf`` resets the PSF. The PSF is *not* refitted
        automatically after a ``background`` change.
        """
        if step not in self.params:
            raise KeyError(f'Unknown step {step!r}; steps: {list(self.params)}')
        unknown = set(kwargs) - set(self.params[step])
        if unknown:
            raise KeyError(f'Unknown parameter(s) {sorted(unknown)} for {step!r}; '
                           f'known: {sorted(self.params[step])}')
        if order is None:
            self.params[step].update(kwargs)
        else:
            self.order_params.setdefault(order, {}).setdefault(step, {}).update(kwargs)
        if step in ('columns', 'trace'):
            self.traces, self.psfs = {}, {}
            self.field_model = None
        if step == 'psf':
            self.psfs = {}
        if step in ('background', 'columns', 'trace'):
            self.bkg_mask = None
        return self

    def show(self):
        """Print the current parameters."""
        print(f'{self.mode} | orders {self.orders}')
        for step, kw in self.params.items():
            print(f'  {step:<11s} ' + ', '.join(f'{k}={v}' for k, v in kw.items()))
        for o, steps in self.order_params.items():
            for step, kw in steps.items():
                if kw:
                    print(f'  order {o} {step}: ' + ', '.join(f'{k}={v}' for k, v in kw.items()))

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------
    def _wavelength(self, order):
        w = self.src.wavelength
        if w is None:
            return None
        if w.ndim == 3:
            return w[order - 1] if order - 1 < len(w) else None
        return w

    def _reference_traces(self):
        """Reference trace positions ``{order: y(x)}`` from the Stage 2 wavelength file."""
        if self.wavelength_file is None:
            raise FileNotFoundError('No Stage 2 wavelength file with reference traces found; '
                                    'set trace reference=None.')
        out = {}
        nspec = self.src.shape[1]
        with fits.open(self.wavelength_file) as h:
            for hdu in h:
                if hdu.name.startswith('TRACE_O'):
                    o = int(hdu.name[7:])
                    xt, yt = np.asarray(hdu.data[0], float), np.asarray(hdu.data[1], float)
                    k = np.argsort(xt)
                    x = np.arange(nspec)
                    y = np.interp(x, xt[k], yt[k], left=np.nan, right=np.nan)
                    out[o] = y
        return out

    def find_trace(self):
        """Static trace and column range of every order (from the exposure median)."""
        src = self.src
        good = src.badpix
        std = src.cr_sigma if src.cr_sigma is not None else src.std
        for o in self.orders:
            pt, pc, pp = self.p('trace', o), self.p('columns', o), self.p('psf', o)
            hw = pt['halfwidth'] or max(3., 0.5 * pp['halfwidth'])
            wl = self._wavelength(o)
            kw = dict(halfwidth=hw, std=std, wavelength=wl, snr_min=pc['snr_min'],
                      min_good_frac=pc['min_good_frac'], nknots=pt['nknots'], clip=pt['clip'],
                      order=o, gap=pc['gap'])
            cols = pc['columns']
            if cols is None and pc['wavelength_range'] is not None and wl is not None:
                tr0 = find_trace(src.median, good, columns=(0, src.shape[1]), **kw)
                lam = _sample_spatial(wl, np.interp(np.arange(src.shape[1]), tr0.cols, tr0.y0))
                lo, hi = pc['wavelength_range']
                ok = (lam >= lo) & (lam <= hi) & tr0.info['valid_all']
                if not ok.any():
                    raise ValueError(f'No valid column in wavelength range {pc["wavelength_range"]}.')
                idx = np.where(ok)[0]
                cols = (idx.min(), idx.max() + 1)
            if pt.get('reference'):
                refs = self._reference_traces()
                tr = trace_from_reference(
                    src.median, good, refs[o], others=[v for k, v in refs.items() if k != o],
                    halfwidth=hw, std=std, wavelength=wl, columns=cols, snr_min=pc['snr_min'],
                    min_good_frac=pc['min_good_frac'], ref_order=pt['ref_order'], clip=pt['clip'],
                    order=o, gap=pc['gap'])
            else:
                tr = find_trace(src.median, good, columns=cols, **kw)
            if wl is not None:
                tr.info['wavelength'] = _sample_spatial(wl[:, tr.cols], tr.y0)
            self.traces[o] = tr
            if self.verbose:
                lam = tr.info.get('wavelength')
                ltxt = (f', {np.nanmin(lam):.3f}-{np.nanmax(lam):.3f} um' if lam is not None
                        and np.isfinite(lam).any() else '')
                print(f'  order {o}: columns {tr.cols[0]}-{tr.cols[-1]}{ltxt}; trace rows '
                      f'{tr.y0.min():.2f}-{tr.y0.max():.2f}; centroid rms about the fit '
                      f'{tr.info["rms"]:.3f} px')
            self.log.add('find_trace', order=o, columns=[int(tr.cols[0]), int(tr.cols[-1]) + 1],
                         halfwidth=hw, rms=tr.info['rms'])
        self.bkg_mask = None
        self.psfs = {}
        return self.traces

    # ------------------------------------------------------------------
    # Background and per-chunk preparation
    # ------------------------------------------------------------------
    def _ensure_background_region(self):
        if self.bkg_mask is not None:
            return
        if not self.traces:
            self.find_trace()
        pb = self.p('background')
        avoid = list(self.traces.values()) + self._field_trace_objs()
        if any(self.p('trace', o).get('reference') for o in self.orders):
            nspat, nspec = self.src.shape
            for k, y in self._reference_traces().items():
                if k in self.traces:
                    continue
                ok = np.isfinite(y) & (y > -10) & (y < nspat + 10)
                if ok.sum() > 1:
                    cols = np.where(ok)[0]
                    avoid.append(Trace(order=k, cols=cols, y0=y[cols]))
        self.bkg_mask, self.bkg_ok = background_region(
            self.src.shape, avoid, pb['inner'], pb['outer'],
            good=self.src.badpix, min_pixels=pb['min_pixels'])
        # reference quantities from the background-subtracted median frame
        med = self.src.median
        if self.contam_mask is not None and self.contam_mask.any():
            # masked contaminants: interpolate along the dispersion (the trace is smooth there)
            med = _fill_along_dispersion(med, self.contam_mask, self._badpix0)
        med = fill_bad(med, self._badpix0)[None]
        if self.field_model is not None:
            med = med - self.field_model[None]
        bk, _ = self._background(med, np.ones_like(med), np.ones(med.shape, bool))
        med = med.copy()
        self._subtract(med, bk)
        self._median_bsub = med[0]
        for o, tr in self.traces.items():
            h = self.p('psf', o)['halfwidth']
            y = tr.y(dy=[0.])
            rows, u, inside = window(y, self.src.shape[0], h)
            D = _gather(med, rows, tr.cols)
            self.shat[o] = np.sum(np.where(inside, edge_weights(u, h) * D, 0.), 1)[0]
            hc = self.p('trace', o)['halfwidth'] or max(3., 0.5 * h)
            cof, wsum = reference_cof(med[0], tr, hc)
            w = np.clip(tr.snr, 0, None) ** 2 * np.isfinite(cof)
            w = w * np.asarray(tr.info.get('clean', True), float)
            self.ref[o] = (hc, np.nan_to_num(cof), w)

    def _bkg_cols(self):
        lo = min(tr.cols[0] for tr in self.traces.values())
        hi = max(tr.cols[-1] for tr in self.traces.values()) + 1
        return np.arange(lo, hi)

    def _background(self, data, err, good):
        pb = self.p('background')
        cols = self._bkg_cols()
        return estimate_background(data, err, good, self.bkg_mask, method=pb['method'],
                                   nsigma=pb['nsigma'], maxiters=pb['maxiters'],
                                   poly_order=pb['poly_order'], cols=cols, ok=self.bkg_ok)

    def _subtract(self, data, bkg):
        cols = self._bkg_cols()
        if bkg.ndim == 2:
            data[:, :, cols] -= bkg[:, None, :].astype(data.dtype)
        else:
            data[:, :, cols] -= bkg.astype(data.dtype)

    def _prepare(self, data, err, good):
        """Background-subtract a chunk (in place) and measure centre-of-flux offsets.

        A static model of field-star traces (:meth:`add_field_trace`) is
        subtracted first.
        """
        if self.field_model is not None:
            data -= self.field_model[None].astype(data.dtype)
        bkg, vb = self._background(data, err, good)
        self._subtract(data, bkg)
        c0 = self._bkg_cols()[0]
        out = {}
        for o, tr in self.traces.items():
            hc, ref, w = self.ref[o]
            dy = cof_offsets(data, tr, hc, ref, w)
            sl = tr.cols - c0
            b = bkg[:, sl] if bkg.ndim == 2 else bkg[:, :, sl]
            out[o] = dict(dy=dy, var_bkg=vb[:, sl], bkg=b)
        return out

    # ------------------------------------------------------------------
    # PSF
    # ------------------------------------------------------------------
    def _sigma2(self, err):
        """Pixel variances used as PSF-fit weights (exposure noise map if available)."""
        if self.src.cr_sigma is not None:
            s = self.src.cr_sigma.astype(np.float64)
            return np.where(np.isfinite(s) & (s > 0), s ** 2, np.inf)[None]
        return err.astype(np.float64) ** 2

    def _psf_table(self, o, data, err, good, gidx, dy, exclude=None):
        """Normalised pixel table of order ``o`` for a chunk.

        ``y = D / F`` with ``F(t, x) = Shat(x) * wl(t)`` (median spectrum x
        white-light factor) and weights ``w = F^2 / sigma^2``; ``key`` is the
        detector pixel (for the pixel-response map) and ``grp`` the
        (integration, column) group.
        """
        tr, h = self.traces[o], self.p('psf', o)['halfwidth']
        nspat, nspec = self.src.shape
        y = tr.y(dy=dy)
        rows, u, inside = window(y, nspat, h)
        D = _gather(data, rows, tr.cols)
        G = _gather(good, rows, tr.cols)
        if exclude is not None:
            G = G & ~_gather(exclude[None], rows, tr.cols)
        S2 = _gather(self._sigma2(err), rows, tr.cols)
        e = edge_weights(u, h)
        sh = self.shat[o]
        wl = np.sum(np.where(inside, e * D, 0.), axis=(1, 2)) / np.sum(sh[sh > 0])
        F = sh[None, :] * wl[:, None]
        sel = inside & G & (np.abs(u) <= h) & np.isfinite(D) & np.isfinite(S2) & (F[:, None, :] > 0)
        n, nwin, ncol = D.shape
        col = np.broadcast_to(np.arange(ncol)[None, None, :], D.shape)
        if self.p('psf', o).get('cv', 'phase') == 'phase':
            # cross-validation folds by sub-pixel phase of the offset from the trace
            par = (np.floor(2 * np.mod(u + 0.25, 1.)) % 2).astype(np.int64)
        else:
            par = np.broadcast_to((gidx % 2)[:, None, None], D.shape)
        loc = np.broadcast_to(np.arange(n)[:, None, None], D.shape)
        Fb = np.broadcast_to(F[:, None, :], D.shape)
        key = rows * nspec + tr.cols[None, None, :]
        return dict(u=u[sel], col=col[sel], y=(D[sel] / Fb[sel]), w=(Fb[sel] ** 2 / S2[sel]),
                    par=par[sel], key=key[sel], grp=(loc * ncol + col)[sel], ngrp=n * ncol, wl=wl)

    def _accumulate_var(self, err, good):
        e2 = np.where(good, err.astype(np.float64) ** 2, 0.)
        self._vsum += e2.sum(0)
        self._vcnt += good.sum(0)

    @staticmethod
    def _group_flux(T, S, Q, keep):
        """Per-(integration, column) flux (in units of F) of the model ``S * Q``."""
        m = S * Q
        w = np.where(keep, T['w'], 0.)
        num = np.bincount(T['grp'], w * T['y'] * m, T['ngrp'])
        den = np.bincount(T['grp'], w * m * m, T['ngrp'])
        f = num / np.where(den > 0, den, np.inf)
        return f[T['grp']]

    def fit_psf(self, verbose=None):
        """Fit the spline PSF (and the pixel-response map) of every order.

        1. A first fit, in memory, on ``init_ints`` integrations spread over
           the exposure, with iterative outlier rejection. With
           ``pixel_correction=True`` (default) a static response factor ``Q``
           of every detector pixel is estimated at the same time (the model of
           a pixel is ``f * P(u, x) * Q``). ``Q`` absorbs pixel-to-pixel
           sensitivity differences (the data are not flat-fielded; ~2 per cent
           rms on NIRSpec G395H), which no smooth profile can describe.
           With several orders (NIRISS/SOSS) the first fit uses only pixels
           away from the other orders; it is then repeated on data from which
           the other orders' models (joint extraction) are subtracted. ``Q``
           is estimated only from pixels outside the other orders' PSF
           windows (``Q = 1`` where orders overlap): there the two models are
           not separately constrained and ``Q`` would absorb their
           inconsistencies.
        2. One streaming pass over all integrations (``ints='all'``, default)
           or ``ints`` integrations spread over the exposure: pixels deviating
           from the first model by more than ``clip`` (robust) sigma are
           rejected, the rest are added to the normal equations of the PSF and
           to the sums that give ``Q``.
        3. The final solve, with smoothing chosen by cross-validation
           (``smooth='cv'``). ``second_pass=True`` repeats step 2 against the
           final model.

        The same pass also accumulates the time-averaged variance of every
        pixel (used as extraction weights with ``variance='mean'``).
        """
        verbose = self.verbose if verbose is None else verbose
        t0 = time.time()
        self._ensure_background_region()
        src = self.src
        nspat, nspec = src.shape
        multi = len(self.orders) > 1
        pp = {o: self.p('psf', o) for o in self.orders}
        p0 = pp[self.orders[0]]
        pixcorr = p0['pixel_correction']
        models = {o: SplinePSF(pp[o]['halfwidth'], self.traces[o].ncol, pp[o]['oversample'],
                               pp[o]['knot_col']) for o in self.orders}
        qmap = np.ones(nspat * nspec)
        other = self._other_order_mask() if multi else {}
        # field-star traces (add_field_trace): their model is fitted in turn with the PSF during
        # the first stage. The PSF and Q are not fitted where the field star dominates
        # (field model > 5% of the target model; at first: a band around the declared trace)
        fields = bool(self.field_traces)
        fmask = None
        # with field-star pixels excluded, parts of the profile window can lack data in every
        # column: a weak ridge lets the PSF decay to zero there instead of extrapolating
        ridge = 1e-3 if fields else 1e-10
        if fields:
            self.field_model = None
            self.bkg_mask = None
            self._ensure_background_region()
            fmask = self._field_exclusion()

        def q_exclude(o):
            m = other.get(o)
            if fmask is not None:
                m = fmask if m is None else m | fmask
            return m
        # ---- 1. first fit on a sample, in memory
        sample = src.sample_ints(p0['init_ints'])
        chunk = self._chunk()
        zscale = {}
        ndec = max(3 if multi else 1, 4 if fields else 1)
        for dec in range(ndec):
            tabs = {o: [] for o in self.orders}
            nloc = 0
            for a in range(0, len(sample), chunk):
                d, e, g, gi = src.read_ints(sample[a:a + chunk])
                prep = self._prepare(d, e, g)          # subtracts the current field model
                if dec > 0 and multi:
                    self.psfs, self.qmap = models, qmap.reshape(nspat, nspec)
                    dsub = self._decontaminate(d, e, g, prep)
                for o in self.orders:
                    if dec > 0:
                        T = self._psf_table(o, dsub[o] if multi else d, e, g, gi, prep[o]['dy'],
                                            exclude=fmask)
                    else:
                        T = self._psf_table(o, d, e, g, gi, prep[o]['dy'], exclude=q_exclude(o))
                    T['grp'] = T['grp'] + nloc * self.traces[o].ncol
                    tabs[o].append(T)
                nloc += len(gi)
            for o in self.orders:
                T = {k: np.concatenate([t[k] for t in tabs[o]])
                     for k in ('u', 'col', 'y', 'w', 'par', 'key', 'grp')}
                T['ngrp'] = nloc * self.traces[o].ncol
                keep = np.ones(len(T['u']), bool)
                m = models[o]
                for it in range(5):
                    Q = qmap[T['key']]
                    sums = NormalSums(m.ncol, m.tu)
                    sums.add(T['u'][keep], T['col'][keep], (T['y'] / Q)[keep],
                             (T['w'] * Q * Q)[keep], T['par'][keep])
                    # cross-validate the smoothing once, keep it for the rejection iterations
                    m.fit(sums, smooth=pp[o]['smooth'] if it == 0 else m.smooth, cv=self._cv(o),
                          ridge=ridge)
                    S = m(T['u'], T['col'], normalise=False)
                    f = self._group_flux(T, S, Q, keep)
                    if pixcorr:
                        qx = q_exclude(o)
                        qkeep = keep if qx is None else keep & ~qx.ravel()[T['key']]
                        qmap = self._update_q(qmap, T, S, f, qkeep)
                        Q = qmap[T['key']]
                    z = (T['y'] - f * S * Q) * np.sqrt(T['w'])
                    sc = max(mad_sigma(z[keep]), 1.)
                    new = np.abs(z) < pp[o]['clip'] * sc
                    if it > 1 and np.array_equal(new, keep):
                        break
                    keep = new
                zscale[o] = sc
                if verbose and dec == ndec - 1:
                    print(f'  order {o}: first PSF from {len(sample)} integrations; residual scale '
                          f'{sc:.2f} sigma; {100 * (1 - keep.mean()):.3f}% of pixels rejected')
            del tabs
            if fields:
                self.psfs, self.qmap = models, qmap.reshape(nspat, nspec)
                self._fit_field_model(niter=2, wide=dec == 0, verbose=verbose and dec == ndec - 1)
                fmask = self._field_dominated()
        q_sample = qmap.copy()
        init = {o: copy.deepcopy(models[o]) for o in self.orders}
        # ---- 2. streaming pass
        npass = 2 if p0['second_pass'] else 1
        for ipass in range(npass):
            sums = {o: NormalSums(models[o].ncol, models[o].tu) for o in self.orders}
            qnum, qden = np.zeros(nspat * nspec), np.zeros(nspat * nspec)
            nrej = {o: 0 for o in self.orders}
            ntot = {o: 0 for o in self.orders}
            self._vsum = np.zeros(src.shape)
            self._vcnt = np.zeros(src.shape)
            ref = init if ipass == 0 else {o: copy.deepcopy(models[o]) for o in self.orders}
            for d, e, g, gi in self._iter_psf_ints(p0['ints']):
                prep = self._prepare(d, e, g)
                self._accumulate_var(e, g)
                if multi:
                    self.psfs, self.qmap = ref, q_sample.reshape(nspat, nspec)
                    dsub = self._decontaminate(d, e, g, prep)
                for o in self.orders:
                    T = self._psf_table(o, dsub[o] if multi else d, e, g, gi, prep[o]['dy'],
                                        exclude=fmask)
                    Q = q_sample[T['key']]
                    S = ref[o](T['u'], T['col'], normalise=False)
                    f = self._group_flux(T, S, Q, np.ones(len(S), bool))
                    z = (T['y'] - f * S * Q) * np.sqrt(T['w'])
                    ok = np.abs(z) < pp[o]['clip'] * zscale[o]
                    f = self._group_flux(T, S, Q, ok)
                    sums[o].add(T['u'][ok], T['col'][ok], (T['y'] / Q)[ok], (T['w'] * Q * Q)[ok],
                                T['par'][ok])
                    if pixcorr:
                        mm = f * S
                        qx = q_exclude(o)
                        qok = ok if qx is None else ok & ~qx.ravel()[T['key']]
                        w = np.where(qok, T['w'], 0.)
                        qnum += np.bincount(T['key'], w * T['y'] * mm, nspat * nspec)
                        qden += np.bincount(T['key'], w * mm * mm, nspat * nspec)
                    nrej[o] += int((~ok).sum())
                    ntot[o] += len(ok)
            for o in self.orders:
                models[o].fit(sums[o], smooth=pp[o]['smooth'], cv=self._cv(o), ridge=ridge)
                models[o].fit_info['rejected_frac'] = nrej[o] / max(ntot[o], 1)
                if verbose:
                    fi = models[o].fit_info
                    print(f'  order {o}: PSF pass {ipass + 1}: {fi["npix"]} pixels from '
                          f'{self._nint_used} integrations, '
                          f'{models[o].ncoef} coefficients, smoothing (u, x) = '
                          f'({fi["smooth"][0]:.0e}, {fi["smooth"][1]:.0e}), reduced chi2 '
                          f'{fi["chi2_red"]:.3f}, rejected {100 * fi["rejected_frac"]:.3f}%')
            if pixcorr:
                q_sample = self._finish_q(q_sample, qnum, qden)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.var_mean = np.where(self._vcnt > 0, self._vsum / self._vcnt, np.inf)
        self.qmap = q_sample.reshape(nspat, nspec)
        self.psfs = models
        if fields:
            # final field model with the final PSF
            self._fit_field_model(niter=2, wide=False, verbose=verbose)
        for o in self.orders:
            fi = models[o].fit_info
            self.log.add('fit_psf', order=o, params=pp[o], npix=fi['npix'], smooth=fi['smooth'],
                         chi2_red=fi['chi2_red'], rejected_frac=fi['rejected_frac'],
                         cv=fi.get('cv'))
        if verbose:
            if pixcorr:
                inq = self.qmap != 1
                print(f'  pixel-response map: {inq.sum()} pixels, rms {np.std(self.qmap[inq]):.4f}')
            print(f'fit_psf done in {time.time() - t0:.0f} s')
        return self.psfs

    def _cv(self, o):
        """Cross-validation scheme of the PSF smoothing for order ``o``."""
        cv = self.p('psf', o).get('cv', 'phase')
        return 'columns' if cv == 'columns' else 'parity'

    def _other_order_mask(self):
        """For each order, the pixels within the PSF window of any other order."""
        nspat, nspec = self.src.shape
        rows = np.arange(nspat)[:, None]
        near = {}
        for o, tr in self.traces.items():
            m = np.zeros((nspat, nspec), bool)
            h = self.p('psf', o)['halfwidth']
            m[:, tr.cols] = np.abs(rows - tr.y0[None, :]) <= h + 1
            near[o] = m
        return {o: np.any([near[k] for k in near if k != o], axis=0) for o in self.traces}

    def _order_model(self, o, flux, dy, out):
        """Add the model ``flux * P * Q`` of order ``o`` to the full-frame array ``out``."""
        tr, psf = self.traces[o], self.psfs[o]
        nspat, nspec = self.src.shape
        n = len(flux)
        y = tr.y(dy=dy)
        rows, u, inside = window(y, nspat, psf.halfwidth)
        Qg = _gather(self.qmap[None], rows, tr.cols) if self.qmap is not None else 1.
        P = psf(u, np.arange(tr.ncol)[None, None, :]) * Qg
        M = np.where(inside & (np.abs(u) <= psf.halfwidth),
                     np.nan_to_num(flux)[:, None, :] * P, 0.)
        idx = (np.arange(n)[:, None, None] * nspat + rows) * nspec + tr.cols[None, None, :]
        np.add.at(out.reshape(-1), idx.ravel(), M.ravel().astype(out.dtype))
        return out

    def _joint(self, data, err, good, prep, niter=3, refine=None, aperture=None, method='optimal'):
        """Joint extraction of overlapping orders by alternating decontamination.

        Each order is extracted from the data minus the current models of
        all other orders; repeated ``niter`` times (block coordinate descent
        on the joint least-squares problem). Returns ``(results, models)``.
        """
        models = {o: np.zeros(data.shape, np.float32) for o in self.orders}
        total = np.zeros(data.shape, np.float32)
        dys = {o: np.array(prep[o]['dy'], float) for o in self.orders}
        res = {}
        for it in range(niter):
            last = it == niter - 1
            for o in self.orders:
                dsub = data - (total - models[o])
                r = self._extract_order(o, dsub, err, good, dys[o], prep[o]['var_bkg'],
                                        aperture=aperture, method=method if last else 'optimal',
                                        refine=(refine if last else 0))
                if last:
                    dys[o] = r['dy']
                total -= models[o]
                models[o][:] = 0.
                self._order_model(o, r['flux'], dys[o], models[o])
                total += models[o]
                res[o] = r
        return res, models, total

    def _decontaminate(self, data, err, good, prep):
        """``{order: data minus the models of all other orders}`` (for the PSF fit)."""
        res, models, total = self._joint(data, err, good, prep, niter=2, refine=0)
        return {o: data - (total - models[o]) for o in self.orders}

    def _update_q(self, qmap, T, S, f, keep):
        """Least-squares pixel response ``Q = sum(w y m) / sum(w m^2)`` with ``m = f S``."""
        mm = f * S
        w = np.where(keep, T['w'], 0.)
        n = len(qmap)
        num = np.bincount(T['key'], w * T['y'] * mm, n)
        den = np.bincount(T['key'], w * mm * mm, n)
        return self._finish_q(qmap, num, den)

    def _finish_q(self, qmap, num, den, window_cols=51, ubin=0.5):
        """Turn the sums into ``Q`` keeping only its pixel-scale structure.

        Any smooth function of (offset from the trace, column) can be moved
        between the PSF and ``Q`` (the trace hardly moves), so the local
        average of ``Q`` over neighbouring columns at the same offset is
        divided out: the PSF (which follows the trace) keeps all smooth
        structure and ``Q`` (fixed to the detector) only the pixel-to-pixel
        response.

        Each pixel's factor is then shrunk towards 1 according to its own
        precision (Wiener / empirical Bayes): ``1 + (q - 1) s^2 / (s^2 +
        var_q)`` with ``var_q = 1 / den`` and ``s`` the pixel-to-pixel rms
        measured on well-exposed pixels. Faint pixels (far wings) would
        otherwise get noisy factors, which matter when an extraction has to
        rely on them (masked trace cores).
        """
        nspat, nspec = self.src.shape
        q = np.array(qmap, dtype=float, copy=True)
        ok = den > 0
        q[ok] = np.clip(num[ok] / den[ok], 0.5, 2.)
        q = q.reshape(nspat, nspec)
        wgt = np.where(ok, den, 0.).reshape(nspat, nspec)
        out = q.copy()
        rows = np.arange(nspat)[:, None]
        for o, tr in self.traces.items():
            h = self.p('psf', o)['halfwidth']
            u = rows - tr.y0[None, :]                       # (nspat, ncol) static offsets
            sub_q, sub_w = q[:, tr.cols], wgt[:, tr.cols]
            inwin = (np.abs(u) <= h + 1) & (sub_w > 0)
            b = np.floor((u + h + 1) / ubin).astype(int)
            nb = int(np.ceil(2 * (h + 1) / ubin)) + 1
            b = np.clip(b, 0, nb - 1)
            ncol = tr.ncol
            cidx = np.broadcast_to(np.arange(ncol)[None, :], u.shape)
            wq = np.zeros((nb, ncol))
            ww = np.zeros((nb, ncol))
            np.add.at(wq, (b[inwin], cidx[inwin]), (sub_w * sub_q)[inwin])
            np.add.at(ww, (b[inwin], cidx[inwin]), sub_w[inwin])
            k = window_cols
            cs = lambda a: np.concatenate([np.zeros((nb, 1)), np.cumsum(a, 1)], 1)
            cq, cw = cs(wq), cs(ww)
            lo = np.clip(np.arange(ncol) - k // 2, 0, ncol)
            hi = np.clip(np.arange(ncol) + k // 2 + 1, 0, ncol)
            trend = (cq[:, hi] - cq[:, lo]) / np.where(cw[:, hi] - cw[:, lo] > 0,
                                                        cw[:, hi] - cw[:, lo], np.inf)
            tpix = np.where(inwin, trend[b, cidx], 0.)
            corr = np.where(inwin & (tpix > 0), sub_q / np.where(tpix > 0, tpix, 1.), sub_q)
            with np.errstate(divide='ignore', invalid='ignore'):
                vq = np.where(inwin & (tpix > 0), 1. / (sub_w * tpix ** 2), np.inf)
            fin = np.isfinite(vq)
            if fin.sum() > 20:
                best = fin & (vq <= np.percentile(vq[fin], 25))
                dev = corr[best] - 1.
                s2 = (1.4826 * np.median(np.abs(dev - np.median(dev)))) ** 2 - np.median(vq[best])
                s2 = max(s2, 1e-6)
                corr = np.where(fin, 1. + (corr - 1.) * s2 / (s2 + vq), corr)
            out[:, tr.cols] = np.where(inwin, corr, out[:, tr.cols])
        return np.clip(out, 0.5, 2.).ravel()

    def _iter_psf_ints(self, ints):
        """Chunks of the integrations used for the PSF (all, or a spread sample)."""
        src = self.src
        chunk = self._chunk()
        if isinstance(ints, str) and ints == 'all':
            self._nint_used = src.nint
            for _, d, e, g, gi in src.chunks(chunk=chunk):
                yield d, e, g, gi
        else:
            sel = src.sample_ints(int(ints))
            self._nint_used = len(sel)
            for a in range(0, len(sel), chunk):
                yield src.read_ints(sel[a:a + chunk])

    def _chunk(self):
        nspat, nspec = self.src.shape
        per = nspat * nspec * 4 * (10 + (4 * len(self.orders) if len(self.orders) > 1 else 0))
        for o, tr in self.traces.items():
            h = self.p('psf', o)['halfwidth']
            per += (2 * int(np.ceil(h)) + 3) * tr.ncol * 8 * 24
        return max(1, int(self.src.max_mem_gb * 1024 ** 3 / per))

    # ------------------------------------------------------------------
    # Field-star contamination
    # ------------------------------------------------------------------
    def find_contaminants(self, image=None, nsigma=5., low=2., grow=2, min_pixels=5, max_size=60,
                          smooth=1.5, apply=True):
        """Find compact field-star images (zeroth orders) and mask them.

        Parameters
        ----------
        image : str, ndarray or None
            Detection image: a ``rate`` / ``rateints`` file (or array, detector
            orientation) of the **F277W exposure** taken with a NIRISS/SOSS time
            series (recommended: process its ``uncal`` with
            :class:`~jstark.reduction.Stage1`). The target's own light in it
            (the red end of order 1) is excluded automatically. ``None`` uses
            the exposure median minus the models of the target's orders: then
            only contaminants *away* from the traces can be found, because one
            on a trace is absorbed by the target's profile and flux.
        nsigma, low : float
            Detection threshold (peak S/N of the smoothed image) and the
            level down to which a source region is grown.
        grow : int
            Dilate the mask by this many pixels (positions in F277W and in the
            CLEAR exposure differ by ~1-3 pixels).
        min_pixels, max_size : int
            Smallest region (pixels); largest extent (pixels) along either
            axis (longer structures are residuals along traces, not sources).
        smooth : float
            Gaussian smoothing (pixels) of the detection image.
        apply : bool
            Treat the masked pixels as bad from now on (background, PSF fit,
            pixel-response map, extraction). The PSF is refitted on the next
            :meth:`fit_psf` / :meth:`run`.

        Returns
        -------
        list of dict
            ``x``, ``y``, ``npix``, ``peak`` (S/N), ``flux``, ``bbox``, and
            ``orders``: the orders whose aperture the source touches.
        """
        src = self.src
        if not self.traces:
            self.find_trace()
        nspat, nspec = src.shape
        rows = np.arange(nspat)[:, None]
        if image is not None:
            img = load_detection_image(image, src)
            exclude = self._target_light_mask(img)
            valid = np.isfinite(img)
            what = os.path.basename(image) if isinstance(image, str) else 'array'
        else:
            self._ready()
            med = fill_bad(src.median, self._badpix0)[None]
            if self.field_model is not None:
                med = med - self.field_model[None]
            bk, _ = self._background(med, np.ones_like(med), np.ones(med.shape, bool))
            med = med.copy()
            self._subtract(med, bk)
            img = med[0] - self._static_target_model(med[0], good=self._badpix0)
            # residual steps at the edges of the PSF windows are not sources
            exclude = np.zeros(src.shape, bool)
            for o, tr in self.traces.items():
                h = self.psfs[o].halfwidth
                d = np.abs(np.abs(rows - tr.y0[None, :]) - h)
                exclude[:, tr.cols] |= d <= 2
            valid = self._badpix0
            what = 'exposure median'
        mask, sources, _ = find_sources(img, valid=valid, exclude=exclude, nsigma=nsigma, low=low,
                                        smooth=smooth, min_pixels=min_pixels, max_size=max_size)
        if grow:
            from scipy import ndimage
            mask = ndimage.binary_dilation(mask, iterations=int(grow))
        for s in sources:
            y0, y1, x0, x1 = s['bbox']
            s['orders'] = []
            for o, tr in self.traces.items():
                sel = (tr.cols >= x0 - grow) & (tr.cols < x1 + grow)
                if not sel.any():
                    continue
                a = self.p('extract', o)['aperture']
                yc = tr.y0[sel]
                if np.any((yc + a >= y0 - grow) & (yc - a < y1 + grow)):
                    s['orders'].append(int(o))
        self.contaminants = sources
        self.contam_mask = mask
        self.log.add('find_contaminants', image=what, nsigma=nsigma, low=low, grow=grow,
                     n=len(sources), npix=int(mask.sum()),
                     sources=[{k: v for k, v in s.items()} for s in sources])
        if self.verbose:
            print(f'find_contaminants ({what}): {len(sources)} source(s), {int(mask.sum())} pixels masked')
            for s in sorted(sources, key=lambda s: -s['peak']):
                on = (f"in the aperture of order(s) {s['orders']}" if s['orders']
                      else 'outside the apertures')
                print(f"  x={s['x']:7.1f} y={s['y']:6.1f}  S/N {s['peak']:5.1f}  {s['npix']:4d} px  {on}")
        if apply:
            self._apply_contam()
        return sources

    def set_contaminant_mask(self, mask, add=False):
        """Set (or, with ``add=True``, extend) the contaminant mask by hand.

        ``mask`` is a boolean image in canonical orientation (detector
        orientation except for MIRI/LRS, which is transposed).
        """
        mask = np.asarray(mask, bool)
        if mask.shape != tuple(self.src.shape):
            raise ValueError(f'Mask shape {mask.shape} differs from {tuple(self.src.shape)}.')
        if add and self.contam_mask is not None:
            mask = mask | self.contam_mask
        self.contam_mask = mask
        self._apply_contam()

    def clear_contaminants(self):
        """Remove the contaminant mask."""
        self.contam_mask, self.contaminants = None, []
        self._apply_contam()

    def _set_badpix(self):
        m = self.contam_mask
        self.src.badpix = self._badpix0 & ~m if m is not None else self._badpix0.copy()

    def _apply_contam(self):
        """Apply the contaminant mask; everything fitted to the old pixel set is reset."""
        self._set_badpix()
        self.psfs, self.qmap, self.var_mean = {}, None, None
        self.bkg_mask = None
        self.field_model = None

    def _target_light_mask(self, img, pad=None, min_run=50, nsig=10.):
        """Pixels holding the target's own light in a detection image (e.g. F277W).

        Columns where the trace core of an order is bright, in contiguous runs
        of at least ``min_run`` columns (zeroth orders are narrower), are
        excluded within ``halfwidth + pad`` of the trace (``pad`` defaults to
        the PSF half-width: the far wings of a bright trace are still above
        the noise), extended by ``pad`` columns (and to the frame edge if the
        run reaches the end of the order).
        """
        nspat, nspec = self.src.shape
        rows = np.arange(nspat)[:, None]
        allx = np.arange(nspec)
        mask = np.zeros((nspat, nspec), bool)
        for o, tr in self.traces.items():
            h = self.p('psf', o)['halfwidth']
            pd = max(10, int(np.ceil(h))) if pad is None else pad
            u = rows - tr.y0[None, :]
            sub = img[:, tr.cols]
            core = np.where(np.abs(u) <= 2, sub, np.nan)
            far = np.where(np.abs(u) > h + pd, sub, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                bg = np.nanmedian(far, 0)
                noise = 1.4826 * np.nanmedian(np.abs(far - bg[None]), 0)
                sig = np.nanmean(core, 0) - bg
            lit = np.nan_to_num(sig) > nsig * np.nan_to_num(noise, nan=np.inf) / np.sqrt(5.)
            edges = np.diff(np.concatenate([[0], lit.astype(int), [0]]))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                if b - a < min_run:
                    continue
                x0 = 0 if a == 0 else tr.cols[a] - pd
                x1 = nspec if b == tr.ncol else tr.cols[b - 1] + pd + 1
                xs = allx[max(x0, 0):min(x1, nspec)]
                yc = np.interp(xs, tr.cols, tr.y0)
                mask[:, xs] |= np.abs(rows - yc[None, :]) <= h + pd
        return mask

    def add_field_trace(self, order, dx, dy, columns=None, refine=True):
        """Model and subtract the spectral trace of a field star.

        The field star's trace of order ``order`` is the target's trace of
        that order shifted by ``dx`` columns and ``dy`` rows (positions from
        the F277W exposure or a catalogue). Its spatial profile is the
        target's PSF; its spectrum (one static flux per column) is fitted to
        the exposure median frame together with the target's orders, and the
        model is subtracted from every integration before the background and
        extraction. ``refine=True`` refines ``dy`` (+-1.5 pixels).
        ``columns=(x0, x1)`` limits the model to detector columns where the
        field star's trace is present.

        The PSF is refitted (twice: before and after the field model is
        known) on the next :meth:`fit_psf` / :meth:`run`. The field-star flux
        left in each aperture is saved as ``FIELD_O<n>``.
        """
        if order not in self.orders:
            raise ValueError(f'Order {order} is not extracted (orders {self.orders}).')
        self.field_traces.append(dict(order=int(order), dx=float(dx), dy=float(dy),
                                      columns=None if columns is None else tuple(int(c) for c in columns),
                                      refine=bool(refine)))
        self.field_model = None
        self.psfs = {}
        self.bkg_mask = None

    def clear_field_traces(self):
        """Remove all field-star trace models."""
        self.field_traces, self.field_model = [], None
        self.psfs, self.bkg_mask = {}, None

    def _field_geometry(self, ft):
        if ft['order'] not in self.traces:
            return None
        tr = self.traces[ft['order']]
        nspat, nspec = self.src.shape
        h = self.p('psf', ft['order'])['halfwidth']
        x = np.arange(nspec)
        xs = x - ft['dx']
        # the target's trace, extended linearly beyond its column range (the field star's
        # trace usually covers columns the target's extraction does not)
        k = min(20, tr.ncol)
        y = np.interp(xs, tr.cols, tr.y0)
        if k >= 2:
            lo, hi = np.polyfit(tr.cols[:k], tr.y0[:k], 1), np.polyfit(tr.cols[-k:], tr.y0[-k:], 1)
            y = np.where(xs < tr.cols[0], np.polyval(lo, xs), y)
            y = np.where(xs > tr.cols[-1], np.polyval(hi, xs), y)
        y = y + ft['dy']
        ok = np.ones(nspec, bool)
        if ft.get('columns') is not None:
            ok &= (x >= ft['columns'][0]) & (x < ft['columns'][1])
        ok &= (y > -h) & (y < nspat - 1 + h)
        if not ok.any():
            return None
        cidx = np.clip(np.searchsorted(tr.cols, np.round(xs).astype(int)), 0, tr.ncol - 1)
        return dict(x=x[ok], y=y[ok], cidx=cidx[ok])

    def _field_exclusion(self, scale=1.):
        """Pixels near the declared field-star traces (``scale`` x the trace-core half-width)."""
        nspat, nspec = self.src.shape
        rows = np.arange(nspat)[:, None]
        m = np.zeros((nspat, nspec), bool)
        for ft in self.field_traces:
            g = self._field_geometry(ft)
            if g is None:
                continue
            o = ft['order']
            w = self.p('trace', o)['halfwidth'] or max(3., 0.5 * self.p('psf', o)['halfwidth'])
            m[:, g['x']] |= np.abs(rows - g['y'][None, :]) <= scale * w + 1
        return m

    def _field_dominated(self, frac=0.05):
        """Pixels where the field-star model exceeds ``frac`` of the target's model."""
        C, T = self.field_model, getattr(self, '_field_target', None)
        if C is None or T is None:
            return self._field_exclusion()
        return C > frac * np.clip(T, 0, None) + 1e-12

    def _field_trace_objs(self):
        out = []
        for ft in self.field_traces:
            g = self._field_geometry(ft)
            if g is not None:
                out.append(Trace(order=ft['order'], cols=g['x'], y0=g['y']))
        return out

    def _static_target_model(self, img, good=None):
        """Model of all target orders fitted to one (median) frame."""
        shape = self.src.shape
        good = self.src.badpix if good is None else good
        good = (good & np.isfinite(img))[None]
        err = np.sqrt(self._sigma2(np.ones((1,) + tuple(shape))))
        err = np.broadcast_to(err, (1,) + tuple(shape))
        prep = {o: dict(dy=np.zeros(1), var_bkg=np.zeros((1, self.traces[o].ncol)))
                for o in self.orders}
        data = np.nan_to_num(img)[None].astype(np.float32)
        _, _, total = self._joint(data, err, good, prep, niter=2 if len(self.orders) > 1 else 1,
                                  refine=0, method='optimal')
        return total[0].astype(float)

    def _field_fit(self, R, g, ft, off, good, s2):
        """Static spectrum of one field trace fitted to the residual image ``R``."""
        nspat, nspec = self.src.shape
        psf = self.psfs[ft['order']]
        h = psf.halfwidth
        rows, u, inside = window((g['y'] + off)[None], nspat, h)
        rows, u, inside = rows[0], u[0], inside[0]
        x = np.broadcast_to(g['x'][None, :], rows.shape)
        P = psf(u, g['cidx'][None, :])
        if self.qmap is not None:
            P = P * self.qmap[rows, x]
        use = inside & (np.abs(u) <= h)
        W = np.where(use & good[rows, x], 1. / s2[rows, x], 0.)
        D = np.nan_to_num(R[rows, x])
        den = np.sum(W * P * P, 0)
        num = np.sum(W * P * D, 0)
        a = np.where(den > 0, num / np.where(den > 0, den, 1.), 0.)
        gain = np.sum(np.where(den > 0, num ** 2 / np.where(den > 0, den, 1.), 0.))
        C = np.zeros((nspat, nspec))
        np.add.at(C, (rows[use], x[use]), (a[None, :] * P)[use])
        return a, C, gain

    def _fit_field_model(self, niter=2, wide=True, dy_range=1.5, dy_step=0.1, verbose=None):
        """Fit the static models of all field-star traces (see :meth:`add_field_trace`).

        Uses the current PSFs. ``dy`` is refined over +-``dy_range`` pixels
        (``wide``) or +-3 steps of ``dy_step / 2``.
        """
        verbose = self.verbose if verbose is None else verbose
        src = self.src
        self.field_model = None
        self.bkg_mask = None
        self._ensure_background_region()
        med = self._median_bsub.astype(float)
        s2 = self._sigma2(np.ones((1,) + tuple(src.shape)))[0]
        good = src.badpix & np.isfinite(med)
        geos = [self._field_geometry(ft) for ft in self.field_traces]
        comps = [np.zeros(src.shape) for _ in geos]
        for it in range(niter):
            T = self._static_target_model(med - sum(comps))
            self._field_target = T
            for j, ft in enumerate(self.field_traces):
                if geos[j] is None:
                    continue
                R = med - T - sum(c for k, c in enumerate(comps) if k != j)
                if ft['refine']:
                    rng, step = (dy_range, dy_step) if (wide and it == 0) else (3 * dy_step, dy_step / 2)
                    offs = np.arange(-rng, rng + step / 2, step)
                    gains = [self._field_fit(R, geos[j], ft, d, good, s2)[2] for d in offs]
                    ft['dy'] += float(offs[int(np.argmax(gains))])
                    geos[j] = self._field_geometry(ft)
                a, C, _ = self._field_fit(R, geos[j], ft, 0., good, s2)
                comps[j] = C
                ft['flux'], ft['cols'] = a, geos[j]['x']
        self.field_model = sum(comps).astype(np.float32)
        self.bkg_mask = None
        self._ensure_background_region()
        self.log.add('field_traces', traces=[{k: v for k, v in ft.items() if k not in ('flux', 'cols')}
                                             for ft in self.field_traces])
        if verbose:
            for ft in self.field_traces:
                if 'flux' in ft:
                    print(f"  field trace (order {ft['order']}, dx {ft['dx']:.1f}, dy {ft['dy']:.2f}): "
                          f"columns {ft['cols'][0]}-{ft['cols'][-1]}, median flux "
                          f"{np.median(ft['flux']):.3g} per column")

    def field_in_aperture(self, order):
        """Static field-star flux (traces modelled with :meth:`add_field_trace`) in each column's aperture."""
        if self.field_model is None or order not in self.psfs:
            return None
        shape = (1,) + tuple(self.src.shape)
        W = self._window(order, self.field_model[None].astype(float), np.ones(shape),
                         np.ones(shape, bool), np.zeros(1))
        return np.sum(W['e'] * np.where(W['inside'], W['D'], 0.), axis=1)[0]

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def _ready(self):
        if not self.traces:
            self.find_trace()
        self._ensure_background_region()
        if not self.psfs:
            self.fit_psf()

    def _window(self, o, data, err, good, dy, aperture=None, deriv=False):
        """Window arrays of order ``o`` around the trace for offsets ``dy``.

        Returns a dict with ``rows``, ``u``, ``inside``, ``D``, ``G``, ``V``,
        ``P`` (profile x pixel response), ``e`` (aperture weights) and, with
        ``deriv``, ``dP``.
        """
        tr, psf = self.traces[o], self.psfs[o]
        pe = self.p('extract', o)
        a = pe['aperture'] if aperture is None else aperture
        h = psf.halfwidth
        if pe['variance'] == 'mean' and self.var_mean is not None:
            var_full = self.var_mean[None]
        else:
            var_full = err.astype(np.float64) ** 2
        col = np.arange(tr.ncol)[None, None, :]
        y = tr.y(dy=dy)
        rows, u, inside = window(y, self.src.shape[0], max(h, a))
        Qg = _gather(self.qmap[None], rows, tr.cols) if self.qmap is not None else 1.
        W = dict(rows=rows, u=u, inside=inside,
                 D=_gather(data, rows, tr.cols).astype(np.float64),
                 G=_gather(good, rows, tr.cols) & inside,
                 V=_gather(var_full, rows, tr.cols),
                 P=psf(u, col) * Qg,
                 e=np.where(inside, edge_weights(u, a), 0.))
        if deriv:
            W['dP'] = psf(u, col, deriv=1) * Qg
        return W

    def _extract_order(self, o, data, err, good, dy, var_bkg, aperture=None, method=None,
                       refine=None):
        """Extract order ``o`` from a background-subtracted chunk."""
        pe = self.p('extract', o)
        method = method or pe['method']
        refine = pe['refine_shift'] if refine is None else refine
        h = self.psfs[o].halfwidth
        dy = np.array(dy, float)
        for it in range(refine + 1):
            W = self._window(o, data, err, good, dy, aperture, deriv=True)
            D, V, G, P, e, u = W['D'], W['V'], W['G'], W['P'], W['e'], W['u']
            off = None
            if pe['reject'] and pe.get('reject_mode', 'temporal') == 'temporal' and len(D) >= 5:
                r0 = optimal(D, V, G, P, e, fit_background=pe['fit_background'])
                off = self._static_residual(D - r0['flux'][:, None, :] * P, G, W['rows'])
            res = optimal(D, V, G, P, e, reject=pe['reject'], niter=pe['reject_iters'],
                          fit_background=pe['fit_background'], var_bkg=var_bkg,
                          resid_offset=off)
            last = it == refine
            gfit = (res['good'] | (G & (e == 0))) & (np.abs(u) <= h)   # good pixels of the PSF window
            ddy, ds = refine_shift(D, V, gfit & ~res['rejected'], P, W['dP'], u, res['flux'],
                                   width=last)
            ddy = np.where(np.isfinite(ddy), np.clip(ddy, -0.5, 0.5), 0.)
            if not last:
                dy = dy + ddy
        out = dict(flux=res['flux'], var=res['var'], maskfrac=res['maskfrac'], nrej=res['nrej'],
                   dy=dy, ds=ds, fbkg=res['bkg'], chi2=res['chi2'])
        bf, bv = box(D, V, G, P, e, res['flux'], var_bkg=var_bkg)
        out['box'], out['box_var'] = bf, bv
        if method == 'optimal':
            pass
        elif method == 'box':
            out['opt'], out['opt_var'] = res['flux'], res['var']
            out['flux'], out['var'] = bf, bv
        elif method == 'covariance':
            from .covariance import covariance_extract
            if o not in self.cov:
                self.estimate_covariance(orders=[o])
            Gc = G & ~res['rejected']
            cf, cv = covariance_extract(self, o, D, V, Gc, P, e, W['rows'],
                                        scale=pe.get('cov_scale', 1.))
            out['opt'], out['opt_var'] = res['flux'], res['var']
            out['flux'], out['var'] = cf, cv
        else:
            raise ValueError(f'Unknown extraction method {method!r}')
        # (almost) all of the profile on masked pixels: nothing left to measure
        lost = res['maskfrac'] > pe.get('max_maskfrac', 1.)
        if lost.any():
            for k in ('flux', 'var', 'box', 'box_var', 'opt', 'opt_var'):
                if k in out:
                    out[k] = np.where(lost, np.nan, out[k])
        return out

    def _static_residual(self, R, G, rows):
        """Median residual of every detector pixel over the integrations of a chunk.

        ``R`` and ``G`` are window arrays; the median is taken per detector
        pixel (row, column), so a trace that moves by a fraction of a pixel
        still compares each pixel with itself.
        """
        n, nwin, ncol = R.shape
        nspat = self.src.shape[0]
        frame = np.full((n, nspat, ncol), np.nan, np.float32)
        ii = np.broadcast_to(np.arange(n)[:, None, None], R.shape)
        cc = np.broadcast_to(np.arange(ncol)[None, None, :], R.shape)
        frame[ii[G], rows[G], cc[G]] = R[G]
        import bottleneck as bn
        med = np.nan_to_num(bn.nanmedian(frame, axis=0))
        return np.take_along_axis(np.broadcast_to(med[None], (n, nspat, ncol)), rows, axis=1)

    def estimate_covariance(self, orders=None, nblocks=3, block_len=300, segments=None):
        """Estimate the row-offset covariance used by ``method='covariance'``.

        See :mod:`jstark.extraction.covariance`. Uses blocks of consecutive
        integrations spread over the exposure (like :meth:`optimise`).
        """
        from .covariance import estimate_row_covariance
        self._ready()
        for o in (orders or self.orders):
            self.cov[o] = estimate_row_covariance(self, o, nblocks=nblocks, block_len=block_len,
                                                  segments=segments)
            if self.verbose:
                K = self.cov[o]['K']
                d = np.sqrt(np.clip(np.diagonal(K, axis1=1, axis2=2), 0, None))
                print(f'  order {o}: row-offset covariance from {self.cov[o]["nint"]} integrations; '
                      f'rms row offset per amplifier: ' +
                      ', '.join(f'{np.median(x[x > 0]):.3g}' if (x > 0).any() else '0' for x in d))
            self.log.add('estimate_covariance', order=o, nint=self.cov[o]['nint'])
        return self.cov

    def extract_chunk(self, data, err, good, aperture=None, method=None):
        """Background-subtract and extract a chunk (in memory). Returns ``{order: dict}``.

        With several orders and ``method='joint'`` (NIRISS/SOSS default) the
        orders are extracted jointly (:meth:`_joint`), and the contamination of
        each order by the others (model flux of the other orders inside the
        aperture, relative to the order's own) is returned as ``contam``.
        """
        self._ready()
        data = np.array(data, dtype=np.float32, copy=True)
        prep = self._prepare(data, err, good)
        method = method or self.p('extract', self.orders[0])['method']
        out = {}
        if len(self.orders) > 1 and method == 'joint':
            niter = self.p('extract', self.orders[0]).get('joint_iters', 3)
            res, models, total = self._joint(data, err, good, prep, niter=niter,
                                             aperture=aperture)
            for o in self.orders:
                r = res[o]
                W = self._window(o, total - models[o], err, good, r['dy'], aperture)
                own = np.sum(W['e'] * np.nan_to_num(r['flux'])[:, None, :] * W['P'], 1)
                r['contam'] = np.sum(W['e'] * W['D'], 1) / np.where(own > 0, own, np.nan)
        else:
            m = None if method == 'joint' else method
            res = {o: self._extract_order(o, data, err, good, prep[o]['dy'], prep[o]['var_bkg'],
                                          aperture=aperture, method=m) for o in self.orders}
        for o in self.orders:
            r = res[o]
            b = prep[o]['bkg']
            r['bkg'] = b if b.ndim == 2 else np.nanmedian(b, axis=1)
            r['dy_cof'] = prep[o]['dy']
            out[o] = r
        return out

    def extract_segment(self, k, save=True):
        """Extract segment ``k`` (0-based) chunk by chunk; optionally save it."""
        self._ready()
        src = self.src
        parts = []
        for _, d, e, g, gi in src.chunks(segments=[k], chunk=self._chunk()):
            res = self.extract_chunk(d, e, g)
            parts.append((gi, res))
        gidx = np.concatenate([p[0] for p in parts])
        out = {o: {key: np.concatenate([p[1][o][key] for p in parts])
                   for key in parts[0][1][o]} for o in self.orders}
        times = src.times[gidx] if src.times is not None else None
        if save:
            fn = output_filename(src.files[k], self.outdir, 'spec')
            self._write_segment(fn, k, gidx, times, out)
            self.outputs[k] = fn
        if self.verbose:
            txt = '; '.join(f'order {o}: {np.mean(out[o]["nrej"]):.1f} rejected px/int, '
                            f'dy rms {np.nanstd(out[o]["dy"]) * 1e3:.1f} mpx' for o in self.orders)
            print(f'  segment {k + 1}/{src.nseg}: {len(gidx)} integrations; {txt}')
        return gidx, out

    def run(self, segments=None, combine=True, keep_segments=True):
        """Extract all (or the given) segments, save them, and combine the spectra.

        Parameters
        ----------
        segments : list of int or None
            Segments to extract (default: all).
        combine : bool
            Write ``<exposure>_spectra.fits``.
        keep_segments : bool
            Keep the per-segment files (``*_spec.fits``). They hold the same
            data as the combined file; ``False`` deletes them after combining.

        Returns
        -------
        Spectra or None
            The combined product (``combine=True``).
        """
        t0 = time.time()
        self._ready()
        self.save_model()
        segs = range(self.src.nseg) if segments is None else segments
        for k in segs:
            self.extract_segment(k)
        self.log.add('run', segments=list(segs), params=self.params,
                     order_params=self.order_params, seconds=time.time() - t0)
        if self.verbose:
            print(f'extraction done in {time.time() - t0:.0f} s')
        if not combine:
            return None
        out = self.combine()
        if not keep_segments:
            remove_files(self.outputs, self.verbose)
        return out

    # ------------------------------------------------------------------
    # Aperture / background optimisation
    # ------------------------------------------------------------------
    def _blocks(self, nblocks, block_len, segments):
        src = self.src
        if isinstance(segments, str) and segments == 'all':
            return [(k, slice(0, src.seg_nint[k])) for k in range(src.nseg)]
        if segments is not None:
            segs = list(segments)
            return [(k, slice(0, min(src.seg_nint[k], block_len))) for k in segs]
        # short exposures: shrink the blocks so that they do not overlap (an overlapping
        # integration would be counted twice)
        block_len = max(1, min(block_len, src.nint // nblocks))
        out = []
        for i in range(nblocks):
            c = int((i + 0.5) / nblocks * src.nint)
            k = src.segment_of(c)
            loc = c - src.seg_start[k]
            n = src.seg_nint[k]
            a = int(np.clip(loc - block_len // 2, 0, max(0, n - block_len)))
            out.append((k, slice(a, min(n, a + block_len))))
        # merge duplicates (short exposures)
        uniq = []
        for b in out:
            if b not in uniq:
                uniq.append(b)
        return uniq

    def optimise(self, apertures=None, bkg_inner=None, nblocks=3, block_len=300, segments=None,
                 order=None, apply=True, plot=False):
        """Choose the extraction aperture and background region by light-curve scatter.

        For every combination, the background is re-estimated and the
        spectra are extracted (the PSF is kept) from blocks of consecutive
        integrations; the white light of each block is normalised by its own
        median, and the robust point-to-point scatter of all blocks together
        is the metric (insensitive to transits and to flux differences
        between blocks).

        Parameters
        ----------
        apertures : sequence of float
            Extraction half-widths to test.
        bkg_inner : sequence of float or None
            Background inner distances to test (default: the current one).
        nblocks, block_len : int
            Blocks of consecutive integrations spread over the exposure.
        segments : None, 'all' or list of int
            ``'all'`` uses the whole exposure; a list uses (the first
            ``block_len`` integrations of) those segments.
        order : int or None
            Order whose white light is used (default: the first).
        apply : bool
            Set the best combination as the current parameters.

        Returns
        -------
        dict
            ``apertures``, ``bkg_inner``, ``sigma`` (relative p2p scatter,
            shape ``(nbkg, nap)``), ``ratio`` (to the propagated errors) and
            ``best``.
        """
        self._ready()
        o = order or self.orders[0]
        pe, pb = self.p('extract', o), self.p('background')
        aps = list(apertures if apertures is not None else
                   np.arange(max(1., pe['aperture'] - 2), pe['aperture'] + 2.5, 1.))
        inner = list(bkg_inner if bkg_inner is not None else [pb['inner']])
        blocks = self._blocks(nblocks, block_len, segments)
        saved = (self.bkg_mask, self.bkg_ok, copy.deepcopy(self.params['background']))
        wl = {(i, j): [[] for _ in blocks] for i in range(len(inner)) for j in range(len(aps))}
        wle = {key: [[] for _ in blocks] for key in wl}
        t0 = time.time()
        try:
            for b, (k, sl) in enumerate(blocks):
                for d, e, g, gi in self._block_chunks(k, sl):
                    for i, rin in enumerate(inner):
                        self.params['background']['inner'] = rin
                        self.bkg_mask = None
                        self._ensure_background_region()
                        dd = np.array(d, np.float32, copy=True)
                        prep = self._prepare(dd, e, g)
                        # shift refinement once (default aperture), reused for every aperture
                        meth = 'optimal' if pe['method'] == 'joint' else pe['method']
                        r0 = self._extract_order(o, dd, e, g, prep[o]['dy'], prep[o]['var_bkg'],
                                                 method='optimal')
                        for j, a in enumerate(aps):
                            r = self._extract_order(o, dd, e, g, r0['dy'], prep[o]['var_bkg'],
                                                    aperture=a, refine=0, method=meth)
                            f, fe = white_light(r['flux'], r['var'])
                            wl[(i, j)][b].append(f)
                            wle[(i, j)][b].append(fe)
        finally:
            self.bkg_mask, self.bkg_ok, self.params['background'] = saved[0], saved[1], saved[2]
            if self.bkg_mask is None:
                self._ensure_background_region()
        sigma = np.zeros((len(inner), len(aps)))
        ratio = np.zeros_like(sigma)
        for (i, j), bl in wl.items():
            fl = [np.concatenate(x) for x in bl if x]
            er = [np.concatenate(x) for x in wle[(i, j)] if x]
            sigma[i, j], ratio[i, j] = p2p_noise(fl, er)
        i, j = np.unravel_index(np.argmin(sigma), sigma.shape)
        ndiff = sum(sl.stop - sl.start - 1 for _, sl in blocks)
        # statistical uncertainty of a MAD-based scatter estimate from ndiff differences
        sigma_err = 1.17 * sigma / np.sqrt(max(ndiff, 1))
        best = dict(aperture=float(aps[j]), bkg_inner=float(inner[i]), sigma=float(sigma[i, j]),
                    ratio=float(ratio[i, j]), sigma_err=float(sigma_err[i, j]))
        res = dict(apertures=np.asarray(aps, float), bkg_inner=np.asarray(inner, float),
                   sigma=sigma, ratio=ratio, sigma_err=sigma_err, best=best, blocks=[(k, sl.start, sl.stop) for k, sl in blocks])
        self.optimisation = res
        if self.verbose:
            print(f'optimise ({time.time() - t0:.0f} s, {len(blocks)} block(s), '
                  f'{sum(s.stop - s.start for _, s in blocks)} integrations): white-light p2p scatter [ppm]')
            print('  bkg_inner \\ aperture ' + ' '.join(f'{a:7.1f}' for a in aps))
            for ii, rin in enumerate(inner):
                print(f'  {rin:20.1f} ' + ' '.join(f'{1e6 * s:7.1f}' for s in sigma[ii]))
            print(f'  best: aperture {best["aperture"]}, bkg_inner {best["bkg_inner"]}: '
                  f'{1e6 * best["sigma"]:.1f} +- {1e6 * best["sigma_err"]:.1f} ppm '
                  f'({best["ratio"]:.2f} x propagated); settings within the +- are equivalent')
        if apply:
            self.set('extract', aperture=best['aperture'])
            if best['bkg_inner'] != pb['inner']:
                self.params['background']['inner'] = best['bkg_inner']
                self.bkg_mask = None
                self._ensure_background_region()
        self.log.add('optimise', order=o, apertures=aps, bkg_inner=inner,
                     sigma_ppm=(1e6 * sigma).round(2).tolist(), best=best, applied=apply)
        if plot:
            from .diagnostics import plot_optimisation
            plot_optimisation(res)
        return res

    def _block_chunks(self, k, sl):
        chunk = self._chunk()
        for a in range(sl.start, sl.stop, chunk):
            d, e, g, gi = self.src.read(k, slice(a, min(sl.stop, a + chunk)))
            yield d, e, g, gi

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _write_segment(self, fn, k, gidx, times, out):
        hdr = fits.Header()
        hdr['JSMODE'] = self.mode
        hdr['EXPOSURE'] = self.src.exposure
        hdr['SEGMENT'] = k + 1
        hdr['SEGSTART'] = int(gidx[0])
        hdr['TRANSPOS'] = (bool(self.src.transpose), 'spectral axis is detector y (columns=rows)')
        hdus = [fits.PrimaryHDU(header=hdr)]
        for o, r in out.items():
            s = f'_O{o}'
            for key, name in (('flux', 'FLUX'), ('var', 'VAR'), ('box', 'BOX'), ('box_var', 'BOXVAR'),
                              ('maskfrac', 'MASKFRAC'), ('bkg', 'BKG'), ('opt', 'OPT'),
                              ('opt_var', 'OPTVAR'), ('contam', 'CONTAM')):
                if key in r:
                    hdus.append(fits.ImageHDU(np.asarray(r[key], np.float64 if key in
                                                         ('flux', 'var', 'box', 'box_var', 'opt',
                                                          'opt_var') else np.float32),
                                              name=name + s))
            wl = self.traces[o].info.get('wavelength')
            hw = fits.ImageHDU(np.asarray(wl if wl is not None else np.full(self.traces[o].ncol, np.nan),
                                          np.float64), name='WAVE' + s)
            hw.header['BUNIT'] = 'micron'
            hw.header['COMMENT'] = 'wavelength of each extracted column (at the trace)'
            hdus.append(hw)
            cols = [fits.Column(name=n, format='D', array=np.asarray(r[key], float))
                    for n, key in (('dy', 'dy'), ('ds', 'ds'), ('dy_cof', 'dy_cof'), ('chi2', 'chi2'))]
            cols.append(fits.Column(name='nrej', format='K', array=np.asarray(r['nrej'])))
            hdus.append(fits.BinTableHDU.from_columns(cols, name='REG' + s))
        tcols = [fits.Column(name='gidx', format='K', array=gidx)]
        if times is not None:
            tcols.append(fits.Column(name='int_mid_BJD_TDB', format='D', array=times))
        hdus.append(fits.BinTableHDU.from_columns(tcols, name='TIMES'))
        fits.HDUList(hdus).writeto(fn, overwrite=True)

    def model_file(self):
        return os.path.join(self.outdir, f'{self.src.exposure}_extract_model.fits')

    def save_model(self, fname=None):
        """Save traces, PSFs, the mean-variance map and the parameters (to resume later)."""
        fname = fname or self.model_file()
        hdr = fits.Header()
        hdr['JSMODE'] = self.mode
        hdr['EXPOSURE'] = self.src.exposure
        hdr['ORDERS'] = ','.join(str(o) for o in self.orders)
        hdus = [fits.PrimaryHDU(header=hdr)]
        for o in self.orders:
            tr = self.traces[o]
            cols = [fits.Column(name='col', format='K', array=tr.cols),
                    fits.Column(name='y0', format='D', array=tr.y0),
                    fits.Column(name='snr', format='D', array=tr.snr)]
            if 'wavelength' in tr.info:
                cols.append(fits.Column(name='wavelength', format='D', array=tr.info['wavelength']))
            hdus.append(fits.BinTableHDU.from_columns(cols, name=f'TRACE_O{o}'))
            if o in self.psfs:
                hdus += self.psfs[o].to_hdus(f'_O{o}')
        if self.var_mean is not None:
            hdus.append(fits.ImageHDU(np.where(np.isfinite(self.var_mean), self.var_mean, 0.)
                                      .astype(np.float32), name='VARMEAN'))
        if self.qmap is not None:
            hdus.append(fits.ImageHDU(self.qmap.astype(np.float32), name='QMAP'))
        if self.contam_mask is not None:
            hdus.append(fits.ImageHDU(self.contam_mask.astype(np.uint8), name='CONTAM'))
        if self.field_model is not None:
            hdus.append(fits.ImageHDU(self.field_model.astype(np.float32), name='FIELD'))
        par = json.dumps({'params': self.params, 'order_params': {str(k): v for k, v in
                                                                  self.order_params.items()},
                          'contaminants': self.contaminants,
                          'field_traces': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in ft.items()} for ft in self.field_traces]})
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name='params', format=f'{len(par)}A', array=np.array([par]))], name='PARAMS'))
        fits.HDUList(hdus).writeto(fname, overwrite=True)
        return fname

    def load_model(self, fname=None):
        """Restore traces, PSFs, variance map and parameters saved by :meth:`save_model`."""
        fname = fname or self.model_file()
        with fits.open(fname) as h:
            par = json.loads(h['PARAMS'].data['params'][0])
            self.params = par['params']
            self.order_params = {int(k): v for k, v in par['order_params'].items()}
            self.orders = [int(x) for x in h[0].header['ORDERS'].split(',')]
            for o in self.orders:
                t = h[f'TRACE_O{o}'].data
                tr = Trace(order=o, cols=np.asarray(t['col'], int), y0=np.asarray(t['y0'], float),
                           snr=np.asarray(t['snr'], float))
                if 'wavelength' in t.names:
                    tr.info['wavelength'] = np.asarray(t['wavelength'], float)
                self.traces[o] = tr
                if f'PSF_O{o}' in h:
                    self.psfs[o] = SplinePSF.from_hdu(h[f'PSF_O{o}'])
            if 'VARMEAN' in h:
                v = np.asarray(h['VARMEAN'].data, float)
                self.var_mean = np.where(v > 0, v, np.inf)
            self.qmap = np.asarray(h['QMAP'].data, float) if 'QMAP' in h else None
            self.contam_mask = np.asarray(h['CONTAM'].data, bool) if 'CONTAM' in h else None
            self.field_model = np.asarray(h['FIELD'].data, np.float32) if 'FIELD' in h else None
        self.contaminants = par.get('contaminants', [])
        self.field_traces = [{k: (np.asarray(v) if k in ('flux', 'cols') else
                                  (tuple(v) if k == 'columns' and v is not None else v))
                              for k, v in ft.items()} for ft in par.get('field_traces', [])]
        self._set_badpix()
        self.bkg_mask = None
        self._ensure_background_region()
        return self

    def combine(self, fname=None):
        """Combine the per-segment spectra into ``<exposure>_spectra.fits``."""
        fname = fname or os.path.join(self.outdir, f'{self.src.exposure}_spectra.fits')
        segs = sorted(self.outputs)
        data = {}
        for k in segs:
            with fits.open(self.outputs[k]) as h:
                for hdu in h[1:]:
                    if isinstance(hdu, fits.BinTableHDU):
                        tab = {n: np.asarray(hdu.data[n]) for n in hdu.data.names}
                        data.setdefault(hdu.name, []).append(tab)
                    else:
                        data.setdefault(hdu.name, []).append(np.asarray(hdu.data))
        hdr = fits.Header()
        hdr['JSMODE'] = self.mode
        hdr['EXPOSURE'] = self.src.exposure
        hdr['ORDERS'] = ','.join(str(o) for o in self.orders)
        hdr['NSEGMENT'] = len(segs)
        hdr['TRANSPOS'] = (bool(self.src.transpose), 'spectral axis is detector y (columns=rows)')
        hdus = [fits.PrimaryHDU(header=hdr)]
        for name, parts in data.items():
            if isinstance(parts[0], dict):
                cat = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
                if name.startswith('REG_O'):
                    o = int(name.split('_O')[1])
                    flux = np.concatenate(data[f'FLUX_O{o}'])
                    cat['dx'], cat['dx_scale'] = spectral_shift(flux)
                    f, _ = white_light(flux, np.concatenate(data[f'VAR_O{o}']))
                    cat['white'] = f / np.nanmedian(f)
                    cat['maskfrac'] = np.nanmean(np.concatenate(data[f'MASKFRAC_O{o}']), axis=1)
                    cat['bkg'] = np.nanmedian(np.concatenate(data[f'BKG_O{o}']), axis=1)
                cols = [fits.Column(name=n, format='K' if v.dtype.kind in 'iu' else 'D', array=v)
                        for n, v in cat.items()]
                hdus.append(fits.BinTableHDU.from_columns(cols, name=name))
            elif name.startswith('WAVE_O'):
                hw = fits.ImageHDU(parts[0], name=name)     # the same for every segment
                hw.header['BUNIT'] = 'micron'
                hdus.append(hw)
            else:
                hdus.append(fits.ImageHDU(np.concatenate(parts), name=name))
        for o in self.orders:
            tr = self.traces[o]
            cols = [fits.Column(name='col', format='K', array=tr.cols),
                    fits.Column(name='y0', format='D', array=tr.y0)]
            wl = tr.info.get('wavelength')
            cols.append(fits.Column(name='wavelength', format='D',
                                    array=wl if wl is not None else np.full(tr.ncol, np.nan)))
            hdus.append(fits.BinTableHDU.from_columns(cols, name=f'TRACE_O{o}'))
            if o in self.psfs:
                hdus += self.psfs[o].to_hdus(f'_O{o}')
            fa = self.field_in_aperture(o)
            if fa is not None:
                hf = fits.ImageHDU(fa, name=f'FIELD_O{o}')
                hf.header['COMMENT'] = 'static field-star flux in the aperture (modelled and subtracted)'
                hdus.append(hf)
        par = json.dumps({'params': self.params,
                          'order_params': {str(k): v for k, v in self.order_params.items()},
                          'optimisation': None if self.optimisation is None else
                          {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in self.optimisation.items()}})
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name='params', format=f'{len(par)}A', array=np.array([par]))], name='PARAMS'))
        fits.HDUList(hdus).writeto(fname, overwrite=True)
        if self.verbose:
            print(f'Saved spectra to {fname}')
        self.log.add('combine', file=fname, segments=segs)
        return Spectra(fname)


def _fill_along_dispersion(img, bad, good):
    """Copy of ``img`` with ``bad`` pixels linearly interpolated along each row."""
    out = np.array(img, dtype=float, copy=True)
    src_ok = good & ~bad & np.isfinite(out)
    x = np.arange(out.shape[1])
    for r in np.where(bad.any(axis=1))[0]:
        ok = src_ok[r]
        if ok.sum() >= 2:
            out[r, bad[r]] = np.interp(x[bad[r]], x[ok], out[r, ok])
    return out


def remove_files(outputs, verbose=True):
    """Delete the files of an ``{segment: path}`` dict (and macOS ``._`` companions); empties it."""
    for k in sorted(outputs):
        fn = outputs.pop(k)
        for f in (fn, os.path.join(os.path.dirname(fn), '._' + os.path.basename(fn))):
            if os.path.isfile(f):
                os.remove(f)
        if verbose:
            print(f'  removed {os.path.basename(fn)}')


def spectral_shift(flux):
    """Shift ``dx`` (pixels) of every spectrum along the dispersion relative to the median.

    Linearised fit ``s_t = a T - a dx T'`` of each spectrum to the median
    template ``T``; returns ``dx`` and the scale ``a``.
    """
    f = np.asarray(flux, float)
    ok = np.all(np.isfinite(f), axis=0)
    f = f[:, ok]
    T = np.nanmedian(f, axis=0)
    dT = np.gradient(T)
    A = np.stack([T, -dT], 1)
    coef = np.linalg.lstsq(A, f.T, rcond=None)[0]
    a, adx = coef
    return adx / np.where(a != 0, a, np.nan), a


class Spectra:
    """Extracted spectral time series (``<exposure>_spectra.fits``).

    Attributes
    ----------
    orders : list of int
    times : ndarray
        Mid-integration BJD_TDB - 2400000.5.
    flux, err, box, box_err, maskfrac, wavelength, cols, regressors : dict
        Per order.
    contam : dict
        Per order (joint multi-order extraction only): flux of the other
        orders' models inside the aperture relative to the order's own.
    field : dict
        Per order, if field-star traces were modelled
        (:meth:`SpectralExtractor.add_field_trace`): their static flux in the
        aperture of each column (already subtracted). Divide by the median
        spectrum for the dilution of each column.
    """

    def __init__(self, fname):
        self.fname = fname
        with fits.open(fname) as h:
            hdr = h[0].header
            self.mode = hdr['JSMODE']
            self.orders = [int(x) for x in hdr['ORDERS'].split(',')]
            t = h['TIMES'].data
            self.gidx = np.asarray(t['gidx'])
            self.times = np.asarray(t['int_mid_BJD_TDB']) if 'int_mid_BJD_TDB' in t.names else None
            self.flux, self.err, self.box, self.box_err = {}, {}, {}, {}
            self.maskfrac, self.wavelength, self.cols, self.regressors = {}, {}, {}, {}
            self.contam, self.extra, self.field = {}, {}, {}
            for o in self.orders:
                s = f'_O{o}'
                self.flux[o] = np.asarray(h['FLUX' + s].data, float)
                self.err[o] = np.sqrt(np.asarray(h['VAR' + s].data, float))
                if 'BOX' + s in h:
                    self.box[o] = np.asarray(h['BOX' + s].data, float)
                    self.box_err[o] = np.sqrt(np.asarray(h['BOXVAR' + s].data, float))
                if 'OPT' + s in h:
                    self.extra[('opt', o)] = (np.asarray(h['OPT' + s].data, float),
                                              np.sqrt(np.asarray(h['OPTVAR' + s].data, float)))
                self.maskfrac[o] = np.asarray(h['MASKFRAC' + s].data, float)
                if 'CONTAM' + s in h:
                    self.contam[o] = np.asarray(h['CONTAM' + s].data, float)
                if 'FIELD' + s in h:
                    self.field[o] = np.asarray(h['FIELD' + s].data, float)
                tr = h['TRACE' + s].data
                self.cols[o] = np.asarray(tr['col'])
                self.wavelength[o] = (np.asarray(h['WAVE' + s].data, float) if 'WAVE' + s in h
                                      else np.asarray(tr['wavelength']))
                reg = h['REG' + s].data
                self.regressors[o] = {n: np.asarray(reg[n]) for n in reg.names}
            self.params = json.loads(h['PARAMS'].data['params'][0])

    @property
    def bjd(self):
        return None if self.times is None else self.times + 2400000.5

    def white_light(self, order=None, method='optimal', normalise=True):
        """White-light curve and its error (columns with any NaN are left out)."""
        o = order or self.orders[0]
        f, e = (self.flux[o], self.err[o]) if method == 'optimal' else (self.box[o], self.box_err[o])
        wl, we = white_light(f, e ** 2)
        if normalise:
            m = np.nanmedian(wl)
            wl, we = wl / m, we / m
        return wl, we

    def bin(self, order=None, nbins=None, edges=None, method='optimal', normalise=True):
        """Spectroscopic light curves summed in wavelength bins.

        Parameters
        ----------
        nbins : int or None
            Number of equal-width bins in column (pixel) space.
        edges : sequence or None
            Bin edges in wavelength (micron); overrides ``nbins``.

        Returns
        -------
        wave : ndarray ``(nb,)``
            Mean wavelength of each bin.
        lc, lc_err : ndarray ``(nint, nb)``
        """
        o = order or self.orders[0]
        f, e = (self.flux[o], self.err[o]) if method == 'optimal' else (self.box[o], self.box_err[o])
        lam = self.wavelength[o]
        if edges is not None:
            idx = np.digitize(lam, edges) - 1
            nb = len(edges) - 1
        else:
            nb = nbins or 20
            idx = (np.arange(len(lam)) * nb) // len(lam)
        good = np.all(np.isfinite(f) & np.isfinite(e), axis=0)
        wave, lc, le = np.zeros(nb), np.zeros((len(f), nb)), np.zeros((len(f), nb))
        for b in range(nb):
            sel = (idx == b) & good
            wave[b] = np.nanmean(lam[sel]) if sel.any() else np.nan
            lc[:, b] = np.sum(f[:, sel], 1)
            le[:, b] = np.sqrt(np.sum(e[:, sel] ** 2, 1))
        if normalise:
            m = np.nanmedian(lc, axis=0)
            lc, le = lc / m, le / m
        return wave, lc, le
