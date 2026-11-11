"""Cleaning of integration-level data (``rateints``) into a 3D cube ready for extraction.

The :class:`CubeCleaner` keeps two masks:

``badpix``
    2D ``(ny, nx)`` map of *persistent* bad pixels (``1`` = good, ``0`` = bad;
    the same convention as :class:`jstark.ApPhoto`). Built from the DQ
    flags (pixels flagged in more than a given fraction of integrations),
    from hot / noisy pixel detection on the median and scatter images, or
    supplied by the user.
``crmask``
    3D ``(nint, ny, nx)`` boolean map of *transient* bad pixels (``True`` =
    bad): cosmic-ray hits found in the time domain, pixels flagged in the DQ
    of only some integrations, and invalid values (NaN data, zero / NaN
    errors).

Both masks are kept after bad pixels are replaced so that the spectral
extraction can down-weight replaced pixels.
"""
import json
import os
import warnings

import numpy as np
from astropy.io import fits
from scipy import ndimage

from .instruments import BAD_DQ_FLAGS, identify_mode, get_mode_config
from .io import find_exposures, select_exposure, read_int_times, output_filename
from .masks import build_trace_mask, ensure_min_background
from .onef import group_background
from .robust import running_median, nanmedian, mad_sigma, MAD_TO_SIGMA
import bottleneck as bn

__all__ = ['CubeCleaner', 'ExposureCleaner', 'hot_pixel_map', 'neighbour_median', 'residual_sigma']

_HEADER_KEYS = ('TARGPROP', 'TARGNAME', 'INSTRUME', 'DETECTOR', 'EXP_TYPE', 'FILTER', 'GRATING',
                'PUPIL', 'SUBARRAY', 'READPATT', 'NGROUPS', 'NINTS', 'NFRAMES', 'GROUPGAP',
                'TFRAME', 'TGROUP', 'EFFINTTM', 'EFFEXPTM', 'SUBSTRT1', 'SUBSTRT2', 'SUBSIZE1',
                'SUBSIZE2', 'NOUTPUTS', 'PWCPOS', 'FXD_SLIT', 'PROGRAM', 'OBSERVTN', 'VISIT',
                'EXPSTART', 'EXPEND', 'CAL_VER', 'CRDS_CTX')


def _dq_bitmask(flags):
    from stdatamodels.jwst.datamodels import dqflags
    m = 0
    for f in flags:
        m |= dqflags.pixel[f]
    return m


def neighbour_median(img, dispersion_axis, window=5):
    """Median of the *neighbours* of each pixel (the pixel itself excluded).

    Neighbours are the ``window // 2`` pixels on either side along the
    dispersion direction, or the 8 surrounding pixels for imaging data
    (``dispersion_axis=None``). Excluding the central pixel matters: a median
    filter that includes it returns the pixel itself on smooth gradients,
    which makes the residuals (and their MAD) vanish.
    """
    half = max(1, int(window) // 2)
    pad = np.pad(np.asarray(img, dtype=float), half, mode='reflect')
    ny, nx = np.shape(img)
    if dispersion_axis is None:
        offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    elif dispersion_axis == 1:
        offs = [(0, k) for k in range(-half, half + 1) if k != 0]
    else:
        offs = [(k, 0) for k in range(-half, half + 1) if k != 0]
    shifts = [pad[half + dy: half + dy + ny, half + dx: half + dx + nx] for dy, dx in offs]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.stack(shifts), axis=0)


def spatial_outliers(img, dispersion_axis, nsigma=10., window=5, rel_tol=0.):
    """Pixels deviating from their neighbours by ``> nsigma * sigma_line + rel_tol * |neighbours|``."""
    sm = neighbour_median(img, dispersion_axis, window)
    resid = img - sm
    if dispersion_axis is None:
        sig = mad_sigma(resid[np.isfinite(resid)])
    else:
        sig = MAD_TO_SIGMA * bn.nanmedian(np.abs(resid), axis=dispersion_axis)
        pos = sig[sig > 0]
        sig = np.where(sig > 0, sig, np.nanmedian(pos) if pos.size else 1.)
        sig = np.expand_dims(sig, dispersion_axis)
    return np.isfinite(resid) & (np.abs(resid) > nsigma * sig + rel_tol * np.abs(sm))


def hot_pixel_map(median, std, dispersion_axis, nsigma=10., window=5, rel_tol=0.25,
                  noisy_factor=3.):
    """Boolean map of hot / cold (from ``median``) and noisy (from ``std``) pixels.

    See :meth:`CubeCleaner.find_hot_pixels` for the criteria. ``std`` may be
    ``None`` to skip the noisy-pixel test.
    """
    new = spatial_outliers(median, dispersion_axis, nsigma, window, rel_tol)
    if std is not None:
        sm = neighbour_median(std, dispersion_axis, window)
        ratio = std / np.where(sm > 0, sm, np.nan)
        new |= np.isfinite(ratio) & (ratio > noisy_factor)
    return new


def residual_sigma(data, window, mask=None, max_mem_gb=4.):
    """Robust per-pixel noise of a cube around a running median in time.

    This is the noise estimate used for time-domain cosmic-ray detection:
    ``1.4826 * median(|d - running_median(d, window)|)`` for every pixel,
    with a leave-one-out running median (the sample is excluded from its
    own window) so that the noise is not underestimated.
    Pixels in ``mask`` (``True`` = bad, same shape as ``data``) are ignored.
    """
    nint, ny, nx = data.shape
    rows = int(max(1, min(ny, max_mem_gb * 1e9 // (nint * nx * 4 * (window + 3)))))
    out = np.empty((ny, nx), np.float32)
    for y0 in range(0, ny, rows):
        ys = slice(y0, min(ny, y0 + rows))
        d = data[:, ys] if mask is None else np.where(mask[:, ys], np.nan, data[:, ys])
        model = running_median(d, window, axis=0, exclude_center=True)
        out[ys] = MAD_TO_SIGMA * bn.nanmedian(np.abs(d - model), axis=0)
    return out


class CubeCleaner:
    """Clean a time series of 2D images (``rateints``) for spectral extraction.

    Typical use::

        cc = CubeCleaner.from_files('/path/to/stage1/output', wavelength='..._wavelength.fits')
        cc.fix_errors()
        cc.find_hot_pixels()
        cc.find_cosmic_rays()
        cc.replace_bad_pixels()
        cc.subtract_background()       # optional
        cc.save('/path/to/cleaned_cube.fits')

    Parameters
    ----------
    data, err : ndarray
        Flux and uncertainty cubes ``(nint, ny, nx)``.
    badpix : ndarray or None
        2D persistent bad-pixel map (``1`` good, ``0`` bad).
    crmask : ndarray or None
        3D transient bad-pixel mask (``True`` bad).
    times : dict or ndarray or None
        Integration times. Either the ``INT_TIMES`` columns (dict) or a 1D
        array of mid-times. Following the jwst ``INT_TIMES`` convention,
        ``times`` are **BJD_TDB - 2400000.5** (use :attr:`bjd` for full
        BJD_TDB).
    wavelength : ndarray or None
        Wavelength map, ``(ny, nx)`` or ``(norder, ny, nx)`` for SOSS.
    mode : str or None
        jstark mode name (sets the dispersion axis).
    dispersion_axis : int or None
        Overrides the mode default.
    header : dict or None
        Metadata to propagate to the saved product.
    verbose : bool
        Print progress information.
    """

    def __init__(self, data, err, badpix=None, crmask=None, times=None, wavelength=None,
                 mode=None, dispersion_axis='default', header=None, verbose=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.err = np.asarray(err, dtype=np.float32)
        nint, ny, nx = self.data.shape
        self.badpix = np.ones((ny, nx), np.uint8) if badpix is None else np.asarray(badpix, np.uint8)
        self.crmask = np.zeros((nint, ny, nx), bool) if crmask is None else np.asarray(crmask, bool)
        if isinstance(times, dict):
            self.int_times = times
            key = 'int_mid_BJD_TDB'
            self.times = np.asarray(times[key]) if key in times else None
        else:
            self.int_times = {}
            self.times = None if times is None else np.asarray(times)
        self.wavelength = None if wavelength is None else np.asarray(wavelength)
        self.mode = mode
        self.config = get_mode_config(mode) if mode is not None else None
        if dispersion_axis == 'default':
            dispersion_axis = self.config.dispersion_axis if self.config is not None else 1
        self.dispersion_axis = dispersion_axis
        self.header = dict(header or {})
        self.verbose = verbose
        self.trace_mask = None
        self.bkg_mask = None
        self.history = []
        self.diagnostics = {}

    # ------------------------------------------------------------------
    # Construction / IO
    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, inputs, suffix='rateints', exposure=None, segments=None, wavelength=None,
                   dq_flags=BAD_DQ_FLAGS, persistent_frac=0.5, verbose=True):
        """Load (and concatenate) segments of an exposure into one in-memory cube.

        Convenient for a single segment or small exposures. For large
        exposures use :class:`ExposureCleaner`, which never holds more than
        one segment (plus a few overlapping integrations) in memory.

        Parameters
        ----------
        inputs : str or list
            Directory, glob pattern, or list of segment files.
        suffix : str
            File suffix to search for (default ``'rateints'``).
        exposure : str or None
            Exposure selection if several exposures are present.
        segments : list of int or None
            0-indexed segments to load (default: all).
        wavelength : str, ndarray or None
            Wavelength map or the FITS file written by
            :class:`~jstark.reduction.Stage2`.
        dq_flags : sequence of str
            DQ flags considered bad.
        persistent_frac : float
            Pixels flagged in more than this fraction of the integrations
            are treated as persistently bad (``badpix``); otherwise only the
            flagged integrations are masked (``crmask``).
        """
        exps = find_exposures(inputs, suffix=suffix)
        key, files = select_exposure(exps, exposure, verbose=verbose)
        if segments is not None:
            files = [files[i] for i in segments]
        bitmask = _dq_bitmask(dq_flags)
        data, err, dqbad, times = [], [], [], {}
        for f in files:
            if verbose:
                print(f'Loading {os.path.basename(f)}')
            with fits.open(f, memmap=False) as hdul:
                data.append(np.asarray(hdul['SCI'].data, dtype=np.float32))
                err.append(np.asarray(hdul['ERR'].data, dtype=np.float32))
                dqbad.append((np.asarray(hdul['DQ'].data) & bitmask) > 0)
            it = read_int_times(f)
            for k, v in it.items():
                times.setdefault(k, []).append(v)
        data = np.concatenate(data)
        err = np.concatenate(err)
        dqbad = np.concatenate(dqbad)
        times = {k: np.concatenate(v) for k, v in times.items()}
        # keep only the integrations actually present (INT_TIMES may be longer)
        if 'integration_number' in times and len(times['integration_number']) != data.shape[0]:
            warnings.warn('INT_TIMES length does not match the number of integrations.')
        h0 = fits.getheader(files[0], 0)
        header = {k: h0[k] for k in _HEADER_KEYS if k in h0}
        header['EXPOSURE'] = key
        header['NSEGMENT'] = len(files)
        mode = identify_mode(h0)
        badpix = (dqbad.mean(axis=0) <= persistent_frac).astype(np.uint8)
        crmask = dqbad & (badpix[None] == 1)
        if isinstance(wavelength, str):
            wavelength = fits.getdata(wavelength, 'WAVELENGTH')
        obj = cls(data, err, badpix=badpix, crmask=crmask, times=times, wavelength=wavelength,
                  mode=mode, header=header, verbose=verbose)
        obj.files = files
        obj._log('from_files', files=[os.path.basename(f) for f in files], dq_flags=list(dq_flags),
                 persistent_frac=persistent_frac,
                 n_persistent=int((badpix == 0).sum()), n_transient=int(crmask.sum()))
        if verbose:
            print(f'{mode}: cube {data.shape}, {int((badpix == 0).sum())} persistent bad pixels '
                  f'from DQ, {crmask.mean() * 100:.3f}% transient DQ flags')
        return obj

    def save(self, fname, overwrite=True):
        """Save the cleaned cube to a multi-extension FITS file."""
        hdr = fits.Header()
        for k, v in self.header.items():
            try:
                hdr[k] = v
            except ValueError:
                pass
        hdr['JSMODE'] = (self.mode or '', 'jstark mode')
        hdr['DISPAXIS'] = (-1 if self.dispersion_axis is None else self.dispersion_axis,
                           'array axis of dispersion (1=x, 0=y, -1=none)')
        hdus = [fits.PrimaryHDU(header=hdr),
                fits.ImageHDU(self.data, name='SCI'),
                fits.ImageHDU(self.err, name='ERR'),
                fits.ImageHDU(self.badpix.astype(np.uint8), name='BADPIX'),
                fits.ImageHDU(self.crmask.astype(np.uint8), name='CRMASK')]
        if self.int_times:
            cols = [fits.Column(name=k, array=np.asarray(v),
                                format='D' if np.asarray(v).dtype.kind == 'f' else 'K')
                    for k, v in self.int_times.items() if np.asarray(v).dtype.kind in 'fiu']
            hdus.append(fits.BinTableHDU.from_columns(cols, name='INT_TIMES'))
        elif self.times is not None:
            hdus.append(fits.BinTableHDU.from_columns(
                [fits.Column(name='int_mid_BJD_TDB', array=self.times, format='D')], name='INT_TIMES'))
        if self.wavelength is not None:
            hdus.append(fits.ImageHDU(np.asarray(self.wavelength, np.float32), name='WAVELENGTH'))
        if self.trace_mask is not None:
            hdus.append(fits.ImageHDU(self.trace_mask.astype(np.uint8), name='TRACEMASK'))
        hist = json.dumps(self.history)
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name='history', format=f'{max(1, len(hist))}A', array=np.array([hist]))],
            name='HISTORY'))
        fits.HDUList(hdus).writeto(fname, overwrite=overwrite)
        if self.verbose:
            print(f'Saved cleaned cube to {fname}')
        return fname

    @classmethod
    def load(cls, fname, verbose=True):
        """Load a cube written by :meth:`save`."""
        with fits.open(fname) as h:
            hdr = dict(h[0].header)
            mode = hdr.get('JSMODE') or None
            disp = hdr.get('DISPAXIS', 1)
            times = {n: np.array(h['INT_TIMES'].data[n]) for n in h['INT_TIMES'].data.names} \
                if 'INT_TIMES' in h else None
            obj = cls(h['SCI'].data, h['ERR'].data, badpix=h['BADPIX'].data,
                      crmask=h['CRMASK'].data.astype(bool), times=times,
                      wavelength=h['WAVELENGTH'].data if 'WAVELENGTH' in h else None,
                      mode=mode, dispersion_axis=None if disp == -1 else disp, header=hdr,
                      verbose=verbose)
            if 'TRACEMASK' in h:
                obj.trace_mask = h['TRACEMASK'].data.astype(bool)
            if 'HISTORY' in h:
                try:
                    obj.history = json.loads(h['HISTORY'].data['history'][0])
                except Exception:
                    pass
        return obj

    def _log(self, step, **kwargs):
        self.history.append({'step': step, **{k: _plain(v) for k, v in kwargs.items()}})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def bjd(self):
        """Mid-integration times as full BJD_TDB (``times + 2400000.5``)."""
        return None if self.times is None else self.times + 2400000.5

    @property
    def good_mask(self):
        """3D boolean mask of good pixels (``True`` = good)."""
        return (self.badpix[None] == 1) & ~self.crmask

    def masked_data(self):
        """Copy of the data with all bad pixels set to NaN."""
        return np.where(self.good_mask, self.data, np.nan)

    def median_frame(self):
        """Median image over time, ignoring bad pixels."""
        return nanmedian(self.masked_data(), axis=0)

    def summary(self):
        nint = self.shape[0]
        txt = (f'{self.mode}: {self.shape}; persistent bad pixels: {(self.badpix == 0).sum()} '
               f'({(self.badpix == 0).mean() * 100:.2f}%); transient flags: {self.crmask.sum()} '
               f'({self.crmask.mean() * 100:.4f}%, {self.crmask.sum() / nint:.1f} per integration)')
        print(txt)
        return txt

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------
    def fix_errors(self):
        """Flag and repair invalid data / uncertainties.

        Pixels with non-finite data, or non-finite / non-positive errors are
        added to ``crmask``. Their errors are replaced by the temporal median
        error of the pixel (or, if the pixel never has a valid error, by the
        median error of the frame times 1000), so that the error array is
        finite and positive everywhere.
        """
        bad = ~np.isfinite(self.data) | ~np.isfinite(self.err) | (self.err <= 0)
        nbad = int(bad.sum())
        self.crmask |= bad & (self.badpix[None] == 1)
        e = np.where(bad, np.nan, self.err)
        med_err = nanmedian(e, axis=0)
        never = ~np.isfinite(med_err)
        if np.any(never):
            self.badpix[never] = 0
            med_err[never] = 1e3 * np.nanmedian(med_err)
        idx = np.nonzero(bad)
        self.err[idx] = med_err[idx[1], idx[2]]
        self._log('fix_errors', n_invalid=nbad, n_never_valid=int(never.sum()))
        if self.verbose:
            print(f'fix_errors: {nbad} invalid values, {int(never.sum())} pixels never valid')
        return nbad

    def find_hot_pixels(self, nsigma=10., window=5, rel_tol=0.25, noisy_factor=3., use_std=True,
                        median=None, std=None):
        """Find persistently deviant pixels (hot, cold, noisy) and update ``badpix``.

        The median image is compared with the median of each pixel's
        neighbours along the dispersion direction (where the PSF varies
        slowly; the 8 surrounding pixels for imaging data). A pixel is
        flagged when

        ``|median - neighbours| > nsigma * sigma_line + rel_tol * |neighbours|``

        where ``sigma_line`` is the robust scatter of the residuals along
        that line of the detector. The relative tolerance prevents the
        (harmless, temporally stable) pixel-to-pixel QE variations of bright
        pixels from being flagged: the median frame has so much S/N that a
        1 per cent flat-field difference is many sigma.

        If ``use_std``, pixels whose temporal scatter exceeds
        ``noisy_factor`` times that of their neighbours are also flagged
        (noisy / RTN pixels, which do harm relative photometry).

        Parameters
        ----------
        median, std : ndarray or None
            Precomputed median and robust temporal-scatter images (e.g. for
            the whole exposure, see :class:`ExposureCleaner`). Computed from
            this cube when ``None``.

        Returns
        -------
        int
            Number of newly flagged pixels.
        """
        if median is None or (use_std and std is None):
            md = self.masked_data()
            if median is None:
                median = nanmedian(md, axis=0)
            if use_std and std is None:
                std = mad_sigma(md, axis=0)
            del md
        new = hot_pixel_map(median, std if use_std else None, self.dispersion_axis, nsigma,
                            window, rel_tol, noisy_factor)
        new &= self.badpix == 1
        self.badpix[new] = 0
        n = int(new.sum())
        self._log('find_hot_pixels', nsigma=nsigma, window=window, rel_tol=rel_tol,
                  noisy_factor=noisy_factor, use_std=use_std, n_new=n)
        if self.verbose:
            print(f'find_hot_pixels: {n} new persistent bad pixels '
                  f'(total {(self.badpix == 0).sum()})')
        return n

    def find_cosmic_rays(self, nsigma=5., method='running', window=None, niters=2,
                         two_sided=True, bad_int_frac=0.01, max_mem_gb=4., sigma=None):
        """Identify transient outliers (cosmic rays) in the time domain.

        Parameters
        ----------
        nsigma : float
            Detection threshold in units of the per-pixel robust temporal
            scatter.
        method : {'running', 'median'}
            ``'running'``: each pixel is compared with a running median of
            its own time series, excluding the tested integration itself
            (robust against transits, trends and tilt events). ``'median'``: each frame is compared with the
            median frame scaled by a white-light curve (the classic
            approach; sensitive to trace motions).
        window : int or None
            Running-median window (integrations); mode default if ``None``.
        niters : int
            Number of iterations.
        two_sided : bool
            Also flag negative outliers.
        bad_int_frac : float
            Integrations in which more than this fraction of pixels is
            flagged are reported in ``diagnostics['bad_integrations']``.
        sigma : ndarray or None
            2D per-pixel noise to use instead of the MAD of the residuals
            (useful for very short segments).

        Returns
        -------
        int
            Number of newly flagged pixels.
        """
        if window is None:
            window = self.config.nint_window if self.config is not None else 11
        nint, ny, nx = self.shape
        n_start = int(self.crmask.sum())
        good_static = self.badpix == 1
        for it in range(niters):
            new_total = 0
            if method == 'running':
                rows = int(max(1, min(ny, max_mem_gb * 1e9 // (nint * nx * 4 * (window + 4)))))
                for y0 in range(0, ny, rows):
                    ys = slice(y0, min(ny, y0 + rows))
                    d = np.where(self.crmask[:, ys], np.nan, self.data[:, ys])
                    # leave-one-out: the tested integration is not part of its own model
                    model = running_median(d, window, axis=0, exclude_center=True)
                    resid = d - model
                    if sigma is None:
                        sig = MAD_TO_SIGMA * bn.nanmedian(np.abs(resid), axis=0)
                    else:
                        sig = np.asarray(sigma, dtype=np.float32)[ys]
                    sig = np.where(sig > 0, sig, np.inf)
                    z = resid / sig[None]
                    hit = (np.abs(z) if two_sided else z) > nsigma
                    hit &= good_static[None, ys] & np.isfinite(z)
                    new_total += int(hit.sum())
                    self.crmask[:, ys] |= hit
            elif method == 'median':
                d = np.where(self.crmask, np.nan, self.data)
                med = nanmedian(d, axis=0)
                scale = self._white_light_scale(d, med)
                resid = d - scale[:, None, None] * med[None]
                sig = mad_sigma(resid, axis=0) if sigma is None else np.asarray(sigma, np.float32)
                sig = np.where(sig > 0, sig, np.inf)
                z = resid / sig[None]
                hit = (np.abs(z) if two_sided else z) > nsigma
                hit &= good_static[None] & np.isfinite(z)
                new_total = int(hit.sum())
                self.crmask |= hit
                del d, resid, z
            else:
                raise ValueError("method must be 'running' or 'median'.")
            if new_total == 0:
                break
        n_new = int(self.crmask.sum()) - n_start
        frac = self.crmask.reshape(nint, -1).mean(axis=1)
        badints = np.where(frac > bad_int_frac)[0]
        self.diagnostics['cr_frac_per_int'] = frac
        self.diagnostics['bad_integrations'] = badints
        self._log('find_cosmic_rays', nsigma=nsigma, method=method, window=window, niters=niters,
                  n_new=n_new, bad_integrations=badints.tolist())
        if self.verbose:
            print(f'find_cosmic_rays: {n_new} new transient outliers '
                  f'({n_new / nint:.1f} per integration); {len(badints)} integrations with '
                  f'>{bad_int_frac * 100:.1f}% flagged pixels')
        return n_new

    def _white_light_scale(self, d, med):
        sel = np.isfinite(med) & (med > np.nanpercentile(med, 90))
        num = bn.nansum(np.where(np.isfinite(d[:, sel]), d[:, sel], med[sel][None]), axis=1)
        s = num / bn.nansum(med[sel])
        return s / np.nanmedian(s)

    def replace_bad_pixels(self, transient='temporal', persistent='dispersion', window=None,
                           maxdist=5, box=5):
        """Replace bad pixels in ``data`` (the masks are kept).

        Parameters
        ----------
        transient : {'temporal', 'scaled_median', None}
            Replacement of transient bad pixels (``crmask``):
            ``'temporal'`` uses the running median of the pixel's own time
            series; ``'scaled_median'`` uses the median frame scaled by the
            white-light curve.
        persistent : {'dispersion', 'box', None}
            Replacement of persistent bad pixels (``badpix == 0``):
            ``'dispersion'`` linearly interpolates between the nearest good
            pixels along the dispersion direction (within ``maxdist``);
            ``'box'`` takes the median of the good pixels in a
            ``box x box`` neighbourhood (the classic approach). Pixels that
            cannot be filled with ``'dispersion'`` fall back to ``'box'``.
        window : int or None
            Running-median window for ``transient='temporal'``.
        """
        n_t = n_p = 0
        if transient is not None:
            n_t = self._replace_transient(transient, window)
        if persistent is not None:
            n_p = self._replace_persistent(persistent, maxdist, box)
        self._log('replace_bad_pixels', transient=transient, persistent=persistent,
                  n_transient=n_t, n_persistent=n_p)
        if self.verbose:
            print(f'replace_bad_pixels: {n_t} transient and {n_p} persistent pixels replaced')

    def _replace_transient(self, method, window):
        idx = np.nonzero(self.crmask.any(axis=0) & (self.badpix == 1))
        if len(idx[0]) == 0:
            return 0
        if method == 'temporal':
            window = window or (self.config.nint_window if self.config is not None else 11)
            ts = self.data[:, idx[0], idx[1]]
            m = self.crmask[:, idx[0], idx[1]]
            model = running_median(np.where(m, np.nan, ts), window, axis=0)
            # fall back to the pixel median where the window is fully masked
            pix_med = bn.nanmedian(np.where(m, np.nan, ts), axis=0)
            model = np.where(np.isfinite(model), model, pix_med[None])
            ts[m] = model[m]
            self.data[:, idx[0], idx[1]] = ts
        elif method == 'scaled_median':
            d = np.where(self.crmask, np.nan, self.data)
            med = nanmedian(d, axis=0)
            scale = self._white_light_scale(d, med)
            del d
            ii = np.nonzero(self.crmask & (self.badpix[None] == 1))
            self.data[ii] = scale[ii[0]] * med[ii[1], ii[2]]
        else:
            raise ValueError(f'Unknown transient replacement {method}.')
        return int((self.crmask & (self.badpix[None] == 1)).sum())

    def _replace_persistent(self, method, maxdist, box):
        bad = self.badpix == 0
        if not bad.any():
            return 0
        remaining = bad.copy()
        if method == 'dispersion' and self.dispersion_axis is not None:
            remaining = self._interp_along_dispersion(bad, maxdist)
        elif method not in ('dispersion', 'box'):
            raise ValueError(f'Unknown persistent replacement {method}.')
        if remaining.any():
            self._box_median(remaining, box)
        return int(bad.sum())

    def _interp_along_dispersion(self, bad, maxdist):
        """Linear interpolation between nearest good pixels along dispersion."""
        ax = self.dispersion_axis
        good = ~bad
        if ax == 0:
            good_t, bad_t = good.T, bad.T
        else:
            good_t, bad_t = good, bad
        nrow, ncol = good_t.shape
        cols = np.arange(ncol)
        # index of nearest good pixel to the left / right of each pixel
        left = np.where(good_t, cols[None], -1)
        left = np.maximum.accumulate(left, axis=1)
        right = np.where(good_t, cols[None], ncol)
        right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]
        r, c = np.nonzero(bad_t)
        l, rr = left[r, c], right[r, c]
        ok = (l >= 0) & (rr < ncol) & (c - l <= maxdist) & (rr - c <= maxdist)
        # one-sided fill at the edges
        only_l = (l >= 0) & ~ok & (c - l <= maxdist) & ((rr >= ncol) | (rr - c > maxdist))
        only_r = (rr < ncol) & ~ok & (rr - c <= maxdist) & ((l < 0) | (c - l > maxdist))
        def pix(rows, cs):
            return (self.data[:, rows, cs] if ax == 1 else self.data[:, cs, rows],
                    self.err[:, rows, cs] if ax == 1 else self.err[:, cs, rows])
        filled = np.zeros_like(bad_t)
        if ok.any():
            w = ((c[ok] - l[ok]) / (rr[ok] - l[ok])).astype(np.float32)
            dl, el = pix(r[ok], l[ok])
            dr, er = pix(r[ok], rr[ok])
            val = (1 - w) * dl + w * dr
            ev = np.sqrt(((1 - w) * el) ** 2 + (w * er) ** 2)
            self._assign(r[ok], c[ok], val, ev)
            filled[r[ok], c[ok]] = True
        for sel, src in ((only_l, l), (only_r, rr)):
            if sel.any():
                d, e = pix(r[sel], src[sel])
                self._assign(r[sel], c[sel], d, e)
                filled[r[sel], c[sel]] = True
        remaining = bad_t & ~filled
        return remaining.T if ax == 0 else remaining

    def _assign(self, rows, cols, val, ev):
        if self.dispersion_axis == 0:
            self.data[:, cols, rows] = val
            self.err[:, cols, rows] = ev
        else:
            self.data[:, rows, cols] = val
            self.err[:, rows, cols] = ev

    def _box_median(self, bad, box):
        half = box // 2
        nint, ny, nx = self.shape
        good = self.badpix == 1
        for y, x in zip(*np.nonzero(bad)):
            h = half
            while True:  # widen the box until it contains good pixels
                y0, y1 = max(0, y - h), min(ny, y + h + 1)
                x0, x1 = max(0, x - h), min(nx, x + h + 1)
                g = good[y0:y1, x0:x1]
                if g.any() or (y1 - y0 >= ny and x1 - x0 >= nx):
                    break
                h *= 2
            if not g.any():
                continue
            self.data[:, y, x] = np.median(self.data[:, y0:y1, x0:x1][:, g], axis=1)
            self.err[:, y, x] = np.median(self.err[:, y0:y1, x0:x1][:, g], axis=1)

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    def build_trace_mask(self, method='threshold', min_pixels=None, **kwargs):
        """Build source and background masks from the median frame."""
        opts = dict(self.config.trace_mask) if self.config is not None else {}
        opts.update(kwargs)
        mp = opts.pop('min_pixels', 4) if min_pixels is None else min_pixels
        opts.pop('method', None)
        med = self.median_frame()
        valid = self.badpix == 1
        self.trace_mask = build_trace_mask(med, self.dispersion_axis, method=method, valid=valid,
                                           **opts)
        axis = 'row' if self.dispersion_axis == 0 else 'column'
        bkg = ~self.trace_mask & valid & np.isfinite(med)
        self.bkg_mask = ensure_min_background(bkg, med, axis, min_pixels=mp, valid=valid)
        return self.trace_mask, self.bkg_mask

    def subtract_background(self, method='median', axis=None, nsigma=3., maxiters=3,
                            poly_order=1, bkg_mask=None, chunk=200):
        """Integration-level background subtraction along the cross-dispersion direction.

        Parameters
        ----------
        method : {'median', 'poly'}
            Sigma-clipped median, or polynomial fit, of the background pixels.
        axis : {'column', 'row'} or None
            Default: one value per spectral pixel, i.e. ``'column'`` when the
            dispersion is along x and ``'row'`` when it is along y.
        bkg_mask : ndarray or None
            Background pixels; built with :meth:`build_trace_mask` if needed.
        """
        if axis is None:
            axis = 'row' if self.dispersion_axis == 0 else 'column'
        if bkg_mask is not None:
            self.bkg_mask = np.asarray(bkg_mask, bool)
        if self.bkg_mask is None:
            self.build_trace_mask()
        mask = self.bkg_mask & (self.badpix == 1)
        d4 = self.data[:, None]          # view: corrected in place
        _, info = group_background(d4, mask, method=method, axis=axis, nsigma=nsigma,
                                   maxiters=maxiters, poly_order=poly_order, chunk=chunk,
                                   exclude=self.crmask[:, None])
        self.diagnostics['background_profile'] = info['profile'][:, 0]
        self._log('subtract_background', method=method, axis=axis, nsigma=nsigma,
                  poly_order=poly_order)
        if self.verbose:
            print(f'subtract_background: {method} per {axis}')

    def subtract_background_model(self, model, regions=None, bkg_mask=None):
        """Scale and subtract a 2D background model (e.g. NIRISS/SOSS zodi model).

        For every integration, the model is scaled to the data with the
        median ratio over background pixels, optionally separately in
        several column ``regions`` (list of ``(x0, x1)``), and subtracted.

        Returns
        -------
        ndarray
            Scale factors, shape ``(nint, nregion)``.
        """
        model = np.asarray(model, dtype=np.float32)
        if bkg_mask is None:
            if self.bkg_mask is None:
                self.build_trace_mask()
            bkg_mask = self.bkg_mask
        mask = bkg_mask & (self.badpix == 1) & np.isfinite(model) & (model != 0)
        nx = self.shape[2]
        regions = regions or [(0, nx)]
        scales = np.zeros((self.shape[0], len(regions)), np.float32)
        for k, (x0, x1) in enumerate(regions):
            m = np.zeros_like(mask)
            m[:, x0:x1] = mask[:, x0:x1]
            d = np.where(self.crmask[:, m], np.nan, self.data[:, m])
            ratio = d / model[m][None]
            s = bn.nanmedian(ratio, axis=1)
            scales[:, k] = s
            self.data[:, :, x0:x1] -= s[:, None, None] * model[None, :, x0:x1]
        self.diagnostics['background_model_scale'] = scales
        self._log('subtract_background_model', regions=regions)
        return scales

    # ------------------------------------------------------------------
    # Quick-look products
    # ------------------------------------------------------------------
    def to_apphoto(self, **kwargs):
        """Create a :class:`jstark.ApPhoto` object (e.g. for MIRI imaging time series).

        The 3D :attr:`good_mask` is passed as the bad-pixel mask and the
        times are full BJD_TDB. Keyword arguments go to ``ApPhoto``.
        """
        from ..jstark import ApPhoto
        t = self.bjd if self.times is not None else np.arange(self.shape[0], dtype=float)
        return ApPhoto(t, self.data, self.err, self.good_mask.astype(float), **kwargs)

    def white_light(self, mask=None, normalize=True):
        """Quick white-light curve: sum of (bad-pixel-replaced) data in ``mask``."""
        if mask is None:
            mask = self.trace_mask if self.trace_mask is not None else np.ones(self.shape[1:], bool)
        f = bn.nansum(self.data[:, mask], axis=1)
        return f / np.nanmedian(f) if normalize else f


class ExposureCleaner:
    """Clean a multi-segment exposure one segment at a time (two passes).

    A full time-series exposure can have tens of segments and far more data
    than fits comfortably in memory, so the segments are never stitched:

    1. :meth:`compute_global` streams through all segments once and builds
       the exposure-wide quantities: the median frame (median of the
       per-segment medians), the robust temporal-scatter image, the
       persistent bad-pixel map (DQ flags + hot / noisy pixels), the
       source / background masks, and the per-pixel noise used for
       cosmic-ray detection (so short segments are as reliable as long ones).
    2. :meth:`clean_segment` cleans one segment with a :class:`CubeCleaner`,
       using the exposure-wide bad-pixel map and masks. The segment is
       padded with ``overlap`` integrations from the neighbouring segments
       so that the time-domain steps (cosmic-ray detection and replacement
       with a running median) behave at segment boundaries exactly as they
       would in a stitched cube. The padding is removed before saving
       ``<root>-segNNN_<det>_cleaned.fits``.

    :meth:`run` does both. :meth:`save_info` writes a small exposure-level
    file (``<exposure>_cleaninfo.fits``) with the bad-pixel map, masks,
    median frame, wavelength map, all integration times and the list of
    cleaned segment files; :meth:`load` restores the object from it, and
    :meth:`iter_segments` / :meth:`load_segment` give the cleaned segments
    one at a time (e.g. for spectral extraction).

    Parameters
    ----------
    inputs : str or list
        Directory, glob pattern or list of segment files (``*_rateints.fits``).
    outdir : str or None
        Output directory (default: the input directory).
    suffix : str
        Input file suffix.
    exposure : str or None
        Exposure selection if several are present.
    wavelength : str, ndarray or None
        Wavelength map, or the file written by :class:`~jstark.reduction.Stage2`.
    dq_flags : sequence of str
        DQ flags considered bad.
    persistent_frac : float
        Pixels flagged in more than this fraction of all integrations are
        persistently bad.
    verbose : bool
        Print progress information.
    """

    def __init__(self, inputs, outdir=None, suffix='rateints', exposure=None, wavelength=None,
                 dq_flags=BAD_DQ_FLAGS, persistent_frac=0.5, verbose=True):
        exps = find_exposures(inputs, suffix=suffix)
        self.exposure, self.files = select_exposure(exps, exposure, verbose=verbose)
        h0 = fits.getheader(self.files[0], 0)
        self.mode = identify_mode(h0)
        self.config = get_mode_config(self.mode)
        self.dispersion_axis = self.config.dispersion_axis
        self.header = {k: h0[k] for k in _HEADER_KEYS if k in h0}
        self.header['EXPOSURE'] = self.exposure
        self.header['NSEGMENT'] = len(self.files)
        self.outdir = outdir or os.path.dirname(os.path.abspath(self.files[0]))
        os.makedirs(self.outdir, exist_ok=True)
        self.seg_nint = [int(fits.getheader(f, 'SCI')['NAXIS3']) for f in self.files]
        self.seg_start = np.concatenate([[0], np.cumsum(self.seg_nint)[:-1]]).astype(int)
        if isinstance(wavelength, str):
            wavelength = fits.getdata(wavelength, 'WAVELENGTH')
        self.wavelength = None if wavelength is None else np.asarray(wavelength)
        self.dq_flags = tuple(dq_flags)
        self.persistent_frac = persistent_frac
        self.verbose = verbose
        self.badpix = self.median = self.std = self.cr_sigma = None
        self.trace_mask = self.bkg_mask = None
        self.outputs = {}
        self.history = []
        self.diagnostics = {}
        if verbose:
            print(f'ExposureCleaner: {self.mode} | {self.exposure} | {len(self.files)} segment(s), '
                  f'{sum(self.seg_nint)} integrations')

    @property
    def nint(self):
        return int(sum(self.seg_nint))

    def _log(self, step, **kwargs):
        self.history.append({'step': step, **{k: _plain(v) for k, v in kwargs.items()}})

    # ------------------------------------------------------------------
    def read_segment(self, i, ints=None):
        """Read integrations ``ints`` (slice; default all) of segment ``i``.

        Returns
        -------
        data, err : ndarray
        dqbad : ndarray of bool
            Pixels flagged with any of ``dq_flags``.
        times : dict
            ``INT_TIMES`` columns for the selected integrations.
        """
        ints = slice(None) if ints is None else ints
        bitmask = _dq_bitmask(self.dq_flags)
        # ``section`` reads only the requested integrations (and handles scaled DQ arrays)
        with fits.open(self.files[i]) as h:
            data = np.array(h['SCI'].section[ints], dtype=np.float32)
            err = np.array(h['ERR'].section[ints], dtype=np.float32)
            dqbad = (np.asarray(h['DQ'].section[ints]).astype(np.uint32) & bitmask) > 0
            hdr = h[0].header
            times = {}
            if 'INT_TIMES' in h:
                tab = h['INT_TIMES'].data
                num = np.asarray(tab['integration_number'])
                i0 = hdr.get('INTSTART', num[0])
                sel = np.where((num >= i0) & (num < i0 + self.seg_nint[i]))[0]
                times = {n: np.asarray(tab[n])[sel][ints] for n in tab.names}
        return data, err, dqbad, times

    def compute_global(self, hot_pixels=True, hot_kwargs=None, trace_kwargs=None):
        """Pass 1: exposure-wide median / scatter images, bad-pixel map and masks.

        Parameters
        ----------
        hot_pixels : bool
            Flag hot / cold / noisy pixels (see :func:`hot_pixel_map`).
        hot_kwargs : dict or None
            Keyword arguments for :func:`hot_pixel_map`.
        trace_kwargs : dict or None
            Keyword arguments for :func:`~jstark.reduction.masks.build_trace_mask`
            (defaults from the mode configuration).
        """
        nflag, meds, stds, rsig = None, [], [], []
        window = self.config.nint_window
        for i in range(len(self.files)):
            data, err, dqbad, _ = self.read_segment(i)
            bad = dqbad | ~np.isfinite(data) | ~np.isfinite(err) | (err <= 0)
            nflag = bad.sum(0) if nflag is None else nflag + bad.sum(0)
            d = np.where(bad, np.nan, data)
            meds.append(nanmedian(d, axis=0))
            stds.append(mad_sigma(d, axis=0))
            rsig.append(residual_sigma(d, window))
            if self.verbose:
                print(f'  pass 1: segment {i + 1}/{len(self.files)} read')
            del data, err, dqbad, d
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.median = np.nanmedian(np.stack(meds), axis=0).astype(np.float32)
            self.std = np.nanmedian(np.stack(stds), axis=0).astype(np.float32)
            # exposure-wide noise for cosmic-ray detection (weighted by segment length)
            w = np.asarray(self.seg_nint, float)
            rs = np.stack(rsig)
            self.cr_sigma = np.sqrt(np.nansum(w[:, None, None] * rs ** 2, axis=0) /
                                    np.nansum(w[:, None, None] * np.isfinite(rs), axis=0)
                                    ).astype(np.float32)
        frac = nflag / self.nint
        self.badpix = ((frac <= self.persistent_frac) & np.isfinite(self.median)).astype(np.uint8)
        n_dq = int((self.badpix == 0).sum())
        n_hot = 0
        if hot_pixels:
            new = hot_pixel_map(self.median, self.std, self.dispersion_axis, **(hot_kwargs or {}))
            new &= self.badpix == 1
            self.badpix[new] = 0
            n_hot = int(new.sum())
        # source / background masks from the exposure median
        opts = dict(self.config.trace_mask)
        opts.update(trace_kwargs or {})
        mp = opts.pop('min_pixels', 4)
        method = opts.pop('method', 'threshold')
        valid = self.badpix == 1
        self.trace_mask = build_trace_mask(self.median, self.dispersion_axis, method=method,
                                           valid=valid, **opts)
        axis = 'row' if self.dispersion_axis == 0 else 'column'
        bkg = ~self.trace_mask & valid & np.isfinite(self.median)
        self.bkg_mask = ensure_min_background(bkg, self.median, axis, min_pixels=mp, valid=valid)
        self._log('compute_global', n_bad_dq=n_dq, n_hot=n_hot, hot_kwargs=hot_kwargs or {},
                  trace_fraction=float(self.trace_mask.mean()))
        if self.verbose:
            print(f'compute_global: {n_dq} bad pixels from DQ, {n_hot} hot/noisy pixels; '
                  f'trace mask {self.trace_mask.mean() * 100:.1f}% of pixels')

    def clean_segment(self, i, overlap=None, cr_kwargs=None, replace_kwargs=None,
                      background=False, bkg_kwargs=None, save=True):
        """Pass 2: clean segment ``i`` (0-indexed) and optionally save it.

        Parameters
        ----------
        overlap : int or None
            Number of integrations borrowed from each neighbouring segment
            (default: half the cosmic-ray running-median window). ``0``
            disables the padding.
        cr_kwargs : dict or None
            Keyword arguments for :meth:`CubeCleaner.find_cosmic_rays`;
            ``False`` skips cosmic-ray detection.
        replace_kwargs : dict or None
            Keyword arguments for :meth:`CubeCleaner.replace_bad_pixels`;
            ``False`` skips the replacement.
        background : bool
            Run :meth:`CubeCleaner.subtract_background` with the
            exposure-wide background mask.
        bkg_kwargs : dict or None
            Keyword arguments for the background subtraction.
        save : bool
            Write ``<root>-segNNN_<det>_cleaned.fits``.

        Returns
        -------
        CubeCleaner
            The cleaned segment (without the padding).
        """
        if self.badpix is None:
            self.compute_global()
        cr_kwargs = {} if cr_kwargs is None else cr_kwargs
        window = (cr_kwargs or {}).get('window') or self.config.nint_window
        if overlap is None:
            overlap = window // 2 + 1
        parts = []
        n_before = 0
        if overlap and i > 0:
            n_prev = self.seg_nint[i - 1]
            k = min(overlap, n_prev)
            parts.append(self.read_segment(i - 1, slice(n_prev - k, n_prev)))
            n_before = k
        parts.append(self.read_segment(i))
        if overlap and i < len(self.files) - 1:
            parts.append(self.read_segment(i + 1, slice(0, min(overlap, self.seg_nint[i + 1]))))
        data = np.concatenate([p[0] for p in parts])
        err = np.concatenate([p[1] for p in parts])
        dqbad = np.concatenate([p[2] for p in parts])
        keys = parts[1 if n_before else 0][3].keys()          # the segment itself
        times = {k: np.concatenate([p[3][k] for p in parts if k in p[3]]) for k in keys}
        del parts

        cc = CubeCleaner(data, err, badpix=self.badpix.copy(), crmask=dqbad & (self.badpix[None] == 1),
                         times=times, wavelength=None, mode=self.mode, header=dict(self.header),
                         verbose=False)
        cc.trace_mask, cc.bkg_mask = self.trace_mask, self.bkg_mask
        cc._log('exposure_badpix', n_bad=int((self.badpix == 0).sum()))
        cc.fix_errors()
        if cr_kwargs is not False:
            kw = dict(cr_kwargs)
            if 'sigma' not in kw and self.cr_sigma is not None:
                # exposure-wide noise: far more robust than a per-segment estimate
                kw['sigma'] = self.cr_sigma
            cc.find_cosmic_rays(**kw)
        if replace_kwargs is not False:
            cc.replace_bad_pixels(**(replace_kwargs or {}))
        if background:
            cc.subtract_background(bkg_mask=self.bkg_mask, **(bkg_kwargs or {}))

        # remove the padding
        sl = slice(n_before, n_before + self.seg_nint[i])
        out = CubeCleaner(cc.data[sl], cc.err[sl], badpix=cc.badpix, crmask=cc.crmask[sl],
                          times={k: v[sl] for k, v in cc.int_times.items()}, mode=self.mode,
                          header=dict(self.header), verbose=False)
        out.trace_mask, out.bkg_mask = self.trace_mask, self.bkg_mask
        out.history = list(self.history) + cc.history
        out.diagnostics = {k: (v[sl] if isinstance(v, np.ndarray) and v.shape[:1] == (len(cc.data),)
                               else v) for k, v in cc.diagnostics.items()}
        out.header.update({'SEGMENT': i + 1, 'SEGSTART': int(self.seg_start[i]),
                           'SEGNINT': int(self.seg_nint[i])})
        nflag = int(out.crmask.sum())
        self.diagnostics[i] = {'n_transient': nflag,
                               'cr_frac_per_int': out.crmask.reshape(len(out.data), -1).mean(1)}
        if save:
            fn = output_filename(self.files[i], self.outdir, 'cleaned')
            out.save(fn)
            self.outputs[i] = fn
        if self.verbose:
            print(f'  segment {i + 1}/{len(self.files)}: {nflag} transient flags '
                  f'({nflag / len(out.data):.1f} per integration)'
                  + (f' -> {os.path.basename(self.outputs[i])}' if save else ''))
        return out

    def run(self, segments=None, **kwargs):
        """Run both passes on all (or the given) segments and save the products.

        Keyword arguments are passed to :meth:`clean_segment`.

        Returns
        -------
        str
            The exposure-level info file (see :meth:`save_info`).
        """
        if self.badpix is None:
            self.compute_global()
        for i in (range(len(self.files)) if segments is None else segments):
            self.clean_segment(i, **kwargs)
        return self.save_info()

    # ------------------------------------------------------------------
    def save_info(self, fname=None, overwrite=True):
        """Write the exposure-level file: masks, median, wavelength, times, segment list."""
        if fname is None:
            fname = os.path.join(self.outdir, f'{self.exposure}_cleaninfo.fits')
        hdr = fits.Header()
        for k, v in self.header.items():
            try:
                hdr[k] = v
            except ValueError:
                pass
        hdr['JSMODE'] = self.mode
        hdr['DISPAXIS'] = -1 if self.dispersion_axis is None else self.dispersion_axis
        hdus = [fits.PrimaryHDU(header=hdr)]
        for name, arr, dt in (('BADPIX', self.badpix, np.uint8), ('MEDIAN', self.median, np.float32),
                              ('STD', self.std, np.float32), ('CRSIGMA', self.cr_sigma, np.float32),
                              ('TRACEMASK', self.trace_mask, np.uint8),
                              ('BKGMASK', self.bkg_mask, np.uint8),
                              ('WAVELENGTH', self.wavelength, np.float32)):
            if arr is not None:
                hdus.append(fits.ImageHDU(np.asarray(arr, dtype=dt), name=name))
        # all integration times (read from the headers of the input files)
        cols = {}
        for i in range(len(self.files)):
            with fits.open(self.files[i]) as h:
                if 'INT_TIMES' not in h:
                    continue
                tab = h['INT_TIMES'].data
                num = np.asarray(tab['integration_number'])
                i0 = h[0].header.get('INTSTART', num[0])
                sel = (num >= i0) & (num < i0 + self.seg_nint[i])
                for n in tab.names:
                    cols.setdefault(n, []).append(np.asarray(tab[n])[sel])
        if cols:
            hdus.append(fits.BinTableHDU.from_columns(
                [fits.Column(name=n, array=np.concatenate(v),
                             format='D' if np.concatenate(v).dtype.kind == 'f' else 'K')
                 for n, v in cols.items() if np.concatenate(v).dtype.kind in 'fiu'],
                name='INT_TIMES'))
        names = [os.path.basename(f) for f in self.files]
        outs = [os.path.basename(self.outputs.get(i, '')) for i in range(len(self.files))]
        w = max(1, max(len(n) for n in names + outs))
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name='input', format=f'{w}A', array=np.array(names)),
            fits.Column(name='cleaned', format=f'{w}A', array=np.array(outs)),
            fits.Column(name='start', format='K', array=self.seg_start),
            fits.Column(name='nint', format='K', array=np.array(self.seg_nint))], name='SEGMENTS'))
        hist = json.dumps(self.history)
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name='history', format=f'{max(1, len(hist))}A', array=np.array([hist]))],
            name='HISTORY'))
        fits.HDUList(hdus).writeto(fname, overwrite=overwrite)
        self.info_file = fname
        if self.verbose:
            print(f'Saved exposure info to {fname}')
        return fname

    @classmethod
    def load(cls, info_file, verbose=True):
        """Restore an :class:`ExposureCleaner` from its info file."""
        d = os.path.dirname(os.path.abspath(info_file))
        with fits.open(info_file) as h:
            seg = h['SEGMENTS'].data
            inputs = [str(x).strip() for x in seg['input']]
            cleaned = [str(x).strip() for x in seg['cleaned']]
            # input files may have moved; only the cleaned ones are needed
            obj = cls.__new__(cls)
            hdr = h[0].header
            obj.mode = hdr['JSMODE']
            obj.config = get_mode_config(obj.mode)
            disp = hdr.get('DISPAXIS', 1)
            obj.dispersion_axis = None if disp == -1 else disp
            obj.header = {k: hdr[k] for k in _HEADER_KEYS + ('EXPOSURE', 'NSEGMENT') if k in hdr}
            obj.exposure = hdr.get('EXPOSURE', '')
            obj.files = inputs
            obj.outdir = d
            obj.seg_nint = [int(n) for n in seg['nint']]
            obj.seg_start = np.asarray(seg['start'], int)
            get = lambda n, t: np.asarray(h[n].data, t) if n in h else None
            obj.badpix = get('BADPIX', np.uint8)
            obj.median = get('MEDIAN', np.float32)
            obj.std = get('STD', np.float32)
            obj.cr_sigma = get('CRSIGMA', np.float32)
            obj.trace_mask = None if 'TRACEMASK' not in h else get('TRACEMASK', bool)
            obj.bkg_mask = None if 'BKGMASK' not in h else get('BKGMASK', bool)
            obj.wavelength = get('WAVELENGTH', np.float32)
            obj.int_times = {n: np.array(h['INT_TIMES'].data[n]) for n in h['INT_TIMES'].data.names} \
                if 'INT_TIMES' in h else {}
            obj.outputs = {i: os.path.join(d, c) for i, c in enumerate(cleaned) if c}
            try:
                obj.history = json.loads(h['HISTORY'].data['history'][0])
            except Exception:
                obj.history = []
        obj.dq_flags, obj.persistent_frac = BAD_DQ_FLAGS, 0.5
        obj.verbose, obj.diagnostics, obj.info_file = verbose, {}, info_file
        return obj

    def load_segment(self, i):
        """Load cleaned segment ``i`` as a :class:`CubeCleaner` (wavelength attached)."""
        cc = CubeCleaner.load(self.outputs[i], verbose=False)
        if cc.wavelength is None:
            cc.wavelength = self.wavelength
        return cc

    def iter_segments(self):
        """Yield ``(i, CubeCleaner)`` for every cleaned segment, one at a time."""
        for i in sorted(self.outputs):
            yield i, self.load_segment(i)

    def times(self):
        """BJD_TDB - 2400000.5 mid-times of all integrations (from the cleaned segments)."""
        if getattr(self, 'int_times', None):
            return np.asarray(self.int_times['int_mid_BJD_TDB'])
        return np.concatenate([cc.times for _, cc in self.iter_segments()])

    def white_light(self, mask=None):
        """Quick normalised white-light curve over all cleaned segments."""
        mask = self.trace_mask if mask is None else mask
        f = np.concatenate([bn.nansum(cc.data[:, mask], axis=1) for _, cc in self.iter_segments()])
        return f / np.nanmedian(f)


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 50 else f'<array {v.shape}>'
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v
