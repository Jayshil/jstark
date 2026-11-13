"""Field-star contamination: detection of compact sources and static trace models.

Slitless modes (NIRISS/SOSS above all) record the light of other stars in the
field as well as the target's:

* **zeroth orders** of field stars: compact images, a few to ~30 pixels
  across. They are found here as compact sources in a detection image and
  masked (the pixels are treated as bad: excluded from the background, the
  PSF fit and the pixel-response map, and filled with the profile in the
  extraction).

  The best detection image is the short F277W exposure taken after a SOSS
  time series (arXiv:2305.17001, arXiv:2306.04676): the filter blocks most of
  the target's light, so zeroth orders that sit *on* the target's traces
  become visible. In the time series itself a contaminant on a trace is
  absorbed by the target's profile and flux (the profile is free to change
  along the dispersion), so only contaminants away from the traces can be
  found there.

* **spectral traces** of field stars that cross or run along the target's:
  modelled as the target's trace of the same order shifted by ``(dx, dy)``,
  with the target's PSF and a free static spectrum, fitted to the exposure
  median frame and subtracted from every integration. Published SOSS
  analyses subtract such traces with a model of the contaminant's trace and
  spectrum (e.g. WASP-96 b, arXiv:2305.17001 and arXiv:2511.16771;
  LHS 1140 b, arXiv:2406.15136).

Both are static: a field star does not transit, so it only dilutes the
target's signal. :class:`~jstark.extraction.SpectralExtractor` reports what
is left in the aperture (``FIELD_O<n>``) for dilution corrections.
"""
import warnings

import numpy as np
from astropy.io import fits
from scipy import ndimage

__all__ = ['load_detection_image', 'local_noise', 'find_sources']


def load_detection_image(image, src=None):
    """2D image for contaminant detection, in canonical orientation.

    Parameters
    ----------
    image : str or ndarray
        A ``rate`` / ``rateints`` / ``cleaned`` FITS file (``SCI`` extension;
        the median over integrations is taken, ``DO_NOT_USE`` pixels are
        ignored) or an array in detector orientation.
    src : ExposureSource or None
        Gives the orientation and the expected shape.
    """
    if isinstance(image, str):
        with fits.open(image) as h:
            d = np.asarray(h['SCI'].data, float)
            if 'DQ' in h:
                dq = np.asarray(h['DQ'].data).astype(np.uint32)
                d = np.where((dq & 1) > 0, np.nan, d)
    else:
        d = np.asarray(image, float)
    if d.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN pixels
            d = np.nanmedian(d, axis=0)
    if src is not None:
        d = src.canon(d)
        if d.shape != tuple(src.shape):
            raise ValueError(f'Detection image has shape {d.shape}, the exposure {tuple(src.shape)}.')
    return d


def local_noise(img, valid, block=(16, 64)):
    """Robust (MAD) noise of ``img`` in blocks, interpolated to every pixel."""
    ny, nx = img.shape
    by, bx = block
    gy, gx = int(np.ceil(ny / by)), int(np.ceil(nx / bx))
    s = np.full((gy, gx), np.nan)
    for i in range(gy):
        for j in range(gx):
            sl = (slice(i * by, (i + 1) * by), slice(j * bx, (j + 1) * bx))
            v = img[sl][valid[sl]]
            if v.size > 20:
                s[i, j] = 1.4826 * np.median(np.abs(v - np.median(v)))
    ok = np.isfinite(s) & (s > 0)
    if not ok.any():
        return np.full(img.shape, np.inf)
    s = np.where(ok, s, np.median(s[ok]))
    big = ndimage.zoom(s, (by, bx), order=1, mode='nearest', grid_mode=True)
    return big[:ny, :nx]


def find_sources(img, valid=None, exclude=None, nsigma=5., low=2., smooth=1.5, min_pixels=5,
                 max_size=60, bkg_size=(15, 61), block=(16, 64)):
    """Compact sources in ``img``.

    The image is high-pass filtered (minus a running median of
    ``bkg_size`` pixels), cleaned of single-pixel outliers (3x3 median),
    smoothed with a Gaussian of ``smooth`` pixels (matched to extended
    sources) and divided by its local noise. Sources are connected regions
    above ``low`` sigma that contain a pixel above ``nsigma`` and have at
    least ``min_pixels`` pixels; regions longer than ``max_size`` pixels in
    either direction (e.g. residuals along traces) are rejected.

    Returns
    -------
    mask : ndarray of bool
        Pixels of the accepted sources.
    sources : list of dict
        ``x``, ``y`` (flux-weighted centroid), ``npix``, ``peak`` (S/N of the
        smoothed image), ``flux`` (sum of the high-passed image), ``bbox``
        ``(y0, y1, x0, x1)``.
    snr : ndarray
        The detection S/N map.
    """
    a = np.asarray(img, float)
    ok = np.isfinite(a)
    if valid is not None:
        ok &= valid
    use = ok if exclude is None else ok & ~exclude
    fill = np.where(use, a, np.nanmedian(a[use]))
    bkg = ndimage.median_filter(fill, size=bkg_size, mode='nearest')
    hp = np.where(use, a - bkg, 0.)
    hp = ndimage.median_filter(hp, size=3, mode='nearest')
    num = ndimage.gaussian_filter(np.where(use, hp, 0.), smooth, mode='constant')
    den = ndimage.gaussian_filter(use.astype(float), smooth, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        sm = np.where(den > 0.3, num / den, 0.)
    noise = local_noise(sm, use & (den > 0.3), block)
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.where(use & (noise > 0), sm / noise, 0.)
    lab, n = ndimage.label(snr > low)
    mask = np.zeros(a.shape, bool)
    sources = []
    if n == 0:
        return mask, sources, snr
    peak = ndimage.maximum(snr, lab, index=np.arange(1, n + 1))
    npix = ndimage.sum(np.ones_like(snr), lab, index=np.arange(1, n + 1))
    for k, sl in enumerate(ndimage.find_objects(lab)):
        if peak[k] < nsigma or npix[k] < min_pixels:
            continue
        hy, hx = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hy > max_size or hx > max_size:
            continue
        reg = lab[sl] == k + 1
        mask[sl] |= reg
        yy, xx = np.mgrid[sl]
        w = np.clip(sm[sl], 0, None) * reg
        ws = w.sum() if w.sum() > 0 else 1.
        sources.append(dict(x=float((w * xx).sum() / ws), y=float((w * yy).sum() / ws),
                            npix=int(npix[k]), peak=float(peak[k]),
                            flux=float(hp[sl][reg].sum()),
                            bbox=(int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop))))
    return mask, sources, snr
