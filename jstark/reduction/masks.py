"""Construction of trace (source) masks and background-pixel masks."""
import numpy as np
from scipy import ndimage

from .robust import clipped_median, mad_sigma, nanmedian

__all__ = ['quick_signal_image', 'trace_mask_threshold', 'trace_mask_centroid',
           'build_trace_mask', 'ensure_min_background', 'amplifier_bounds']


def quick_signal_image(data, groupdq=None, max_ints=200):
    """A quick, high-S/N image of the source from a ramp cube.

    The median (over up to ``max_ints`` integrations) of the last minus the
    first non-saturated group is computed. Saturated groups are excluded
    using ``groupdq``.

    Parameters
    ----------
    data : ndarray
        Ramp data with shape ``(nint, ngroup, ny, nx)``.
    groupdq : ndarray or None
        Group DQ array with the same shape.

    Returns
    -------
    ndarray
        2D image ``(ny, nx)`` in DN.
    """
    nint = data.shape[0]
    sel = np.linspace(0, nint - 1, min(nint, max_ints)).astype(int)
    d = np.asarray(data[sel], dtype=np.float32)
    if d.shape[1] == 1:
        return nanmedian(d[:, 0], axis=0)
    if groupdq is not None:
        bad = (np.asarray(groupdq[sel]) & 3) > 0  # DO_NOT_USE | SATURATED
        d = np.where(bad, np.nan, d)
    # last valid group - first valid group (per pixel); e.g. MIRI flags group 0 (firstframe)
    valid = np.isfinite(d)
    first_idx = np.argmax(valid, axis=1)
    last_idx = d.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    first = np.take_along_axis(d, first_idx[:, None], axis=1)[:, 0]
    last = np.take_along_axis(d, last_idx[:, None], axis=1)[:, 0]
    nstep = last_idx - first_idx
    cds = np.where(nstep > 0, (last - first) / np.maximum(nstep, 1) * (d.shape[1] - 1), np.nan)
    return nanmedian(cds, axis=0)


def _cross_disp_profile_stats(image, dispersion_axis):
    """Return background level and noise of each cross-dispersion profile."""
    ax = 1 - dispersion_axis  # cross-dispersion axis of the 2D image
    bkg = clipped_median(image, axis=ax, nsigma=2.5, maxiters=5)
    resid = image - np.expand_dims(bkg, ax)
    # noise from the negative side of the distribution (not affected by the trace)
    neg = np.where(resid < 0, resid, np.nan)
    sig = 1.4826 * nanmedian(np.abs(neg), axis=ax)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, nanmedian(np.abs(resid), axis=ax) * 1.4826)
    return bkg, sig, resid


def trace_mask_threshold(image, dispersion_axis=1, nsigma=3., pad=2, smooth=5, valid=None,
                         despike=None):
    """Mask pixels that contain significant source flux.

    For each spectral pixel (cross-dispersion profile), the background and
    its noise are estimated robustly and pixels above
    ``background + nsigma * noise`` are flagged. The mask is then smoothed
    along the dispersion direction and dilated by ``pad`` pixels.

    Parameters
    ----------
    image : ndarray
        2D source image (e.g. from :func:`quick_signal_image`).
    dispersion_axis : int or None
        Axis of the dispersion (``1`` = x, ``0`` = y). ``None`` for imaging,
        in which case a global threshold is used.
    nsigma : float
        Detection threshold.
    pad : int
        Dilation (in pixels) applied to the mask.
    smooth : int
        Width of the maximum filter applied along the dispersion direction.
    valid : ndarray or None
        Boolean mask of pixels that can be used (e.g. not reference pixels).
    despike : int or None
        If given, width of a median filter applied along the dispersion
        direction (3x3 for imaging) before thresholding, to suppress hot
        pixels and short streaks. Off by default: isolated detections are
        already removed by a morphological opening and a size filter.

    Returns
    -------
    ndarray
        Boolean mask, ``True`` on the source.
    """
    img = np.array(image, dtype=float)
    if valid is not None:
        img[~valid] = np.nan
    if despike and despike > 1:
        nan = ~np.isfinite(img)
        filled = np.where(nan, np.nanmedian(img), img)
        if dispersion_axis is None:
            size = (3, 3)
        else:
            size = [1, 1]
            size[dispersion_axis] = int(despike)
            size = tuple(size)
        img = ndimage.median_filter(filled, size=size, mode='nearest')
        img[nan] = np.nan
    if dispersion_axis is None:
        bkg = clipped_median(img.ravel(), axis=0, nsigma=2.5, maxiters=5)
        sig = mad_sigma(img[img < bkg].ravel() - bkg) if np.any(img < bkg) else mad_sigma(img.ravel())
        mask = img > bkg + nsigma * sig
    else:
        bkg, sig, resid = _cross_disp_profile_stats(img, dispersion_axis)
        ax = 1 - dispersion_axis
        mask = resid > nsigma * np.expand_dims(sig, ax)
    mask &= np.isfinite(img)
    # remove isolated detections (noise spikes, warm pixels) *before* smoothing / dilation,
    # otherwise the smoothing stretches them into blobs that survive the size filter
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2))) | _large_components(mask)
    if dispersion_axis is not None and smooth and smooth > 1:
        size = [1, 1]
        size[dispersion_axis] = int(smooth)
        mask = ndimage.maximum_filter(mask, size=tuple(size))
    if pad and pad > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(pad))
    return mask


def _large_components(mask, min_size=10):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_size
    return keep[lab]


def trace_mask_centroid(image, dispersion_axis=1, halfwidth=8, poly_order=3, snr_min=10.,
                        valid=None):
    """Mask a band of ``±halfwidth`` pixels around a polynomial trace.

    The flux-weighted centroid of each cross-dispersion profile with
    sufficient S/N is computed, a polynomial is fitted to the centroids, and
    all pixels within ``halfwidth`` of the trace are masked.

    Returns
    -------
    mask : ndarray
        Boolean mask, ``True`` on the trace.
    trace : ndarray
        Trace position (cross-dispersion pixel) for every spectral pixel.
    """
    img = np.array(image, dtype=float)
    if valid is not None:
        img[~valid] = np.nan
    if dispersion_axis == 0:
        img = img.T
    bkg, sig, resid = _cross_disp_profile_stats(img, 1)
    resid = np.where(np.isfinite(resid), resid, 0.)
    ny, nx = resid.shape
    y = np.arange(ny)
    flux = np.clip(resid, 0, None)
    tot = flux.sum(axis=0)
    cen = (flux * y[:, None]).sum(axis=0) / np.where(tot > 0, tot, np.nan)
    snr = resid.max(axis=0) / np.where(sig > 0, sig, np.nan)
    good = np.isfinite(cen) & (snr > snr_min)
    x = np.arange(nx)
    if good.sum() < poly_order + 2:
        raise ValueError('Not enough high-S/N columns to fit the trace.')
    # iterative polynomial fit with clipping
    for _ in range(5):
        coef = np.polyfit(x[good], cen[good], poly_order)
        res = cen - np.polyval(coef, x)
        s = mad_sigma(res[good])
        new = good & (np.abs(res) < 4 * max(s, 0.05))
        if new.sum() == good.sum():
            break
        good = new
    trace = np.polyval(coef, x)
    mask = np.abs(y[:, None] - trace[None, :]) <= halfwidth
    if dispersion_axis == 0:
        mask = mask.T
    return mask, trace


def build_trace_mask(image, dispersion_axis=1, method='threshold', valid=None, **kwargs):
    """Build a trace mask with the requested ``method``.

    ``method`` is ``'threshold'`` (:func:`trace_mask_threshold`),
    ``'centroid'`` (:func:`trace_mask_centroid`) or ``'both'`` (union of the
    two). Unknown keyword arguments for a method are ignored.
    """
    import inspect
    def _kw(func):
        names = inspect.signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in names}
    if method == 'threshold':
        return trace_mask_threshold(image, dispersion_axis, valid=valid, **_kw(trace_mask_threshold))
    if method == 'centroid':
        return trace_mask_centroid(image, dispersion_axis, valid=valid, **_kw(trace_mask_centroid))[0]
    if method == 'both':
        m1 = trace_mask_threshold(image, dispersion_axis, valid=valid, **_kw(trace_mask_threshold))
        m2 = trace_mask_centroid(image, dispersion_axis, valid=valid, **_kw(trace_mask_centroid))[0]
        return m1 | m2
    raise ValueError(f'Unknown trace-mask method {method}.')


def ensure_min_background(bkg_mask, image, axis, min_pixels=4, valid=None):
    """Guarantee at least ``min_pixels`` background pixels per column/row.

    For every column (``axis='column'``) or row (``axis='row'``) of
    ``bkg_mask`` with fewer than ``min_pixels`` usable pixels, the
    ``min_pixels`` pixels with the lowest signal in ``image`` are added.

    Returns
    -------
    ndarray
        Updated background mask (``True`` = usable background pixel).
    """
    m = np.array(bkg_mask, dtype=bool, copy=True)
    img = np.array(image, dtype=float)
    if valid is not None:
        img[~valid] = np.inf
    img[~np.isfinite(img)] = np.inf
    if axis == 'row':
        m, img = m.T, img.T
    nlow = np.sum(m, axis=0) < min_pixels
    if np.any(nlow):
        order = np.argsort(img[:, nlow], axis=0)[:min_pixels]
        cols = np.where(nlow)[0]
        sub = m[:, cols]
        np.put_along_axis(sub, order, True, axis=0)
        sub &= np.isfinite(img[:, cols])
        m[:, cols] = sub
    return m.T if axis == 'row' else m


def amplifier_bounds(nx, noutputs, xstart=1, amp_width=512):
    """Column boundaries of readout amplifiers in a (sub)array.

    Parameters
    ----------
    nx : int
        Number of columns of the subarray.
    noutputs : int
        Number of outputs (amplifiers) used for the readout.
    xstart : int
        1-indexed first column of the subarray on the full detector
        (``SUBSTRT1``).
    amp_width : int
        Width of an amplifier on the full detector.

    Returns
    -------
    list of (int, int)
        ``(start, stop)`` column ranges in subarray coordinates.
    """
    if noutputs is None or noutputs <= 1:
        return [(0, nx)]
    x0 = xstart - 1
    edges = [0]
    for b in range(amp_width, 2048, amp_width):
        if x0 < b < x0 + nx:
            edges.append(b - x0)
    edges.append(nx)
    return list(zip(edges[:-1], edges[1:]))
