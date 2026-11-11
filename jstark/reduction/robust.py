"""Fast, NaN-aware robust statistics used throughout the reduction."""
import numpy as np
import bottleneck as bn

__all__ = ['nanmedian', 'mad_sigma', 'clipped_median', 'running_median', 'interp_nans_1d']

MAD_TO_SIGMA = 1.482602218505602


def nanmedian(a, axis=None):
    """NaN-ignoring median (bottleneck), silent for all-NaN slices."""
    return bn.nanmedian(a, axis=axis)


def mad_sigma(a, axis=None, center=None):
    """Robust standard deviation from the median absolute deviation."""
    if center is None:
        center = bn.nanmedian(a, axis=axis)
        if axis is not None:
            center = np.expand_dims(center, axis)
    return MAD_TO_SIGMA * bn.nanmedian(np.abs(a - center), axis=axis)


def clipped_median(a, axis, nsigma=3., maxiters=3, return_count=False):
    """Iteratively sigma-clipped median along ``axis`` (NaNs ignored).

    Parameters
    ----------
    a : ndarray
        Input array (it is not modified).
    axis : int
        Axis along which to compute the statistic.
    nsigma : float or None
        Clipping threshold in units of the MAD-based sigma. ``None``
        disables clipping.
    maxiters : int
        Maximum number of clipping iterations.
    return_count : bool
        Also return the number of samples used.

    Returns
    -------
    med : ndarray
        Median with ``axis`` removed.
    n : ndarray, optional
        Number of non-NaN samples remaining after clipping.
    """
    work = np.array(a, dtype=np.float32, copy=True)
    med = bn.nanmedian(work, axis=axis)
    if nsigma is not None:
        for _ in range(maxiters):
            medx = np.expand_dims(med, axis)
            dev = np.abs(work - medx)
            sig = MAD_TO_SIGMA * bn.nanmedian(dev, axis=axis)
            bad = dev > nsigma * np.expand_dims(sig, axis)
            if not np.any(bad):
                break
            work[bad] = np.nan
            med = bn.nanmedian(work, axis=axis)
    if return_count:
        return med, np.sum(np.isfinite(work), axis=axis)
    return med


def running_median(a, window, axis=0, exclude_center=False):
    """Centred running median along ``axis`` with reflective edges.

    NaNs are ignored inside each window. ``window`` is forced to be odd.
    With ``exclude_center=True`` the sample itself is left out of its own
    window (a leave-one-out median of its ``window - 1`` neighbours). This
    is what outlier detection needs: otherwise the median partly follows
    the sample, and the residual scatter (MAD) underestimates the noise.
    """
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return np.array(a, copy=True)
    if exclude_center:
        return _running_median_loo(np.asarray(a), window, axis)
    a = np.moveaxis(np.asarray(a), axis, 0)
    n = a.shape[0]
    half = window // 2
    if n <= half:
        out = np.broadcast_to(bn.nanmedian(a, axis=0), a.shape).copy()
        return np.moveaxis(out, 0, axis)
    pad = np.concatenate([a[half:0:-1], a, a[-2:-half - 2:-1]], axis=0)
    out = bn.move_median(pad, window, axis=0, min_count=1)[2 * half:]
    return np.moveaxis(out, 0, axis)


def _running_median_loo(a, window, axis):
    """Leave-one-out centred running median (see :func:`running_median`)."""
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    half = window // 2
    if n <= half:
        out = np.broadcast_to(bn.nanmedian(a, axis=0), a.shape).copy()
        return np.moveaxis(out, 0, axis)
    pad = np.concatenate([a[half:0:-1], a, a[-2:-half - 2:-1]], axis=0)
    win = sliding_window_view(pad, window, axis=0)            # (n, ..., window)
    keep = np.r_[0:half, half + 1:window]
    out = bn.nanmedian(np.take(win, keep, axis=-1), axis=-1)
    return np.moveaxis(out.astype(a.dtype, copy=False), 0, axis)


def interp_nans_1d(y, axis=-1, max_gap=None):
    """Linearly interpolate NaNs along ``axis`` from the nearest finite neighbours.

    Parameters
    ----------
    y : ndarray
        Input array; a copy is returned.
    axis : int
        Interpolation axis.
    max_gap : int or None
        If given, NaN runs longer than this are left untouched.
    """
    y = np.moveaxis(np.array(y, dtype=float, copy=True), axis, -1)
    shp = y.shape
    y2 = y.reshape(-1, shp[-1])
    x = np.arange(shp[-1])
    for row in y2:
        bad = ~np.isfinite(row)
        if not bad.any() or bad.all():
            continue
        if max_gap is not None:
            # identify NaN runs and skip long ones
            edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            fill = np.zeros_like(bad)
            for s, e in zip(starts, ends):
                if e - s <= max_gap:
                    fill[s:e] = True
        else:
            fill = bad
        row[fill] = np.interp(x[fill], x[~bad], row[~bad])
    return np.moveaxis(y2.reshape(shp), -1, axis)
