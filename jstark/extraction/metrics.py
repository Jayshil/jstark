"""Light-curve noise metrics that are insensitive to transits / eclipses.

A deep transit makes the scatter of a light curve meaningless as a noise
measure, and blocks of integrations taken in and out of transit sit at
different flux levels. The point-to-point scatter avoids both problems:
consecutive differences remove the flux level, and the median ignores the
few large differences at ingress and egress.
"""
import numpy as np

from ..reduction.robust import MAD_TO_SIGMA, running_median

__all__ = ['p2p_sigma', 'p2p_noise', 'binned_noise', 'white_light']


def p2p_sigma(y, axis=-1):
    """Robust point-to-point standard deviation of ``y`` (white-noise equivalent).

    ``1.4826 * median(|d - median(d)|) / sqrt(2)`` with ``d`` the consecutive
    differences.
    """
    d = np.diff(np.asarray(y, float), axis=axis)
    med = np.nanmedian(d, axis=axis, keepdims=True)
    return MAD_TO_SIGMA * np.nanmedian(np.abs(d - med), axis=axis) / np.sqrt(2.)


def p2p_noise(blocks, errors=None):
    """Pooled point-to-point noise of several blocks of a light curve.

    Parameters
    ----------
    blocks : list of 1D arrays
        Flux of consecutive integrations; each block is normalised by its
        own median (so in- and out-of-transit blocks can be combined).
    errors : list of 1D arrays or None
        Propagated uncertainties of the same blocks.

    Returns
    -------
    sigma : float
        Relative point-to-point scatter (e.g. ``1e-4`` = 100 ppm).
    ratio : float
        ``sigma`` divided by the median relative uncertainty (NaN without
        ``errors``).
    """
    diffs, rel = [], []
    for k, b in enumerate(blocks):
        b = np.asarray(b, float)
        m = np.nanmedian(b)
        if not np.isfinite(m) or m == 0:
            continue
        diffs.append(np.diff(b / m))
        if errors is not None:
            rel.append(np.asarray(errors[k], float) / abs(m))
    d = np.concatenate(diffs)
    d = d[np.isfinite(d)]
    sigma = MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / np.sqrt(2.)
    ratio = sigma / np.nanmedian(np.concatenate(rel)) if rel else np.nan
    return float(sigma), float(ratio)


def binned_noise(y, bins=(1, 2, 4, 8, 16, 32, 64), window=51):
    """Scatter of a light curve after removing a running median, versus bin size.

    Returns the robust scatter of the binned residuals for every bin size;
    for white noise it falls as ``1 / sqrt(bin)``.
    """
    y = np.asarray(y, float)
    r = y / running_median(y, window) - 1.
    out = []
    for nb in bins:
        n = len(r) // nb
        if n < 5:
            out.append(np.nan)
            continue
        rb = np.nanmean(r[:n * nb].reshape(n, nb), axis=1)
        out.append(MAD_TO_SIGMA * np.nanmedian(np.abs(rb - np.nanmedian(rb))))
    return np.asarray(bins), np.asarray(out)


def white_light(flux, var, cols=None):
    """Sum of the spectra over columns (NaNs ignored consistently)."""
    f = np.asarray(flux, float)
    v = np.asarray(var, float)
    if cols is not None:
        f, v = f[:, cols], v[:, cols]
    ok = np.all(np.isfinite(f) & np.isfinite(v), axis=0)
    return np.sum(f[:, ok], 1), np.sqrt(np.sum(v[:, ok], 1))
