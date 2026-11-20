"""Light-curve noise metrics that are insensitive to transits / eclipses.

A deep transit makes the scatter of a light curve meaningless as a noise
measure, and blocks of integrations taken in and out of transit sit at
different flux levels. The point-to-point scatter avoids both problems:
consecutive differences remove the flux level, and the median ignores the
few large differences at ingress and egress.
"""
import numpy as np

from ..reduction.robust import MAD_TO_SIGMA, running_median

__all__ = ['p2p_sigma', 'p2p_noise', 'binned_noise', 'white_light', 'find_psf_breaks']


def find_psf_breaks(series, threshold=20., w=5, persist=25, min_sep=10):
    """Abrupt, lasting steps in per-integration PSF regressors (e.g. a mirror tilt).

    For each regressor (e.g. ``{'dy': dy, 'ds': ds}``) the local step
    ``median(x[k:k+w]) - median(x[k-w:k])`` is compared with its noise (from
    the point-to-point scatter). A break is reported where the score exceeds
    ``threshold`` and the step persists: the medians over ``persist``
    integrations on either side differ by at least half of it, in the same
    direction. Calibrated on full exposures (docs/MD/extraction_notes.md
    §5.17): the WASP-39 b G395H tilt scores 63-91 in ``ds``, nothing in nine
    other exposures (transits, ramps, the PRISM high-gain-antenna move)
    exceeds 10.

    Returns
    -------
    list of dict
        ``gidx`` (first integration of the new epoch), ``score``, ``key``,
        ``step``, sorted by integration.
    """
    found = []
    for key, x in series.items():
        x = np.asarray(x, float)
        n = len(x)
        if n < 2 * w + 1:
            continue
        sig = p2p_sigma(x[np.isfinite(x)])
        if not np.isfinite(sig) or sig <= 0:
            continue
        sd = sig * 1.2533 * np.sqrt(2. / w)
        z = np.zeros(n)
        for k in range(w, n - w + 1):
            z[k] = (np.nanmedian(x[k:k + w]) - np.nanmedian(x[k - w:k])) / sd
        az = np.abs(z)
        while True:
            k = int(np.argmax(az))
            if az[k] < threshold:
                break
            az[max(0, k - min_sep):k + min_sep] = 0.
            step = z[k] * sd
            lo, hi = max(0, k - persist), min(n, k + persist)
            big = np.nanmedian(x[k:hi]) - np.nanmedian(x[lo:k])
            if np.sign(big) == np.sign(step) and abs(big) >= 0.5 * abs(step):
                found.append(dict(gidx=k, score=float(abs(z[k])), key=key, step=float(step)))
    # merge detections of the same event in different regressors
    out = []
    for f in sorted(found, key=lambda d: (d['gidx'], -d['score'])):
        if out and abs(f['gidx'] - out[-1]['gidx']) <= min_sep:
            if f['score'] > out[-1]['score']:
                out[-1] = f
            continue
        out.append(f)
    return out


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
