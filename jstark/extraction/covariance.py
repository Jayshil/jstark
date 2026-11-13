"""Covariance-weighted extraction for row-correlated noise (NIRCam grism 1/f).

In NIRCam grism time series the 1/f noise is an offset shared by whole
detector rows (along the dispersion) within each amplifier. Pixels of one
column therefore have correlated noise across rows, and optimal extraction
(which assumes independent pixels) does not weight them optimally.

The row-offset covariance ``K`` is estimated per amplifier from the residuals
of the PSF fit (data - model), averaged along each detector row over the
columns of the amplifier, and their covariance over time (Schlawin et al.
2020, Noise Floor II, arXiv:2010.03576, discuss covariance-weighted
extraction). Each column is then extracted by generalised least squares with
``C = diag(V) + K``.

Limitation: the part of the row offsets that has the shape of the profile is
indistinguishable from a change of the flux and is removed by the fit (it is
absent from the residuals). No weighting can reduce it; ``K`` describes the
remaining part, which is exactly what the weights can act on. The returned
errors therefore do not include the profile-shaped part of the 1/f noise.

Weights for sums of columns (``cov_scale``). A row offset is shared by all
columns of an amplifier, so in a sum of ``B`` columns (a spectral bin, the
white light) it adds up coherently (variance ~ B**2 K) while the pixel noise
adds in quadrature (~ B V). The weights that minimise the noise of the sum use
``C = diag(V) + B K``; per column (B = 1) K is negligible next to V. With
``set('extract', method='covariance', cov_scale=B)`` the faint wings, where
the read noise is small and the row offsets dominate, are down-weighted. On
HD 207496 (one segment) B = 500 lowered the white-light scatter by ~10%.
The returned variances are those of each column for these weights.
"""
import numpy as np

from ..reduction.masks import amplifier_bounds
from .extract import covariance_weighted

__all__ = ['amplifier_groups', 'estimate_row_covariance', 'covariance_extract']


def amplifier_groups(src, cols):
    """Amplifier index of each canonical spectral column (0 if unknown / one output)."""
    hdr = src.header
    nout = int(hdr.get('NOUTPUTS') or 1)
    if src.transpose or nout <= 1:
        return np.zeros(len(cols), int)
    bounds = amplifier_bounds(src.frame_shape[1], nout, int(hdr.get('SUBSTRT1') or 1))
    edges = np.array([b[0] for b in bounds[1:]])
    return np.searchsorted(edges, np.asarray(cols), side='right')


def estimate_row_covariance(ex, order, nblocks=3, block_len=300, segments=None, min_frac=0.9):
    """Row-offset covariance of order ``order`` from PSF-fit residuals.

    Returns
    -------
    dict
        ``K`` ``(ngroup, nspat, nspat)``, ``group`` ``(ncol,)``, ``rows``
        (detector rows with enough data), ``nint``.
    """
    src = ex.src
    tr = ex.traces[order]
    nspat = src.shape[0]
    grp = amplifier_groups(src, tr.cols)
    ngrp = int(grp.max()) + 1
    series, wvar = [], []
    for k, sl in ex._blocks(nblocks, block_len, segments):
        for d, e, g, gi in ex._block_chunks(k, sl):
            dd = np.array(d, np.float32, copy=True)
            prep = ex._prepare(dd, e, g)
            res = ex._extract_order(order, dd, e, g, prep[order]['dy'], prep[order]['var_bkg'],
                                    method='optimal')
            W = ex._window(order, dd, e, g, res['dy'])
            R = np.where(W['G'], W['D'] - res['flux'][:, None, :] * W['P'], 0.)
            w = np.where(W['G'] & (W['e'] > 0), 1. / W['V'], 0.)
            n = len(dd)
            key = ((np.arange(n)[:, None, None] * nspat + W['rows']) * ngrp +
                   grp[None, None, :])
            size = n * nspat * ngrp
            sw = np.bincount(key.ravel(), w.ravel(), size).reshape(n, nspat, ngrp)
            swr = np.bincount(key.ravel(), (w * R).ravel(), size).reshape(n, nspat, ngrp)
            with np.errstate(invalid='ignore', divide='ignore'):
                series.append(swr / sw)
                wvar.append(1. / sw)
    o = np.concatenate(series)                  # (nint, nspat, ngrp) mean residual per row
    vw = np.concatenate(wvar)
    K = np.zeros((ngrp, nspat, nspat))
    good_rows = []
    for a in range(ngrp):
        ok = np.mean(np.isfinite(o[:, :, a]), axis=0) >= min_frac
        rows = np.where(ok)[0]
        good_rows.append(rows)
        if len(rows) < 2:
            continue
        x = o[:, rows, a]
        x = x - np.nanmean(x, axis=0)
        x = np.where(np.isfinite(x), x, 0.)
        cov = x.T @ x / max(len(x) - 1, 1)
        cov -= np.diag(np.nanmean(np.where(np.isfinite(vw[:, rows, a]), vw[:, rows, a], np.nan), 0))
        # nearest positive semi-definite matrix
        lam, vec = np.linalg.eigh(cov)
        cov = (vec * np.clip(lam, 0, None)) @ vec.T
        K[a][np.ix_(rows, rows)] = cov
    return dict(K=K, group=grp, rows=good_rows, nint=len(o))


def covariance_extract(ex, order, D, V, G, P, e, rows, max_mb=400, scale=1.):
    """Covariance-weighted flux of a chunk (window arrays as in :func:`optimal`).

    ``scale`` multiplies the row-offset covariance (number of columns summed).
    """
    cv = ex.cov[order]
    use = np.any(e > 0, axis=(0, 2))            # window rows inside the aperture
    sl = slice(int(np.argmax(use)), len(use) - int(np.argmax(use[::-1])))
    D, V, G, P, e, rows = (a[:, sl] for a in (D, V, G, P, e, rows))
    n, nwin, ncol = D.shape
    per_int = ncol * nwin * nwin * 8 * 4
    step = max(1, int(max_mb * 2 ** 20 / per_int))
    flux = np.full((n, ncol), np.nan)
    var = np.full((n, ncol), np.nan)
    for a in range(0, n, step):
        s = slice(a, a + step)
        flux[s], var[s] = covariance_weighted(D[s], V[s], G[s], P[s], e[s], cv['K'],
                                              (rows[s], cv['group']), scale=scale)
    return flux, var
