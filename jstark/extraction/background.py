"""Integration-level background subtraction before extraction.

The background of every integration is estimated separately for every
spectral pixel (column of the canonical frame) from pixels at a distance of
at least ``inner`` (and at most ``outer``) pixels from every spectral trace.
For imaging, see :mod:`jstark.extraction.photometry`.
"""
import numpy as np
import bottleneck as bn

from ..reduction.robust import MAD_TO_SIGMA

__all__ = ['background_region', 'estimate_background', 'subtract_background']


def background_region(shape, traces, inner, outer=None, good=None, min_pixels=4):
    """Static background mask ``(nspat, nspec)`` from the trace positions.

    Parameters
    ----------
    shape : tuple
        Canonical frame shape.
    traces : list of Trace
        Every order on the detector (pixels near any of them are excluded).
    inner : float
        Minimum distance from every trace.
    outer : float or None
        Maximum distance from the nearest trace.
    good : ndarray of bool or None
        Persistent good-pixel map.
    min_pixels : int
        Columns with fewer background pixels are flagged in the returned
        ``ok`` array (their background is interpolated from neighbours).
    """
    nspat, nspec = shape
    rows = np.arange(nspat)[:, None].astype(float)
    dmin = np.full(shape, np.inf)
    for tr in traces:
        y = np.full(nspec, np.nan)
        y[tr.cols] = tr.y0 + (np.nanmedian(tr.dy) if tr.dy is not None else 0.)
        # extend the trace beyond its column range (pixels there may hold PSF wings)
        yfill = np.interp(np.arange(nspec), tr.cols, tr.y0)
        y = np.where(np.isfinite(y), y, yfill)
        dmin = np.minimum(dmin, np.abs(rows - y[None, :]))
    mask = dmin >= inner
    if outer is not None:
        mask &= dmin <= outer
    if good is not None:
        mask &= good
    ok = mask.sum(axis=0) >= min_pixels
    return mask, ok


def estimate_background(data, err, good, mask, method='median', nsigma=3., maxiters=3,
                        poly_order=1, cols=None, ok=None):
    """Background of every integration and spectral pixel.

    Parameters
    ----------
    data, err : ndarray ``(n, nspat, nspec)``
    good : ndarray of bool ``(n, nspat, nspec)``
    mask : ndarray of bool ``(nspat, nspec)``
        Background region.
    method : {'median', 'poly', 'none'}
        Sigma-clipped median of the background pixels, or a polynomial along
        the spatial direction fitted to them (``poly_order``).
    cols : ndarray or None
        Spectral pixels to process (default: all).
    ok : ndarray of bool or None
        Columns with enough background pixels; the others are interpolated.

    Returns
    -------
    bkg : ndarray
        ``(n, ncol)`` for ``'median'``, ``(n, nspat, ncol)`` for ``'poly'``.
    var : ndarray ``(n, ncol)``
        Variance of the background estimate (at the frame centre for
        ``'poly'``).
    """
    cols = np.arange(data.shape[2]) if cols is None else np.asarray(cols)
    n, nspat = data.shape[:2]
    if method in (None, 'none'):
        return np.zeros((n, len(cols)), np.float32), np.zeros((n, len(cols)), np.float32)
    m = mask[None, :, cols] & good[:, :, cols]
    d = np.where(m, data[:, :, cols], np.nan).astype(np.float32)
    e2 = np.where(m, err[:, :, cols].astype(np.float32) ** 2, np.nan)
    for _ in range(maxiters if nsigma else 0):
        med = bn.nanmedian(d, axis=1)
        dev = np.abs(d - med[:, None, :])
        sig = MAD_TO_SIGMA * bn.nanmedian(dev, axis=1)
        bad = dev > nsigma * np.maximum(sig, 1e-30)[:, None, :]
        if not bad.any():
            break
        d[bad] = np.nan
        e2[bad] = np.nan
    npix = np.sum(np.isfinite(d), axis=1)
    if method == 'median':
        bkg = bn.nanmedian(d, axis=1)
        # variance of a median of npix values: (pi/2) * mean variance / npix
        var = (np.pi / 2) * bn.nanmean(e2, axis=1) / np.maximum(npix, 1)
    elif method == 'poly':
        k = int(poly_order) + 1
        x = (np.arange(nspat) - 0.5 * (nspat - 1)) / max(nspat / 2, 1)
        V = np.vander(x, k, increasing=True)                     # (nspat, k)
        fin = np.isfinite(d)
        w = np.where(fin, 1. / np.where(fin, e2, 1.), 0.)
        dz = np.where(fin, d, 0.)
        G = np.einsum('nrc,ri,rj->ncij', w, V, V)
        rhs = np.einsum('nrc,ri,nrc->nci', w, V, dz)
        G += 1e-12 * np.trace(G, axis1=-2, axis2=-1)[..., None, None] * np.eye(k)
        coef = np.linalg.solve(G, rhs[..., None])[..., 0]          # (n, ncol, k)
        bkg = np.einsum('ri,nci->nrc', V, coef).astype(np.float32)
        cov = np.linalg.inv(G)
        v0 = V[nspat // 2]
        var = np.einsum('i,ncij,j->nc', v0, cov, v0)
    else:
        raise ValueError(f'Unknown background method {method!r}')
    bad_col = npix < 1
    if ok is not None:
        bad_col |= ~ok[cols][None, :]
    if bad_col.any():
        bkg, var = _interp_columns(bkg, var, bad_col)
    return bkg, np.nan_to_num(var).astype(np.float32)


def _interp_columns(bkg, var, bad):
    """Interpolate the background of bad columns from good neighbours."""
    bkg = np.array(bkg, dtype=np.float32)
    var = np.array(var, dtype=np.float32)
    x = np.arange(bad.shape[1])
    for i in range(bad.shape[0]):
        b = bad[i]
        if not b.any() or b.all():
            continue
        if bkg.ndim == 2:
            bkg[i, b] = np.interp(x[b], x[~b], bkg[i, ~b])
        else:
            for r in range(bkg.shape[1]):
                bkg[i, r, b] = np.interp(x[b], x[~b], bkg[i, r, ~b])
        var[i, b] = np.interp(x[b], x[~b], var[i, ~b])
    return bkg, var


def subtract_background(data, bkg, cols):
    """Subtract ``bkg`` (from :func:`estimate_background`) in place from ``data[:, :, cols]``."""
    if bkg.ndim == 2:
        data[:, :, cols] -= bkg[:, None, :]
    else:
        data[:, :, cols] -= bkg
    return data
