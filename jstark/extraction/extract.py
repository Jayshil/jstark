"""Spectral extraction kernels (vectorised over integrations and columns).

All functions take *window* arrays ``(n, nwin, ncol)``: for every integration
and trace column, the pixels of a window around the trace (see
:func:`jstark.extraction.trace.window`).

Optimal extraction follows Horne (1986) with these choices:

* The estimate is ``f = sum(e P D / V) / sum(e P^2 / V)``, where ``e`` is the
  fraction of each pixel inside the aperture (so the estimate changes
  smoothly when the trace moves) and masked pixels are left out.
* Its variance is ``sum(e^2 P^2 / V) / sum(e P^2 / V)^2`` (``= 1 / sum(P^2/V)``
  for whole pixels). ``stark`` / Horne's ``sum(P) / sum(P^2/V)`` is too small
  whenever pixels are masked.
* ``V`` can be the time-averaged variance of each pixel (default in the
  extractor), which avoids the bias of weighting by each pixel's own noisy
  error.
* Remaining outliers are rejected one at a time per column (largest
  normalised residual first).
"""
import numpy as np

__all__ = ['optimal', 'box', 'refine_shift', 'covariance_weighted']


def optimal(D, V, G, P, e, reject=None, niter=3, fit_background=False, var_bkg=None,
            resid_offset=None):
    """Optimal extraction of every (integration, column).

    Parameters
    ----------
    D, V : ndarray ``(n, nwin, ncol)``
        Data and variance.
    G : ndarray of bool
        Usable pixels.
    P : ndarray
        Profile (normalised in each column).
    e : ndarray
        Aperture weights in [0, 1] (0 outside the aperture).
    reject : float or None
        Rejection threshold on ``|D - fP| / sqrt(V)``.
    niter : int
        Maximum number of rejection iterations (one pixel per column each).
    fit_background : bool
        Fit an additive constant per column together with the flux.
    var_bkg : ndarray ``(n, ncol)`` or None
        Variance of the subtracted background (common to all pixels of a
        column), propagated into the flux variance.
    resid_offset : ndarray or None
        Expected (static) residual of every pixel, subtracted before the
        rejection test: pixels are rejected only if they deviate from their
        usual residual, so a static model misfit cannot make a pixel switch
        between rejected and kept from one integration to the next.

    Returns
    -------
    dict
        ``flux``, ``var``, ``maskfrac`` (fraction of the profile in the aperture
        on masked pixels), ``nrej`` (rejected pixels per integration),
        ``good`` (final pixel mask), ``bkg`` (fitted constant, if any) and
        ``chi2`` (mean squared normalised residual of each integration).
    """
    P = np.clip(P, 0., None)
    Vs = np.where(G & (V > 0), V, np.inf)
    G = G & (e > 0) & np.isfinite(D)
    G0 = G.copy()
    nrej = np.zeros(D.shape[0], int)
    for it in range(niter + 1):
        w = np.where(G, e / Vs, 0.)
        if fit_background:
            a11 = np.sum(w * P * P, 1)
            a12 = np.sum(w * P, 1)
            a22 = np.sum(w, 1)
            b1 = np.sum(w * P * np.where(G, D, 0.), 1)
            b2 = np.sum(w * np.where(G, D, 0.), 1)
            det = a11 * a22 - a12 ** 2
            ok = det > 0
            det = np.where(ok, det, np.inf)
            f = (a22 * b1 - a12 * b2) / det
            b = (a11 * b2 - a12 * b1) / det
            model = f[:, None, :] * P + b[:, None, :]
        else:
            den = np.sum(w * P * P, 1)
            f = np.sum(w * P * np.where(G, D, 0.), 1) / np.where(den > 0, den, np.inf)
            b = np.zeros_like(f)
            model = f[:, None, :] * P
        if not reject or it == niter:
            break
        off = 0. if resid_offset is None else resid_offset
        z = np.where(G, np.abs(D - model - off) / np.sqrt(Vs), 0.)
        worst = np.argmax(z, axis=1)
        zmax = np.take_along_axis(z, worst[:, None, :], 1)[:, 0, :]
        hit = zmax > reject
        if not hit.any():
            break
        ii, cc = np.nonzero(hit)
        G[ii, worst[ii, cc], cc] = False
        nrej += hit.sum(1)
    w = np.where(G, e / Vs, 0.)
    if fit_background:
        # variance of f from the 2x2 weighted least squares with weights e/V
        a11 = np.sum(w * P * P, 1)
        a12 = np.sum(w * P, 1)
        a22 = np.sum(w, 1)
        Vg = np.where(G, Vs, 0.)
        q11 = np.sum(w * w * Vg * P * P, 1)
        q12 = np.sum(w * w * Vg * P, 1)
        q22 = np.sum(w * w * Vg, 1)
        det = a11 * a22 - a12 ** 2
        det = np.where(det > 0, det, np.inf)
        # f = (a22 * sum(wPD) - a12 * sum(wD)) / det
        var = (a22 ** 2 * q11 - 2 * a22 * a12 * q12 + a12 ** 2 * q22) / det ** 2
        coef_sum = 0.     # a constant background offset is absorbed by b
    else:
        den = np.sum(w * P * P, 1)
        den = np.where(den > 0, den, np.inf)
        var = np.sum(w * w * np.where(G, Vs, 0.) * P * P, 1) / den ** 2
        coef_sum = np.sum(w * P, 1) / den     # d f / d (background offset)
    if var_bkg is not None:
        var = var + coef_sum ** 2 * var_bkg
    eP = e * P
    tot = np.sum(eP, 1)
    maskfrac = np.sum(np.where(G, 0., eP), 1) / np.where(tot > 0, tot, np.inf)
    bad = ~np.isfinite(var) | (np.sum(G, 1) == 0)
    f = np.where(bad, np.nan, f)
    var = np.where(bad, np.nan, var)
    # goodness of fit of every integration: mean (D - model)^2 / V over the aperture pixels
    m = f[:, None, :] * P + (b[:, None, :] if fit_background else 0.)
    z2 = np.where(G & np.isfinite(m), (D - m) ** 2 / Vs, 0.)
    chi2 = np.sum(z2, axis=(1, 2)) / np.maximum(np.sum(G & np.isfinite(m), axis=(1, 2)), 1)
    return dict(flux=f, var=var, maskfrac=maskfrac, nrej=nrej, good=G, bkg=b,
                rejected=G0 & ~G, chi2=chi2)


def box(D, V, G, P, e, flux, var_bkg=None):
    """Box (aperture-sum) extraction; masked pixels are filled with ``flux * P``."""
    fill = np.where(G, D, flux[:, None, :] * np.clip(P, 0, None))
    f = np.sum(e * np.nan_to_num(fill), 1)
    var = np.sum(e * e * np.where(np.isfinite(V), V, 0.), 1)
    if var_bkg is not None:
        var = var + np.sum(e, 1) ** 2 * var_bkg
    return f, var


def refine_shift(D, V, G, P, dP, u, flux, width=False):
    """Gauss-Newton update of the trace offset of every integration.

    The model of a window is ``f(x) P(u - dy)``; to first order, a shift
    ``dy`` changes it by ``-f P'(u) dy``. Solving for one ``dy`` per
    integration (all columns together) gives the update. With
    ``width=True`` a relative width change ``ds`` (``P(u/s)/s``, derivative
    ``-(P + u P')``) is fitted at the same time and also returned.

    Returns
    -------
    ddy : ndarray ``(n,)``
    ds : ndarray ``(n,)`` (zeros unless ``width``)
    """
    f = flux[:, None, :]
    ok = G & np.isfinite(D) & np.isfinite(f)
    w = np.where(ok, 1. / np.where(V > 0, V, np.inf), 0.)
    R = np.where(ok, D - f * P, 0.)
    J1 = np.where(ok, -f * dP, 0.)
    if not width:
        num = np.sum(w * R * J1, axis=(1, 2))
        den = np.sum(w * J1 * J1, axis=(1, 2))
        return num / np.where(den > 0, den, np.inf), np.zeros(len(D))
    J2 = np.where(ok, -f * (P + u * dP), 0.)
    a11 = np.sum(w * J1 * J1, (1, 2))
    a12 = np.sum(w * J1 * J2, (1, 2))
    a22 = np.sum(w * J2 * J2, (1, 2))
    b1 = np.sum(w * R * J1, (1, 2))
    b2 = np.sum(w * R * J2, (1, 2))
    det = a11 * a22 - a12 ** 2
    det = np.where(det > 0, det, np.inf)
    return (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det


def covariance_weighted(D, V, G, P, e, cov_rows, row_index, scale=1.):
    """Covariance-weighted extraction (generalised least squares per column).

    The noise of the pixels of a column is modelled as
    ``C = diag(V) + K[r_i, r_j]``, where ``K`` is the covariance of
    correlated offsets shared by whole detector rows (e.g. NIRCam 1/f,
    which runs along rows). The flux is
    ``f = P^T C^-1 D / P^T C^-1 P`` with variance ``1 / P^T C^-1 P``.

    With ``scale = B`` the weights use ``C_B = diag(V) + B K`` (optimal for a
    sum of ``B`` columns sharing the row offsets) and the returned variance is
    that of the column for these weights, ``w^T (diag(V) + K) w`` with
    ``w = C_B^-1 P / P^T C_B^-1 P`` (equal to ``1 / P^T C^-1 P`` for B = 1).

    Parameters
    ----------
    D, V, G, P, e : ndarray ``(n, nwin, ncol)``
        As in :func:`optimal`; ``e`` selects the aperture (pixels with
        ``e == 0`` are excluded; fractional weights are not used here).
    cov_rows : ndarray ``(ngroup, nspat, nspat)``
        Row-offset covariance for each column group (e.g. amplifier).
    row_index : tuple ``(rows, group)``
        ``rows`` ``(n, nwin, ncol)`` detector rows of the window pixels and
        ``group`` ``(ncol,)`` the covariance group of every column.
    scale : float
        Multiplier of the row-offset covariance in the weights.

    Returns
    -------
    flux, var : ndarray ``(n, ncol)``
    """
    rows, group = row_index
    n, nwin, ncol = D.shape
    use = G & (e > 0) & np.isfinite(D) & (V > 0)
    big = 1e30
    # build C for every (integration, column): (n, ncol, nwin, nwin)
    r = np.moveaxis(rows, 1, 2)                                # (n, ncol, nwin)
    g = np.broadcast_to(group[None, :], (n, ncol))
    K = cov_rows[g[..., None, None], r[..., :, None], r[..., None, :]]
    Vd = np.moveaxis(np.where(use, V, big), 1, 2)
    C = scale * K + Vd[..., :, None] * np.eye(nwin)
    u = np.moveaxis(use, 1, 2)
    C = np.where(u[..., :, None] & u[..., None, :], C, 0.) + \
        np.where(u, 0., big)[..., :, None] * np.eye(nwin)
    p = np.moveaxis(np.where(use, np.clip(P, 0, None), 0.), 1, 2)
    d = np.moveaxis(np.where(use, D, 0.), 1, 2)
    x = np.linalg.solve(C, np.stack([p, d], -1))                # (n, ncol, nwin, 2)
    a = np.sum(p * x[..., 0], -1)
    b = np.sum(p * x[..., 1], -1)
    a = np.where(a > 0, a, np.nan)
    if scale == 1.:
        return b / a, 1. / a
    w = np.where(u, x[..., 0], 0.) / a[..., None]               # (n, ncol, nwin)
    Vu = np.where(u, Vd, 0.)
    var = np.sum(w * w * Vu, -1) + np.einsum('...i,...ij,...j->...', w, np.where(
        u[..., :, None] & u[..., None, :], K, 0.), w)
    return b / a, var
