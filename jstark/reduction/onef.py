"""Group-level (or integration-level) background and 1/f-noise subtraction.

Three algorithms are provided, all working on a 4D cube
``(nint, ngroup, ny, nx)`` (integration-level data can be passed with
``ngroup = 1``):

``'median'``
    Sigma-clipped median of the background pixels of each column
    (``axis='column'``) or row (``axis='row'``). For rows, the statistic can
    be computed amplifier-by-amplifier (NIRCam, Schlawin et al. 2020), with a
    fall-back to the full row when an amplifier segment of a row has too few
    background pixels. This is the approach of ExoTiC-JEDI / Eureka!.

``'poly'``
    Polynomial (in the cross-stripe coordinate) fitted with sigma-clipping
    to the background pixels of each column/row. Useful when the background
    has a gradient along the column (e.g. NIRSpec G395H).

``'diff'``
    "Difference-image" method (exoTEDRF; Radica 2024). A median frame is
    built for every group, scaled by a rough white-light curve, and
    subtracted from each integration. The residual images are almost pure
    1/f noise, so static background structure (e.g. NIRISS/SOSS zodiacal
    light, contaminating orders) does not bias the stripe estimate. The
    column/row offsets of the residuals are then subtracted from the data.
    Note that this method removes only the time-variable component: the
    static background remains in the data and should be removed at the
    integration level.

``'readout'``
    Readout-time model (for multi-amplifier read-outs, e.g. NIRCam grism).
    Rows (for ``axis='row'``) are read sequentially in time, and all
    amplifiers read the same row simultaneously. In the residual images of
    the ``'diff'`` method, each amplifier's per-row offset is modelled as

        offset_k(row) = S_k(row) + c(row)

    where ``S_k`` is a smooth (polynomial) function of read time specific to
    amplifier ``k`` and ``c`` is a common-mode term shared by all amplifiers
    read at the same moment. ``S_k`` and ``c`` are fitted iteratively to the
    rows where an amplifier has enough background pixels; the model fills
    the rows where it has none (e.g. rows fully covered by the trace),
    instead of simply borrowing the other amplifiers' value. Rows with
    enough background pixels keep their directly measured offset.

``'rowcol'``
    Row then column medians of the background pixels: the amplifiers are
    first equalised (one offset per amplifier from all its background
    pixels), then one row offset common to all amplifiers is subtracted
    (median over the background pixels of the whole row), then a column
    offset (median over the background pixels of each column).

For ``'diff'`` and ``'readout'``, ``wing_max`` (DN per group) or ``wing_frac``
(fraction of the peak signal per group) adds the faint trace pixels
(median signal per group below the threshold) to the pixels used to measure
the offsets. The difference images remove the static stellar light, and in
the faint wings flux changes are far smaller than the 1/f offsets, so the
offsets can be measured there directly; only the bright core is left to the
model.
"""
import warnings

import numpy as np
import bottleneck as bn

from .robust import clipped_median, nanmedian, mad_sigma, MAD_TO_SIGMA
from .masks import amplifier_bounds

__all__ = ['group_background', 'stripe_median', 'stripe_rowcol', 'wing_peak', 'stripe_wmean', 'stripe_poly', 'readout_model',
           'diff_scale_factors']

_BAD_GROUP = 1 | 2  # DO_NOT_USE | SATURATED


def _chunks(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def stripe_median(cube, bkg_mask, axis='column', nsigma=3., maxiters=3, amp_bounds=None,
                  min_pixels=4):
    """Column- or row-wise sigma-clipped median of background pixels.

    Parameters
    ----------
    cube : ndarray
        Array of shape ``(..., ny, nx)``; NaNs are ignored.
    bkg_mask : ndarray
        Boolean ``(ny, nx)`` mask, ``True`` for background pixels. A 3D or
        higher mask broadcastable to ``cube`` is also accepted.
    axis : {'column', 'row'}
        ``'column'``: one value per column; ``'row'``: one value per row
        (per amplifier when ``amp_bounds`` has more than one segment).
    amp_bounds : list of (int, int) or None
        Amplifier column ranges (only used for ``axis='row'``).
    min_pixels : int
        Minimum number of background pixels for a per-amplifier row
        estimate; otherwise the full-row estimate is used.

    Returns
    -------
    model : ndarray
        Background model with the same shape as ``cube``.
    profile : ndarray
        The stripe values: ``(..., nx)`` for columns or
        ``(..., ny, namp)`` for rows.
    """
    masked = np.where(bkg_mask, cube, np.nan).astype(np.float32)
    if axis == 'column':
        prof = clipped_median(masked, axis=-2, nsigma=nsigma, maxiters=maxiters)
        prof = np.where(np.isfinite(prof), prof, 0.)
        return np.broadcast_to(prof[..., None, :], cube.shape), prof
    if axis != 'row':
        raise ValueError("axis must be 'column' or 'row'.")
    nx = cube.shape[-1]
    amp_bounds = amp_bounds or [(0, nx)]
    model = np.zeros(cube.shape, dtype=np.float32)
    # 1) equalise the amplifiers: one offset per amplifier from all its background pixels
    if len(amp_bounds) > 1:
        lead = cube.shape[:-2]
        for (x0, x1) in amp_bounds:
            sub = masked[..., x0:x1].reshape(lead + (-1,))
            off = clipped_median(sub, axis=-1, nsigma=nsigma, maxiters=maxiters)
            off = np.where(np.isfinite(off), off, 0.).astype(np.float32)
            model[..., x0:x1] = off[..., None, None]
        masked = masked - model
    # 2) row-by-row on the equalised data (per amplifier where possible)
    full = clipped_median(masked, axis=-1, nsigma=nsigma, maxiters=maxiters)
    full = np.where(np.isfinite(full), full, 0.)
    prof = np.empty(cube.shape[:-1] + (len(amp_bounds),), dtype=np.float32)
    for k, (x0, x1) in enumerate(amp_bounds):
        if len(amp_bounds) == 1:
            seg = full
        else:
            seg, n = clipped_median(masked[..., x0:x1], axis=-1, nsigma=nsigma, maxiters=maxiters,
                                    return_count=True)
            seg = np.where((n >= min_pixels) & np.isfinite(seg), seg, full)
        model[..., x0:x1] += seg[..., None]
        prof[..., k] = model[..., x0]
    return model, prof


def stripe_rowcol(cube, bkg_mask, amp_bounds=None, nsigma=3., maxiters=3):
    """Amplifier equalisation, a row offset common to all amplifiers, then column offsets.

    Returns
    -------
    model : ndarray
        Background model with the same shape as ``cube``.
    profile : ndarray
        The row part of the model, ``(..., ny, namp)``.
    """
    # a huge min_pixels forces the full-row (all-amplifier) estimate after equalisation
    rows, prof = stripe_median(cube, bkg_mask, axis='row', nsigma=nsigma, maxiters=maxiters,
                               amp_bounds=amp_bounds, min_pixels=np.iinfo(np.int64).max)
    cols, _ = stripe_median(cube - rows, bkg_mask, axis='column', nsigma=nsigma,
                            maxiters=maxiters)
    return rows + cols, prof


def stripe_wmean(cube, weights, axis='column', nsigma=4., maxiters=3, amp_bounds=None):
    """Inverse-variance weighted, sigma-clipped mean of each column / row.

    Used by the ``'diff'`` method with ``use_source_pixels=True``: the
    residual images contain (almost) only noise, so every pixel, including
    those on the source, can contribute with weight ``1 / variance``.
    For rows, each amplifier segment gets its own estimate.

    Parameters
    ----------
    cube : ndarray
        ``(..., ny, nx)``; NaNs are ignored.
    weights : ndarray
        Weights broadcastable to ``cube`` (``0`` excludes a pixel).

    Returns
    -------
    model, profile : ndarray
        As for :func:`stripe_median`.
    """
    w = np.where(np.isfinite(cube), np.broadcast_to(weights, cube.shape), 0.).astype(np.float32)
    x = np.where(w > 0, cube, 0.).astype(np.float32)
    nx = cube.shape[-1]
    segments = [(0, nx)] if axis == 'column' else (amp_bounds or [(0, nx)])
    red = -2 if axis == 'column' else -1
    model = np.zeros(cube.shape, dtype=np.float32)
    profs = []
    for (x0, x1) in segments:
        ws, xs = w[..., x0:x1], x[..., x0:x1]
        for it in range(maxiters + 1):
            sw = ws.sum(axis=red, keepdims=True)
            mu = np.where(sw > 0, (ws * xs).sum(axis=red, keepdims=True) / np.where(sw > 0, sw, 1), 0.)
            if it == maxiters or nsigma is None:
                break
            z = (xs - mu) * np.sqrt(ws)
            clip = (np.abs(z) > nsigma) & (ws > 0)
            if not clip.any():
                break
            ws = np.where(clip, 0., ws)
        model[..., x0:x1] = mu
        profs.append(np.squeeze(mu, axis=red))
    prof = profs[0] if axis == 'column' else np.stack(profs, axis=-1)
    return model, prof


def readout_model(values, counts, min_pixels=10, order=3, niter=4, use_direct=True):
    """Fit per-amplifier smooth readout-time trends plus a common mode.

    Parameters
    ----------
    values : ndarray
        Per-line, per-amplifier offsets ``(..., nline, namp)`` measured from
        background pixels (lines are read sequentially in time).
    counts : ndarray
        Number of background pixels behind each value (same shape).
    min_pixels : int
        Values from fewer pixels are treated as missing.
    order : int
        Polynomial order (in read time) of the amplifier-specific term ``S_k``.
    niter : int
        Number of alternating fit iterations.
    use_direct : bool
        Keep the measured value where it exists (model only the gaps).

    Returns
    -------
    final : ndarray
        Offsets for every line and amplifier ``(..., nline, namp)``.
    smooth : ndarray
        The amplifier-specific smooth terms ``S_k``.
    common : ndarray
        The common-mode term ``c`` ``(..., nline)`` (NaN where no amplifier
        has data).
    """
    v = np.where(counts >= min_pixels, values, np.nan).astype(np.float64)
    nline, namp = v.shape[-2:]
    t = np.linspace(-1., 1., nline)
    V = np.vander(t, order + 1, increasing=True)                        # (nline, p)
    S = np.zeros_like(v)
    c = np.zeros(v.shape[:-1])
    for _ in range(niter):
        # common mode: what all amplifiers read at the same time share
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            c = np.nanmedian(v - S, axis=-1) if namp > 1 else np.zeros(v.shape[:-1])
        cf = np.where(np.isfinite(c), c, 0.)
        # amplifier-specific smooth trend in read time
        y = v - cf[..., None]
        w = np.isfinite(y).astype(np.float64)
        yy = np.where(w > 0, y, 0.)
        A = np.einsum('...lk,lp,lq->...kpq', w, V, V) + np.eye(order + 1) * 1e-9
        b = np.einsum('...lk,lp->...kp', yy, V)
        ok = w.sum(axis=-2) > order                                       # (..., namp)
        A = np.where(ok[..., None, None], A, np.eye(order + 1))
        b = np.where(ok[..., None], b, 0.)
        coef = np.linalg.solve(A, b[..., None])[..., 0]                   # (..., namp, p)
        S = np.einsum('lp,...kp->...lk', V, coef)
    pred = S + np.where(np.isfinite(c), c, 0.)[..., None]
    final = np.where(np.isfinite(v), v, pred) if use_direct else pred
    return final.astype(np.float32), S.astype(np.float32), c.astype(np.float32)


def stripe_poly(cube, bkg_mask, axis='column', order=1, nsigma=3., maxiters=3):
    """Polynomial fit of the background along each column or row.

    For ``axis='column'`` a polynomial in ``y`` is fitted to the background
    pixels of each column (and vice versa for rows).

    Returns
    -------
    model : ndarray
        Background model with the same shape as ``cube``.
    coeffs : ndarray
        Polynomial coefficients, shape ``(..., n_stripes, order + 1)``.
    """
    c = np.swapaxes(cube, -1, -2) if axis == 'row' else cube
    m = np.swapaxes(bkg_mask, -1, -2) if axis == 'row' else bkg_mask
    ncross = c.shape[-2]
    t = np.linspace(-1, 1, ncross)
    V = np.vander(t, order + 1, increasing=True)                   # (ncross, p)
    w = (np.broadcast_to(m, c.shape) & np.isfinite(c)).astype(np.float64)
    d = np.where(w > 0, c, 0.).astype(np.float64)
    for it in range(maxiters + 1):
        A = np.einsum('...ys,yp,yq->...spq', w, V, V)               # (..., nstripe, p, p)
        b = np.einsum('...ys,yp,...ys->...sp', w, V, d)            # (..., nstripe, p)
        A += np.eye(order + 1) * 1e-12
        good = np.linalg.cond(A) < 1e12
        A[~good] = np.eye(order + 1)
        b[~good] = 0.
        coef = np.linalg.solve(A, b[..., None])[..., 0]
        model = np.einsum('yp,...sp->...ys', V, coef)
        if it == maxiters or nsigma is None:
            break
        res = np.where(w > 0, d - model, np.nan)
        sig = MAD_TO_SIGMA * bn.nanmedian(np.abs(res), axis=-2)
        clip = np.abs(res) > nsigma * sig[..., None, :]
        if not np.any(clip & (w > 0)):
            break
        w[clip] = 0.
    model = model.astype(np.float32)
    if axis == 'row':
        model = np.swapaxes(model, -1, -2)
    return model, coef


def diff_scale_factors(data, median_frames, source_mask, groupdq=None):
    """Per-integration white-light scale factors for the difference method.

    The ratio of the summed CDS image (last minus first group) of each
    integration to that of the median frames is computed over the brightest
    source pixels.

    Returns
    -------
    ndarray
        Scale factor of each integration (median-normalised to 1).
    """
    nint, ngroup = data.shape[:2]
    if ngroup < 2:
        img = data[:, 0]
        ref = median_frames[0]
    else:
        img = data[:, -1] - data[:, 0]
        ref = median_frames[-1] - median_frames[0]
    sel = source_mask & np.isfinite(ref) & (ref > 0)
    if groupdq is not None and ngroup >= 2:
        sat = np.any((groupdq[:, -1] & _BAD_GROUP) > 0, axis=0)
        sel &= ~sat
    if sel.sum() < 10:
        return np.ones(nint)
    # keep the brightest half of the source pixels
    thr = np.nanpercentile(ref[sel], 50)
    sel &= ref >= thr
    num = np.nansum(np.where(np.isfinite(img[:, sel]), img[:, sel], 0.), axis=1)
    den = np.nansum(ref[sel])
    s = num / den
    return s / np.nanmedian(s)


def group_background(data, bkg_mask, method='median', axis='column', groupdq=None,
                     nsigma=3., maxiters=3, poly_order=1, amp_bounds=None, min_pixels=4,
                     source_mask=None, chunk=50, return_model=False, inplace=True, exclude=None,
                     use_source_pixels=False, weight_ints=200, readout_order=3,
                     readout_direct=True, wing_max=None, wing_frac=None, wing_min_pixels=100):
    """Subtract a column/row background (including 1/f noise) from every group.

    Parameters
    ----------
    data : ndarray
        Ramp cube ``(nint, ngroup, ny, nx)``. Modified in place unless
        ``inplace=False``.
    bkg_mask : ndarray
        Boolean ``(ny, nx)`` background mask (``True`` = background pixel).
    method : {'median', 'poly', 'diff', 'readout', 'rowcol'}
        Algorithm, see the module documentation.
    axis : {'column', 'row'}
        Direction of the 1/f stripes.
    groupdq : ndarray or None
        Group DQ; saturated / do-not-use groups are excluded from the
        statistics (but still corrected).
    nsigma, maxiters : float, int
        Sigma clipping parameters.
    poly_order : int
        Polynomial order for ``method='poly'``.
    amp_bounds : list of (int, int) or None
        Amplifier column ranges for row-wise, per-amplifier estimates.
    min_pixels : int
        Minimum number of background pixels in an amplifier row segment.
    source_mask : ndarray or None
        Boolean source mask used to compute white-light scale factors for
        ``method='diff'`` (defaults to ``~bkg_mask``).
    chunk : int
        Number of integrations processed at a time (memory control).
    return_model : bool
        If ``True`` also return the full background model (large!).
    exclude : ndarray or None
        Boolean array (broadcastable to ``data``) of pixels to exclude from
        the statistics (they are still corrected).
    use_source_pixels : bool
        ``method='diff'`` only. If ``True``, all valid pixels (also those on
        the source) are used, combined with inverse-variance weights
        estimated from the temporal scatter of each pixel's residuals
        (``bkg_mask`` then only marks pixels to exclude when ``False`` *and*
        not on the source, e.g. reference pixels). This gives an estimate
        for every column / amplifier-row even where the source fills it
        (e.g. NIRCam rows on the trace).
    weight_ints : int
        Number of integrations used to estimate the weights.
    readout_order : int
        ``method='readout'``: polynomial order (in read time) of the
        amplifier-specific term.
    readout_direct : bool
        ``method='readout'``: keep directly measured offsets where an
        amplifier has enough background pixels (``False`` uses the model
        everywhere).
    wing_max : float or None
        ``method='diff'`` / ``'readout'``: also measure the offsets on source
        pixels whose median signal per group (DN, from the median frames) is
        below this value. ``None`` uses only ``bkg_mask`` (unless ``wing_frac``
        is given).
    wing_frac : float or None
        Threshold relative to the brightness of the source: ``wing_max =
        wing_frac * peak``, with ``peak`` from :func:`wing_peak`. A relative
        threshold keeps the same rows measured whatever the brightness.
        Ignored when ``wing_max`` is given. The threshold used is returned in
        ``info['wing_threshold']``.
    wing_min_pixels : int
        With ``wing_max``: an amplifier-row (or column) without enough
        background pixels uses its wing pixels only if at least this many
        qualify; otherwise it is left to the model. A few faint pixels at the
        end of the spectrum in a bright row are sensitive to shifts along the
        dispersion and would set the offset of the whole row.

    Returns
    -------
    data : ndarray
        Corrected data.
    info : dict
        Diagnostics: ``'profile'`` (stripe values for every integration and
        group), and for ``'diff'`` the ``'scale'`` factors and
        ``'median_frames'``.
    """
    if not inplace:
        data = np.array(data, copy=True)
    nint, ngroup, ny, nx = data.shape
    info = {'method': method, 'axis': axis}
    profiles, models = [], []

    median_frames, scale = None, None
    if method in ('diff', 'readout'):
        median_frames = np.empty((ngroup, ny, nx), dtype=np.float32)
        for g in range(ngroup):
            dg = np.array(data[:, g], dtype=np.float32)
            if groupdq is not None:
                dg[(groupdq[:, g] & _BAD_GROUP) > 0] = np.nan
            median_frames[g] = nanmedian(dg, axis=0)
        src = source_mask if source_mask is not None else ~bkg_mask
        scale = diff_scale_factors(data, median_frames, src, groupdq)
        info['scale'] = scale
        info['median_frames'] = median_frames
        if use_source_pixels:
            sel = np.linspace(0, nint - 1, min(nint, weight_ints)).astype(int)
            r = np.array(data[sel], dtype=np.float32) - scale[sel, None, None, None] * median_frames[None]
            if groupdq is not None:
                r[(groupdq[sel] & _BAD_GROUP) > 0] = np.nan
            # remove the common stripes first so that they do not inflate the variance
            r -= nanmedian(r, axis=-2)[..., None, :] if axis == 'column' else nanmedian(r, axis=-1)[..., None]
            var = mad_sigma(r, axis=0) ** 2                      # (ngroup, ny, nx)
            valid = bkg_mask | (source_mask if source_mask is not None else ~bkg_mask)
            weights = np.where((var > 0) & np.isfinite(var) & valid[None], 1. / var, 0.)
            info['weights'] = weights.astype(np.float32)

    fit_mask = bkg_mask
    if wing_max is not None or wing_frac is not None:
        if method not in ('diff', 'readout'):
            raise ValueError("wing_max / wing_frac require method='diff' or 'readout'.")
        rate = (median_frames[-1] - median_frames[0]) / (ngroup - 1) if ngroup > 1 \
            else median_frames[0]
        src = source_mask if source_mask is not None else ~bkg_mask
        if wing_max is None:
            info['wing_peak'] = wing_peak(rate, src)
            wing_max = wing_frac * info['wing_peak']
        info['wing_threshold'] = float(wing_max)
        wing = src & np.isfinite(rate) & (rate < wing_max)
        wing = _drop_sparse_wings(wing, bkg_mask, axis, amp_bounds, min_pixels, wing_min_pixels)
        fit_mask = bkg_mask | wing
        info['fit_mask'] = fit_mask

    for sl in _chunks(nint, chunk):
        d = np.array(data[sl], dtype=np.float32)
        if groupdq is not None:
            d[(groupdq[sl] & _BAD_GROUP) > 0] = np.nan
        if exclude is not None:
            d[np.broadcast_to(exclude[sl], d.shape)] = np.nan
        if method in ('diff', 'readout'):
            d = d - scale[sl, None, None, None] * median_frames[None]
        if method == 'readout':
            model, prof = _readout_stripes(d, fit_mask, axis, amp_bounds, min_pixels, nsigma,
                                           maxiters, readout_order, readout_direct)
        elif method == 'diff' and use_source_pixels:
            model, prof = stripe_wmean(d, weights[None], axis=axis, nsigma=max(nsigma, 4.),
                                       maxiters=maxiters,
                                       amp_bounds=amp_bounds if axis == 'row' else None)
        elif method == 'rowcol':
            model, prof = stripe_rowcol(d, bkg_mask, amp_bounds=amp_bounds, nsigma=nsigma,
                                        maxiters=maxiters)
        elif method in ('median', 'diff'):
            model, prof = stripe_median(d, fit_mask, axis=axis, nsigma=nsigma, maxiters=maxiters,
                                        amp_bounds=amp_bounds if axis == 'row' else None,
                                        min_pixels=min_pixels)
        elif method == 'poly':
            model, prof = stripe_poly(d, bkg_mask, axis=axis, order=poly_order, nsigma=nsigma,
                                      maxiters=maxiters)
        else:
            raise ValueError(f'Unknown group-background method {method}.')
        data[sl] -= model.astype(data.dtype)
        profiles.append(np.asarray(prof, dtype=np.float32))
        if return_model:
            models.append(np.array(model, dtype=np.float32))

    info['profile'] = np.concatenate(profiles, axis=0)
    if return_model:
        info['model'] = np.concatenate(models, axis=0)
    return data, info


def wing_peak(rate, source_mask, percentile=95.):
    """Peak signal of the source: a high percentile of the per-column maxima.

    Parameters
    ----------
    rate : ndarray
        Signal per group ``(ny, nx)``.
    source_mask : ndarray
        Boolean ``(ny, nx)``, ``True`` on the source.
    percentile : float
        Percentile of the column maxima (robust against a few hot columns and
        independent of the spectral shape's faint ends).
    """
    r = np.where(source_mask & np.isfinite(rate), rate, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        colmax = np.nanmax(r, axis=0)
    colmax = colmax[np.isfinite(colmax) & (colmax > 0)]
    return float(np.percentile(colmax, percentile)) if colmax.size else np.nan


def _drop_sparse_wings(wing, bkg_mask, axis, amp_bounds, min_pixels, wing_min_pixels):
    """Remove wing pixels of lines (amplifier-rows / columns) that have too few of them."""
    wing = wing.copy()
    if axis == 'column':
        nb, nw = bkg_mask.sum(0), wing.sum(0)
        wing[:, (nb < min_pixels) & (nw < wing_min_pixels)] = False
        return wing
    for (x0, x1) in (amp_bounds or [(0, wing.shape[1])]):
        nb, nw = bkg_mask[:, x0:x1].sum(1), wing[:, x0:x1].sum(1)
        wing[(nb < min_pixels) & (nw < wing_min_pixels), x0:x1] = False
    return wing


def _readout_stripes(d, bkg_mask, axis, amp_bounds, min_pixels, nsigma, maxiters, order, direct):
    """Per-line, per-amplifier offsets completed with :func:`readout_model`."""
    if axis == 'column':                      # lines are columns: treat as rows of the transpose
        d = np.swapaxes(d, -1, -2)
        bkg_mask = bkg_mask.T
        amp_bounds = None
    nx = d.shape[-1]
    amps = amp_bounds or [(0, nx)]
    masked = np.where(bkg_mask, d, np.nan).astype(np.float32)
    vals, cnts = [], []
    for (x0, x1) in amps:
        v, n = clipped_median(masked[..., x0:x1], axis=-1, nsigma=nsigma, maxiters=maxiters,
                              return_count=True)
        vals.append(v)
        cnts.append(n)
    vals = np.stack(vals, axis=-1)            # (..., nline, namp)
    cnts = np.stack(cnts, axis=-1)
    final, _, _ = readout_model(vals, cnts, min_pixels=min_pixels, order=order, use_direct=direct)
    final = np.where(np.isfinite(final), final, 0.)
    model = np.empty(d.shape, np.float32)
    for k, (x0, x1) in enumerate(amps):
        model[..., x0:x1] = final[..., k][..., None]
    if axis == 'column':
        model = np.swapaxes(model, -1, -2)
    return model, final


def amp_bounds_from_model(model):
    """Amplifier column ranges for a jwst datamodel (uses NOUTPUTS/SUBSTRT1)."""
    nx = model.data.shape[-1]
    nout = getattr(model.meta.exposure, 'noutputs', None) or 1
    xstart = getattr(model.meta.subarray, 'xstart', None) or 1
    return amplifier_bounds(nx, nout, xstart)
