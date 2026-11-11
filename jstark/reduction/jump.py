"""Jump (cosmic-ray) detection in the time domain at the group level.

In a time-series observation the same scene is observed in hundreds to
thousands of integrations. For every pixel and every group difference
(``group[k+1] - group[k]``) we therefore have a time series that is
stationary except for the (small) astrophysical variability and slow
systematics. A jump appears as an outlier in this time series. This is far
more sensitive than the standard up-the-ramp two-point-difference algorithm
when the number of groups is small, which is the usual case for bright
exoplanet host stars (cf. exoTEDRF's ``jumpstep_in_time``; Radica 2024, and
the NIRISS/SOSS technical report, arXiv:2505.05571).

Algorithm
---------
1. Compute group differences ``d[i, k] = data[i, k+1] - data[i, k]``.
2. Model each pixel's difference time series with a centred, leave-one-out
   running median over ``window`` integrations (the tested integration is
   excluded, so the model is not pulled by the outlier and the residual
   scatter is not underestimated).
3. Estimate the noise robustly per pixel and difference index from the
   median absolute deviation of the residuals in time.
4. Flag ``|d - model| > threshold * sigma``.
5. Either *flag* the group (set the ``JUMP_DET`` bit so that ramp fitting
   splits the ramp) or *replace* the affected difference by its running
   median value and rebuild the ramp (no information is lost, which
   matters for ramps with only a few groups).
"""
import numpy as np
from scipy import ndimage

from .robust import running_median, MAD_TO_SIGMA
import bottleneck as bn

__all__ = ['temporal_jump']

DO_NOT_USE = 1
SATURATED = 2
JUMP_DET = 4


def temporal_jump(data, groupdq=None, window=11, threshold=10., mode='replace',
                  two_sided=True, flag_neighbors=False, neighbor_threshold=None,
                  sigma_floor=None, row_chunk=None, max_mem_gb=4.):
    """Detect (and flag or repair) jumps by comparing integrations in time.

    Parameters
    ----------
    data : ndarray
        Ramp cube ``(nint, ngroup, ny, nx)``. Modified in place when
        ``mode='replace'``.
    groupdq : ndarray or None
        Group DQ array (same shape as ``data``). Modified in place when
        ``mode='flag'``. Saturated / do-not-use groups are ignored.
    window : int
        Running-median window, in integrations (made odd).
    threshold : float
        Detection threshold in units of the robust sigma.
    mode : {'replace', 'flag', 'auto'}
        ``'replace'`` repairs the ramp, ``'flag'`` sets ``JUMP_DET``.
        ``'auto'`` uses ``'replace'`` for ramps with fewer than 10 groups and
        ``'flag'`` otherwise.
    two_sided : bool
        If ``True``, negative outliers are also flagged.
    flag_neighbors : bool
        Also treat the 4 nearest neighbours of a jump as jumps when the
        jump significance exceeds ``neighbor_threshold``.
    neighbor_threshold : float or None
        Significance needed to flag neighbours (default ``3 * threshold``).
    sigma_floor : ndarray or float or None
        Lower limit on the noise of a group difference (in DN), e.g.
        ``sqrt(2) * readnoise``. Prevents false detections in pixels whose
        MAD is artificially small.
    row_chunk : int or None
        Number of detector rows processed at a time. Chosen from
        ``max_mem_gb`` when ``None``.

    Returns
    -------
    data : ndarray
        (Possibly repaired) ramp cube.
    groupdq : ndarray or None
        (Possibly updated) group DQ.
    info : dict
        ``'jump_mask'`` (boolean ``(nint, ngroup, ny, nx)``, ``True`` for
        groups that follow a detected jump), ``'n_jumps'``,
        ``'frac_per_int'`` (fraction of pixels with a jump in each
        integration), ``'mode'``.
    """
    nint, ngroup, ny, nx = data.shape
    if mode == 'auto':
        mode = 'replace' if ngroup < 10 else 'flag'
    if mode not in ('replace', 'flag'):
        raise ValueError("mode must be 'replace', 'flag' or 'auto'.")
    jump_mask = np.zeros((nint, ngroup, ny, nx), dtype=bool)
    info = {'mode': mode, 'window': window, 'threshold': threshold}
    if ngroup < 2 or nint < 3:
        info.update(jump_mask=jump_mask, n_jumps=0, frac_per_int=np.zeros(nint))
        return data, groupdq, info
    if neighbor_threshold is None:
        neighbor_threshold = 3 * threshold

    if row_chunk is None:
        bytes_per_row = nint * (ngroup - 1) * nx * 4 * (6 + window)  # work arrays + LOO windows
        row_chunk = int(max(1, min(ny, max_mem_gb * 1e9 // bytes_per_row)))

    for y0 in range(0, ny, row_chunk):
        ys = slice(y0, min(ny, y0 + row_chunk))
        d = np.diff(np.asarray(data[:, :, ys], dtype=np.float32), axis=1)
        if groupdq is not None:
            bad = (np.asarray(groupdq[:, :, ys]) & (DO_NOT_USE | SATURATED)) > 0
            bad_d = bad[:, 1:] | bad[:, :-1]
            d[bad_d] = np.nan
        # leave-one-out: the tested integration is not part of its own model
        model = running_median(d, window, axis=0, exclude_center=True)
        resid = d - model
        sigma = MAD_TO_SIGMA * bn.nanmedian(np.abs(resid), axis=0)  # (ngroup-1, ny_c, nx)
        if sigma_floor is not None:
            floor = sigma_floor if np.isscalar(sigma_floor) else np.asarray(sigma_floor)[ys]
            sigma = np.maximum(sigma, floor)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
        signif = resid / sigma[None]
        if not two_sided:
            hit = signif > threshold
        else:
            hit = np.abs(signif) > threshold
        hit &= np.isfinite(signif)
        if flag_neighbors:
            strong = np.abs(signif) > neighbor_threshold
            strong &= np.isfinite(signif)
            if np.any(strong):
                st = np.zeros((1, 1, 3, 3), bool)
                st[0, 0, 1, :] = st[0, 0, :, 1] = True
                grown = ndimage.binary_dilation(strong, structure=st)
                hit |= grown & np.isfinite(d)
        # a hit in difference k means a jump between groups k and k+1
        jump_mask[:, 1:, ys] = hit
        if mode == 'replace':
            corr = np.where(hit, model - d, 0.).astype(np.float32)
            corr = np.cumsum(corr, axis=1)
            data[:, 1:, ys] += corr.astype(data.dtype)
        elif groupdq is not None:
            gdq = groupdq[:, 1:, ys]
            gdq[hit] |= JUMP_DET
            groupdq[:, 1:, ys] = gdq

    njump = int(jump_mask.sum())
    frac = jump_mask.any(axis=1).reshape(nint, -1).mean(axis=1)
    info.update(jump_mask=jump_mask, n_jumps=njump, frac_per_int=frac)
    return data, groupdq, info
