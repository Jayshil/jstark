"""Custom bias (superbias) corrections.

The CRDS superbias does not always match the bias level of time-series
data, which leaves column structure and offsets in the ramps (most
prominently for NIRSpec, which has no reference pixels in its TSO
subarrays). Three alternatives to the standard ``superbias`` step are
offered:

``'scaled'``
    Scale the CRDS superbias so that it matches the zeroth group in
    background pixels (Eureka!'s ``bias_correction``). The scale factor is
    computed per integration and can be used as is (``mode='per_int'``),
    averaged over the segment (``'mean'``), or smoothed in time
    (``'smooth'``).

``'self'``
    A superbias measured from the data themselves. With a reset–read–read
    pattern and one frame per group (``NFRAMES=1, GROUPGAP=0``), the first
    group contains one group-time of signal on top of the bias, so
    ``bias ≈ median_i(2·g0 − g1)`` over all integrations. This removes the
    signal of the first group correctly, unlike simply subtracting the
    median first group.

``'user'``
    A user-supplied 2D bias image (array or FITS file).
"""
import numpy as np

from .robust import nanmedian, running_median

__all__ = ['self_superbias', 'superbias_scale_factors', 'apply_bias']

_BAD_GROUP = 1 | 2


def self_superbias(data, groupdq=None, nframes=1, groupgap=0):
    """Estimate the bias image from the data: ``median_i(2 g0 - g1)``.

    Parameters
    ----------
    data : ndarray
        Ramp cube ``(nint, ngroup, ny, nx)`` (at least two groups).
    groupdq : ndarray or None
        Group DQ; integrations in which either of the first two groups is
        saturated / do-not-use are excluded per pixel.
    nframes, groupgap : int
        Readout parameters. The extrapolation assumes the first group
        accumulates ``(nframes + 1) / 2`` frame times of signal and the
        group spacing is ``nframes + groupgap`` frames.

    Returns
    -------
    bias : ndarray
        2D bias image ``(ny, nx)``; NaN where no estimate is possible.
    """
    if data.shape[1] < 2:
        raise ValueError('The self-calibrated bias needs at least two groups.')
    g0 = np.array(data[:, 0], dtype=np.float32)
    g1 = np.array(data[:, 1], dtype=np.float32)
    if groupdq is not None:
        bad = ((groupdq[:, 0] | groupdq[:, 1]) & _BAD_GROUP) > 0
        g0[bad] = np.nan
        g1[bad] = np.nan
    frac = ((nframes + 1) / 2.) / (nframes + groupgap)   # signal in g0 / (g1 - g0)
    return nanmedian(g0 - frac * (g1 - g0), axis=0)


def superbias_scale_factors(group0, superbias, bkg_mask, mode='smooth', window=51):
    """Scale factors matching the superbias to the zeroth group.

    ``SF_i = median(group0_i[bkg]) / median(superbias[bkg])``.

    Parameters
    ----------
    group0 : ndarray
        Zeroth group of every integration ``(nint, ny, nx)``.
    superbias : ndarray
        2D superbias.
    bkg_mask : ndarray
        Boolean mask of background pixels used to compute the ratio.
    mode : {'per_int', 'mean', 'smooth'}
        How the per-integration factors are combined.
    window : int
        Running-median window for ``mode='smooth'``.

    Returns
    -------
    ndarray
        Scale factor of each integration.
    """
    sel = bkg_mask & np.isfinite(superbias)
    ref = np.nanmedian(superbias[sel])
    sf = nanmedian(group0[:, sel], axis=1) / ref
    if mode == 'mean':
        sf = np.full_like(sf, np.nanmedian(sf))
    elif mode == 'smooth':
        sf = running_median(sf, window, axis=0)
    elif mode != 'per_int':
        raise ValueError("mode must be 'per_int', 'mean' or 'smooth'.")
    return sf


def apply_bias(data, bias, scale=None):
    """Subtract ``scale_i * bias`` from every group of every integration (in place)."""
    bias = np.where(np.isfinite(bias), bias, 0.).astype(np.float32)
    if scale is None:
        data -= bias[None, None].astype(data.dtype)
    else:
        for i in range(data.shape[0]):
            data[i] -= (scale[i] * bias)[None].astype(data.dtype)
    return data
