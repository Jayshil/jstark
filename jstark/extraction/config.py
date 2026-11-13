"""Default extraction parameters per observing mode.

Parameters are grouped by step:

``columns``
    Column (spectral pixel) range of each order. ``columns=(x0, x1)`` or
    ``wavelength_range=(lam0, lam1)`` override the automatic choice (largest
    contiguous run of columns with a valid wavelength, enough good pixels and
    a per-integration S/N above ``snr_min``). MIRI/LRS defaults to
    ``wavelength_range=(5, 12)``; ``ex.set('columns', wavelength_range=None)``
    restores the automatic choice.
``trace``
    ``halfwidth`` of the centre-of-flux window, ``nknots`` of the spline
    describing the trace shape, ``clip``; ``reference='stage2'`` places the
    reference traces of the Stage 2 wavelength file (NIRISS/SOSS) with a
    fitted polynomial correction of degree ``ref_order``.
``background``
    ``method`` ('median', 'poly' or 'none'), ``inner`` / ``outer`` distance
    from the traces, ``nsigma``, ``poly_order``, ``min_pixels``.
``psf``
    ``halfwidth`` of the profile window, ``oversample`` (knots per pixel
    across the trace), ``knot_col`` (knots along the dispersion; 0 = 1D
    profile), ``smooth`` ('cv', a number, a pair, or 0 for none), ``clip``
    (outlier rejection, in sigma), ``init_ints`` (integrations of the
    first, in-memory fit), ``ints`` ('all' or a number of integrations
    spread over the exposure), ``second_pass``, ``pixel_correction`` (fit a
    static response factor per detector pixel, see
    :meth:`SpectralExtractor.fit_psf`), ``cv`` (cross-validation folds for
    ``smooth='cv'``: 'phase' = sub-pixel phase of the offset from the trace,
    'ints' = even/odd integrations, 'columns' = column blocks).
``extract``
    ``method`` ('optimal', 'box' or 'covariance'), ``aperture`` (half-width),
    ``variance`` ('mean' = time-averaged per-pixel variance, or 'data'),
    ``reject`` (sigma), ``reject_iters`` and ``reject_mode`` ('temporal':
    a pixel is rejected only if its residual deviates from its median
    residual over the chunk, so static model misfit never causes rejections;
    'model': deviation from the model itself, as in Horne 1986),
    ``fit_background``,
    ``refine_shift`` (Gauss-Newton iterations of the trace offset),
    ``joint_iters`` (alternating iterations of the joint multi-order
    extraction, ``method='joint'``), ``max_maskfrac`` (columns of an
    integration with a larger fraction of the profile on masked pixels get
    NaN: the remaining pixels hold too little of the light, e.g. under a
    masked field-star contaminant; saturated cores of bright NIRSpec traces
    reach ~0.95 and are still measured well from the wings).

The values below were chosen from one-segment tests (see
``docs/extraction_notes.md``).
"""
import copy

__all__ = ['DEFAULTS', 'MODE_DEFAULTS', 'get_defaults']

DEFAULTS = {
    'columns': dict(columns=None, wavelength_range=None, snr_min=5., min_good_frac=0.5, gap=5),
    'trace': dict(halfwidth=None, nknots=8, clip=3., reference=None, ref_order=1),
    'background': dict(method='median', inner=8., outer=None, nsigma=3., maxiters=3,
                       poly_order=1, min_pixels=4),
    'psf': dict(halfwidth=7., oversample=2., knot_col=20, smooth='cv', clip=6.,
                init_ints=200, ints='all', second_pass=False, pixel_correction=True, cv='phase'),
    'extract': dict(method='optimal', aperture=5., variance='mean', reject=6., reject_iters=3,
                    reject_mode='temporal', fit_background=False, refine_shift=2, joint_iters=3,
                    max_maskfrac=0.98),
}

MODE_DEFAULTS = {
    'NIRSpec': {'orders': [1]},
    'NIRSpec/PRISM': {'orders': [1]},
    'NIRISS/SOSS': {
        'orders': [1, 2],
        'trace': dict(halfwidth=12., reference='stage2', ref_order=1),
        'background': dict(method='poly', inner=20., min_pixels=10),
        'psf': dict(halfwidth=16., oversample=2., knot_col=100),
        'extract': dict(aperture=15., method='joint'),
    },
    'NIRCam': {
        'orders': [1],
        'background': dict(inner=12., min_pixels=10),
        'psf': dict(halfwidth=9., oversample=2., knot_col=20),
        'extract': dict(aperture=6.),
    },
    'MIRI/LRS': {
        'orders': [1],
        # below ~4.5 um the prism dispersion folds back (several wavelengths on the
        # same pixels) and 5-6 um receives some shorter-wavelength light; beyond
        # ~12 um the throughput is low and the background dominates
        'columns': dict(wavelength_range=(5., 12.)),
        'background': dict(inner=10., min_pixels=10),
        'psf': dict(halfwidth=7., oversample=2., knot_col=15),
        'extract': dict(aperture=5.),
    },
}


def get_defaults(mode):
    """Merged default parameters (and orders) for a jstark mode name."""
    p = copy.deepcopy(DEFAULTS)
    orders = [1]
    for key in (mode.split('/')[0], mode):
        over = MODE_DEFAULTS.get(key, {})
        orders = over.get('orders', orders)
        for step, kw in over.items():
            if step != 'orders':
                p[step].update(kw)
    return p, list(orders)
