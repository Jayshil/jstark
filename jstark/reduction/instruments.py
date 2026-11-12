"""Instrument-mode definitions and default reduction recipes.

Every supported JWST time-series mode is described by a :class:`ModeConfig`.
It stores the detector geometry relevant to the reduction (dispersion axis,
1/f-noise direction, amplifier layout) together with the *default* Stage 1
recipe, i.e. the ordered list of steps (official ``jwst`` steps and custom
``jstark`` steps) and their default parameters.

The defaults follow typical choices in the published literature:

* Group-level background / 1/f subtraction for all near-infrared modes
  (Eureka!: Bell et al. 2022; ExoTiC-JEDI: Alderson et al. 2023;
  exoTEDRF: Radica 2024; see also the NIRISS/SOSS technical report,
  arXiv:2505.05571, which recommends group-level correction).
  Column-by-column for NIRSpec and NIRISS (fast-read direction along the
  DMS y-axis); row-by-row for NIRCam grism (Schlawin et al. 2020), using
  the ``'readout'`` model: in NIRCam ~96 per cent of the row-to-row 1/f is
  common to the four simultaneously read amplifiers, so rows in which an
  amplifier has no background pixels are filled with the common mode
  measured in the other amplifiers plus that amplifier's own read-time
  trend.
* For NIRISS/SOSS, the "difference-image" 1/f method of exoTEDRF is the
  default because there is very little background area on the detector.
* Consistent saturation flags: each pixel uses the same set of groups in
  every integration of the exposure. Otherwise partially saturated pixels
  switch between e.g. 4 and 5 fitted groups from one integration to the next,
  and their rates jump by several per cent (measured on NIRSpec/PRISM).
* Jump detection in the time domain (as in exoTEDRF's ``jumpstep_in_time``)
  instead of the default up-the-ramp two-point-difference algorithm.
* Dark-current subtraction is skipped for NIRSpec and NIRCam (dark current
  is negligible and the reference files add noise), and applied for
  NIRISS/SOSS and MIRI.
* MIRI: ``emicorr`` (390 Hz EMI), ``firstframe``, ``lastframe``, ``reset``
  and ``rscd`` are applied; no group-level background subtraction. The ramp
  fit uses every group those steps leave usable. The early groups are
  distorted by RSCD, but on GJ 3929 F1500W the distortion was the same in
  every integration (it only rescales the flux), while dropping the first 12
  groups raised the photometric scatter by 28 per cent. ``firstgroup`` (an
  integer, or ``'auto'``) remains available for tests on a full exposure.

All of these can be changed by the user through :class:`~jstark.reduction.Stage1`.
"""
import copy
from dataclasses import dataclass, field

__all__ = ['StepSpec', 'ModeConfig', 'identify_mode', 'get_mode_config', 'list_modes',
           'JSTARK_STEPS', 'BAD_DQ_FLAGS']

# Custom (non-jwst) Stage 1 steps implemented in jstark.
JSTARK_STEPS = ('consistent_saturation', 'custom_bias', 'group_bkg', 'temporal_jump')

# Default DQ flags treated as "bad" when building bad-pixel maps.
# SATURATED is deliberately *not* included: in rateints products jwst sets it for any
# pixel that saturated in some group, even when a valid slope was fitted from the
# unsaturated groups (partial saturation, e.g. the core of a bright NIRSpec/PRISM trace).
# Pixels without a valid slope carry DO_NOT_USE (or a NaN rate) and are still caught.
BAD_DQ_FLAGS = ('DO_NOT_USE', 'DEAD', 'HOT', 'NONLINEAR', 'NO_LIN_CORR',
                'NO_GAIN_VALUE', 'REFERENCE_PIXEL', 'NON_SCIENCE', 'UNRELIABLE_SLOPE',
                'OTHER_BAD_PIXEL')


@dataclass
class StepSpec:
    """One step in a Stage 1 recipe.

    Parameters
    ----------
    name : str
        Name of the step. Either a ``jwst`` Detector1 step name (e.g.
        ``'superbias'``) or one of the custom jstark steps
        (``'custom_bias'``, ``'group_bkg'``, ``'temporal_jump'``).
    kwargs : dict
        Parameters passed to the step.
    skip : bool
        If ``True`` the step is not executed.
    """
    name: str
    kwargs: dict = field(default_factory=dict)
    skip: bool = False

    @property
    def is_custom(self):
        return self.name in JSTARK_STEPS


@dataclass
class ModeConfig:
    """Configuration of one JWST time-series observing mode.

    Attributes
    ----------
    name : str
        Mode identifier, e.g. ``'NIRSpec/G395H'``.
    instrument : str
        ``'NIRSPEC'``, ``'NIRISS'``, ``'NIRCAM'`` or ``'MIRI'``.
    spectroscopic : bool
        ``False`` for imaging (photometric) time series.
    dispersion_axis : int or None
        Array axis of a 2D frame along which the wavelength changes:
        ``1`` (x, columns) or ``0`` (y, rows). ``None`` for imaging.
    onef_axis : str or None
        Direction of 1/f stripes in DMS orientation. ``'column'`` means a
        stripe is constant along a column (correction computed for each
        column), ``'row'`` means constant along a row.
    per_amplifier : bool
        Whether background/1/f statistics should be computed separately
        for each readout amplifier (only matters when NOUTPUTS > 1).
    recipe : list of StepSpec
        Ordered default Stage 1 steps.
    trace_mask : dict
        Default keyword arguments for building the trace mask used to
        select background pixels.
    nint_window : int
        Default running window (in integrations) used for time-domain
        outlier detection.
    notes : str
        Free-form notes about the mode.
    """
    name: str
    instrument: str
    spectroscopic: bool = True
    dispersion_axis: int = 1
    onef_axis: str = 'column'
    per_amplifier: bool = False
    recipe: list = field(default_factory=list)
    trace_mask: dict = field(default_factory=dict)
    nint_window: int = 11
    notes: str = ''

    def copy(self):
        return copy.deepcopy(self)

    def step(self, name):
        """Return the :class:`StepSpec` called ``name``."""
        for s in self.recipe:
            if s.name == name:
                return s
        raise KeyError(f'Step {name} is not part of the recipe of {self.name}.')


# ---------------------------------------------------------------------------
# Default recipes
# ---------------------------------------------------------------------------

def _nir_recipe(onef_axis, onef_method='median', dark=False, refpix_kwargs=None,
                charge_migration=False, onef_kwargs=None):
    """Default near-infrared (NIRSpec, NIRISS, NIRCam) Stage 1 recipe."""
    return [
        StepSpec('group_scale'),
        StepSpec('dq_init'),
        StepSpec('saturation', {'n_pix_grow_sat': 1}),
        StepSpec('consistent_saturation', {'min_frac': 0.01, 'scope': 'exposure'}),
        StepSpec('superbias'),
        StepSpec('custom_bias', {'method': 'scaled', 'mode': 'smooth'}, skip=True),
        StepSpec('refpix', refpix_kwargs or {}),
        StepSpec('group_bkg', {'method': onef_method, 'axis': onef_axis, **(onef_kwargs or {})}),
        StepSpec('linearity'),
        StepSpec('dark_current', skip=not dark),
        StepSpec('charge_migration', skip=not charge_migration),
        StepSpec('jump', {'rejection_threshold': 15., 'maximum_cores': 'half'}, skip=True),
        StepSpec('temporal_jump', {'window': 11, 'threshold': 10., 'mode': 'auto'}),
        StepSpec('clean_flicker_noise', skip=True),
        StepSpec('ramp_fit', {'algorithm': 'OLS_C', 'maximum_cores': 'half'}),
        StepSpec('gain_scale'),
    ]


def _miri_recipe():
    """Default MIRI Stage 1 recipe."""
    return [
        StepSpec('group_scale'),
        StepSpec('dq_init'),
        StepSpec('emicorr'),
        StepSpec('saturation'),
        StepSpec('consistent_saturation', {'min_frac': 0.01, 'scope': 'exposure'}),
        StepSpec('firstframe', {'bright_use_group1': True}),
        StepSpec('lastframe'),
        StepSpec('reset'),
        StepSpec('linearity'),
        StepSpec('rscd'),
        StepSpec('dark_current'),
        StepSpec('refpix'),
        StepSpec('group_bkg', {'method': 'median', 'axis': 'row'}, skip=True),
        StepSpec('jump', {'rejection_threshold': 15., 'maximum_cores': 'half'}, skip=True),
        StepSpec('temporal_jump', {'window': 11, 'threshold': 10., 'mode': 'auto'}),
        # RSCD distorts the first MIRI groups (group-to-group increases ~50-120% too high in
        # group 1, linear to ~1% from group 12; WASP-43 LRS, GJ 3929 F1500W). Measured on
        # F1500W, the distortion is the same in every integration (a flux scale only), and
        # starting the fit at group 4 / 8 / 12 raised the scatter by 8 / 14 / 28% without
        # changing the early-visit ramp or the red noise. So all usable groups are fitted;
        # set firstgroup (int or 'auto') to test trimming on a full exposure.
        StepSpec('ramp_fit', {'algorithm': 'OLS_C', 'maximum_cores': 'half'}),
        StepSpec('gain_scale'),
    ]


def _nirspec(grating):
    return ModeConfig(
        name=f'NIRSpec/{grating}', instrument='NIRSPEC', dispersion_axis=1,
        onef_axis='column', per_amplifier=False,
        recipe=_nir_recipe('column', 'median', dark=False),
        trace_mask={'method': 'threshold', 'nsigma': 3., 'pad': 2, 'min_pixels': 4},
        notes='NIRSpec BOTS (S1600A1). 1/f stripes run along DMS columns. '
              'G395H/G235H use NRS1 and NRS2; M gratings and PRISM use NRS1.')


def _niriss_soss():
    return ModeConfig(
        name='NIRISS/SOSS', instrument='NIRISS', dispersion_axis=1, onef_axis='column',
        per_amplifier=False,
        recipe=_nir_recipe('column', 'diff', dark=True),
        trace_mask={'method': 'threshold', 'nsigma': 3., 'pad': 3, 'min_pixels': 10},
        notes='NIRISS SOSS (GR700XD). 1/f is corrected with the difference-image method '
              'because the three spectral orders leave little background area.')


def _nircam(filt):
    return ModeConfig(
        name=f'NIRCam/{filt}', instrument='NIRCAM', dispersion_axis=1, onef_axis='row',
        per_amplifier=True,
        recipe=_nir_recipe('row', 'readout', dark=False,
                           refpix_kwargs={'odd_even_columns': True, 'use_side_ref_pixels': True},
                           onef_kwargs={'readout_order': 1}),
        trace_mask={'method': 'threshold', 'nsigma': 3., 'pad': 3, 'min_pixels': 10},
        notes='NIRCam long-wave grism time series (GRISMR). 1/f stripes run along DMS rows. '
              'Default group-level method: readout-time model (per-amplifier read-time trend '
              '+ common mode across amplifiers), which fills rows where an amplifier has no '
              'background pixels; ~15% lower white-light scatter than row-by-row medians.')


def _miri_lrs():
    return ModeConfig(
        name='MIRI/LRS', instrument='MIRI', dispersion_axis=0, onef_axis='row',
        per_amplifier=False, recipe=_miri_recipe(),
        trace_mask={'method': 'threshold', 'nsigma': 3., 'pad': 3, 'min_pixels': 10},
        nint_window=11,
        notes='MIRI LRS slitless (SLITLESSPRISM). Dispersion is along the y-axis.')


def _miri_imaging(filt):
    return ModeConfig(
        name=f'MIRI/{filt}', instrument='MIRI', spectroscopic=False, dispersion_axis=None,
        onef_axis=None, per_amplifier=False, recipe=_miri_recipe(),
        trace_mask={'method': 'threshold', 'nsigma': 3., 'pad': 5, 'min_pixels': 10},
        notes='MIRI imaging time series (e.g. F1500W eclipse photometry).')


_SUPPORTED = {
    'NIRSpec/PRISM': lambda: _nirspec('PRISM'),
    'NIRSpec/G395H': lambda: _nirspec('G395H'),
    'NIRSpec/G235H': lambda: _nirspec('G235H'),
    'NIRSpec/G395M': lambda: _nirspec('G395M'),
    'NIRSpec/G235M': lambda: _nirspec('G235M'),
    'NIRISS/SOSS': _niriss_soss,
    'NIRCam/F444W': lambda: _nircam('F444W'),
    'NIRCam/F322W2': lambda: _nircam('F322W2'),
    'MIRI/LRS': _miri_lrs,
    'MIRI/F1500W': lambda: _miri_imaging('F1500W'),
}


def list_modes():
    """Return the names of all supported modes."""
    return list(_SUPPORTED.keys())


def identify_mode(header):
    """Identify the jstark mode name from a primary FITS header (or dict).

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        Primary header of a JWST file.

    Returns
    -------
    str
        Mode name, e.g. ``'NIRSpec/G395H'``.
    """
    instrument = str(header.get('INSTRUME', '')).upper()
    exp_type = str(header.get('EXP_TYPE', '')).upper()
    filt = str(header.get('FILTER', '')).upper()
    if instrument == 'NIRSPEC':
        return f"NIRSpec/{str(header.get('GRATING', '')).upper()}"
    if instrument == 'NIRISS' and exp_type == 'NIS_SOSS':
        return 'NIRISS/SOSS'
    if instrument == 'NIRCAM' and exp_type == 'NRC_TSGRISM':
        return f'NIRCam/{filt}'
    if instrument == 'MIRI' and exp_type.startswith('MIR_LRS'):
        return 'MIRI/LRS'
    if instrument == 'MIRI' and exp_type == 'MIR_IMAGE':
        return f'MIRI/{filt}'
    raise ValueError(f'Unsupported mode: INSTRUME={instrument}, EXP_TYPE={exp_type}, FILTER={filt}.')


def get_mode_config(mode):
    """Return a fresh :class:`ModeConfig` for ``mode``.

    Unsupported filters/gratings of a supported instrument mode fall back to
    the closest generic configuration (with a warning).
    """
    if mode in _SUPPORTED:
        return _SUPPORTED[mode]()
    import warnings
    inst, sub = mode.split('/')
    warnings.warn(f'Mode {mode} is not explicitly supported; using generic {inst} defaults.')
    if inst == 'NIRSpec':
        return _nirspec(sub)
    if inst == 'NIRCam':
        return _nircam(sub)
    if inst == 'MIRI':
        return _miri_imaging(sub)
    raise ValueError(f'Unknown mode {mode}.')
