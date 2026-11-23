"""Light-curve fitting with juliet (needs the ``fitting`` extra: juliet, corner, radvel).

Main classes
------------
LightCurves
    One dataset (one visit x one mode/order/detector): times, flux and
    errors per channel, and a pool of named regressors. Read from jstark
    spectra (``from_spectra``) or photometry (``from_photometry``) files, from
    tables of other observatories (``from_table``), or built from arrays.
BroadbandFit
    Joint juliet fit of the broadband light curve (white light of the
    spectra, or the photometric light curve) of every dataset in a dict
    ``{name: LightCurves}``. The fit decides which regressors enter the
    linear model (``linear=``), which is the GP input (``gp=``) and which
    enter your own functions (``non_linear=``). Optionally with radial
    velocities (``rvs=``).
SpectroscopicFit
    One joint fit per wavelength bin over all datasets that cover it
    (parallel, resumable); results as a :class:`Spectrum` table.
julietPlots
    Plots of a juliet fit (moved from ``jstark.jstark``).

Priors: :func:`jstark.fitting.priors.build_priors` adds the nuisance priors
of each dataset (flux offset, jitter, linear coefficients, GP); planet,
orbit and limb-darkening priors are always yours.
"""
from .lightcurve import LightCurves, as_lightcurves, check_name  # noqa: F401
from .priors import build_priors, normalise_priors, model_report  # noqa: F401
from .fit import BroadbandFit, red_noise_beta  # noqa: F401
from .spectroscopic import SpectroscopicFit, Spectrum, restrict_priors, make_bins  # noqa: F401
from .plots import julietPlots  # noqa: F401

__all__ = ['LightCurves', 'BroadbandFit', 'SpectroscopicFit', 'Spectrum', 'julietPlots',
           'restrict_priors', 'make_bins', 'build_priors', 'normalise_priors',
           'model_report', 'as_lightcurves', 'check_name', 'red_noise_beta']
