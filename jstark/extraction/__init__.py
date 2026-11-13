"""Spectral extraction and photometry of cleaned JWST time-series data.

The spline-PSF optimal extraction is a port of ``stark``
(A. Brandeker & J. A. Patel; https://github.com/Jayshil/stark), extended to
work segment by segment on exposures of any length (see
:mod:`jstark.extraction.psf` for the differences).

Main classes
------------
SpectralExtractor
    Trace, background, PSF and extraction of every order of a spectroscopic
    exposure; writes one spectra file per segment and a combined file.
Spectra
    The combined extracted spectra (white light, binned light curves).
Photometry
    Aperture and PSF-weighted photometry of imaging exposures (e.g. MIRI
    F1500W), segment by segment; ``ApPhoto`` is the in-memory aperture/PLD
    helper (moved here from ``jstark.jstark``).
ExposureSource
    Chunked, segment-by-segment access to the cleaned data.

Field-star contamination (NIRISS/SOSS): ``SpectralExtractor.find_contaminants``
masks zeroth orders found in the F277W exposure, ``add_field_trace`` models and
subtracts field-star traces (:mod:`jstark.extraction.contamination`).
"""
from .source import ExposureSource  # noqa: F401
from .trace import Trace, find_trace  # noqa: F401
from .psf import SplinePSF  # noqa: F401
from .extractor import SpectralExtractor, Spectra  # noqa: F401
from .metrics import p2p_sigma, p2p_noise  # noqa: F401
from .photometry import Photometry, PhotometryResult, ApPhoto  # noqa: F401

__all__ = ['SpectralExtractor', 'Spectra', 'Photometry', 'PhotometryResult', 'ApPhoto',
           'ExposureSource', 'Trace', 'SplinePSF']
