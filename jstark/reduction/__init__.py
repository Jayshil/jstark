"""JWST time-series reduction: from ``uncal`` ramps to cleaned 3D cubes.

Main classes
------------
Stage1
    Detector-level processing (``uncal`` -> ``rateints``) with custom
    bias, group-level background / 1/f and time-domain jump steps.
Stage2
    Wavelength solution (``assign_wcs``) and optional Stage 2 steps.
ExposureCleaner
    Segment-by-segment cleaning of a whole exposure (two passes, no
    stitching); writes one cleaned file per segment plus a small info file.
CubeCleaner
    In-memory cleaning of one cube (one segment): error fixing, bad-pixel / cosmic-ray identification and correction,
    and integration-level background subtraction.
"""
from .instruments import *  # noqa: F401,F403
from .io import find_exposures, select_exposure, StepLog  # noqa: F401
from .stage1 import Stage1  # noqa: F401
from .stage2 import Stage2, wavelength_map  # noqa: F401
from .cleaning import CubeCleaner, ExposureCleaner  # noqa: F401
