Tutorials
=========

Complete analyses of public JWST data, from ``uncal`` files to the
published quantities. Besides showing how to use jstark, each tutorial
explains *why* the defaults are what they are: the tests on real data that
chose each method, and the comparison with published results.

:doc:`wasp39b_prism`
   NIRSpec/PRISM transit of WASP-39 b: every stage in detail, saturated
   pixels and columns, the 1/f correction, the PSF model and pixel-response
   map, and the transmission spectrum against the ERS benchmark.

:doc:`wasp39b_nircam`
   NIRCam/F322W2 transit of WASP-39 b: 1/f noise across four amplifiers,
   extraction weights for summed columns, per-amplifier fits, and a default
   (offsets measured on the trace wings) that the transit overturned.

:doc:`wasp43b_miri_lrs`
   MIRI/LRS phase curve of WASP-43 b: the MIRI ramp and the ``rscd`` step,
   three exposures with one PSF, the 2024 wavelength-calibration change,
   and a phase-curve fit against four published reductions.

:doc:`gj3929b_miri_f1500w`
   MIRI/F1500W eclipses of the rocky planet GJ 3929 b: imaging photometry,
   a joint fit of five datasets, and why a free eclipse time is risky on
   weak eclipses.

Each tutorial runs Stage 1, cleaning and extraction on one segment, as you
would when trying out settings, and loads the products of the full
exposures, made with the same code (shown in the notebook, behind
``RUN_FULL = False``). The raw data are tens of GB per data set (from
`MAST <https://mast.stsci.edu>`_) and the full reductions and fits take
hours, so the notebooks were run locally and are shown with their outputs.
To run them yourself, download the data and change the paths in the setup
cell.

.. toctree::
   :maxdepth: 1
   :hidden:

   wasp39b_prism
   wasp39b_nircam
   wasp43b_miri_lrs
   gj3929b_miri_f1500w
