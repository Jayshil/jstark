jstark
======

**jstark** is a modular Python toolkit for JWST time-series observations of
transiting exoplanets. It takes raw ``uncal`` files to transmission,
emission and phase-curve spectra: detector processing, spectral extraction
or photometry, and light-curve fitting.

It is built on the official `jwst <https://jwst-pipeline.readthedocs.io>`_
pipeline and adds its own steps where tests on real data showed that the
standard ones fall short. Every step can be stopped, inspected, changed and
resumed, and every intermediate product is saved.

.. code-block:: text
   :class: pipeline

   *_uncal.fits  (one file per segment)
        │
        ▼  jstark.reduction.Stage1        detector processing, group-level 1/f, jumps
   *_rateints.fits
        │
        ├─► jstark.reduction.Stage2      wavelength map on the full subarray
        ▼  jstark.reduction.ExposureCleaner   bad pixels, cosmic rays (segment by segment)
   *_cleaned.fits  +  <exposure>_cleaninfo.fits
        │
        ▼  jstark.extraction             spline-PSF optimal extraction / photometry
   <exposure>_spectra.fits   or   <exposure>_photometry.fits
        │
        ▼  jstark.fitting                juliet fits: white light, spectra, phase curves
   transit / eclipse depths, spectra, phase-curve parameters

Supported modes
---------------

================  =============================================
Instrument        Modes
================  =============================================
NIRSpec BOTS      PRISM, G395H, G395M, G235H, G235M
NIRISS            SOSS (orders 1 and 2, extracted jointly)
NIRCam            long-wave grism (GRISMR): F444W, F322W2
MIRI              LRS slitless, F1500W imaging
================  =============================================

G235H and G235M are supported but have not yet been tested on real data.

What jstark does differently
----------------------------

* **Segment by segment.** Exposures of any length are processed one segment
  at a time and never stitched into one cube, so memory use does not grow
  with the length of the observation. Quantities that need the whole
  exposure (saturation map, cosmic-ray noise, the PSF) are built in
  streaming passes over all segments.
* **Group-level 1/f removal** tuned per instrument: column medians on
  NIRSpec, difference images on NIRISS/SOSS, and a read-time model with a
  common mode across the four amplifiers on NIRCam.
* **Consistent saturation**: a partially saturated pixel uses the same
  groups in every integration, so its rate does not jump by several per
  cent between integrations.
* **Time-domain cosmic-ray detection** at the group and integration level,
  with a leave-one-out running median.
* **Spline-PSF optimal extraction** (a vectorised port of
  `stark <https://github.com/Jayshil/stark>`_): a tensor-product spline
  profile fitted to all integrations, smoothed by cross-validation, with a
  per-pixel response map measured from the data instead of a flat field.
  One PSF can be shared by all exposures of a visit.
* **Joint light-curve fits** of several visits, detectors or instruments
  with `juliet <https://juliet.readthedocs.io>`_: broadband and
  spectroscopic fits, common-mode correction, phase curves.

Each default was chosen by a test on real data; the
:doc:`tutorials <tutorials/index>` show those tests next to the analysis.

Installation
------------

jstark needs Python 3.11 or later. A fresh environment is recommended:

.. code-block:: bash

   conda create -n jstark python=3.12
   conda activate jstark
   pip install "jwstark[fitting] @ git+https://github.com/Jayshil/jstark.git"

The ``fitting`` extra installs juliet, corner and radvel, needed for
:mod:`jstark.fitting`; leave it out (``pip install git+https://github.com/Jayshil/jstark.git``) if
you only reduce and extract. The package is installed as ``jwstark`` (the
name ``jstark`` is taken on PyPI) and imported as ``jstark``.
For development, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/Jayshil/jstark.git
   cd jstark
   pip install -e ".[fitting,test]"

The jwst pipeline downloads its reference files from CRDS. Tell it where to
keep them (several GB) before the first run:

.. code-block:: bash

   export CRDS_PATH=$HOME/crds_cache
   export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

See the :doc:`user_guide/quickstart` for a first run.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   tutorials/index

Credits
-------

jstark is developed by Jayshil A. Patel. The spectral extraction is a port of
``stark`` (A. Brandeker & J. A. Patel). jstark builds on the
`jwst <https://github.com/spacetelescope/jwst>`__ calibration pipeline
(STScI), `juliet <https://github.com/nespinoza/juliet>`__ (Espinoza et al.
2019) and the packages it uses (batman, dynesty), and on numpy, scipy,
astropy and bottleneck. Please cite these packages when you use jstark.
