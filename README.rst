jstark
======

``jstark`` reduces and analyses JWST time-series observations of transiting
exoplanets, from raw ``uncal`` files to light curves, transit and eclipse
fits, and spectra:

* **Reduction** (``jstark.reduction``): the jwst Detector1 steps plus
  group-level 1/f correction, time-domain jump detection, wavelength maps
  and per-exposure cleaning.
* **Extraction** (``jstark.extraction``): optimal spectral extraction with a
  spline PSF, and aperture and PSF-weighted photometry.
* **Fitting** (``jstark.fitting``): joint white-light and spectroscopic
  light-curve fits with `juliet <https://juliet.readthedocs.io>`_.

Supported modes: NIRSpec PRISM and gratings, NIRISS SOSS, NIRCam grism
time series, MIRI LRS and MIRI imaging.

Installation
------------

.. code-block:: bash

    pip install "jwstark[fitting] @ git+https://github.com/Jayshil/jstark.git"

    # optional: JAX samplers for the light-curve fits (NUTS, GPUs)
    pip install "jwstark[fitting,jax] @ git+https://github.com/Jayshil/jstark.git"

The package is installed as ``jwstark`` and imported as ``jstark``.

Documentation
-------------

The quick start, user guide, API and tutorials (WASP-39 b, WASP-43 b and
GJ 3929 b) are at https://jstark.readthedocs.io.

License
-------

Copyright (c) Jayshil A. Patel, licensed under the GNU GPL v3+ (see the
``licenses`` folder). Based on the
`OpenAstronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_.
