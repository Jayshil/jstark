Tool to reduce, extract, and analyze JWST timeseries data
---------------------------------------------------------

``jstark`` is a modular toolkit for JWST time-series observations of
transiting exoplanets, from raw ``uncal`` files to light curves and spectra.

Reduction (``jstark.reduction``)
================================

* ``Stage1`` -- ``uncal`` -> ``rateints``: all jwst Detector1 steps, plus
  custom bias correction, group-level background / 1/f subtraction
  (column, row-by-amplifier, polynomial, or difference-image methods) and
  time-domain jump detection.
* ``Stage2`` -- wavelength maps on the full subarray (``assign_wcs``;
  PASTASOSS for NIRISS/SOSS).
* ``ExposureCleaner`` / ``CubeCleaner`` -- error fixing, bad-pixel and
  cosmic-ray identification and replacement, background subtraction;
  processed and written segment by segment (no stitching of segments).

Supported modes: NIRSpec PRISM / G395H / G235H / G395M / G235M, NIRISS SOSS,
NIRCam F444W / F322W2 grism, MIRI LRS and MIRI F1500W imaging. See
``docs/reduction.rst`` for a walk-through.

.. code-block:: python

    from jstark.reduction import Stage1, Stage2, ExposureCleaner
    s1 = Stage1('/data/NRS1', outdir='/data/NRS1/reduction'); s1.run()
    s2 = Stage2('/data/NRS1/reduction'); wfile = s2.run()['wavelength']
    ec = ExposureCleaner('/data/NRS1/reduction', wavelength=wfile)
    info = ec.run()          # cleaned segment files + <exposure>_cleaninfo.fits

Extraction (``jstark.extraction``)
==================================

``SpectralExtractor`` (spline-PSF optimal extraction, segment by segment),
``Photometry`` (aperture and PSF-weighted photometry of imaging data) and
``ApPhoto`` (in-memory aperture photometry and PLD). See
``docs/extraction.rst``.

Light-curve fitting (``jstark.fitting``)
========================================

``LightCurves`` (one dataset: times, fluxes per channel and named
regressors, read from jstark spectra or photometry), ``BroadbandFit``
(joint juliet fit of the white-light or photometric light curves of several
visits or instruments, with linear, GP and ramp systematics),
``SpectroscopicFit`` (one joint fit per wavelength bin over all datasets
covering it, in parallel and resumable; results as a ``Spectrum`` table)
and ``julietPlots``. See ``docs/fitting.rst``.

License
-------

This project is Copyright (c) Jayshil A. Patel and licensed under
the terms of the GNU GPL v3+ license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! jstark is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
jstark based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
