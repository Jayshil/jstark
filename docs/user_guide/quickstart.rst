Quick start
===========

This page installs jstark and takes one exposure from ``uncal`` files to a
transit depth with the default settings of every stage. The following
pages explain each stage; the :doc:`tutorials <../tutorials/index>` go
through complete analyses of real data.

Installation
------------

.. code-block:: bash

   conda create -n jstark python=3.12
   conda activate jstark
   pip install "jwstark[fitting] @ git+https://github.com/Jayshil/jstark.git"

   # where the jwst pipeline keeps its reference files (several GB)
   export CRDS_PATH=$HOME/crds_cache
   export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

Check the installation with ``python -c "import jstark.reduction, jstark.extraction, jstark.fitting"``.

The data
--------

Download the ``*_uncal.fits`` files of your time series from
`MAST <https://mast.stsci.edu>`_ into one folder. Long exposures come in
segments (``..._00001-seg001_nrs1_uncal.fits``, ``-seg002``, ...); jstark
finds and processes all segments of an exposure automatically. If the
folder holds several exposures (e.g. the NIRISS/SOSS F277W exposure next to
the time series, or both NIRSpec detectors), the one with the most
integrations is used, or choose with ``exposure='nrs2'``.

From uncal to a transit depth
-----------------------------

The example uses NIRSpec/PRISM; the same code runs every supported mode, as
the mode is read from the file headers and each mode has its own defaults.

.. code-block:: python

    from jstark.reduction import Stage1, Stage2, ExposureCleaner
    from jstark.extraction import SpectralExtractor
    from jstark.fitting import LightCurves, BroadbandFit

    data = '/data/wasp39b/NRSPR'            # folder with the *_uncal.fits files
    red = data + '/reduction'               # all products go here

    if __name__ == '__main__':              # needed in scripts (multiprocessing)

        # 1. Detector processing: uncal -> rateints (one file per segment)
        s1 = Stage1(data, outdir=red)
        s1.show()                           # the steps and parameters for this mode
        s1.run()

        # 2. Wavelength map
        wfile = Stage2(red).run()['wavelength']

        # 3. Cleaning: bad pixels and cosmic rays, segment by segment
        info = ExposureCleaner(red, wavelength=wfile).run()

        # 4. Spectral extraction
        ex = SpectralExtractor(info)
        ex.find_trace()
        ex.fit_psf()                        # one PSF from all integrations
        spec = ex.run()                     # -> <exposure>_spectra.fits

        # 5. White-light transit fit with juliet
        lc = LightCurves.from_spectra(spec.fname)   # the fit sums the columns (white light)
        priors = {'P_p1': ('fixed', 4.0552941),
                  't0_p1': ('normal', [2459771.3356, 0.001]),
                  'a_p1': ('uniform', [5., 20.]),
                  'b_p1': ('uniform', [0., 1.]),
                  'p_p1': ('uniform', [0.05, 0.25]),
                  'ecc_p1': ('fixed', 0.), 'omega_p1': ('fixed', 90.),
                  'q1_PRISM': ('uniform', [0., 1.]),
                  'q2_PRISM': ('uniform', [0., 1.])}
        fit = BroadbandFit({'PRISM': lc}, priors=priors, out_folder=red + '/white',
                           linear=['time', 'dy'], ramp=1)
        fit.fit(nthreads=8)
        print(fit.summary())                # includes the depth (Rp/R*)^2 in ppm

Things to know
--------------

* **Scripts need** ``if __name__ == '__main__':``. Ramp fitting uses
  worker processes; without the guard each worker re-runs the whole script.
  Notebooks do not need it.
* **Memory and time.** Stage 1 takes 1--7 minutes per segment and can use
  16--20 GB of memory with four cores. Run large exposures one at a time.
* **Running again.** Finished steps are loaded, not repeated; changed settings
  raise an error unless ``overwrite=True`` is given (see :ref:`overwrite`).
* **Testing on one segment.** ``s1.run(segments=[0])`` processes only the
  first segment. Build the saturation map from the same segment first with
  ``s1.compute_saturation_map(segments=[0])``; by default it is built from
  all segments.
* **Disk space.** A NIRSpec segment needs about 3 GB for its ``rateints``
  and cleaned files. ``s1.run(save_variances=False)``,
  ``ExposureCleaner(...).run(remove_inputs=True)`` and
  ``ex.run(keep_segments=False)`` keep the total small
  (see :ref:`disk-space`).
* **Change any default** with ``set``: ``s1.set('group_bkg',
  method='poly')``, ``ex.set('extract', aperture=6)``. ``show()`` lists
  what can be set.
* **Imaging** (MIRI F1500W) uses :class:`~jstark.extraction.Photometry`
  instead of the spectral extractor, and
  :meth:`LightCurves.from_photometry <jstark.fitting.LightCurves.from_photometry>`
  to read its light curve.
* **Several exposures of one visit** (e.g. a phase curve) are extracted
  together with :class:`~jstark.extraction.VisitExtractor`, so that they
  share one PSF.

Products
--------

Everything is written to the output folder:

====================================  =======================================
File                                  Content
====================================  =======================================
``*_rateints.fits``                   Stage 1 rates, one file per segment
``*_stage1diag.npz``                  Stage 1 diagnostics (masks, 1/f, jumps)
``<exposure>_saturation_map.fits``    first saturated group of every pixel
``<exposure>_wavelength.fits``        wavelength map (Stage 2)
``*_cleaned.fits``                    cleaned cubes, one per segment
``<exposure>_cleaninfo.fits``         masks, median frame, times: the input
                                      of the extraction
``<exposure>_extract_model.fits``     traces, PSF, pixel-response map
``<exposure>_spectra.fits``           spectra of every integration, with
                                      regressors (``dy``, ``ds``, ...)
``<exposure>_*_log.json``             every step and parameter that was run
``<exposure>_products.json``          what each file was made with
                                      (see :ref:`overwrite`)
====================================  =======================================

Next steps
----------

* :doc:`reduction`, :doc:`extraction` and :doc:`fitting` describe every
  stage and its options.
* The :doc:`tutorials <../tutorials/index>` are complete analyses
  (WASP-39 b with NIRSpec/PRISM and NIRCam, WASP-43 b's MIRI/LRS phase
  curve, GJ 3929 b's MIRI eclipses), with the tests behind each default.
