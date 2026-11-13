JWST time-series reduction
==========================

``jstark.reduction`` takes raw JWST time-series data (``*_uncal.fits``) to a
cleaned 3D cube (time x 2D image) ready for spectral extraction or
photometry. It is modular: every step can be run on its own and every
intermediate product can be inspected.

Supported modes: NIRSpec BOTS (PRISM, G395H, G235H, G395M, G235M), NIRISS
SOSS, NIRCam long-wave grism (F444W, F322W2), MIRI LRS (slitless) and MIRI
imaging (F1500W).

Overview
--------

1. :class:`~jstark.reduction.Stage1` -- detector processing, ``uncal`` ->
   ``rateints``. Wraps every ``jwst`` Detector1 step and adds custom steps:

   * ``consistent_saturation`` -- partially saturated pixels use the same
     set of groups in every integration of the exposure (a quick pre-pass
     over all segments builds the map; ``scope='segment'`` avoids the
     pre-pass). Otherwise a pixel that saturates at group 5 in some
     integrations and not in others switches between 4- and 5-group slopes,
     which differ by several per cent.
   * ``custom_bias`` -- scaled CRDS superbias (Eureka!-style), a superbias
     measured from the data (``median(2 g0 - g1)``), or a user bias.
   * ``group_bkg`` -- group-level background / 1/f subtraction:
     column-by-column (NIRSpec, NIRISS), row-by-row and
     amplifier-by-amplifier (NIRCam), with ``'median'``, ``'poly'``, the
     ``'diff'`` (difference-image, exoTEDRF-like) or the ``'readout'``
     (read-time model with a common mode across amplifiers) algorithm. The
     direction and method are only defaults, e.g.
     ``s1.set('group_bkg', axis='column', method='poly')``.
   * ``temporal_jump`` -- jump detection in the time domain: each pixel's
     group differences are compared with a running median over
     integrations. Jumps are either repaired (default for short ramps) or
     flagged for ramp fitting.

2. :class:`~jstark.reduction.Stage2` -- wavelength solution only
   (``assign_wcs``), evaluated on the full subarray (PASTASOSS maps for
   SOSS). Optional ``flat_field`` / ``photom`` etc.

3. :class:`~jstark.reduction.ExposureCleaner` -- cleaning of the whole
   exposure *segment by segment* (segments are never stitched, so memory
   use does not grow with the length of the observation):

   * pass 1 streams through all segments and builds the exposure-wide
     median frame, temporal-scatter image, persistent bad-pixel map (DQ +
     hot / noisy pixels), trace / background masks and the per-pixel noise
     used for cosmic-ray detection;
   * pass 2 cleans each segment (error fixing, time-domain cosmic-ray
     detection with a leave-one-out running median, bad-pixel replacement,
     optional background subtraction), padded with a few integrations from
     the neighbouring segments so that segment boundaries are handled as in
     a stitched cube, and writes ``<root>-segNNN_<det>_cleaned.fits``
     (``SCI``, ``ERR``, ``BADPIX``, ``CRMASK``, ``INT_TIMES``, ``TRACEMASK``);
   * a small ``<exposure>_cleaninfo.fits`` holds the masks, median frame,
     wavelength map, all times and the list of segment files.

   :class:`~jstark.reduction.CubeCleaner` does the same for a single
   in-memory cube (one segment) and is what ``ExposureCleaner`` uses
   internally.

All segments (``-seg001``, ``-seg002``, ...) of an exposure are found and
processed automatically. If a directory contains several exposures (e.g. the
SOSS F277W exposure next to the time series), the one with most
integrations is used unless ``exposure=`` is given.

Quick start
-----------

.. code-block:: python

    from jstark.reduction import Stage1, Stage2, ExposureCleaner, CubeCleaner

    # --- Stage 1 ---------------------------------------------------------
    s1 = Stage1('/data/NRS1', outdir='/data/NRS1/reduction')
    s1.show()                                   # the recipe for this mode
    s1.set('group_bkg', method='poly', poly_order=1)   # change a step
    s1.set('dark_current', skip=False)          # switch a step on
    s1.run()                                    # all segments

    # --- Stage 2 ---------------------------------------------------------
    s2 = Stage2('/data/NRS1/reduction')
    s2.run()                                    # writes *_wavelength.fits

    # --- Cleaning (segment by segment) ------------------------------------
    ec = ExposureCleaner('/data/NRS1/reduction', wavelength=s2.save_wavelength())
    ec.compute_global()                         # pass 1: masks, bad pixels, noise
    info = ec.run(background=True)              # pass 2: clean + save every segment

    # later, e.g. for extraction: one segment at a time
    ec = ExposureCleaner.load(info)
    for i, seg in ec.iter_segments():
        ...                                     # seg is a CubeCleaner (data, err, masks, times)

    # interactive inspection of a single segment
    cc = CubeCleaner.from_files('/data/NRS1/reduction', segments=[0])
    cc.fix_errors(); cc.find_hot_pixels(); cc.find_cosmic_rays(); cc.replace_bad_pixels()

When running from a script (not a notebook), wrap the code in
``if __name__ == '__main__':`` because ramp fitting uses multiprocessing.

Inspecting intermediate steps
-----------------------------

The pipeline can be stopped after any step and resumed from the in-memory
model:

.. code-block:: python

    m = s1.run(segments=[0], stop='refpix')        # returns the RampModel
    # ... look at m.data, m.groupdq ...
    s1.build_trace_mask(m)                         # or s1.set_trace_mask(mask)
    s1.run(segments=[0], start='group_bkg', model=m)

Intermediate ramps can be written with ``s1.run(save_after=['refpix'])``.
Diagnostics of the custom steps (removed 1/f stripes, jump statistics, the
trace and background masks) are stored in ``s1.diagnostics`` and in a
``*_stage1diag.npz`` file per segment. :mod:`jstark.reduction.diagnostics`
provides plotting functions for all of them.

Disk space
----------

Each stage reads the previous stage's files, so intermediate products are
needed until the next stage has run. Three options keep the total small:

* ``s1.run(save_variances=False)``: the ``rateints`` files are written without
  ``VAR_POISSON`` and ``VAR_RNOISE`` (40 per cent smaller; jstark uses only
  ``SCI``, ``ERR`` and ``DQ``).
* ``ExposureCleaner(...).run(remove_inputs=True)``: each ``rateints`` file is
  deleted as soon as it is no longer needed (after the next segment, which
  borrows integrations from it, is cleaned). The info file is rewritten after
  every segment, so the products on disk stay consistent if the run stops.
  Re-cleaning then requires re-running Stage 1.
* ``SpectralExtractor.run(keep_segments=False)`` /
  ``Photometry.run(keep_segments=False)``: the per-segment spectra are deleted
  after they are combined (the combined file holds the same data).

The cleaned cubes (``*_cleaned.fits``) are the input of the extraction; keep
them while extraction settings are explored.

Times
-----

``CubeCleaner.times`` follow the jwst ``INT_TIMES`` convention
(BJD_TDB - 2400000.5); ``CubeCleaner.bjd`` gives full BJD_TDB.

Default recipes
---------------

========================  =============================  ===========================
Mode                      Group-level background / 1/f   Notes
========================  =============================  ===========================
NIRSpec (all gratings)    column median, trace masked    dark skipped
NIRISS SOSS               column, difference-image       dark applied
NIRCam F444W / F322W2     row, readout-time model        side reference pixels used
MIRI LRS / imaging        none                           emicorr, firstframe,
                                                         lastframe, reset, rscd, dark
========================  =============================  ===========================

Jump detection is ``temporal_jump`` (window 11 integrations, 10 sigma) for
all modes; the standard ``jump`` step can be enabled instead with
``s1.set('jump', skip=False); s1.set('temporal_jump', skip=True)``.
