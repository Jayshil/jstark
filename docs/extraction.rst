Spectral extraction and photometry
==================================

``jstark.extraction`` turns the cleaned cubes written by
:class:`~jstark.reduction.ExposureCleaner` into stellar spectra (one per
integration and spectral order) or, for imaging, into light curves. Like the
reduction, it works **segment by segment**: an exposure is never stitched
into one cube, so memory use does not depend on the length of the
observation.

The spectral extraction is a port of ``stark`` (A. Brandeker & J. A. Patel,
https://github.com/Jayshil/stark): the spatial profile (PSF) of the trace is
modelled with splines and the spectrum is extracted optimally with it.

Overview
--------

:class:`~jstark.extraction.SpectralExtractor` runs four steps per order:

1. **Trace** (:meth:`~jstark.extraction.SpectralExtractor.find_trace`): the
   static trace ``y0(x)`` is measured on the exposure median frame (iterative
   centre of flux, smoothed with a cubic spline). The column range is chosen
   automatically: the largest contiguous run of columns with a valid
   wavelength, enough good pixels and a per-integration S/N above
   ``snr_min``. Override with ``columns=(x0, x1)`` or
   ``wavelength_range=(lam0, lam1)``; MIRI/LRS defaults to
   ``wavelength_range=(5, 12)`` (below ~4.5 micron the prism dispersion folds
   back, several wavelengths falling on the same pixels, and 5-6 micron
   receives some shorter-wavelength light; beyond ~12 micron the throughput
   is low), ``ex.set('columns', wavelength_range=None)`` restores the
   automatic choice. For NIRISS/SOSS the reference traces of
   the Stage 2 wavelength file are shifted to the data with a fitted
   correction measured only in columns where the orders are well separated.
   Every integration gets an offset ``dy(t)``: first from the centre of flux,
   then refined by fitting the PSF model (Gauss-Newton).

2. **Background**: for every integration and spectral pixel, a clipped
   median (or a polynomial) of the pixels at least ``inner`` (and at most
   ``outer``) pixels away from every trace. Its uncertainty is propagated.

3. **PSF** (:meth:`~jstark.extraction.SpectralExtractor.fit_psf`): a cubic
   tensor-product spline ``S(u, x)`` of the offset from the trace ``u`` and the
   column ``x`` (``knot_col=0`` gives stark's column-independent 1D profile),
   fitted to pixel values normalised by a noise-free flux estimate (median
   spectrum x white-light factor).

   * **All integrations** are used by default: the least-squares sums are
     accumulated chunk by chunk, so the result equals a fit to the whole
     stitched exposure while only one chunk is in memory.
   * **Smoothness penalties** (P-splines) along ``u`` and ``x`` suppress the
     oscillations that fine knot spacings produce where the data do not
     constrain the profile. Their strengths are chosen by
     **cross-validation between sub-pixel phases**: the model fitted to the
     pixels whose offset from the trace has a fractional part below one half
     must predict those above, and vice versa. This tests exactly how well
     the profile interpolates between sampled phases, which is what matters
     when the trace moves (a wavy profile couples the flux to the pointing),
     without penalising real changes of the profile along the dispersion.
   * A **pixel-response map** ``Q`` (one factor per detector pixel; the model
     is ``f P Q``) absorbs pixel-to-pixel sensitivity differences: the data
     are not flat-fielded, and on NIRSpec/G395H these differences are ~2 per
     cent, several times the photon noise of a bright pixel, which no smooth
     profile can describe. Only the pixel-scale part of ``Q`` is kept (its
     local average is divided out), so the profile, which follows the trace
     motion, keeps all smooth structure. Each factor is shrunk towards 1
     according to its own precision (pixels far in the wings, with little
     light, would otherwise get noisy factors).

4. **Extraction** (:meth:`~jstark.extraction.SpectralExtractor.run`):

   * ``method='optimal'`` (default): ``f = sum(e P D / V) / sum(e P^2 / V)``
     with aperture weights ``e`` (fractional at the aperture edges, so the
     estimate changes smoothly when the trace moves), masked pixels left
     out, the time-averaged variance of each pixel as ``V`` (no bias from
     weighting by each pixel's own noisy error), and the correct variance
     ``sum(e^2 P^2 / V) / sum(e P^2 / V)^2``. Remaining outliers are rejected
     one at a time per column, *temporally*: a pixel is rejected only if its
     residual deviates from its own median residual over the chunk, so a
     static model misfit never makes pixels switch between kept and rejected
     (``reject_mode='model'`` gives Horne's original test).
     ``fit_background=True`` fits a constant per column together with the
     flux (it costs precision in every mode tested).
   * Columns of an integration with more than ``max_maskfrac`` (default
     0.98) of the profile on masked pixels are set to NaN: too little light
     is left to measure (e.g. under a masked zeroth order covering the whole
     trace). Saturated cores of bright NIRSpec traces reach ~0.95 and are
     still measured well from the wings.
   * ``method='box'``: aperture sum, masked pixels filled with the model.
     The box spectra are always saved as well.
   * ``method='covariance'`` (NIRCam default): generalised least squares per
     column with ``C = diag(V) + cov_scale * K``, ``K`` being the covariance
     of row offsets (1/f) per amplifier estimated from the residuals of the
     PSF fit (:meth:`~jstark.extraction.SpectralExtractor.estimate_covariance`).
     A row offset is shared by all columns of an amplifier, so in a sum of
     ``B`` columns (a spectral bin, the white light) it adds up coherently
     while the pixel noise adds in quadrature; the weights that minimise the
     noise of the sum use ``C = diag(V) + B K``. Per column (``cov_scale=1``)
     ``K`` is negligible and the result equals optimal extraction. The NIRCam
     default ``cov_scale=500`` (the columns of one amplifier) down-weights
     the faint wings and lowered the white-light scatter by ~10% on the
     F444W / F322W2 test data. The saved variance is that of each column
     for these weights.
   * ``method='joint'`` (NIRISS/SOSS default): the orders are extracted
     together. Each order is extracted from the data minus the current models
     of the other orders, alternating until convergence; the PSF of each
     order is fitted on decontaminated data in the same way. The
     contamination of each order by the others inside its aperture is saved
     (``CONTAM_O<n>``) for dilution corrections.

Quick start
-----------

.. code-block:: python

    from jstark.extraction import SpectralExtractor

    ex = SpectralExtractor('/data/NRS1/reduction/<exposure>_cleaninfo.fits')
    ex.show()                                    # parameters of every step
    ex.set('psf', oversample=2, knot_col=20)     # any parameter of any step
    ex.find_trace()
    ex.fit_psf()                                 # one pass over all integrations
    ex.optimise(apertures=[3, 4, 5, 6], bkg_inner=[6, 8, 10])
    spec = ex.run()                              # all segments, then combined

    wl, wl_err = spec.white_light()              # normalised white light
    wave, lc, lc_err = spec.bin(nbins=30)        # spectroscopic light curves

``run()`` writes ``<root>-segNNN_<det>_spec.fits`` per segment
(``run(keep_segments=False)`` deletes them after combining) and
``<exposure>_spectra.fits`` with, per order ``n``: ``WAVE_On`` (wavelength of
each column, micron), ``FLUX_On``, ``VAR_On`` (a variance),
``BOX_On``, ``BOXVAR_On``, ``MASKFRAC_On`` (fraction of the profile in the
aperture on masked pixels, e.g. saturated cores), ``BKG_On``, ``TRACE_On``
(columns, trace, wavelength), the PSF model, and ``REG_On``: per-integration
regressors ``dy`` (trace offset), ``ds`` (relative PSF width change),
``dx`` (shift along the dispersion), ``bkg``, ``maskfrac``, ``white``.
``FIELD_On`` holds the static flux of modelled field-star traces in the
aperture (below). ``<exposure>_extract_model.fits`` holds the traces, PSFs,
pixel-response map, contaminant mask, field-star model and parameters, so a
run can be resumed with
:meth:`~jstark.extraction.SpectralExtractor.load_model`.

For quick tests, ``ex.set('psf', ints=300)`` fits the PSF on 300 integrations
spread over the exposure and ``ex.run(segments=[0])`` extracts one segment.

White light from the saved spectra
----------------------------------

Sum the columns (``Spectra.white_light()`` does this), or average the
spectra *normalised per column* with weights that do not change with time.
An inverse-variance weighted mean of the raw fluxes is dominated by the
faintest columns (weights ~ 1/flux) and picks up the small changes of the
per-integration variances; on G395H it gives ~900 ppm instead of 175.

Field-star contamination (NIRISS/SOSS)
--------------------------------------

Stars in the field add light to slitless data: compact **zeroth orders** and,
sometimes, **spectral traces** crossing or running along the target's. Both
are static and only dilute the transit or eclipse.

.. code-block:: python

    # F277W exposure taken with the SOSS time series: Stage 1 only
    s1 = Stage1('/data/NIRISS/<f277w>_uncal.fits', outdir='/data/NIRISS/f277w')
    f277w = s1.run(save_variances=False)[0]

    ex = SpectralExtractor(info)
    ex.find_trace()
    ex.find_contaminants(f277w)     # zeroth orders -> masked
    ex.add_field_trace(order=1, dx=-120, dy=35)  # a field star's trace (if any)
    spec = ex.run()

:meth:`~jstark.extraction.SpectralExtractor.find_contaminants` finds compact
sources in a detection image and masks them: the pixels are treated as bad
(excluded from the background, the PSF fit and the pixel-response map, and
filled with the profile in the extraction); columns where the mask covers
(almost) the whole profile become NaN (``max_maskfrac``). The F277W exposure
is the right detection image: its filter blocks most of the target's light,
so zeroth orders *on* the target's traces become visible (the target's own
F277W light, the red end of order 1, is excluded automatically). Without it
(``image=None``) only contaminants away from the traces can be found: in the
time series a contaminant on a trace is absorbed by the target's profile and
flux. ``set_contaminant_mask`` adds a mask by hand.

:meth:`~jstark.extraction.SpectralExtractor.add_field_trace` models the trace
of a field star as the target's trace of the same order shifted by
``(dx, dy)`` (positions from the F277W image or a catalogue; ``dy`` is
refined), with the target's PSF and a free static spectrum. It is fitted to
the median frame in turn with the target's PSF, and subtracted from every
integration; the target's PSF and pixel-response map are not fitted where the
field star dominates. ``FIELD_On`` gives the field-star flux left in each
aperture (subtracted) for dilution checks.

``diagnostics.plot_contaminants(ex, image)`` shows the mask and the traces.

Choosing the aperture and background region
-------------------------------------------

:meth:`~jstark.extraction.SpectralExtractor.optimise` extracts blocks of
consecutive integrations spread over the exposure (``nblocks`` x
``block_len``; ``segments='all'`` uses everything) for every combination of
aperture half-width and background inner radius, keeping the PSF. The metric
is the robust point-to-point scatter of the white light, each block
normalised by its own median: consecutive differences remove the flux level
(so in- and out-of-transit blocks can be combined) and the median ignores
ingress and egress. It reports the statistical uncertainty of the metric;
settings within it are equivalent.

Imaging: :class:`~jstark.extraction.Photometry`
-----------------------------------------------

.. code-block:: python

    from jstark.extraction import Photometry

    ph = Photometry('/data/F1500W/reduction/<exposure>_cleaninfo.fits')
    ph.set('background', r_in=15, r_out=25)
    ph.fit_psf()                                  # empirical PSF from all integrations
    ph.optimise(radii=[3, 4, 5, 6], psf_radii=[6, 8, 10])
    res = ph.run()                                # -> <exposure>_photometry.fits
    f_psf, e_psf = res.flux['psf'], res.err['psf']
    f_ap, e_ap = res.flux['aperture'], res.err['aperture']
    phat, pcs = res.pld_basis(npc=5)              # PLD regressors

For every integration: sky from a clipped-median annulus (or a plane),
centroid (centre of flux refined by a PSF fit), exact-aperture photometry and
PSF-weighted (optimal) photometry with an empirical PSF, the time-averaged
normalised image shifted to each centroid with a bicubic spline. PSF
weighting helps most when the background dominates the noise (MIRI at 15
micron). The PLD regressors should be fitted together with the eclipse
model.

:class:`~jstark.extraction.ApPhoto` is the in-memory aperture photometry and
PLD helper (moved from ``jstark.jstark``; ``CubeCleaner.to_apphoto()``
creates one).

Diagnostics
-----------

:mod:`jstark.extraction.diagnostics`: ``plot_trace``, ``plot_psf``,
``plot_psf_width``, ``plot_residuals``, ``plot_optimisation``,
``plot_white_light``, ``plot_regressors``, ``plot_spectra``,
``plot_photometry``, ``plot_contaminants``.

API
---

.. automodapi:: jstark.extraction
   :no-inheritance-diagram:
