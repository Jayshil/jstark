Light-curve fitting
===================

``jstark.fitting`` fits light curves with `juliet
<https://juliet.readthedocs.io>`_ (it needs the ``fitting`` extra: juliet,
corner). It takes the products of :mod:`jstark.extraction` (or any arrays)
and fits one or several *datasets* jointly: several visits of a planet, both
NIRSpec detectors, or observations with different instruments.

Broadband light curves (the white light of spectra, or a photometric light
curve) are fitted with :class:`~jstark.fitting.BroadbandFit`; spectroscopic
light curves (one fit per wavelength bin, all visits jointly) with
:class:`~jstark.fitting.SpectroscopicFit`, normally with the orbit fixed
from the broadband fit.

Datasets
--------

A :class:`~jstark.fitting.LightCurves` object holds one dataset: one visit
observed with one mode, order or detector. It contains the times
(BJD_TDB), a flux and an error per integration and channel, and a pool of
named *regressors* (per-integration time series)::

    from jstark.fitting import LightCurves

    # spectra (one order): channels = columns; regressors dy, ds, dx, chi2, ...
    lc = LightCurves.from_spectra('<exposure>_spectra.fits', order=1)
    # photometry (one channel); regressors xc, yc, width, bkg (+ PLD with pld=n)
    ph = LightCurves.from_photometry('<exposure>_photometry.fits', method='psf')
    # anything else
    lc = LightCurves(times, flux2d, err2d, wavelength=wave, regressors={'roll': roll})

Useful methods (each returns a new object):

* ``white_light()``: the **sum** of the channels and its error (channels
  that are NaN in most integrations, e.g. saturated or contaminated
  columns, are left out). For a photometric dataset it is the light curve
  itself.
* ``select_channels(wave_range=(2, 5.35))``: keep a wavelength range.
* ``bin(nbins=...)`` or ``bin(edges=...)``: sum channels into wavelength
  bins.
* ``exclude(ints=[(0, 800), (20750, 20766)])``: remove integrations by
  number (e.g. the start of a visit, a high-gain-antenna move), or by
  ``time_range``.
* ``clip(nsigma=5, window=31)``: remove outliers against a running median
  of the white light.
* ``add_regressor(name, values)``: any array ``(nint,)`` or ``(nint, k)``;
  ``per_channel=True`` for an array ``(nint, nchan)`` with one column per
  channel (``bin`` averages it over the summed channels, weighted by their
  flux).
* ``split()``: one dataset per channel group (NIRCam: per readout
  amplifier, ``A1`` ... ``A4``).

Built-in regressors: ``time`` (days from the middle of the dataset),
``time2``, ``time3`` and ``phase4`` (three indicator columns for
integrations 1, 2 and 3 modulo 4: the NIRCam fast-frame-reset pattern).

Several datasets are passed as a dict; the keys are juliet instrument names
and must be letters and digits only (``'V1'``, ``'V4a'``, ``'NRS2'``; juliet
splits names on ``_`` and ``-``)::

    lcs = {'V1': ph1, 'V2': ph2, 'V3': ph3}

Plain dicts of arrays are accepted too: ``{'V1': {'times': t, 'flux': f,
'err': e, 'regressors': {...}}}``.

Linear and GP regressors
------------------------

The datasets only *store* regressors. The fit decides which enter the
linear model and which is the GP input, per dataset::

    fit = BroadbandFit(lcs, priors=pri, out_folder='eclipse_fit',
                       linear={'V1': ['time', 'xc', 'yc'], 'V2': ['time', 'xc']},
                       gp={'V3': 'time'}, gp_kernel='matern')

A list (or a single name) applies to every dataset. The same quantity can
be in both (e.g. a linear slope in time plus a GP in time).

* **Linear**: the chosen arrays form juliet's linear-regressor matrix, one
  coefficient ``theta<i>_<dataset>`` per column. They are standardised
  (mean 0, standard deviation 1), so a coefficient is the relative flux
  change for a one-sigma change of the regressor; the means and standard
  deviations are saved in ``jstark_fit.json``.
* **GP**: one name gives a 1D input (celerite kernels), a list a
  multi-dimensional one (george). juliet picks the kernel from the names of
  the GP priors; ``gp_kernel`` (``'matern'``, ``'exp'``, ``'matern-exp'``,
  ``'sho'``) only sets the default priors added when you give none. A 1D
  input that is not sorted (e.g. a centroid) is handled by ordering the
  dataset by it, as celerite requires.
* **Ramp**: ``ramp={'V1': 1}`` multiplies the model of a dataset by
  ``1 + A exp(-t / tau)`` (``ramp=2``: two terms), with ``t`` the time since
  its first integration; for start-of-visit settling (MIRI) when the start
  is not trimmed.

Priors
------

Give the astrophysical priors in juliet naming, as ``(distribution,
hyperparameters)`` pairs, juliet dicts, or a juliet prior file. Parameters
without a dataset suffix are shared by all datasets (``fp_p1``); a suffix
makes them specific (``p_p1_V1``) or shared by a subset (``q1_V1_V2``).

For every dataset, jstark adds what you did not give:

=========================  ==========================================
``mdilution_<ds>``         fixed to 1
``mflux_<ds>``             normal(0, 0.1)
``sigma_w_<ds>``           log-uniform(0.1, 1e5) ppm
``theta<i>_<ds>``          uniform(-1, 1), one per linear column
``GP_*_<ds>``              broad priors for the chosen kernel
``ramp[2]amp/tau_<ds>``    uniform(-0.1, 0.1); log-uniform(2 cadences, duration)
=========================  ==========================================

Change these defaults with ``nuisance={'sigma_w': ('loguniform', [1, 1e4]),
'theta_V1': ...}`` or by giving the prior yourself.

**Limb darkening is never added.** In juliet a dataset gets a transit model
only if a ``q1``/``u1``/``c1`` prior names it. Eclipse-only data need none;
for transits give e.g. ``q1_V1_V2`` / ``q2_V1_V2`` (shared) or one pair per
dataset. ``BroadbandFit`` prints the model of each dataset and warns when an
``Rp/R*`` prior applies to a dataset without limb-darkening priors.

Running the fit
---------------

::

    fit.fit()                                     # dynamic dynesty
    fit.fit(sampler='dynesty', nthreads=8, n_live_points=1000, dlogz=0.1)
    fit.fit(light_travel_delay=True, stellar_radius=0.32)   # eclipses

The default sampler is ``dynamic_dynesty``. Keyword arguments go to
``juliet.load.fit``: juliet's own options and any keyword of the sampler
(juliet forwards the ones the sampler accepts).

Eclipse timing: with ``light_travel_delay=True`` juliet computes the eclipse
time from ``t0``, ``ecc`` and ``omega`` (plus the light-travel time). To fit
the eclipse time instead, give a ``t_secondary_p1`` prior and leave
``light_travel_delay`` off; one ``t_secondary`` serves all visits (the model
is periodic in ``P``).

A free eclipse time on low signal-to-noise eclipses can give a multimodal
posterior (each visit's red noise prefers its own time), and the sampler
with default settings may settle in one mode, even one with a negative
depth. Check the ``t_secondary`` posterior; if it is multimodal, use more
live points (``n_live_points=...``) or fix the time from an independent
constraint (e.g. radial velocities, or a published joint fit), and compare
fixed-time fits by their evidence (``lnZ`` in juliet's results).

juliet writes its files into ``out_folder`` and never overwrites them: run
again with the same inputs and the finished fit is loaded. jstark saves a
fingerprint of the data, regressors and priors in ``jstark_fit.json`` and
refuses to reuse a folder that holds a *different* fit, unless
``overwrite=True`` (which deletes the old juliet files).

Results
-------

::

    fit.summary()          # table: fitted (median, +, -), fixed, derived
    fit.diagnostics()      # per dataset: p2p of residuals, errors, sigma_w, beta
    fit.residuals('V1')    # data, model, residuals
    p = fit.plots()        # julietPlots
    p.full_model_lc(); p.phase_folded_lc(window=2.); p.plot_gp(); p.plot_corner()
    pri_ch = fit.channel_priors()          # orbit fixed to the posterior medians
    again = BroadbandFit.load('eclipse_fit')   # later, from the folder

``summary`` adds the transit depth ``(Rp/R*)^2`` and eclipse depth in ppm,
``a/R*`` from ``rho`` and the inclination from ``b``; it is saved as
``jstark_summary.ecsv``. ``diagnostics`` gives the point-to-point scatter of
the residuals against the propagated errors (alone and with the fitted
jitter) and the red-noise factor beta for several bin sizes.

Example: several eclipses observed with MIRI
--------------------------------------------

.. code-block:: python

    from jstark.fitting import LightCurves, BroadbandFit

    files = {'V1': '.../V1/reduction/..._photometry.fits', 'V2': ..., 'V3': ...}
    lcs = {k: LightCurves.from_photometry(f).exclude(ints=[(0, 100)]) for k, f in files.items()}
    pri = {'P_p1': ('fixed', P), 't0_p1': ('normal', [T0, sT0]),
           't_secondary_p1': ('uniform', [Tsec - 0.05, Tsec + 0.05]),
           'p_p1': ('fixed', 0.0318), 'a_p1': ('normal', [17.04, 0.5]),
           'b_p1': ('truncatednormal', [0.21, 0.12, 0., 1.]),
           'ecc_p1': ('fixed', 0.), 'omega_p1': ('fixed', 90.),
           'fp_p1': ('uniform', [-1e-3, 1e-3])}
    fit = BroadbandFit(lcs, priors=pri, out_folder='gj3929_eclipses',
                       linear=['time', 'xc', 'yc'], ramp=1)
    fit.fit(nthreads=8)

The ``Rp/R*`` prior without limb darkening triggers the warning above; that
is intended here (no transit in eclipse data).

Spectroscopic fits
------------------

:class:`~jstark.fitting.SpectroscopicFit` puts all datasets on one
wavelength grid and runs one juliet fit per bin (a ``BroadbandFit`` of the
bin's light curves). In each bin it sums the channels of every dataset that
covers the bin, and fits those datasets jointly: planet parameters are
shared as the priors say (one depth per bin unless you give per-dataset
ones), and each dataset keeps its own offset, jitter and systematics. So
several visits of one mode are fitted jointly in each bin, and NRS1/NRS2 or
two SOSS orders fall into their own bins by wavelength.

.. code-block:: python

    from jstark.fitting import BroadbandFit, SpectroscopicFit

    bb = BroadbandFit(lcs, priors=pri, out_folder='white', linear=['time', 'dy'], ramp=1)
    bb.fit(nthreads=8)

    # orbit (and, here, the ramp timescale) fixed to the white-light medians
    pri_ch = bb.channel_priors(fix=('P', 't0', 'a', 'b', 'ecc', 'omega', 'ramptau'))
    pri_ch.update({'p_p1': ('uniform', [0.05, 0.25]),
                   'q1_V1_V2': ('uniform', [0, 1]), 'q2_V1_V2': ('uniform', [0, 1])})
    sf = SpectroscopicFit(lcs, priors=pri_ch, out_folder='spectrum', nbins=60,
                          linear=['time', 'dy'], ramp=1)
    spec = sf.run(nproc=12)          # in a script: under if __name__ == '__main__':
    spec.plot(white=bb)

``channel_priors(fix=...)`` takes name prefixes that end at an underscore:
``'GP'`` carries every GP hyperparameter, ``'GP_rho'`` only the Matern
timescale, ``'ramptau'`` the ramp timescale, ``'GP_rho_V1'`` one parameter
of one dataset. Fixing the GP timescale from the white light and leaving
the amplitude free in each bin (added automatically when you give none) is
the usual way to model red noise shared by all wavelengths:

.. code-block:: python

    bb = BroadbandFit(lcs, priors=pri, out_folder='white', gp='time')
    bb.fit(nthreads=8)
    pri_ch = bb.channel_priors(fix=('P', 't0', 'a', 'b', 'ecc', 'omega', 'GP_rho'))
    pri_ch['p_p1'] = ('uniform', [0.05, 0.25])
    sf = SpectroscopicFit(lcs, priors=pri_ch, out_folder='spectrum', nbins=60, gp='time')

With ``mode='normal'`` the white-light posteriors become normal priors
instead of fixed values. ``planet='p1'`` restricts the planet parameters to
one planet; dataset parameters (GP, ramp, ...) are kept.

Common-mode correction (off by default). The common mode of a dataset is
its white light divided by the white-light planet model
(:meth:`~jstark.fitting.BroadbandFit.common_mode`): the systematics shared
by all wavelengths, plus the white-light noise. It is matched to the
spectroscopic integrations by time; integrations the white-light fit did
not use are dropped.

.. code-block:: python

    sf = SpectroscopicFit(lcs, priors=pri_ch, out_folder='spectrum_cm', nbins=60,
                          common_mode=bb, cm_mode='divide')      # or 'regressor'

* ``cm_mode='divide'``: every channel is divided by the common mode before
  binning (the usual approach).
* ``cm_mode='regressor'``: the common mode is the linear regressor ``cm``
  of every bin, with a free coefficient per bin, so the correction may
  scale with wavelength.
* ``common_mode`` may also be ``{name: BroadbandFit or (times, cm)}`` per
  dataset; datasets without one are fitted as they are.

**Per amplifier (NIRCam).** The 1/f noise left after Stage 1 is mostly
shared by the columns of one readout amplifier (512 columns, ~0.5 µm of
the F322W2 or F444W spectrum) and differs between amplifiers, so one
white-light common mode cannot remove it. NIRCam spectra record the
amplifier of each column (:attr:`LightCurves.groups`);
:meth:`~jstark.fitting.LightCurves.split` gives one dataset per amplifier
(``A1`` ... ``A4``). Fit their white lights jointly, each with its own depth
and limb darkening (a common planet model would leave the difference in
depth between amplifiers in the common mode and shift whole amplifiers in
the spectrum), and pass that fit as the common mode: every channel is then
corrected by the common mode of its own amplifier, and bins may straddle
amplifier boundaries.

.. code-block:: python

    lc = LightCurves.from_spectra('<exposure>_nrcalong_spectra.fits', name='NRC')
    amps = lc.split()                                   # {'A1': ..., 'A4': ...}
    pri_amp = dict(orbit, **{f'p_p1_{a}': ('uniform', [0.05, 0.25]) for a in amps},
                   **{f'{q}_{a}': ('uniform', [0, 1]) for a in amps for q in ('q1', 'q2')})
    bb = BroadbandFit(amps, priors=pri_amp, out_folder='white_amps', linear=['time', 'phase4'])
    bb.fit(nthreads=8)
    sf = SpectroscopicFit({'NRC': lc}, priors=pri_ch, out_folder='spectrum_cm', edges=edges,
                          linear=['time', 'phase4'], common_mode=bb, cm_mode='divide')

The level of each amplifier's part of the spectrum then comes from that
amplifier's white-light depth (the first consequence below, per amplifier).
For several visits name the amplifier datasets ``<visit>A1`` ...
(``lc.split(prefix='V1A')``). :meth:`BroadbandFit.channel_common_mode`
returns the per-channel common mode ``(times, (nint, nchan))`` directly.

Two consequences to keep in mind:

* The spectrum keeps its shape but takes its level from the white light.
  An error in the white-light depth (e.g. from systematics the white-light
  model does not describe) shifts every bin by the same amount.
* Each bin is part of the white light, so dividing by the common mode also
  removes a share of the bin's own noise (about the bin's fraction of the
  white light). Its residuals then look slightly too quiet. This is
  negligible for many bins (1% of the white light per bin for 100 bins),
  but not for a handful of wide bins.

When it helps: the depths hardly change (the bin's share of the
white-light transit is replaced by the fitted one, which agrees with it);
what changes is the noise left in the bins. On WASP-39 b (PRISM, SOSS and
NIRCam at R = 100) depths moved by a few ppm and errors by under 2%: the
shared noise is small next to each bin's own. With a few wide bins on
NIRCam, the residuals without a common mode were red (beta ~2.2, errors
too small) and the common mode made them white (errors then ~25% smaller,
but relative to the white-light level). Check ``sf.diagnostics()`` (beta)
without it first.

Bins:

* ``edges``: 1D contiguous edges, or an ``(nbin, 2)`` array of lower and
  upper edges (gaps allowed, e.g. the bins of a published spectrum);
* ``nbins``: bins with equal numbers of good channels over all datasets;
* neither: the native channels (datasets must share one channel grid).

A dataset enters a bin when its good channels (not NaN in most
integrations) cover at least ``min_cover`` (0.5) of the bin width, so a
detector edge or a run of saturated columns does not add a half-covered
light curve to a bin.

Priors are a dict used for every bin, or a function ``priors(info)``
returning the dict of one bin (``info``: ``index``, ``wave``, ``lo``,
``hi``, ``datasets``), e.g. for limb-darkening coefficients from a model.
Names of datasets that do not cover a bin are removed from the parameter
names (``q1_V1_V2`` becomes ``q1_V1``). ``linear``, ``gp``, ``ramp`` and
the other options are those of ``BroadbandFit``.

``run(nproc=..., sampler=..., **fit_kwargs)`` fits the bins in ``nproc``
processes (juliet single-threaded in each; dynesty's progress output off
unless ``print_progress=True``), and skips bins already finished with the
same inputs: an interrupted run continues where it stopped. Bins whose
folder holds a different fit stop the run before anything is fitted, unless
``overwrite=True``. Bin ``i`` lives in ``<out_folder>/bin<i>`` (a complete
``BroadbandFit`` folder: ``sf.bin_result(i)`` reloads it).

Results:

::

    spec = sf.spectrum()               # Spectrum table, also spectrum.ecsv
    spec['wave'], spec['depth_p1_ppm'], spec['depth_p1_ppm_errp'], spec['depth_p1_ppm_errm']
    sf.diagnostics()                   # per bin and dataset: p2p, err, sigma_w, beta
    sf.plot_spectrum(white=bb)
    sf.plot_2d('V1')                   # detrended data, planet model, residuals vs time
    sf.plot_noise()                    # rms vs bin size; p2p / error vs wavelength
    sf = SpectroscopicFit.load('spectrum')   # later, from the folder alone

The ``Spectrum`` table has one row per finished bin, with every fitted or
derived planet and limb-darkening parameter ``x`` as ``x``, ``x_errp`` and
``x_errm``; ``Spectrum.read('spectrum/spectrum.ecsv')`` reads it back.

API
---

.. automodapi:: jstark.fitting
   :no-inheritance-diagram:
