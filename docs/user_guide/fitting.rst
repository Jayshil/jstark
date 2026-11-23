Light-curve fitting
===================

``jstark.fitting`` fits light curves with `juliet
<https://juliet.readthedocs.io>`_ (it needs the ``fitting`` extra: juliet,
corner, radvel). It takes the products of :mod:`jstark.extraction` (or any arrays)
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
    # any other observatory, from a table (see below)
    ts = LightCurves.from_table('TESS51.dat', 0, 1, 2)
    # anything else
    lc = LightCurves(times, flux2d, err2d, wavelength=wave, regressors={'roll': roll})

``white_light()`` returns the **sum** of the channels and its error as
arrays (channels that are NaN in most integrations, e.g. saturated or
contaminated columns, are left out; for a photometric dataset it is the
light curve itself). :class:`~jstark.fitting.BroadbandFit` computes it
itself, so pass the dataset, not its white light. Useful methods that return
a new dataset:

* ``select_channels(wave_range=(2, 5.35))``: keep a wavelength range.
* ``bin(nbins=...)`` or ``bin(edges=...)``: sum channels into wavelength
  bins.
* ``exclude(ints=[(0, 800), (20750, 20766)])``: remove integrations by
  number (e.g. the start of a visit, a high-gain-antenna move), or by
  ``time_range``.
* ``clip(nsigma=5, window=31)``: remove outliers against a running median
  of the white light. The window (in integrations) must be short compared
  with the ingress: at long cadences it is not, and the transit itself is
  removed (there is a warning); use ``protect=(t0, per, half_width)``.
* ``around(t0, per, half_width)``: keep only the data near the transits (or
  eclipses) of a long time series.
* ``split()``: a dict with one dataset per channel group (NIRCam: per readout
  amplifier, ``A1`` ... ``A4``).

``add_regressor(name, values)`` adds a regressor to the dataset in place:
any array ``(nint,)`` or ``(nint, k)``, or with ``per_channel=True`` an
array ``(nint, nchan)`` with one column per channel (``bin`` averages it
over the summed channels, weighted by their flux).

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
* **Non-linear**: your own function of regressors and parameters, per
  dataset (next section).
* **Ramp**: ``ramp={'V1': 1}`` multiplies the model of a dataset by
  ``1 + A exp(-t / tau)`` (``ramp=2``: two terms), with ``t`` the time since
  its first integration; for start-of-visit settling (MIRI) when the start
  is not trimmed.

Non-linear models
-----------------

::

    def roll_model(x, parameters):           # at the top level of the script
        phi = np.radians(x) + parameters['phase_CHEOPS']
        return 1. + parameters['amp_CHEOPS'] * np.sin(phi)

    pri.update({'amp_CHEOPS': ('uniform', [0., 0.01]), 'phase_CHEOPS': ('uniform', [-3.14, 3.14])})
    fit = BroadbandFit(lcs, priors=pri, out_folder='fit',
                       non_linear={'CHEOPS': {'function': roll_model, 'regressor': 'roll',
                                              'multiplicative': True}})

The function gets the regressor (a stored or built-in name, a list of names
for a ``(nint, k)`` array, or an array with one row per integration) and a
dict with the value of every parameter; it returns the model of each
integration. The regressor follows the data when integrations are dropped,
binned or sorted. The model is ``(planet + linear) * f`` with
``multiplicative=True`` and ``planet + linear + f`` otherwise (juliet's
default). Choose parameter names yourself (with a dataset suffix, so they do
not look like juliet parameters) and give their priors; a missing prior is
reported before the fit starts.

* juliet allows one function per dataset: ``ramp`` can be combined with a
  multiplicative function, not with an additive one.
* The source text of the function is part of the fit's fingerprint (changing
  even a comment makes it a different fit); data it reads from global
  variables is not.
* With ``nthreads > 1`` the function must be importable by the worker
  processes: define it in a module, or at the top level of a script.
* ``BroadbandFit.load('fit', non_linear={'CHEOPS': roll_model})``: the
  functions are not stored, so give them again.

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
``Rp/R*`` prior applies to a dataset without limb-darkening priors. It also
warns about priors that name no dataset of the fit (juliet ignores them;
free ones are only unused dimensions).

**Groups** (optional) are short names for several datasets::

    groups={'TESS': ['TESS51', 'TESS91']}

In a prior name a group is one parameter *shared* by its datasets:
``q1_TESS`` is ``q1_TESS51_TESS91``. In the per-dataset options (``linear``,
``gp``, ``ramp``, ``supersample``, ``ld_laws``, ``non_linear``, ``nuisance``)
the setting goes to each of the datasets; a dataset's own entry takes
precedence. Full names always work, alone or mixed with groups, and give the
same fit. Results accept group names too (``fit.values('p_p1_TESS')``,
``phase_folded_lc(instruments='TESS')``).

Joint fits with other observatories
-----------------------------------

Photometry of other observatories (TESS, CHEOPS, ground-based, ...) is
fitted like any other dataset. You prepare the data; ``from_table`` reads
them from a file (text, CSV, FITS, ...), an astropy or pandas table, or a
dict of columns, with any columns as regressors. Columns are given by name
or number::

    from astropy.table import Table
    tab = Table.read('tess2022112184951-s0051-..._lc.fits', hdu=1)
    tab = tab[tab['QUALITY'] == 0]                       # quality cuts are yours
    tess51 = LightCurves.from_table(tab, 'TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR',
                                    time_offset=2457000.)    # BTJD -> BJD
    tess51 = tess51.around(T0, P, 0.35)                  # days around the transits
    cheops = LightCurves.from_table('visit.fits', 'BJD_TIME', 'FLUX', 'FLUXERR',
                                    regressors={'roll': 'ROLL', 'bg': 'BG', 'pca': ['U0', 'U1']})

    lcs = {'NRS1': nrs1, 'NRS2': nrs2, 'TESS51': tess51, 'TESS91': tess91, 'CHEOPS': cheops}
    fit = BroadbandFit(lcs, priors=pri, out_folder='joint',
                       groups={'NRS': ['NRS1', 'NRS2'], 'TESS': ['TESS51', 'TESS91']},
                       linear={'NRS': ['time', 'dy'], 'CHEOPS': ['bg', 'pca']},
                       gp={'TESS': 'time', 'CHEOPS': 'roll'})

* **Times must be BJD_TDB in all datasets.** Times that are not full Julian
  dates raise an error; whether they are TDB cannot be checked (UTC differs
  by 69 s). TESS ``TIME`` is BJD_TDB - 2457000. The ``BJD_TIME`` of CHEOPS
  DRP and PIPE files is BJD_TDB (``MJD_TIME`` is TT and not barycentric).
* **Long exposures**: ``supersample={'TESS': (15, 30 / 1440)}`` evaluates the
  planet model on 15 points per 30-min exposure (exposure time in days;
  without it the median cadence is used). There is a warning from about 10
  min without supersampling.
* **Bandpasses**: give each its own ``p_p1_<dataset or group>`` (or one
  ``p_p1``) and limb darkening. A shared orbit and ``P``, ``t0`` need no suffix.
* **Run time** grows with the number of datasets: each has an offset, a
  jitter and its systematics. Above about 20 parameters dynesty switches to
  slice sampling; ``sample='rwalk'`` is usually faster. Many visits (tens
  of CHEOPS visits) are too many for one nested-sampling fit.
* Independent normal priors from a fit of the other data are not a
  substitute for a joint fit: they lose the correlation of ``a/R*`` and ``b``.
* Plots: use ``phase_folded_lc(instruments='TESS', one_plot=True)`` and so on, one
  bandpass (and one kind of event) at a time.
* ``beta`` in :meth:`~jstark.fitting.BroadbandFit.diagnostics` bins
  consecutive points and ignores gaps, so it is only indicative for data with
  gaps (CHEOPS) or cut around transits.
* ``SpectroscopicFit`` stays a JWST fit:
  ``fit.channel_priors(datasets=['NRS1', 'NRS2'])`` gives the priors of the joint fit that
  apply to the JWST datasets.

Transit timing variations are not supported.

Radial velocities
-----------------

::

    fit = BroadbandFit(lcs, priors=dict(pri, K_p1=('uniform', [0., 200.])), out_folder='joint_rv',
                       rvs={'SOPHIE': (t, rv, err), 'CORALIE': (t2, rv2, err2)})
    fit.fit(nthreads=8)
    fit.rv_diagnostics()          # per instrument: offset, rms, error, jitter, chi2
    fit.rv_residuals('SOPHIE')    # velocities, model, residuals
    fit.plots().plot_rv()         # time series and phase-folded orbit of every planet

Times are BJD_TDB, velocities in m/s (or any unit, then also that of ``K``,
the offsets and the jitters). The orbit uses the ``P``, ``t0`` and
eccentricity priors of the planet; a ``K_p<n>`` prior is required. Missing
offsets ``mu_<instrument>`` and jitters ``sigma_w_<instrument>`` are added (the
latter in the unit of the velocities, not ppm). Names must differ from those
of the datasets.

* Planets without ``p_p<n>`` are fitted to the velocities only. juliet
  requires increasing periods (``P_p1 < P_p2 < ...``).
* ``rv_slope`` (and ``rv_quad``) priors add a linear (quadratic) trend, per
  day (per day squared), about the middle of the RV times (or ``fit(ta=...)``);
  ``rv_intercept`` is fixed to 0, as it is degenerate with the offsets.
* With a free eccentricity, keep it below about 0.9 through the priors or
  ``fit(ecclim=...)``: batman fails near 1. ``summary`` derives ``ecc`` and
  ``omega`` from ``secosomega``/``sesinomega``.
* GPs and linear regressors are not available for the velocities.

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

**Do not fix** ``t_secondary`` **to a value away from** ``t0 + P/2`` while
the orbit is fitted (e.g. free ``a`` or ``b``). batman turns ``t_secondary``
into a conjunction time only when ``t_secondary`` changes between calls,
and juliet writes ``t0`` into the model parameters on every call, so a fixed
``t_secondary`` is ignored and the eclipse lands at the time implied by
``t0``, ``ecc`` and ``omega`` (``t0 + P/2`` for a circular orbit).
:meth:`~jstark.fitting.BroadbandFit.fit` refuses this combination. To impose
an eclipse time measured elsewhere, give it a narrow free prior instead,
e.g. ``'t_secondary_p1': ('normal', [t_sec, sigma])``, which also carries
its uncertainty into the depth.

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
refuses to reuse a folder that holds a *different* fit. ``overwrite=True``
deletes the old files and fits again, also if the fit is the same. The sampler
settings are not part of the fingerprint.

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

The complete analysis is the tutorial :doc:`../tutorials/gj3929b_miri_f1500w`.

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

Phase curves
------------

A complete example is the tutorial :doc:`../tutorials/wasp43b_miri_lrs`.

jstark uses juliet's Cowan & Agol (2008) model. The planet flux relative to
the star is

.. math::

    F(t) = f_p + C_1(\cos\omega t - 1) + D_1\sin\omega t
           + C_2(\cos 2\omega t - 1) + D_2\sin 2\omega t,
    \qquad \omega t = 2\pi (t - t_\mathrm{sec}) / P,

hidden during the eclipse (juliet multiplies it by the normalised eclipse
model) and multiplied by the transit model. ``fp`` is the planet flux at the
eclipse. Give ``fp_p1``, ``C1_p1`` and ``D1_p1`` priors; if you leave out
``C2``/``D2`` they are fixed to 0 (first-order model; juliet always
evaluates the second-order terms). Rules that jstark checks before fitting:

* juliet computes phase curves **only for datasets with a transit model**
  (limb-darkening priors). Without them it would silently fit a plain
  eclipse, so jstark raises an error. If the transit is not covered, give
  fixed ``q1``/``q2``.
* the ``C1`` ... ``D2`` names carry the instrument suffix of the ``fp``
  prior (``fp_p1_LRS`` -> ``C1_p1_LRS``), because juliet reads them that way.
* juliet needs the eclipse time: either give no ``t_secondary`` prior and
  fit with ``light_travel_delay=True, stellar_radius=<R*/Rsun>`` (juliet
  then computes the eclipse time from ``P``, ``t0``, ``ecc``, ``omega`` and
  the light-travel time), or give a ``t_secondary_p1`` prior to fit it.

.. code-block:: python

    lc = LightCurves.from_spectra(spec).select_channels(wave_range=(5., 10.5))
    lc = lc.exclude(ints=[(0, 779)])
    pri = {'P_p1': ('fixed', P), 't0_p1': ('normal', [T0, 0.01]),
           'a_p1': ('uniform', [3., 7.]), 'b_p1': ('uniform', [0., 1.]),
           'p_p1': ('uniform', [0.1, 0.2]), 'ecc_p1': ('fixed', 0.), 'omega_p1': ('fixed', 90.),
           'q1_LRS': ('uniform', [0., 1.]), 'q2_LRS': ('uniform', [0., 1.]),
           't_secondary_p1': ('normal', [T0 + P / 2, 0.007]),
           'fp_p1': ('uniform', [0., 0.012]),
           'C1_p1': ('uniform', [-0.01, 0.01]), 'D1_p1': ('uniform', [-0.01, 0.01])}
    fit = BroadbandFit({'LRS': lc}, priors=pri, out_folder='white_pc',
                       linear=['time', 'dy', 'ds'], ramp=1, clip={'nsigma': 5})
    fit.fit(nthreads=8)
    # or: no 't_secondary_p1' prior and fit.fit(light_travel_delay=True, stellar_radius=0.667)
    fit.summary()            # + fnight, fmax, fmin, pcamp (ppm), pcoffset, pcoffmin (deg)
    fit.phase_curve('_p1')   # posterior samples of the same quantities
    fit.plots().phase_folded_lc()

Derived quantities (``summary``, :func:`jstark.fitting.utils.phase_curve_quantities`):
the nightside flux ``fnight`` (at the transit, ``fp - 2 C1``), the maximum
and minimum planet flux, the peak-to-peak amplitude ``pcamp``, and the
phase offsets of the maximum before the eclipse (``pcoffset``) and of the
minimum before the transit (``pcoffmin``), in degrees, positive = eastward.
In a :class:`SpectroscopicFit` they become spectrum columns
(``fnight_p1_ppm``, ...); ``channel_priors()`` also fixes ``t_secondary`` when it was fitted.

**Time binning (off by default).** ``BroadbandFit(..., bin_time=n)`` (and
``SpectroscopicFit``) averages ``n`` consecutive integrations before
fitting (:meth:`LightCurves.bin_time`; bins never span gaps). Long bins smear
ingress and egress: add ``supersample=k`` to evaluate the planet model on
``k`` sub-exposures per bin. Note that ``sigma_w`` is a jitter per fitted
point, so its value changes with binning.

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
folder holds a different fit stop the run before anything is fitted.
``overwrite=True`` fits all bins again: their old files are deleted at the
start, so if that run is interrupted, continue it with ``overwrite=False``.
Bin ``i`` lives in ``<out_folder>/bin<i>`` (a complete
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
