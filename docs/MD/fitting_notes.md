# jstark light-curve fitting: development notes

As of 2026-11-27. The record of what was discussed, decided, implemented
and measured while building the light-curve fitting stage of jstark
(`jstark/fitting/`), for anyone continuing the work, including future work
sessions. The code and its docstrings remain the reference for details;
`docs/fitting.rst` is the user-facing walk-through. The stages before this
one: `docs/detector_processing_notes.md` and `docs/extraction_notes.md`.

---

## 1. Decisions agreed with the user (2026-11-14)

- **Tool:** juliet (2.2.10 installed; dynesty 3.1). Replace `SpectroscopicLC`
  and `julietPlots` in `jstark/jstark.py`; `jstark.py` and `jstark/utils.py`
  may be deleted (no back-compatibility needed).
- **Multi-dataset fits:** inputs are dicts keyed by dataset (visit /
  instrument / detector); the keys are juliet instrument names.
- **White light = sum of the channels**, never an inverse-variance weighted
  mean (extraction_notes.md §5.12: weights that change in time move the
  mean, and the faintest columns dominate it).
- **Sampler:** `dynamic_dynesty` by default; any juliet or sampler keyword
  passes through `fit(**fit_kwargs)`.
- **Limb darkening is the user's choice:** jstark never adds LD priors
  (eclipse-only data need none).
- **Photometry must work:** imaging datasets have no white light; the light
  curve itself is the broadband curve.
- **Linear vs GP regressors:** datasets store a pool of named regressors;
  the fit assigns roles with `linear=` and `gp=` (per dataset or for all).
  The same quantity may be in both (e.g. a slope and a GP in time).
- **GJ 3929 (multi-visit test data):** V4 holds two exposures, obs 004 and
  obs 005 (starts 16 min after 004 ends; continuation of visit 4). They are
  fitted as two datasets, `V4a` and `V4b`, each with its own baseline and
  ramp.
- **Phases:** 1 = `LightCurves`, priors, `BroadbandFit`, `julietPlots`
  move + fixes, real-data checks (this document); 2 = multi-visit
  spectroscopic fit (`SpectroscopicFit`, `Spectrum`, spectrum plots;
  replaces `SpectroscopicLC`); 3 = common-mode correction, NIRCam
  phase-of-4 / per-amplifier regressors, fixed GP hyperparameters for
  channels, phase curves.

## 2. Architecture (`jstark/fitting/`)

| Module | Contents |
| --- | --- |
| `lightcurve.py` | `LightCurves` (one dataset): `from_spectra`, `from_photometry`, `white_light` (sum), `select_channels`, `bin` (sum), `exclude`, `outliers`/`clip`, `add_regressor` (also per channel), `regressor`, `design_matrix` (standardised); built-ins `time`, `time2`, `time3`, `phase4`; `check_name`, `as_lightcurves`, `MIRI_PIVOT` |
| `priors.py` | `normalise_priors` (short form, juliet dict or prior file), `build_priors` (adds nuisance priors), `model_report` (what juliet will fit per dataset), `covers`, `GP_KERNELS`, `priors_to_text` |
| `fit.py` | `BroadbandFit`, `ExpRamp`, `NonLinear`, `red_noise_beta` |
| `spectroscopic.py` | `SpectroscopicFit` (one `BroadbandFit` per wavelength bin, parallel, resumable), `Spectrum` (astropy `Table` subclass), `make_bins`, `restrict_priors` |
| `plots.py` | `julietPlots` (moved from `jstark.py`, fixed) |
| `jaxfit/` | optional JAX backend (5.10-5.14): `PriorSet` (juliet priors, transforms), `orbit` (batman geometry), `planet` (limb darkening, eclipse, phase curves), `JaxModel` (juliet's model and likelihood), `samplers` (`run_sampler`: blackjax / numpyro NUTS), `JaxResults` (chains, diagnostics, files), `JaxEvaluator` (posterior models) |
| `evaluators.py` | the model interface of the result methods and plots; `JulietEvaluator` (5.11, 5.13) |
| `utils.py` | helpers moved from `jstark/utils.py`: `t14`, `b_to_inc`, `inc_to_b`, `rho_to_ar`, `lcbin`, `pipe_mad`, `rms`, `fake_allan_deviation`, `corner_plot`, `make_psd`, `psd_fitting_kernel`, `planck_func`, `BrightnessTemperatureCalculator` |

Dropped from `utils.py`: `clip_outliers` (replaced by `LightCurves.clip`),
`standarize_data` / `classic_PCA` (copies live in
`extraction/photometry.py`), `generate_times_with_gaps` (CHEOPS
simulator), `trapz2d`, `autocorrelation_function`. `jstark/jstark.py` (`SpectroscopicLC`) was
deleted in Phase 2 (2026-11-15).

Tests: `jstark/tests/test_fitting.py` (12 tests, ~2 min; small dynesty
runs; the spectroscopic test spawns 2 worker processes).

## 3. How `BroadbandFit` works

1. Each dataset's broadband light curve: `white_light()` (channels NaN in
   > 50% of integrations left out; for photometry the light curve itself).
2. Linear regressors: `design_matrix(names)`, standardised (mean 0, sd 1;
   the means and sds are saved in `jstark_fit.json`). GP input: the named
   regressor(s), unscaled.
3. Integrations with any non-finite flux, error, linear or GP regressor
   value are dropped (for that dataset only); the flux is normalised by its
   median.
4. A 1D GP input that is not sorted (e.g. a centroid) sorts the whole
   dataset by it: juliet's per-instrument celerite GP is silently *not
   initialised* when the regressor is unsorted (juliet sorts only global
   GPs).
5. Priors: user priors first; then per dataset `mdilution` fixed 1, `mflux`
   N(0, 0.1), `sigma_w` LU(0.1, 1e4) ppm (1e5 until 2026-11-28), `theta<i>` U(-1, 1), GP
   hyperparameters (relative-flux amplitude LU(1e-6, 1e-2); timescales
   LU(cadence, 10 x span)), ramp amplitude U(-0.1, 0.1) and timescale
   LU(2 cadences, duration). `nuisance=` overrides the defaults.
6. `model_report`: transit if an LD prior names the dataset (juliet's own
   rule), eclipse if an `fp` prior applies, phase curve if `phaseoffset` /
   `C1` / `D1` apply. Warns when an `Rp/R*` prior applies to a dataset
   without LD priors (juliet then fits no transit to it).
7. `juliet.load(...)` with dicts of times, fluxes, errors, linear and GP
   regressors and, for ramps, `non_linear_functions` (`ExpRamp`, a
   module-level class so it pickles for `nthreads`); then
   `dataset.fit(sampler=..., **fit_kwargs)`.
8. `summary()` (fitted, fixed, derived: `(Rp/R*)^2` and `fp` in ppm, `a`
   from `rho`, `inc` from `b`) -> `jstark_summary.ecsv`; `diagnostics()`
   (p2p of residuals, median error, `sigma_w`, ratios, beta) ->
   `jstark_diagnostics.ecsv`; `channel_priors()` (orbit fixed or normal
   from the posteriors, for Phase 2); `BroadbandFit.load(folder)`.

### juliet behaviour found while building it

- **juliet never overwrites its files** (`lc.dat`, `priors.dat`,
  regressor files, `*posteriors.pkl`): a re-run in the same folder with
  changed data or priors silently returns the *old* fit. jstark stores a
  fingerprint (data, regressors, priors, LD laws, ramps) in
  `jstark_fit.json` and refuses a folder holding a different fit unless
  `overwrite=True` (which deletes the juliet files). An identical earlier
  fit is loaded instead of re-run. Since 2026-11-22 `overwrite=True` also
  re-runs an *identical* fit (user decision), and `SpectroscopicFit.run`
  clears all requested bins first, so an interrupted `overwrite=True` run is
  continued with `overwrite=False`.
- **Instrument names** must not contain `_` (parameter names are split on
  it) or `-` (per-instrument LD laws are written `inst-law`), and must not
  equal a parameter part (`p1`, `q1`, `w`, `theta0`, ...): juliet matches
  instruments against the parts of every parameter name.
- **Transit model only with LD priors:** juliet decides per instrument
  whether to fit a transit from the presence of `q1`/`u1`/`c1` priors.
- **GP amplitudes are in relative flux, `sigma_w` in ppm.**
- **`non_linear_functions` with `input_folder`:** `juliet.load` keeps the
  functions but leaves `nlm_lc_boolean` False, so the ramp is ignored.
  `BroadbandFit.load` switches it on (juliet 2.2.10 bug).
- **Eclipse timing:** with `light_travel_delay=True` juliet computes the
  eclipse time from `t0`, `ecc`, `omega`; a `t_secondary_p1` prior is used
  only without it. One `t_secondary` serves all visits.
- **400 NumPy "array to scalar" deprecation warnings** per eclipse test come
  from a juliet code path inside a try/except (they do not occur in jstark
  code; with the warning made an error the tests still pass).

### 3.1 How `SpectroscopicFit` works (Phase 2)

1. **Grid:** `make_bins`: given edges (1D contiguous, or `(nbin, 2)` pairs
   with gaps, e.g. a published table), `nbins` (equal numbers of good
   channels over all datasets, edges half-way between neighbouring
   channels), or the native channels (refused if the datasets' grids
   differ). Single-channel datasets are left out with a warning.
2. **Coverage:** a dataset enters a bin when its good channels (finite in
   >= 50% of integrations) with centres in the bin cover >= `min_cover`
   (0.5) of the bin width. Bin wavelength = mean of the summed channels'
   centres (averaged over datasets).
3. **Per bin:** `LightCurves.bin(edges=[lo, hi])` per covering dataset
   (sum), priors from a dict or `priors(info)`, then `restrict_priors`:
   dataset names not in the bin are removed from parameter names
   (`q1_V1_V2` -> `q1_V1`), parameters left with no dataset are dropped.
   Per-dataset options (`linear`, `gp`, ...) are restricted the same way;
   then a `BroadbandFit(..., out_folder=<out>/bin<i>, verbose=False)`.
   Warnings raised while building the bins are shown once.
4. **Resume:** for each bin, `done` = same fingerprint and summary,
   diagnostics and residual files present -> skipped; `different` = files
   of another fit -> the run stops before fitting anything unless
   `overwrite=True`; otherwise fitted (a bin killed mid-fit restarts from
   scratch: juliet writes its posteriors only at the end).
5. **Parallel:** `ProcessPoolExecutor` with the `spawn` context (macOS
   safe; the `BroadbandFit` objects are pickled, `ExpRamp` is module
   level); BLAS/OpenMP threads set to 1 in the workers; `nthreads` dropped
   with a warning; dynesty `print_progress=False` unless given. The worker
   (`_run_bin`) fits, writes `jstark_summary.ecsv`,
   `jstark_diagnostics.ecsv` and `jstark_residuals.npz` (data, median
   model, planet-only model from juliet's `return_components`, detrended =
   y - model + planet, residuals, integration numbers), and returns only
   `ok` / error text and the time. Failed bins are listed; the others are
   kept.
6. **Collect:** `spectrum()` reads the bin summaries: every fitted or
   derived parameter that is not a nuisance term (`mflux`, `mdilution`,
   `sigma_w`, `GP_*`, `theta*`, `ramp*`) becomes `x`, `x_errp`, `x_errm`;
   plus `bin`, `wave`, `wave_lo`, `wave_hi`, `datasets`, `lnZ` ->
   `spectrum.ecsv`; diagnostics of all bins -> `spectrum_diagnostics.ecsv`.
   `SpectroscopicFit.load(folder)` needs only `jstark_spectroscopic.json`
   and the bin folders.
7. **Plots:** `Spectrum.plot` (with a broadband value from a
   `BroadbandFit` or `(value, error)`), `plot_2d` (per dataset: detrended
   data, planet model, residuals; wavelength vs hours; optional time
   binning), `plot_noise` (rms vs bin size of every bin; p2p / error and
   p2p / (error + jitter) vs wavelength).
8. **Fixing nuisance terms from the white light:** `channel_priors(fix=...)`
   takes any parameter type, e.g. `fix=(..., 'ramptau')` fixes the ramp
   timescale per dataset (used for PRISM, §5.4).

`BroadbandFit.residuals(components=True)` and `save_residuals()` were added
for this (planet model, detrended data, integration numbers).

## 4. `julietPlots` fixes (moved to `fitting/plots.py`)

| Problem | Fix |
| --- | --- |
| `lm_lc_arguments.keys()` crashed without linear regressors (juliet stores None) | `self._lm`, `self._gp` default to `{}` |
| GP regressors indexed `[:, 0]` although juliet keeps 1D inputs 1D | `_gpx()` |
| `mflux`/`mdilution`/`sigma_w` read from the posteriors only (KeyError when fixed or shared, e.g. `sigma_w_V1_V2`) | `_samples()`: posterior, else fixed prior, else default; matches shared names |
| a single instrument name as a string was iterated character by character | `_instruments()` |
| `phase_folded_lc(one_plot=...)` with `instruments=None` crashed | instruments resolved first |
| high-resolution workaround read `<input_folder>/priors.dat` (None when built from objects) and re-ran juliet *in the fit folder* | in-memory priors, scratch folder with a copy of the posteriors, same sampler and options |
| planet-only models kept ramps (non-linear models) | `_planet_only_samples()` makes `ExpRamp` neutral; detrended data and GP data divide out the ramp |
| bare `except:` | `except Exception:` |

## 5. Results

### 5.1 Synthetic tests

- Two transit visits (4 channels, one dead; linear in time + `dy`): Rp/R*
  0.1001 +- 0.0003 (injected 0.1), a/R* 10.02 +- 0.07 (10), b 0.17 +- 0.08
  (0.175); `dy` coefficients equal 3e-4 x sd(dy); residual p2p equals the
  error of the 4-channel sum.
- Three eclipse visits (photometry, no LD, a GP on one, a ramp on one):
  fp 503 +- 18 ppm (500); ramp amplitude recovered.

### 5.2 WASP-39 b NIRSpec PRISM white light (full exposure)

Script: `.../jwst-ers/Data/NRSPR/reduction/fits/fit_white_prism.py`;
output `.../fits/white_2.0-5.35/`. White light 2.0-5.35 um (313 columns,
outside the saturated region), all 4 segments, HGA move (integrations
20750-20765) removed, no trimming; linear in time + `dy`, one exponential
ramp. `ds` and `dx` were left out on purpose: both carry a transit-shaped
signal (brighter-fatter, extraction_notes.md §5.14) and would bias the
depth. P fixed (4.0552941 d), e = 0, quadratic LD with free q1, q2.
Dynamic dynesty, 8 threads: 13 min (1.7 million likelihood calls on
21,484 points).

| Parameter | jstark | ERS benchmark (Carter & May 2024) |
| --- | --- | --- |
| t0 (BJD_TDB) | 2459771.335679 +- 0.000008 | 2459771.335647 |
| a/R* | 11.419 +- 0.013 | 11.390 (NIRISS, Feinstein et al.: 11.406) |
| b | 0.4448 +- 0.0027 | 0.4498 |
| i (deg) | 87.767 +- 0.016 | (NIRISS: 87.735) |
| Rp/R* | 0.14617 +- 0.00009 (depth 21,367 +- 27 ppm) | |
| ramp | +882 ppm, tau 1.4 h | (they trimmed 2000-3000 integrations) |

- Geometry within ~2 sigma of the benchmark (different wavelength range and
  systematics model); t0 differs by 2.8 s.
- Residuals: p2p 454 ppm, propagated error 433 ppm (ratio 1.05), jitter
  `sigma_w` 155 ppm (ratio with jitter 0.99); beta 1.04 / 1.06 / 1.11 /
  1.14 / 1.22 at 8 / 16 / 32 / 64 / 128 integrations (the extraction
  comparison found 1.12 at 64 for both reductions).
- Plots (`full_model.png`, `phase_folded.png`, `allan.png`) work with
  21,484 points and the ramp removed.

### 5.3 GJ 3929 b MIRI F1500W eclipses

Data: Rocky Worlds DDT program 9235, 5 exposures (V1, V2, V3, V4a = obs
004, V4b = obs 005), reduced with jstark defaults (Stage 1 all groups,
PSF-weighted photometry; `<visit>/reduction/`); script
`.../gj3929/Data/run_gj3929_full.py`, Stage 1 ~16-18 min per exposure.
Fits: `.../gj3929/Data/fit_gj3929_eclipses.py` -> `.../Data/fits/`.
References: Xue et al. 2025 (arXiv:2508.12516; V1+V2, aperture 4 px,
groups 5-45, 800 integrations trimmed, linear + exponential ramp +
centroids, eclipse time free per visit): 177 +47/-45 (V1), 143 +34/-35
(V2), joint 160 +26/-27 ppm. Connors et al. 2026 (arXiv:2606.07511; all
four visits, FN-PCA, 500 integrations trimmed, V4a 1500 trimmed for a
suspected tilt event, V4b with its own exponential ramp): 190 +- 40, 155
+- 34, 60 +- 34, 97 +- 78 ppm per visit, joint 118 +- 22 ppm; **eclipse 54
+- 2 min before phase 0.5** (e cos w = -0.0225 +- 0.0010).

**Single visit (V1) and eclipse timing.**
- With e = 0 and the light-travel delay (eclipse at phase 0.5), V1 gave
  fp = -125 +- 36 ppm (100 integrations trimmed) or -21 +- 60 ppm (800
  trimmed): the window at phase 0.5 sits on a positive excursion of the
  residuals. With the eclipse time free, one visit does not constrain it
  (PSF: 9 +- 170 ppm; aperture: -371 ppm at the edge of the window).
- A least-squares baseline fitted outside Xue's V1 window leaves -60 +- 31
  (PSF) and -81 +- 32 ppm (aperture) inside it (white-noise errors; beta
  1.2-1.6): a weak dip, within ~2 sigma of 177 +- 46 ppm.
- PSF photometry scatters less than aperture photometry (747 vs 792 ppm per
  integration; propagated 635 ppm) and has less red noise (beta at 128
  integrations 1.19 vs 1.63).

**V1 + V2 jointly, Xue et al. setup** (800 trimmed, ramp + time + xc +
yc, shared t_secondary free within +-0.05 d of phase 0.5; dynamic dynesty,
225 s): **fp = 143 +- 33 ppm** (Xue: 160 +26/-27), eclipse **53 +- 3 min
before phase 0.5** (Connors: 54 +- 2; Xue's per-visit times are ~11 min
early). 99% of the posterior is in that mode; the 1% near phase 0.5 has
fp 63 +- 47 ppm. Residual p2p 737 / 735 ppm, beta(128) 1.21 / 1.03.

**Plotting lesson:** with a `t_secondary` prior, batman shifts
`params.t0` to the conjunction implied by the eclipse time, and
`julietPlots` used that `t0` whenever `t0` was not in the posteriors, so
an offset eclipse appeared at phase 0.5. `julietPlots._pmed` now prefers
fixed priors, and the eclipse panel is centred on the fitted eclipse
phase.

**All five datasets** (2026-11-15; linear in time + xc + yc, one
exponential ramp per dataset, dynamic dynesty with defaults, 8 threads,
18-27 min each; V4a/V4b: visit-4 eclipse falls on the gap between them,
centre near the end of V4a, V4b starts in eclipse and its ramp overlaps the
egress):

| Variant | Trim | Eclipse time | fp (ppm) | lnZ |
| --- | --- | --- | --- | --- |
| A | 100 | free: +66 min (100%) | -136 +- 22 | (other trim) |
| C | 100 | free, per-visit depths: +2 +- 2 min | V1 -131 +- 46, V2 107 +- 34, V3 168 +- 35, V4 165 +- 100 | (other trim) |
| D | 500 (V4a 1500) | free: -8 +- 1 min (99%) | 128 +- 20 | 70364.9 +- 0.6 |
| E | 500 (V4a 1500) | fixed -54 min, per-visit depths | V1 7 +- 39, V2 110 +- 31, V3 184 +- 30, V4 189 +- 90 | 70353.6 +- 0.6 |
| F | 500 (V4a 1500) | fixed -54 min | **117 +- 19** | **70368.7 +- 0.6** |

Connors et al.: 118 +- 22 joint (-54 min); per visit 190 +- 40, 155 +- 34,
60 +- 34, 97 +- 78.

- **Joint depth reproduced:** F (Connors' trimming and eclipse time) gives
  117 +- 19 ppm vs 118 +- 22; D (time free) 128 +- 20.
- **Per-visit depths are not:** V1 7 vs 190 and V3 184 vs 60 ppm (each
  ~3 sigma), V2 and V4 agree. V1 keeps a long negative ramp after 500
  integrations (-1200 to -1800 ppm, tau ~1-1.4 h, the opposite sign to the
  other visits) that trades off with its depth (Xue et al., 800 trimmed and
  aperture photometry: 177 ppm). Connors used FN-PCA systematics, so this
  is model dependence on 4-sigma-per-visit signals, not a code issue; the
  per-visit split is disfavoured (lnZ E - F = -15).
- **The eclipse time is multimodal** (modes near -54, -34, -8 and +66 min).
  A least-squares profile per visit (scratch script, ramp tau on a grid):
  V2 prefers -54, V1 -34, V3 -6 (171 ppm there, ~0 at -54: the V3
  per-visit difference above), V4 split by the gap. In juliet, max lnL is
  70640.1 near -8 min (D) and 70636.6 at -54 min (F): only 3.5 apart.
- **dynamic dynesty with default settings does not sample this reliably:**
  D's best sample in the -70..-35 min region has lnL 70620.1, 16.5 below
  F's maximum there, so D's live points left the -54 mode before finding
  its peak (posterior weight there 0.01%; ~3% expected) and lnZ_D is
  probably underestimated (lnZ_F > lnZ_D by 3.8, although F is D restricted
  to one time). A (100 trimmed) collapsed onto the +66 min mode with a
  negative depth, driven by the start of V1. Lesson for users (now in
  fitting.rst): with a free eclipse time on low-S/N eclipses, fix the time
  from an independent constraint or compare fixed-time fits by evidence /
  max likelihood, and use more live points; 100 integrations of trimming
  are not enough for these MIRI visits.
- Residuals: p2p 737-758 ppm per integration in all variants; beta(128)
  0.88-1.37 (V3 highest).
- Plots: `fits/eclipses_D/phase_folded.png`, `full_model.png`.
  `phase_folded_lc` has a new `window` keyword (half-width in T14, default
  1.0; was a fixed 0.7, which hid the eclipse baseline). Its y-limits come
  from the model, so binned points of a low-S/N eclipse can fall outside
  (open).

**Correction (2026-11-19): E and F were not at -54 min.** With a *fixed*
`t_secondary` and a varying orbit (here free a/R*, b), juliet + batman put
the eclipse at the time implied by t0 (phase 0.5 for e = 0), not at
`t_secondary`. batman (2.5.1) converts `t_secondary` into a conjunction time
only when `t_secondary` differs from its value at the previous call, and
juliet (2.2.10) writes the given t0 into the batman parameters on every call;
whenever a/R* or b change, batman then recomputes the orbit from the raw t0.
A free `t_secondary` changes on every call and is handled correctly (D: the
evaluated eclipse sits at its posterior time, -8 min). Checked on the loaded
fits (juliet's own model for F has `params.t_secondary` = -54.0 min, but its
eclipse is centred at phase 0.5) and with a minimal batman reproduction.
Consequences:
- E and F are fits at **phase 0.5**, not at Connors' time. "Joint depth
  reproduced" (F 117 +- 19 vs 118 +- 22) and the E per-visit depths are
  therefore not comparisons at the same eclipse time.
- The sampling statements above that use F (max lnL 70636.6 "at -54 min",
  lnZ_F > lnZ_D, "D lost the -54 min mode") compare D with a phase-0.5 fit and
  no longer hold. The per-visit least-squares profile (scratch script, no
  juliet) is unaffected.
- jstark: `BroadbandFit.fit` now refuses a fixed `t_secondary` while the orbit
  varies (unless it equals t0 + P/2 for a circular orbit with t0 and P
  fixed); test `test_fixed_t_secondary_with_varying_orbit_is_refused`;
  fitting.rst explains it. To impose an eclipse time, use a narrow free prior.
- Re-done on the reduction with the current MIRI defaults (Stage 1
  2026-11-19, `<visit>/reduction_current/`; `fit_gj3929_current.py`,
  `fits/current_eclipses_<D|F|E>`), with `t_secondary` normal(-54, 2 min) for
  F and E: see below.
- To report upstream (juliet), with the minimal reproduction.

**Redo on the current-default reduction (2026-11-19).** Stage 1 8-18 min per
exposure (sequential, ramp_fit 4 cores); each fit ~18-22 min (dynesty, 8
threads). Same models as before (trim 500/1500, ramp 1, linear time, xc, yc).
Juliet-only check of the bug first (`juliet_fixed_tsec_repro.py`, injected
500 ppm at -54 min, free a/R*): fixed `t_secondary` -> 42 +- 20 ppm;
normal(-54 min, 30 s) -> 498 +- 20 ppm at -54.0 min.

| Fit | Eclipse time (min from phase 0.5) | fp (ppm) | lnZ | max lnL |
| --- | --- | --- | --- | --- |
| D | uniform +-72 -> -8.5 +- 1.4 (94%; ~3% at +60..+72, prior edge) | 134 +- 20 | 70697.3 | 70969.1 |
| F | normal(-54, 2) -> -51.4 +- 1.6 | 111 +- 18 | 70693.8 | 70958.9 |
| E | normal(-54, 2), fp per visit -> -53.6 | V1 65 +- 41, V2 160 +- 27, V3 77 +- 30, V4 89 +- 64 | 70668.4 | 70960.9 |

- Joint depth at Connors' time 111 +- 18 vs their 118 +- 22; with the time
  free 134 +- 20 (old reduction 128 +- 20, also at -8 min).
- The data (this reduction and systematics model) prefer -8 min: max lnL
  10.3 above F's, and lnZ_D - lnZ_F = +3.6 despite D's 144-min prior. In D
  the best sample within -60..-45 min is 70950.9, 8 below F's max there:
  dynesty does not resolve the maximum of a mode with negligible posterior
  weight, as expected, but that mode is genuinely disfavoured here.
  Red noise (beta_128 up to 1.6, V3) inflates lnL differences by ~beta^2, so
  the -8 vs -54 preference should not be over-read.
- Per visit (E): V2, V3, V4 agree with Connors (155, 60, 97); V1 lower
  (Connors 190). lnZ_E - lnZ_F = -25.
- The tutorial (gj3929b_miri_f1500w.ipynb) uses these fits.

### 5.4 WASP-39 b NIRSpec PRISM transmission spectrum vs the ERS benchmark

Scripts: `.../jwst-ers/Data/NRSPR/reduction/fits/fit_spec_prism.py` (fit)
and `compare_spec_prism.py` (comparison); output `.../fits/spec_ers147/`
(`spectrum.ecsv`, `comparison_ers.json`, `compare_ers.png`, `lc2d.png`,
`noise.png`).

- Bins: the 147 published bins of the benchmark (`Fit_LimbDarkening/
  NIRSpec_PRISM/bins_scale1.csv`, 0.517-5.351 um). Same data selection and
  systematics as the white light (§5.2): all 4 segments, HGA move removed,
  linear in time + `dy`, one ramp. P, t0, a/R*, b, e, omega and the ramp
  timescale fixed to the white-light medians (`channel_priors(fix=...)`);
  free Rp/R*, q1, q2, ramp amplitude, linear coefficients, jitter.
  Freeing tau per bin was tried on bin 100: the ramp is then poorly
  constrained (tau 9 min against 1.4 h) with no change in depth beyond the
  error, so tau stays fixed.
- Run: 145 bins (bins 26 and 29 at 1.08 and 1.24 um are dropped because
  more than half of their columns are group-0-only, NaN by `min_groups=2`),
  12 processes, dynamic dynesty defaults, 6721 s total (~400 s per bin,
  ~0.8 million likelihood calls each). No failures.

| Bins | n | median offset (ppm) | scatter after offset (ppm) | reduced chi2 | error ratio jstark/ERS | > 3 sigma |
| --- | --- | --- | --- | --- | --- | --- |
| unsaturated | 112 | -152 | 90 | 0.23 | 1.14 | 0 |
| unsaturated, 2.0-5.35 um | 102 | -149 | 91 | 0.20 | 1.14 | 0 |
| saturated (ERS flag) | 33 | +244 | 682 | 103 | 0.98 | 24 |

- **Unsaturated bins reproduce the benchmark** up to a constant offset of
  -150 ppm. The shape agrees bin by bin: no bin beyond 3 sigma. The reduced
  chi2 of 0.2 is expected well below 1 because both reductions use the
  same photons, so the differences are much smaller than the quadrature
  sum of the errors. The offset is a common-mode change. It most likely
  comes from the different white-light geometry (b 0.445 against 0.450,
  a/R* 11.42 against 11.39) and the different systematics/trimming
  choices (not tested). Independent reductions of the same data often
  differ by offsets of a few hundred ppm; an offset does not change the
  spectral shape.
- jstark errors are 14% larger (median over the 112 bins of the per-bin
  ratio of the symmetrised depth errors; 5-95%: 1.08-1.22, the same at
  all wavelengths). Checks:
  - White-noise floor per bin (measured scatter, depth template from the
    fit, baseline constant + slope + ramp, limb darkening fixed): jstark
    errors are 1.17x the floor, ERS errors 1.02x, although ERS also fitted
    the limb darkening.
  - Refitting bins 60, 100 and 130 with q1, q2 fixed
    (`fits/fit_spec_prism_fixld.py`, `spec_ers147_fixLD/`) shrinks the
    errors by 9-21%. The depths stay the same. The ERS fixed-LD errors
    shrink by the same amount. Like with like, jstark/ERS is still
    1.09-1.13.
  - Published spectroscopic light curves (`S4_wasp39b_ap0_bg0_LCData.h5`)
    against ours, same bins, out of transit: p2p scatter ratio
    jstark/ERS 0.993 (5-95%: 0.979-1.012); both reductions' per-point
    errors match their scatter (p2p/err 0.99 and 1.01).
  - The floor depends on the systematics model. With ERS's own scatter,
    ERS errors are 1.02x the floor of "full exposure + ramp" but 1.12x the
    floor of "first 2000 integrations trimmed, no ramp" (their approach).
    Fitting the ramp costs more depth precision than discarding 2000
    integrations (the trimmed floor is ~9% lower).
  - Confirmed by refitting bins 60, 100, 130 with the first 2000
    integrations trimmed and no ramp (`fits/fit_spec_prism_trim.py`,
    `spec_ers147_trim2000/`): errors 61 / 129 / 197 ppm, i.e. 0.98-1.02x
    ERS (full + ramp: 1.07-1.12x). Depths moved by +43 / +137 / -15 ppm.
  - **So the 14% is the cost of fitting the ramp instead of trimming.**
    ERS errors are not underestimated (an earlier draft of this note said
    they were; that compared them with the wrong floor). The trimmed fits
    might also explain part of the -150 ppm offset (not tested on all
    bins).
- **Saturated bins are biased, as predicted from the light curves**
  (extraction_notes.md §5.14). 0.62-0.75 um agrees (offset ~-250 ppm).
  From 0.75 um the jstark depth rises above the benchmark: up to +1500 ppm
  at 1.0 um and +2200 ppm at 1.2 um, with a saw-tooth pattern across
  1.3-1.8 um that follows how many groups the cores have before
  saturating. It returns to the unsaturated offset by ~1.85 um.
- Noise (`noise.png`): residuals bin down as white noise in all bins. Before
  the jitter is added, the scatter-to-error ratio is ~1.0 above 1.8 um but
  1.1-1.27 below 1.5 um, so the propagated errors of the saturated columns
  are too small and the jitter absorbs it. One bin at 4.5 um has a ratio of
  0.82 (scatter below the errors, not chased).
- 2D residuals (`lc2d.png`): no structure in time or wavelength beyond the
  noise increase at the blue and red ends.

**Conclusion:** `SpectroscopicFit` is validated on the unsaturated part of
the PRISM spectrum. The saturated region (0.75-1.85 um) needs one of the
extraction fixes (b)-(d) of extraction_notes.md §6. This spectrum is the
baseline to compare those fixes against.

### 5.5 Common mode on real data (Phase 3 steps 2-3, 2026-11-16)

Same bins, priors and systematics as the baseline in every case; only the
common mode changes. Columns: depth change vs no common mode (median, rms
after the median), median error ratio vs no common mode, median per-bin
p2p/err and beta_32, and chi2_red vs the ERS benchmark after its median
offset. Scripts: `compare_cm.py` (in `NRSPR/reduction/fits/`), outputs in
`*/fits/compare_cm/`.

| Data (bins) | common mode | depth change (ppm) | err ratio | p2p/err | beta_32 | chi2 vs ERS |
| --- | --- | --- | --- | --- | --- | --- |
| PRISM (ERS 147) | none | - | 1 | 0.996 | 1.00 | 0.234 |
| | divide | -0.2 ± 3.5 | 1.002 | 0.991 | 1.00 | 0.237 |
| | regressor | -3.3 ± 4.2 | 0.999 | 0.992 | 1.00 | 0.235 |
| SOSS O1 (R100, 108) | none | - | 1 | 1.29 | 0.83 | 0.50 |
| | divide | -1.5 ± 2.4 | 0.985 | 1.28 | 0.81 | 0.51 |
| | regressor | 0.0 ± 2.9 | 0.984 | 1.27 | 0.81 | 0.51 |
| SOSS O2 (R100, 27) | none | - | 1 | 1.44 | 0.79 | 0.33 |
| | divide | -1.3 ± 2.6 | 0.989 | 1.44 | 0.80 | 0.33 |
| | regressor | -0.3 ± 2.5 | 0.983 | 1.41 | 0.79 | 0.33 |
| NIRCam F322W2 (R100, 49) | none | - | 1 | 0.98 | 1.15 | 0.30 |
| | per-amp divide | 3.4 ± 3.1 | 0.991 | 0.93 | 0.77 | 0.31 |
| | per-amp regressor | 3.5 ± 2.8 | 0.989 | 0.92 | 0.72 | 0.31 |
| | white-light divide | -2.1 ± 2.6 | 0.988 | 0.96 | 0.76 | 0.30 |

(beta for white noise with N = 365 (NIRCam) has a median of 0.89 at 32
integrations, 68% range 0.69-1.10, so values below 1 are expected there.)

**At native or R100 resolution the common mode changes nothing that
matters** (depths by a few ppm, errors by <= 1.7%) in any of the three
modes. The shared noise is real (the residuals averaged over all bins drop
from 139 to 81 ppm in SOSS O1 and from 227 to 23 ppm in NIRCam), but per
bin it is small next to the bin's own noise (NIRCam ~200 ppm shared vs
~1400 ppm per bin and integration), so it adds little in quadrature and
the per-bin jitter absorbs it. The SOSS per-bin excess (p2p/err 1.29 /
1.44) is not common mode either: it stays the same after the correction
(it correlates over neighbouring columns; SOSS memory notes).

**Why the depths cannot move much.** Dividing by data / planet model of
the white light replaces the bin's share of the white-light transit as
measured by the data with the fitted one; with the same systematics
model both agree. What changes is the error: without a common mode it is
the error of the absolute depth (for white residuals); with it, the error
of the depth relative to the white-light level, which carries its own
error (and any white-light red noise) for all bins at once.

**Wide bins (NIRCam, 7 bins, two per amplifier, `*_wide`).** Without a
common mode the residuals are red (beta_32 2.1-2.4: the reported errors
are too small for the absolute depths). Both the per-amplifier and the
single white-light common mode whiten them (beta 0.5-1.1) and reduce the
errors by ~24% (55 -> 41 ppm), with depths unchanged within 8 ppm. The
per-amplifier version gives lower p2p (242 vs 299 ppm in bin 1) but the
same errors: the difference is self-noise removal (each wide bin is half
of its amplifier's light, but ~1/7 of the total). The last bin
(3.89-4.03 um) is almost all of amplifier 4, so the per-amplifier common
mode divides it by itself (p2p 407 -> 101 ppm): meaningless for that bin.

**NIRCam per amplifier vs one common mode:** no measurable gain on WASP-39
b F322W2 (the Stage 1 `readout` correction leaves a remainder that is
mostly common to all amplifiers, as in nircam 1/f findings). Kept as an
option.

**Full NIRCam F322W2 reduction (by the way).** 39 min end to end with
defaults. Per amplifier white-light depths 21505 / 21251 / 20927 / 20958
ppm (so a common planet model for all amplifiers would indeed have left
transit-shaped residuals). Orbit a/R* 11.36 ± 0.03, b 0.452 ± 0.006
(PRISM 11.42, 0.445; SOSS 11.40, 0.450). Spectrum vs the ERS benchmark at
R100: median offset -108 ppm, chi2_red 0.30 after it, errors 0.99x. Light
curves vs the ERS (Eureka!) time series, same integrations: white light p2p
205 vs 271 ppm (-24%); R100 bins 4% lower; p2p/err 1.06 (ours) vs 1.26
(theirs). The white light has red noise (beta_32 3.2, a dip before
ingress); this affects the white-light depth/orbit errors, not the shape
of the spectrum.

**Full SOSS reduction (by the way).** 3.8 h end to end (PSF fit 3.4 h with
the F277W contaminant model). Joint white-light fit of both orders
(a/R* 11.40, b 0.450); spectra vs the SOSS benchmark at R100: offsets +67
(O1) / +47 ppm (O2), chi2_red 0.50 / 0.33, errors 1.07x / 1.04x.

### 5.6 Why jstark is offset from the ERS benchmark (2026-11-16)

Offsets vs ERS in unsaturated bins: PRISM -152, NIRCam -108, SOSS +67 / +47
ppm (constant with wavelength). Mixed signs, so not one pipeline bias.

**Sensitivity.** Noise-free batman test: with LD free, Delta b = 0.01 moves the
depth by ~210-240 ppm (fixed LD: 80-120 ppm); our white-light b error
(+-0.006) alone allows ~+-130 ppm. Moves along the a/R*-b degeneracy cost
only tens of ppm. The per-bin errors exclude this (orbit fixed), so the
absolute level carries a common error of ~100 ppm not in the error bars.

**The ERS orbit** is one joint fit of all JWST + TESS + NGTS + RVs
(ZENODO/2_WHITE_LIGHT_CURVES/priors-and-posteriors/posteriors.dat):
P 4.0552842518, a/R* 11.3905, b 0.4498, t0 2459791.61207. Same for all
instruments.

**PRISM decomposition** (`fits/fit_spec_prism_orbit.py`, `fit_spec_ers_lc.py`,
`compare_orbit.py`; 12 unsaturated bins 2.2-5.3 um, LD free everywhere):

| Run | jstark - ERS median (ppm) | rms after offset | err / ERS |
| --- | --- | --- | --- |
| ours: our orbit, full exposure + ramp | -160 | 80 | 1.14 |
| ERS orbit, full + ramp | -137 | 82 | 1.12 |
| our orbit, first 2000 ints trimmed, no ramp | -94 | 48 | 1.04 |
| ERS orbit, trimmed | -73 | 48 | 1.04 |
| ERS's own light curves, our fitter, ERS orbit, trimmed | -21 | 24 | 1.02 |

- Orbit: +24 ppm. Ramp vs trimming: +76 ppm (the larger part; the ramp
  model and the trimming give depths ~80 ppm apart, a systematics-model
  dependence of the absolute level). Extraction (our light curves vs
  theirs, same fit): -52 ppm (~0.25% of the depth, dilution-like; not
  traced). Fitter details left: -21 ppm.
- Fitting ERS light curves with jstark reproduces ERS depths (-21 +- 24
  ppm): the fitting code is consistent with theirs.
- Conclusion: the offset is analysis choices, not a bug; the shape is
  unaffected (but see 5.6.1: the extraction part is wavelength dependent).

#### 5.6.1 The extraction part, traced (2026-11-16)

Scripts in `NRSPR/reduction/fits/extraction_offset/`. Method: fit both
reductions' light curves with the same fixed transit shape (ERS orbit, ERS
exotic-ld quadratic LD), and the depth difference from the *difference*
light curve (ours/oot - ERS/oot), which cancels the shared photon noise
(residual correlation 0.96), errors with a time-binning beta.
(`depth_lin.py`, `diffdepth.py`, `allbins.py`; a first regression of ours on
ERS, `compare_lcs.py`, is biased by noise in the regressor - ignore it.)

- Reproduced at light-curve level: optimal -57, box -64 ppm (12-bin median).
- Not the extractor: a plain box summed straight from the cleaned 2D
  frames with the extractor's background (median of rows |d| > 8) matches
  the extractor's box bin by bin (`growth2.py`). PSF fit, optimal weights,
  rejection play no role. The background we subtract is ~0.016% of the flux
  (~3 ppm of depth).
- Wavelength dependent (all 102 unsaturated bins, weighted means):
  2.1-3.0 um -9 +- 2; 3.0-3.8 -20 +- 3; 3.8-4.4 -33 +- 7; 4.4-5.0 -96 +- 10;
  5.0-5.4 -66 +- 20 ppm. The "-52 ppm median" is mostly the red end.
  ERS per-bin errors there are 166-222 ppm, so ~0.5 sigma per bin but
  coherent over ~40 bins (CO2/CO region).
- Behaves like an additive, non-transiting term: converted to flux, the
  implied excess in our light curves is ~15-25 DN/s per column at every
  wavelength (stellar flux varies 20x). Either our frames keep something
  near the trace that ERS removed, or ERS over-subtracted. Not uniform along
  the column (our |d| > 8 background would remove that).
- Brighter-fatter is present: fitted PSF width ds follows the transit,
  width ~ F^0.055 (narrower by 0.12% in transit); trace position dy does
  not. Row-resolved depths in the core differ by up to ~1500 ppm (flux
  moving between pixels), but summed over an aperture this cancels:
  depth vs box half-width changes by <~15 ppm at 2-4 um.
- In the red, wider apertures are shallower: D(h=6) - D(h=3) ~ -45 ppm at
  4.4-5.0 um and ~ -70 ppm at 5.0-5.4 um (~2 sigma each); wing rows
  |d| = 4-6 are ~1300-1800 ppm shallower than the core. A hint of a
  non-transiting signal in the PSF wings of our frames; it explains at most
  half of the red offset (a 3-px half-width box is still ~-100 ppm vs ERS
  in the red bins).
- Open: whether the excess is in our stage 1 (candidates: group-level
  column background, linearity, no dark step) or in ERS's. Next tests:
  stage-1 variants on one segment, then the difference-LC depth.

#### 5.6.2 Offsets, Step 0 and Step 1.1 (2026-11-17)

**Step 0: one tool for all comparisons.**
`jwst-ers/Data/tools/ratio_depth.py`: `align` (integrations matched by
time), `transit_shape` / `eclipse_shape` (unit-depth batman shapes),
`depth_difference` (ratio or difference light curve, polynomial in time,
fixed shape; errors from the residual scatter x the largest time-binning
beta at 32-256 points; also the residual correlation and noise of each),
`band_summary`. Validated on PRISM (`tools/validation/validate_prism.py`):
ours - ERS by band -8.5 +- 2.0, -20.4 +- 3.5, -33.5 +- 7.2, -97.6 +- 10.6,
-69.1 +- 20.1 ppm (5.6.1: -9, -20, -33, -96, -66); ratio and difference
forms agree to < 3 ppm; residual correlation 0.96.

**Step 1.1: official Stage 1.** `NRSPR/reduction_official/` (scripts
`run_official.py`, `compare_11.py`): jwst 3.0.0 `calwebb_detector1` on
seg002-seg003 (integrations 6101-18300; transit 8257-15782) with the CRDS
defaults (includes `dark_current`; no group-level background), except
`clean_flicker_noise`: CRDS `pars-cleanflickernoisestep_0002` turns it on
for NRS BOTS, it is a 1/f correction (not wanted here) and it crashed in
its internal `assign_wcs`, so it was skipped. Then jstark ExposureCleaner
(defaults) + SpectralExtractor (PRISM defaults), as for our reduction.
115 s per segment; 8 min total. The PSF fit shows the missing 1/f
correction (reduced chi2 6.4, residual scale 1.27 sigma), but the binned
light curves are only 2% noisier than ours (p2p ratio 1.02).

| Band (um) | ours - ERS | official - ERS | official - ours | official box - ours box |
| --- | --- | --- | --- | --- |
| 2.1-3.0 | -9.8 +- 2.4 | -13.0 +- 3.3 | -2.9 +- 3.0 | -8.3 +- 3.2 |
| 3.0-3.8 | -19.1 +- 4.1 | -25.0 +- 5.6 | -5.5 +- 4.6 | -2.0 +- 5.4 |
| 3.8-4.4 | -26.5 +- 8.6 | -44.0 +- 11.8 | -18.6 +- 9.5 | -18.6 +- 11.2 |
| 4.4-5.0 | -106.7 +- 12.6 | -104.8 +- 15.8 | +2.5 +- 11.4 | +6.7 +- 13.5 |
| 5.0-5.4 | -59.1 +- 24.2 | +11.5 +- 30.4 | +62.7 +- 21.6 | +87.7 +- 24.9 |

(ppm; weighted means over the 102 unsaturated ERS bins, integrations of
seg 2-3 within ERS's window.) Frames (out-of-transit medians,
background-subtracted like the extractor): official - ours summed over the
aperture is -1.6, -1.1, -0.5, -0.8, +1.1 DN/s per column by band, against
the 15-25 DN/s per column that the offset would need; both backgrounds are
0.0-0.5 DN/s per pixel.

**Conclusion: our Stage 1 is not the cause.** The official Stage 1, with
dark subtraction and without group-level background, gives the same
offset against ERS at 2.1-5.0 um, and official - ours is consistent with
zero there (the 5.0-5.4 um band differs by +63 +- 22 ppm, 2.9 sigma; the
lowest-flux band). The offset sits in steps common to both of our runs
(ExposureCleaner; the extractor was already excluded in 5.6.1) or on the
ERS (FIREFLy) side.

#### 5.6.3 Offsets, Steps 1.2-1.4 (2026-11-17): it is reduction-to-reduction scatter

Scripts in `NRSPR/reduction_official/scripts/` (`step12_cleaner.py`,
`step1_wave.py`, `step14_growth.py`, `step13_referee.py`), logs in
`reduction_official/step1/`. All ppm, medians or weighted means over the
102 unsaturated ERS bins, segments 2-3 (or the full window where noted).

- **1.2 ExposureCleaner is not the cause.** A plain 2D box (rows |d| > 8
  background) summed from our rateints and from the cleaned frames gives
  the same depths to <= 1 ppm. The box from our and from the official
  rateints both reproduce the offset against ERS (medians about -7 to -10,
  -23 to -40, -43, -110 to -125 ppm per band).
- **Wavelength solution is not the cause.** Ours vs the ERS time-series
  spectra (`1_TIME_SERIES_SPECTRA/NIRSpec_PRISM_Spectra.h5`): -1.7 nm
  (0.17 px) median, -3.3 to -0.4 nm; every ERS bin contains the same
  columns in both. The ERS spectra summed in the ERS bins reproduce the ERS
  light curves exactly.
- **1.4 Background region and aperture.** The background region (|d| > 8,
  |d| > 10, outer 3 rows) changes nothing. The aperture matters in the red:
  half-width 2 -> 8 px moves 4.4-5.0 um from -103 to -165 ppm and 5.0-5.4 um
  from -82 to -197 ppm (flat at 2.1-3.8 um): the PSF wings transit less
  deeply than the core there. A constant per-pixel non-transiting term does
  not fit (implied 0.2-4.3 DN/s per pixel, varying with band). Adding the
  PSF-width regressor `ds` makes the offset worse (-150 to -320 ppm): `ds`
  follows the transit (brighter-fatter) and absorbs part of it, confirming
  that it must stay out.
- **1.3 Referees: four independent reductions** (Rustamkulov et al. 2023,
  Zenodo 7388032, `NRSPR/published_rustamkulov2023/`; normalised per-column
  light curves without metadata; column and integration offsets recovered
  from the shared photon noise: FIREFLy and tshirt 0/0, Eureka! columns +14,
  Tiberius columns +20 and integrations +2000). Bins = the same
  flux-weighted mean of normalised columns for every reduction.

  | Band (um) | FIREFLy - jstark | tshirt - jstark | Eureka! - jstark | Tiberius - jstark |
  | --- | --- | --- | --- | --- |
  | 2.1-3.0 | +8.7 +- 2.2 | -3.6 +- 3.2 | +53.1 +- 4.8 | +18.3 +- 5.1 |
  | 3.0-3.8 | +20.7 +- 3.7 | -4.0 +- 4.5 | +67.4 +- 6.4 | -11.4 +- 6.5 |
  | 3.8-4.4 | +36.8 +- 7.4 | -11.4 +- 8.4 | +83.2 +- 10.3 | -122.5 +- 13.4 |
  | 4.4-5.0 | +93.8 +- 10.9 | -22.2 +- 11.3 | +120.6 +- 12.1 | -223.0 +- 21.2 |
  | 5.0-5.4 | +69.3 +- 20.4 | -49.0 +- 20.1 | +148.6 +- 24.0 | -412.0 +- 40.6 |

  FIREFLy - ERS benchmark = 0.0 in every band (correlation 1.000): **the
  ERS benchmark PRISM light curves are the FIREFLy reduction.**

**Conclusion.** The PRISM depth offset is ordinary scatter between
reductions of the same data, not a jstark error. jstark agrees with tshirt
to within 4-22 ppm at 2.1-5.0 um (the reddest band -49 +- 20). The four
published reductions alone span ~340 ppm at 4.4-5.0 um and ~560 ppm at
5.0-5.4 um (Tiberius shallowest, Eureka! deepest); the ERS benchmark
(FIREFLy) sits between them. The aperture dependence in the red (wings
shallower than the core) is a plausible reason the reductions differ
(different apertures and weights), but which depth is "true" cannot be
decided from these data alone. Note that the Rustamkulov reductions used
the 2022 calibration.

#### 5.6.4 Offsets, Step 2: NIRCam F322W2 and NIRISS SOSS (2026-11-17)

Scripts and logs in `jwst-ers/Data/tools/step2/` (`decompose.py`,
`referee_nircam.py`, `nircam_waveshift.py`, `nircam_selfcheck.py`,
`nircam_official.py`, `referee_soss.py`); NIRCam official Stage 1 in
`NRC_LW/reduction_official/`.

**Decomposition** (ERS R100 bins; linear fixed-shape fits reproduce both
published spectra to <= 8 ppm, so the parts add up; medians per bin, ppm):

| Mode | total | orbit | LD (ours fitted vs ERS fitted) | light curves | remainder |
| --- | --- | --- | --- | --- | --- |
| NIRCam F322W2 | -108 | -4 | +20 | -119 (wmean -110 +- 11) | +2 |
| SOSS O1 | +67 | +6 | +25 | +40 (wmean +34 +- 4) | 0 |
| SOSS O2 | +47 | +5 | +24 | +12 (wmean +21 +- 9) | +2 |

The orbit is negligible; our fitted limb darkening adds +20-25 ppm in every
mode; the rest is in the light curves.

**Referees.** NIRCam: Ahrer et al. 2023 (Zenodo 7101283) Eureka!, HANSOLO,
tshirt binned raw light curves (110 bins). Their wavelength scales differ
from ours (measured from the shared noise): Eureka! and tshirt -1.4 to -3.2
nm, HANSOLO -13 to -56 nm (a different/older dispersion solution); bins
mapped to our scale. Reduction - jstark: Eureka! +114 +- 9, tshirt +51 +- 9,
HANSOLO +27 +- 8 ppm (ERS benchmark +110 +- 11: the benchmark matches the
Eureka! reduction). SOSS: Feinstein et al. 2023 (Zenodo 7709712) nirHiss
spectroscopic light curves (wavelength scale within 1-3 nm): nirHiss -
jstark -7.5 +- 5.2 (O1), -33 +- 11 ppm (O2); ERS - jstark -34 +- 4, -21 +- 9.
SOSS is within the scatter of reductions (10-30 ppm). NIRCam is not:
jstark is shallower than all three reductions.

**NIRCam: our Stage 1 is the cause.** Official `calwebb_detector1` (CRDS
defaults, `clean_flicker_noise` skipped) + jstark extraction (defaults):
official - ours = +70 +- 9 ppm (covariance extraction), +110 +- 13 (box),
in all three bands; official - ERS = -39 +- 7 ppm, inside the referee
spread. Our box extraction is 23 +- 4 ppm shallower than our covariance
extraction (so the extraction weighting is not the cause). The jstark
NIRCam recipe differs from the official one mainly in the group-level
`readout` 1/f correction (`group_bkg`, `method='readout'`, `axis='row'`,
`wing_frac=0.04`; plus `temporal_jump` and no `dark_current`). Hypothesis:
the difference images (median frame scaled by a rough white light) leave a
transit-shaped residual in the wing pixels used with `wing_frac` (a wing
depth that differs from the white light), which the row-offset model
subtracts from whole rows. To test: our Stage 1 with `wing_frac=None`, and
with `method='diff'` / `'median'`, on the same data.

**NIRCam Stage 1 variants** (2026-11-17; `NRC_LW/reduction_variants/`,
`run_variant.py`; full exposure, everything as the default except
`group_bkg`; `tools/step2/nircam_variants.py`, `nircam_variants_noise.py`).
Depth - official Stage 1, R100 bins (covariance extraction; box similar,
larger), and white-light noise (residuals about a linear trend + fixed
transit):

| `group_bkg` | depth - official (ppm) | white rms / p2p (ppm) | beta_8 / beta_32 |
| --- | --- | --- | --- |
| official (none) | 0 | 241 / 217 | 1.42 / 2.14 |
| readout + wing_frac 0.04 (default) | -70 +- 9 | 230 / 178 | 1.89 / 3.25 |
| readout, no wings | -14 +- 5 (median -9) | 201 / 187 | 1.41 / 2.20 |
| diff, no wings | -25 +- 5 | 190 / 173 | 1.41 / 2.07 |
| median | -46 +- 5 | 200 / 190 | 1.30 / 1.80 |

**The wing pixels cause most of the dilution** (+48 to +56 ppm back
without them) and add red noise (beta_32 3.25): they let the transit into
the 1/f model. The lower p2p that made `wing_frac` the default
(detector_processing_notes 6.4.x, HD 207496 / 55 Cnc) was measured without a
transit or eclipse in the data, so this leakage could not show there. Every
group-level correction still dilutes a little (readout -14, diff -25,
median -46 ppm), least for `readout` without wings. Per-bin noise at R100 is
the same for all (ratio 1.00-1.02).

**Decision (user, 2026-11-17): NIRCam default = `readout` without wing
pixels** (`instruments._nircam`: `onef_kwargs={'readout_order': 1}`;
`wing_frac` stays available, documented as biasing transit/eclipse depths;
reduction tests pass). The `readout_nowing` variant has exactly these
settings, so it is the new-default reduction
(`NRC_LW/reduction_variants/readout_nowing/`; fits in its `fits/`, made with
`fit_white_nrc.py` / `fit_spec_nrc.py none` and `NRC_RED` pointing there).

| WASP-39 b F322W2, R100 | old default | new default |
| --- | --- | --- |
| white-light orbit a/R*, b | 11.362, 0.4522 | 11.403, 0.4455 (ERS 11.3905, 0.4498) |
| spectrum - ERS: median offset | -108 ppm | -60 ppm |
| rms after offset / chi2_red / error ratio | 77 / 0.31 / 0.99 | 75 / 0.29 / 0.98 |
| decomposition: orbit / LD / light curves | -4 / +20 / -110 +- 11 | -18 / +6 / -54 +- 8 |
| Eureka! - jstark (light curves) | +114 +- 9 | +56 +- 7 |
| tshirt - jstark | +51 +- 9 | **+1.5 +- 7.6** |
| HANSOLO - jstark | +27 +- 8 | -36 +- 8 |
| ERS benchmark - jstark | +110 +- 11 | +54 +- 8 |

With the new default the NIRCam light curves sit inside the spread of the
published reductions (HANSOLO -36 to Eureka! +56) and agree with tshirt,
as on PRISM. The remaining -60 ppm against the ERS spectrum is the light
curves (-54, i.e. ERS = Eureka!-like) plus the orbit (-18; our b 0.4455).

### 5.7 Phase curves (Phase 3 step 4, 2026-11-16)

**juliet behaviour (2.2.10), checked on a synthetic WASP-43 b-like phase
curve** (`fp` 5500, `C1` 2500, `D1` -800 ppm; fit recovers all within 1
sigma; planet component within 12 ppm of the truth):

- Cowan & Agol: `F = fp + C1 (cos wt - 1) + D1 sin wt + C2 (cos 2wt - 1) +
  D2 sin 2wt`, `wt = 2 pi (t - t_secondary) / P`, multiplied by the
  normalised eclipse model and by the transit model. `fp` = planet flux at
  the eclipse.
- juliet always reads `C2`, `D2` (KeyError without them) -> jstark adds
  them fixed at 0 (first-order model; listed in `added`).
- `C1`..`D2` are looked up with the instrument suffix of `fp`
  (`fp_iname`) -> `C1_p1_LRS` needs `fp_p1_LRS`; checked.
- Phase curves are evaluated **only in the joint transit + eclipse
  branch** (`TranEclFit`); a dataset without LD priors gets a plain
  eclipse and the `C1`/`D1` priors are silently ignored (synthetic test:
  "fp" = 119 ppm). jstark raises instead (`check_phase_curves`); fixed LD
  is fine when the transit is not covered.
- `t_secondary` is **not mandatory**: with `light_travel_delay=True,
  stellar_radius=<R*/Rsun>` (keywords of `juliet.fit`, not priors) juliet
  computes the eclipse time from P, t0, ecc, omega (batman
  `get_t_secondary` + light-travel time) and uses it in the phase-curve
  `omega_t` too. Only with `light_travel_delay` off does juliet read
  `t_secondary_p<n>` (KeyError if absent) -> `BroadbandFit.fit` checks this
  and names both routes. Verified on the synthetic phase curve without a
  `t_secondary` prior (2026-11-16): fp 5493+-36 (truth 5500), fnight
  494+-34 (500), amp 5253+-18 (5250), offset 17.89+-0.19 deg (17.74); the
  planet component differs from the delay-free truth by <=105 ppm, only at
  eclipse ingress/egress (the ~15 s delay). `phase_folded_lc` now takes the
  eclipse phase from juliet's computed `t_secondary` when there is no
  `t_secondary` posterior.
- The per-planet component (`return_components`) *is* set in the joint
  branch; the earlier note claiming otherwise was wrong.

**Code:** `utils.cowan_agol`, `utils.phase_curve_quantities` (per sample:
`fday`, `fnight` = fp - 2 C1, `fmax`, `fmin`, `amp`, `offset` of the maximum
before the eclipse and `offset_min` of the minimum before the transit,
degrees, positive = east; grid + parabolic refinement);
`BroadbandFit.phase_curve(suffix)`; `summary()` derived rows `fnight`,
`fmax`, `fmin`, `pcamp` (ppm), `pcoffset`, `pcoffmin` (deg) -> also
`Spectrum` columns; `t_secondary` added to `channel_priors()` geometry;
`LightCurves.bin_time(n | dt)` (means, gap-aware, `exptime` in meta) and
`bin_time=` / `supersample=` on `BroadbandFit` and `SpectroscopicFit`
(off by default; juliet supersampling with the bin span); 
`LightCurves.concatenate` (exposures of one visit -> one dataset and one
baseline; visit-wide `index`; `expstep` regressor for per-exposure steps).
Plots: `phase_folded_lc` already had phase-curve panels; fixed for fits
whose data miss the transit (empty transit window: juliet's `bin_data`
with step 0, NaN y-limits) via `_bin_data`, `_robust_ylim`.
Tests: 5 new (binning, prior checks, synthetic phase-curve fit + derived
quantities, `t_secondary` check, concatenation).

**WASP-43 b MIRI LRS (ERS 1366 obs 11).** The phase curve is three
exposures (`04103_00001`-`_00003`, 3072 integrations of 10.34 s each, 10
segments each); only exposure 1 had been downloaded (phases 0.29-0.74:
first eclipse and dayside, no transit, no nightside). Exposures 2-3 are
downloading (`download_wasp43_exp23.log`, ~28 GB).
Benchmark: Bell et al. 2024 (Nat. Astron., arXiv:2401.13027), white light
5-10.5 um, Eureka! v1: Fday 5752 +- 19, Fnight 1636 +- 37, peak-to-trough
4180 +- 33 ppm, offset 7.34 +- 0.38 deg east (7.34-9.60 across four
reductions); first 779 integrations trimmed, one exponential ramp, linear
in time, position and PSF width; two HGA moves (8.83 h, 17.66 h) removed.

Exposure 1 alone (`fits/fit_exp1_pc.py`; orbit and LD fixed to
literature, free `fp`, `C1`, `D1`, `t_secondary`, ramp, linear time, dy,
ds): **fp = 5749 +- 30 ppm** (Bell 5752 +- 19). Nightside 3494 +- 860 and
offset 33 +- 10 deg are unconstrained without the nightside, as expected.
Residuals show ingress/egress structure (fixed a/R*, b, Rp/R*; beta ~2.5
at 64-128 points); the full phase curve fits the geometry from the transit.

**Full phase curve (all 3 exposures, 2026-11-16;
`reduction/fits/fit_white_pc.py <order> 0 <trim> <optimal|box> <step|nostep>`).**
Exposures joined with `LightCurves.concatenate(common_channels=True)`,
5-10.5 um, first 779 integrations trimmed, ramp 1, linear time/dy/ds,
t0, a/R*, b, Rp/R*, q1, q2 free, t_secondary normal(+-10 min).

1. *Optimal flux, one baseline* (`white_pc_o1`, `_o2`): nightside 243 /
   -99 ppm, amplitude 5552 / 6092 ppm, beta_128 ~7. Cause: flux steps at
   the exposure boundaries (= Bell's "HGA move" times), -2.48 ppt at
   exp1->2 and +0.39 ppt at exp2->3, absorbed by the phase curve.
2. *The steps come from the optimal extraction*: box flux steps -0.46 and
   -0.01 ppt, background change < 0.01 ppt, trace shift < 0.001 px, QMAP
   mean equal to 0.08 ppt; the per-exposure PSF profiles differ by ~1%
   (|dP|/P ~ 10-11 ppt). Bell et al. see no step (the HGA moves "affect only
   a couple of integrations that we removed"). -> **jstark extraction
   issue**: the PSF is fitted per exposure and its normalisation mismatch
   rescales the optimal flux by up to 2.5 ppt. Candidate fix: one PSF
   model per visit (fit it from all exposures, or `load_model` of exposure
   1 for the others). Not done yet (needs the user's decision).
3. Results (second order preferred: dlnZ = +67 over first order on the
   same data; Bell: 16 sigma):

   | | optimal + expstep (o2) | box, no step (o2) | Bell+24 |
   |---|---|---|---|
   | Fday (= fp) | 5643 +- 20 | 5589 +- 20 | 5752 +- 19 |
   | Fnight | 1458 +- 54 | 1442 +- 45 | 1636 +- 37 |
   | amplitude | 4237 +- 52 | 4196 +- 41 | 4180 +- 33 |
   | offset (deg E) | 8.2 +- 0.7 | 7.6 +- 0.5 | 7.34 +- 0.38 |
   | C2 (ppm) | 104 +- 10 | 109 +- 9 | |
   | sigma_w / beta_128 | 237 / 1.54 | 256 / 1.67 | |

   Box fits: expstep coefficients -32 +- 16 and -74 +- 25 ppm, evidence
   prefers no step (dlnZ = +22). Trimming 30 integrations after each
   boundary changes nothing (first order: identical to < 0.2 sigma).
4. **Shape agrees, absolute level does not**: amplitude and offset match
   Bell; Fday and Fnight are both ~160-190 ppm lower (additive in planet
   flux, not a dilution: the amplitude is unchanged). Both eclipses agree
   locally (detrended data, +-0.25 h vs 0.75-1.25 h: 5458 +- 35 and
   5531 +- 35 ppm), so it is the eclipse depth itself. Exposure 1 alone with
   fixed literature geometry gave 5749 +- 30 (different geometry, ramp
   degenerate with the phase curve). Bell give no per-reduction Fday spread
   and note that long-wavelength transit depths differ between pipelines.
   Open; compare with the PRISM finding (our depths shallower than ERS by
   an additive pedestal, §5.6.1).
5. **One PSF per visit fixes the step** (2026-11-16,
   `reduction/scripts/extract_sharedpsf.py`: exposures 2-3 re-extracted with
   `load_model` of exposure 1's extraction model -> `reduction/sharedpsf/`;
   20 s per exposure). Boundary steps of the optimal flux: exp1->2
   -2.48 -> **-0.48 ppt**, exp2->3 +0.39 -> **+0.11 ppt**, now equal to the box
   steps (-0.45, -0.01). Second-order fits, trim 30:

   | | shared PSF, no step | shared PSF + expstep | Bell+24 |
   |---|---|---|---|
   | Fday | 5683 +- 19 | 5676 +- 19 | 5752 +- 19 |
   | Fnight | 1523 +- 47 | 1524 +- 54 | 1636 +- 37 |
   | amplitude | 4214 +- 43 | 4210 +- 51 | 4180 +- 33 |
   | offset (deg E) | 6.8 +- 0.5 | 8.8 +- 0.8 | 7.34 +- 0.38 |
   | sigma_w / beta_128 | 236 / 1.54 | 235 / 1.45 | |
   | lnZ | 54198.5 | 54185.7 | |

   expstep coefficients -37 +- 16, -100 +- 25 ppm; the evidence prefers no
   step (dlnZ = +12.8). lnZ is 163 above the per-exposure-PSF + expstep fit
   (same data, 8379 vs 8380 integrations). Differences from Bell: Fday
   -69 (2.6 sigma combined), Fnight -113 (1.9 sigma), amplitude +34 (0.6
   sigma), offset -0.5 deg (0.8 sigma); Bell's own offsets span 7.34-9.60 deg
   across four reductions. Optimal flux keeps its lower white noise
   (sigma_w 236 vs 256 ppm for box).
6. **Joint visit PSF** (`VisitExtractor`, default `shared_psf=True`;
   `reduction/visitpsf/`; extraction_notes §5.16). Steps -0.475 / +0.118 ppt
   (= box). Fits, trim 30:

   | | o2, no step | o2 + expstep | o1, no step | Bell+24 |
   |---|---|---|---|---|
   | Fday | 5646 +- 19 | 5638 +- 19 | 5555 +- 18 | 5752 +- 19 |
   | Fnight | 1457 +- 42 | 1452 +- 52 | 1552 +- 38 | 1636 +- 37 |
   | amplitude | 4239 +- 39 | 4237 +- 50 | 4051 +- 31 | 4180 +- 33 |
   | offset (deg E) | 6.6 +- 0.4 | 8.1 +- 0.7 | 8.8 +- 0.4 | 7.34 +- 0.38 |
   | beta_128 | 1.59 | 1.54 | 2.25 | |
   | lnZ | 54052.4 | 54035.3 | 53960.4 | |

   Second order preferred (dlnZ +92 over first order), no step preferred
   (+17; step terms -32 +- 16, -84 +- 24 ppm). Against Bell (o2, no step):
   Fday -106 (3.9 sigma combined), Fnight -179 (3.2 sigma), amplitude +59
   (1.2 sigma), offset -0.75 deg (1.3 sigma). **The absolute level depends
   on the extraction at the ~100 ppm level**: Fday 5589 (box), 5646 (joint
   PSF), 5683 (exposure 1's PSF); Fnight 1442-1523. The shape (amplitude
   4196-4239 ppm, offset 6.6-7.6 deg) is robust. Since an eclipse depth is
   a relative measurement within an exposure, a profile choice can only
   change it if the profile mismatch changes with flux or time (e.g. the
   brighter-fatter-like width change seen on PRISM, §5.6.1). Open.
7. **Definitions and band** (2026-11-16). *Definitions are identical*:
   juliet's planet flux is `fp + C1 (cos wt - 1) + D1 sin wt + C2 (cos 2wt
   - 1) + D2 sin 2wt` (wt = 0 at mid-eclipse); Eureka! (Bell's v1 fit,
   v0.10 `differentiable_models/SinusoidPhaseCurve.py` + `StarryModel.py`)
   uses `fp * [1 + A1 (cos phi - 1) + B1 sin phi + A2 (cos 2phi - 1) + B2
   sin 2phi]` (phi = 0 at t0 + P/2; starry planet amplitude set so the
   full-dayside flux is fp), i.e. C1 = fp A1. Fday = flux at mid-eclipse =
   fp in both; Fnight = model at phi = pi (transit) = fp - 2 C1 in both;
   amplitude = peak-to-trough in both. The only model difference is
   `T * (1 + vis Fp)` (juliet) vs `T + vis Fp` (Eureka!): 37 ppm, in transit
   only. A synthetic Eureka!-form light curve (our posterior medians, real
   sampling, sigma 238 ppm) fitted with jstark/juliet recovers Fday +3 +- 12,
   Fnight -6 +- 13, amplitude +10 +- 8 ppm, offset -0.1 +- 0.2 deg; the
   difference goes into the transit depth (-37 ppm) -> not the cause
   (`scratchpad .../w43/eform/eureka_form.py`).
   *The band is not the same*: the LRS slitless wavelength calibration was
   replaced in Aug/Sep 2024 (CRDS jwst_1263/jwst_1293; specwcs_0006 in
   jwst_1174 -> specwcs_0015 now). At the detector row the old solution
   calls 5.000 um the new one gives 5.174 um (+174 nm; 6.0 -> 6.083, 7.0 ->
   7.026, 8.0 -> 7.995, 10.5 -> 10.488). Bell (arXiv Jan 2024) used the old
   one, so their "5-10.5 um" is our **5.174-10.488 um**; our 5-10.5 adds the
   bluest columns (5-6 um holds 33% of the white-light flux, lowest
   contrast). Band sensitivity (joint PSF, o2): blue edge 5.0 / 5.1 / 5.2 ->
   Fday 5646 / 5688 / 5707. In Bell's band:

   | | optimal, joint PSF | box | Bell+24 |
   |---|---|---|---|
   | Fday | 5704 +- 20 (-48, 1.7 sigma) | 5645 +- 21 | 5752 +- 19 |
   | Fnight | 1520 +- 45 (-116, 2.0 sigma) | 1495 +- 50 | 1636 +- 37 |
   | amplitude | 4236 +- 41 (+56, 1.1 sigma) | 4200 +- 46 | 4180 +- 33 |
   | offset (deg E) | 6.7 +- 0.5 (-0.7, 1.2 sigma) | 7.6 +- 0.5 | 7.34 +- 0.38 |
   | sigma_w / beta_128 | 247 / 1.51 | 267 / 1.59 | |

   Matching the band halves the dayside gap (-106 -> -48 ppm) and reduces
   the nightside gap (-179 -> -116 ppm); everything is now within ~2 sigma
   of Bell's fiducial reduction. Optimal vs box still differ by ~60 ppm in
   Fday. Comparisons with pre-Aug-2024 MIRI LRS papers must convert the
   band edges to the current wavelength solution.
8. **Step 3: Bell et al.'s four reductions** (2026-11-17; Zenodo 10525170,
   `MIRI/published_bell2024/`; white light curves of Eureka! v1 (fiducial),
   Eureka! v2, SPARTA, TEATRO, 9216 integrations, 5-10.5 um old calibration;
   scripts in `MIRI/reduction/fits/step3/`).

   *(a) jstark fits of their light curves* (`fit_bell_white.py`; our
   fiducial model: second order, 779 + 30-after-boundaries trimmed, ramp,
   linear time / centroid / PSF width; SPARTA and TEATRO files have no
   centroid or width, so time only):

   | Light curve | Fday | Fnight | amplitude | offset (deg) |
   | --- | --- | --- | --- | --- |
   | Eureka! v1 (Bell published: 5752 +- 19, 1636 +- 37, 4180 +- 33, 7.34 +- 0.38) | 5751 +- 19 | 1648 +- 45 | 4169 +- 40 | 7.5 +- 0.4 |
   | Eureka! v2 | 5716 +- 20 | 1739 +- 47 | 4038 +- 41 | 9.8 +- 0.6 |
   | SPARTA | 5719 +- 18 | 1570 +- 30 | 4216 +- 27 | 7.2 +- 0.3 |
   | TEATRO | 5742 +- 23 | 1608 +- 57 | 4197 +- 53 | 8.5 +- 0.7 |
   | jstark (ours, joint visit PSF, Bell's band) | 5704 +- 20 | 1520 +- 45 | 4236 +- 41 | 6.7 +- 0.5 |

   **The fitter is not the cause**: jstark on Bell's Eureka! v1 light curve
   reproduces their published values (Fday -1, Fnight +12, amplitude -11
   ppm, offset +0.2 deg). The orbit agrees (b 0.656-0.664 for their light
   curves, 0.655 ours; Bell's Eureka! v1 priors i = 82.33 +- 0.20, a/R* 4.93
   +- 0.10 give b ~ 0.66). Their four reductions alone span 35 ppm in Fday
   and 170 ppm in Fnight (1570-1739). Ours: Fday 12 ppm below their lowest,
   Fnight 50 ppm below their lowest (SPARTA).

   *(b) Ratio light curves* (`ratio_bell.py`, ours / theirs, linear model
   with slow terms (linear or quadratic time, ramp, dy, ds, optional
   exposure steps) + dD transit + d(fp, C1, D1, C2, D2) x visible fraction;
   orbit and shapes from our fit). Optimal extraction, linear slow terms:

   | ours - | Eureka! v1 | Eureka! v2 | SPARTA | TEATRO |
   | --- | --- | --- | --- | --- |
   | dFday | -27 +- 18 | +2 +- 24 | +1 +- 20 | +16 +- 16 |
   | dFnight | -121 +- 21 | -185 +- 29 | -85 +- 23 | -83 +- 19 |

   Robust to the slow-term model: with a quadratic and/or exposure steps
   dFday stays -26 to +20 and dFnight -64 to -159 (larger errors). The
   ratio vs time (`ratio_vs_time.png`) shows a smooth ~cos-shaped variation
   over the orbit, minimum ~-100 ppm near the transit (where Fnight is
   measured), plus a -100 ppm mismatch in the first hour (ramp) and a
   +100-150 ppm bump decaying over ~1 h after the first exposure boundary.
   Our box flux shows the same nightside deficit (dFnight -77 to -179; its
   dFday is -36 to -79), so the slow variation is upstream of the
   extraction: Stage 1 or the cleaner's integration-level background
   subtraction (MIRI runs `background=True`: per-row median of the
   background pixels). The extraction-level `bkg` regressor is ~0 (the
   cleaner removed it), and the cleaner's background profile was not saved
   (rateints deleted), so that could not be tested here.

   **Conclusion.** Fday: ours agrees with all four reductions at the light
   curve level; the -48 ppm against Bell's fiducial number is within their
   own 35 ppm spread plus our fit. Fnight: ours is 50 ppm below the lowest of
   their reductions and 128 below their fiducial, from a slow (~orbit-long)
   variation in our light curve that is in our Stage 1 or cleaning, not in
   the fitter, orbit, band or extraction. Next test: our MIRI Stage 1 +
   cleaning with and without the integration-level background, and the
   official Stage 1 (item 9).

9. **Step 3b: MIRI Stage 1 test** (2026-11-18; `MIRI/reduction_stage1test/`,
   `stage1_test.py`, scripts and logs in `MIRI/reduction/fits/step3/`).
   Our Stage 1 and the official `calwebb_detector1` (jwst 3.0.0,
   `clean_flicker_noise` skipped, everything else at the CRDS defaults), all
   three exposures, then `ExposureCleaner` with `background=True` and
   `False`, then `VisitExtractor` (joint visit PSF). Bell's band, linear
   baseline, ratio method as in item 8.

   | Product | - Eureka! v1 | - Eureka! v2 | - SPARTA | - TEATRO |
   |---|---|---|---|---|
   | dFnight, jstark Stage 1 | -121 +- 21 | -185 +- 29 | -85 +- 23 | -83 +- 19 |
   | dFnight, official Stage 1 | -98 +- 18 | -162 +- 28 | -62 +- 21 | -61 +- 21 |
   | dFday, jstark Stage 1 | -27 +- 18 | +2 +- 24 | +1 +- 20 | +16 +- 16 |
   | dFday, official Stage 1 | -29 +- 16 | +2 +- 23 | -3 +- 18 | +13 +- 18 |
   | dTransit, jstark Stage 1 | +88 +- 25 | +79 +- 34 | +97 +- 28 | +172 +- 23 |
   | dTransit, official Stage 1 | +23 +- 22 | +18 +- 33 | +41 +- 25 | +102 +- 25 |

   - jstark Stage 1 + cleaner background reproduces the earlier products
     exactly (same numbers as item 8), so the comparison is like for like.
   - **Cleaner background: no effect.** `background=False` gives the same
     light curves to < 2 e-6 relative: the extractor's own local background
     (`BKG_O1`, now ~139 DN/s larger) absorbs what the cleaner no longer
     removes. Adding that full background time series as a regressor
     (`lin+bkg`) moves dFnight by <= 3 ppm (coefficient about -6000 ppm per
     unit relative background change, ~4 sigma, but not degenerate with the
     phase curve).
   - **Stage 1: about 25 ppm of the nightside, and the transit depth.**
     Direct ratio jstark/official Stage 1 (`ratio_stage1.py`, 177 ppm p2p,
     beta 1.00): dFday +7 +- 9, dFnight -25 +- 10 (linear) / -3 +- 21
     (quadratic), dTransit +63 +- 12 (both). The binned ratio is flat apart
     from the transit and a small bump after the first exposure boundary
     (`ratio_stage1_vs_time.png`). Official Stage 1 p2p is 3% lower
     (374 vs 387 ppm). Which of our MIRI steps deepens the transit by
     63 ppm is not yet traced.
   - **Exposure starts: 20-40 ppm.** Dropping a further 0.5 / 1 / 2 h after
     each exposure start (`TRIMH` in `ratio_bell.py`) moves dFnight against
     Eureka! v1 from -121 to -103 / -98 / -83 (ours) and from -98 to
     -92 / -88 / -75 (official); against SPARTA, official goes -62 to -41.
     The ratio to Bell's light curves is low in the first hour and dips
     after the second boundary, so the ramp handling (our fixed-tau
     exponential and 30-integration trims) leaks partly into the
     nightside.
   - **Bell's reductions among themselves** (`ratio_bell_pairs.py`):
     dFnight v1 - v2 -65 +- 20, v1 - SPARTA +34 +- 13, v1 - TEATRO +32 +- 20,
     v2 - SPARTA +95 +- 18, SPARTA - TEATRO +4 +- 25. Consistent with our
     pairwise numbers (ours - v1 minus ours - SPARTA = -36). The published
     nightside spread is ~100 ppm.

   **Conclusion.** The nightside gap against Bell's fiducial (Eureka! v1,
   -121 +- 21) splits into about -25 from our Stage 1, -20 to -40 from the
   exposure-start ramps, and about -60 to -75 that remains. The remainder
   is comparable to the spread among Bell's own reductions: with the official
   Stage 1 and 2 h extra trimming we are -41 +- 36 below SPARTA and
   -74 +- 28 below TEATRO. The cleaner's background subtraction is not
   responsible. Open: which Stage 1 step deepens the transit by 63 ppm
   (and the ~25 ppm nightside), and whether a free ramp timescale per
   exposure in the real fit reduces the exposure-start leakage (item 10
   answers the first).

10. **MIRI Stage 1 variants** (2026-11-18; `MIRI/reduction_stage1var/`,
    `run_variant.py`, `run_all.sh`, `compare.py`, `compare.log`). Starting
    from the jstark defaults, one step at a time is changed to match the
    official run: CRDS skipped `rscd` and `firstframe` and used the
    two-point jump at 4 sigma. Exposure 2 only (it holds the whole
    transit); each variant is cleaned and extracted (visit PSF on exposure
    2 alone). Transit-depth differences come from the ratio method
    (quadratic baseline, ramp, both products' dy/ds, cos/sin of the phase).

    | Variant | dTransit - ours, optimal | box | - official, optimal |
    |---|---|---|---|
    | official Stage 1 | -74 +- 13 | -64 +- 14 | 0 |
    | `rscd` skipped | **-55 +- 11** | **-50 +- 12** | +20 +- 6 |
    | `firstframe` skipped | 0 (identical) | 0 | +74 +- 13 |
    | jwst two-point jump (4 sigma) instead of `temporal_jump` | -7 +- 4 | -5 +- 4 | +66 +- 13 |
    | `consistent_saturation` skipped | 0 (identical) | 0 | +74 +- 13 |

    - **The cause is `rscd`.** In jwst 3.0 it is a group-flagging step: it
      sets the first `group_skip` groups to DO_NOT_USE
      (`jwst_miri_rscd_0026`, SLITLESSPRISM FASTR1: 5 in integration 1, 7 in
      later integrations). So our default fits groups 8-63, while the
      official run (and Bell's Eureka! v1: `skip_rscd True`,
      `skip_firstframe True` in `0_ECFs/Eureka_v1/S1_WASP-43b.ecf`) fits
      from group 1 or 2. The early groups rise too steeply (RSCD), and
      including them raises the rates by a flux-dependent fraction: ours /
      official is 0.986 at 50-100 DN/s, 0.990 at 300-1000 and 0.991 above
      1000 (exposure 2, segment 5). That non-proportional response makes
      the transit about 55 ppm shallower.
    - `firstframe` is redundant once `rscd` has flagged the first 7
      groups, and `consistent_saturation` changes nothing here: both give
      bit-identical light curves.
    - The jump algorithm is worth about 5-7 ppm. rscd + jump = -62 of the
      -74 ppm; the remaining ~12 ppm is within the errors.
    - Noise: fitting the early groups lowers the white-light p2p by 5%
      (360 vs 380 ppm), as with the official run.
    - Not yet settled: which depth is right. The early groups are
      distorted, so dropping them should give the more linear response,
      but our docstring notes (GJ 3929 F1500W) that the distortion only
      rescaled the flux there. Test: start the fit at group 10 or 12 (and 1)
      and see whether the depth converges by group 8.
    - **User decision 2026-11-18:** MIRI follows the official TSO
      configuration (`firstframe`, `rscd`, `refpix` skipped; `lastframe`
      kept). See item 11.
    - Peak memory per variant (main process) was 17-20 GB with
      `maximum_cores='4'`. Scripts that run Stage 1 with several cores must
      have an `if __name__ == '__main__':` guard; without it, the
      four-in-parallel first attempt re-spawned whole pipelines and crashed
      the machine.

11. **WASP-43 b rerun with the new MIRI defaults** (2026-11-18;
    `MIRI/reduction/rscdoff/`, `rerun.py`, `run_all.sh`). All three
    exposures through Stage 1 (`firstframe`, `rscd`, `refpix` skipped),
    `ExposureCleaner(background=True)` and `VisitExtractor` (joint visit
    PSF); white-light fit `fit_white_pc.py 2 0 30 optimal nostep rscdoff`
    with `WRANGE=5.174,10.488` (Bell's band), the same setup as the
    previous fiducial (`white_pc_o2_trim30_visitpsf_w5.174-10.488`).
    About 85 min for the reduction (27 min per exposure), 6 min for the fit;
    the main process peaked at 16.5 GB and free memory never went below 77%.

    | Quantity | Old defaults (rscd on) | New defaults | Bell Eureka! v1 (published) |
    |---|---|---|---|
    | Transit depth (ppm) | 24754 +- 53 | 24683 +- 48 | |
    | Fday (ppm) | 5704 +- 20 | 5696 +- 19 | 5752 |
    | Fnight (ppm) | 1520 | 1518 +- 39 | 1636 |
    | Amplitude (ppm) | 4236 | 4234 +- 35 | 4180 |
    | Offset (deg) | 6.7 | 6.70 +- 0.39 | 7.34 |
    | White noise sigma_w (ppm) | 247 +- 5 | 233 +- 4 | |

    Ratio method (Bell's band, linear baseline, item 8 setup):

    | jstark (new) - | dFday | dFnight | dTransit |
    |---|---|---|---|
    | Eureka! v1 | -32 +- 16 | -100 +- 19 | +22 +- 22 |
    | Eureka! v2 | -2 +- 23 | -163 +- 28 | +16 +- 33 |
    | SPARTA | -6 +- 18 | -64 +- 21 | +39 +- 25 |
    | TEATRO | +10 +- 17 | -63 +- 21 | +103 +- 25 |
    | jstark old defaults | -9 +- 8 | +23 +- 10 | -63 +- 11 |

    - **Transit depth:** 63 ppm shallower than with the old defaults, and
      now consistent with Eureka! v1, v2 and SPARTA (+16 to +39 ppm, within
      1.6 sigma; was +79 to +97). TEATRO is still +103, but TEATRO is
      itself 71-87 ppm deeper than the other three (item 9).
    - **White noise:** 5.6% lower (247 -> 233 ppm), from the 7 extra groups.
    - **Phase curve:** essentially unchanged in the full fit (Fday -8,
      Fnight -2 ppm, amplitude and offset the same). In the fixed-shape
      ratio, dFnight vs the old reduction is +23 +- 10, and the gap to
      Eureka! v1 shrinks from -121 to -100 ppm (to SPARTA/TEATRO: -85 -> -64).
      The full fit refits orbit, LD and ramp, so the two agree within the
      fit's 40 ppm nightside error.
    - So the rscd change fixes the transit depth and the noise, but not
      the nightside. The rest of the nightside gap is still the
      exposure-start ramps (item 9, 20-40 ppm) plus inter-reduction spread
      (Bell's own nightsides span ~100 ppm).

### 5.8 WASP-39 b NIRSpec G395H, full exposure (2026-11-20)

Reduction and extraction: extraction_notes.md §5.17 (optimal extraction with
PSF epochs, `psf breaks=[269]` at the mirror tilt). Scripts in
`/Volumes/Jayshil SSD/JWST/jwst-ers/NRSG395H/Outputs/scripts/`
(`fit_white_g395h.py`, `fit_spec_g395h.py`, `compare_spec_g395h.py`);
fits in `Outputs/fits/`, comparisons in `Outputs/comparison/`.

**Set-up (user decisions 2026-11-20).** One fit per detector. Integrations
0-14 (settling) and 269 (tilt during the integration) removed. Systematics:
linear in time + `dy` + a step at the tilt, which is the extraction's
`epoch` regressor (0 before 269, 1 after). No ramp. `ds` and `dx` left out
(transit-shaped, §5.2). Optimal extraction (box gives the same light curves
here, extraction_notes §5.17). P = 4.0552842518 d and e = 0 fixed; t0, a/R*,
b, Rp/R*, q1, q2, jitter free. White light over the ERS bin range of each
detector (2.725-3.716 / 3.829-5.172 um). Dynamic dynesty, 8 threads: 40 s.

| | NRS1 | NRS2 | ERS benchmark (joint fit) |
| --- | --- | --- | --- |
| Rp/R* | 0.14631 +- 0.00016 | 0.14664 +- 0.00020 | 0.14613 / 0.14678 |
| depth (ppm) | 21,405 +- 47 | 21,503 +- 58 | |
| b | 0.4466 +- 0.0049 | 0.4466 +- 0.0062 | 0.4498 |
| a/R* | 11.421 +- 0.024 | 11.393 +- 0.032 | 11.390 |
| t0 - ERS (s) | -3.4 +- 1.4 | +5.1 +- 1.9 | |
| q1, q2 | 0.040, 0.19 | 0.032, 0.18 | 0.049, 0.11 / 0.024, 0.31 |
| step at the tilt (standardised coefficient) | -483 +- 60 ppm | -428 +- 76 ppm | |
| p2p / err; p2p / (err + jitter) | 1.15; 0.92 | 1.25; 0.98 | |
| jitter (ppm) | 74 | 104 | |

- Rp/R* within ~1 sigma of the benchmark on each detector, geometry too.
- The white light scatters 15-25% above its propagated errors (native bins
  only 4-6%): a noise common to the columns, absorbed by the jitter.
- The two detectors' t0 differ by 8.5 +- 2.4 s (3.5 sigma), although they
  record the same photons at the same times. Not followed up. The tilt sits
  in the flat bottom (mid-transit is integration ~259, egress ends ~337).
- **Integration 270** is a +721 / +729 ppm outlier on both detectors
  (5-7 sigma): the tilt had not fully settled. The ExoTiC-JEDI light
  curves (Alderson et al. 2023) drop 269-271. Refitting without 269-271
  changes the depths by +21 / +16 ppm (0.4 sigma), the geometry not at all,
  and lowers the jitter (74 -> 65, 104 -> 98 ppm). Kept as the user
  specified (only 269 dropped); for future tilts drop the tilt integration
  and the next one or two.

**Spectra in the ERS bins** (`fit_spec_g395h.py`): orbit fixed to each
detector's white-light medians (`channel_priors(fix=(P, t0, a, b, ecc,
omega))`), per bin Rp/R*, q1, q2 free, linear time + dy + epoch, jitter.
R100 (30 + 29 bins): 260 s per detector with 8 processes (~70 s per bin).

| R100 | offset ours - ref (ppm) | rms after offset | chi2_red | error ratio ours/ref | > 3 sigma |
| --- | --- | --- | --- | --- | --- |
| NRS1 vs ERS | +20 | 50 | 0.17 | 0.98 | 0 |
| NRS2 vs ERS | -40 | 40 | 0.08 | 1.00 | 0 |
| NRS1 vs Alderson+23 weighted average | -142 | 56 | 0.37 | 1.09 | 0 |
| NRS2 vs Alderson+23 weighted average | -50 | 47 | 0.13 | 1.11 | 0 |
| 11 Alderson+23 reductions: range of offsets | NRS1 -357 to +77, NRS2 -211 to +151 | 47-130 | | 0.68-1.39 | 0 |

- The spectrum reproduces the benchmark bin by bin (SO2 at 4.05 um, CO2 at
  4.3 um, H2O on NRS1) with the same error bars, and sits inside the spread
  of the 11 published reductions. (Alderson reductions rebinned onto R100 by
  inverse-variance weighting of their 10-pixel bins.)

Native ERS bins (`bins_scale1`, 488 + 675 bins of ~3 columns; same setup;
3312 s + 4304 s with 8 processes, ~55 s per bin; no failed bins). Plot and
numbers: `Outputs/comparison/spectrum_bins_scale1.png/.json`.

| native bins | offset ours - ERS (ppm) | rms after offset | chi2_red | error ratio ours/ERS | > 3 sigma |
| --- | --- | --- | --- | --- | --- |
| NRS1 vs ERS (fitted LD) | +40 | 235 | 0.09 | 0.96 | 0 |
| NRS2 vs ERS (fitted LD) | +3 | 287 | 0.10 | 0.80 | 0 |
| NRS1 vs ERS (fixed LD) | +64 | 215 | 0.13 | 1.03 | 0 |
| NRS2 vs ERS (fixed LD) | -12 | 297 | 0.12 | 0.81 | 0 |

- chi2_red ~0.1: the two reductions share photons and errors, so their
  differences are much smaller than the errors would imply (as at R100).
  The few large differences are ERS outliers with large errors (e.g. 3.06 um).
- **Error bars on NRS2.** Ours are 9% smaller than ERS at 3.96 um,
  rising to 28% at 5.04 um. On NRS1 they are 3-13% smaller. The light
  curves are not the cause: the p2p scatter of our native-bin light
  curves matches the ERS light curves to within 1% at every wavelength.
  Limb darkening is not the cause either: the ERS fixed-LD errors are
  almost the same as its fitted-LD errors. Check against a fixed-LD
  least-squares floor (residual rms, depth + offset + slope + step, no
  dy):

  | wavelength | rms/err | floor | ours | ERS fitted LD |
  | --- | --- | --- | --- | --- |
  | NRS1 2.82-3.62 um | 1.02-1.04 | 223-292 | 1.08-1.10 x floor | 1.12-1.24 x |
  | NRS2 3.96 um | 1.05 | 293 | 1.07 x | 1.17 x |
  | NRS2 4.51 um | 1.05 | 410 | 1.06 x | 1.33 x |
  | NRS2 5.04 um | 1.05 | 557 | 1.06 x | 1.47 x |

  Our errors are what white noise at the measured scatter gives,
  plus the LD and systematics freedom. Our residuals show no red noise
  (beta_32 ~0.75-1). The ERS errors grow with wavelength beyond that
  level. That points to inflation on their side (red-noise or
  systematics treatment we cannot see from the released tables), not to
  underestimated errors here. At R100 the two error sets agree (0.98-1.00).
  Not followed up.

### 5.9 Other observatories and radial velocities in `BroadbandFit` (2026-11-23)

**Decisions (2026-11-22/23).** No TESS or CHEOPS readers and no downloads:
the user prepares the data, `LightCurves.from_table` reads them. Roles of
regressors (linear, GP, non-linear) are decided in `BroadbandFit` as before.
`groups` is optional. RVs in a basic version (no GP, no linear regressors).
TTVs left out. `SpectroscopicFit` is not changed. Validation on WASP-39 b
only. Phase-folded plots are not changed: `instruments=` selects one bandpass
and kind of event (checked on a TESS transit + eclipse-only fit; only
`one_plot=False` without `instruments` puts an eclipse in a transit window).

**Implementation.**
- `LightCurves.from_table`, `around`, `clip(protect=)`; `check_bjd` (error in
  `BroadbandFit` for times that are not full JD; BTJD was accepted silently
  before). `clip` is unchanged by default but warns: 31 integrations of 30
  min removed all 35 in-transit points of a synthetic TESS sector.
- `groups`: `check_groups`, `expand_group_priors` (a group in a prior name is
  one shared parameter; overlaps and transit times refused),
  `expand_group_options` (per-dataset options; `nuisance` full names go to
  every member). Same fingerprint as the full names.
- `non_linear`: `NonLinear` wraps the user function (ramp x function when
  multiplicative; juliet has one function per dataset; additive + ramp
  refused; aliases for group names). Called once at construction with prior
  centres. Fingerprint: source text, simple state, regressor values.
  Regressors saved in `jstark_nonlinear.npz`; `load(non_linear=)` needs the
  functions. Planet-only plots switch the functions off
  (`_without_non_linear`).
- `ExpRamp` finds parameters shared by several datasets (`ramptau_V1_V2`
  would have failed in juliet with a KeyError).
- `stray_priors` warning. juliet ignores priors of absent datasets (ten kinds
  tested, free and fixed: Rp/R* unchanged); free ones are unused dimensions and
  prior-only `Spectrum` columns. Not dropped, to keep fingerprints.
  `channel_priors(datasets=)` leaves them out.
- `supersample=(n, exposure time)`; warning for points of >= ~10 min.
- `rvs`: passed to juliet as `t_rv`, ...; `mu` and `sigma_w` priors added,
  `rv_intercept` fixed to 0 with a trend, `ta` = middle of the RV times;
  `rv_residuals`, `rv_diagnostics`, `julietPlots.plot_rv`; `radvel` added to
  the `fitting` extra. `summary` derives `ecc`, `omega` for all
  parametrisations, and the inclination now uses them (with
  `secosomega`/`sesinomega` it assumed e = 0).
- `BroadbandFit.load` failed (KeyError `t_secondary_p1`) for fits made with
  `light_travel_delay=True`: the options that change the model are now stored
  (`model_options`; older records are parsed) and passed on.

**Checks.** 13 new tests; the fingerprint of a deterministic fit is pinned to the
value of the 2026-11-22 code, and the G395H white-light fits and 59 R100 bins
still match their stored fingerprints. Noise-free RVs: juliet's model within
0.03 m/s of a sine; eccentric two-planet RVs recovered (truth from an
independent Kepler solver). CHEOPS `BJD_TIME` (DRP, PIPE) is BJD_TDB to 0.01 s
(recomputed with astropy), `MJD_TIME` is TT.
Synthetic NRS1 + NRS2 + 2 TESS sectors + CHEOPS: 33-37 free parameters, 6-9
min on 4 processes (static dynesty, rwalk), recovered within 2.2 sigma.

**WASP-39 b** (`Outputs/scripts/fit_joint_g395h_tess_rv.py`,
`compare_joint.py`; `NRSG395H/Outputs/joint/`). A: jstark NRS1 + NRS2 (data and
systematics of 5.8); B: + TESS 51 and 91 (SPOC PDCSAP 2 min, quality 0, +-0.35
d around 3 + 5 transits, Matern GP in time with shared hyperparameters); C: +
the ERS RVs (CORALIE 9, SOPHIE 8). ERS priors on the orbit and RVs. Dynamic
dynesty, rwalk, 8 processes: 150 / 532 / 617 s (20 / 29 / 34 parameters).
Reference: the ERS joint fit (all JWST modes, TESS 51, NGTS, RVs). The data
overlap, so differences in combined sigma are only indicative.

| | ERS joint | A | B | C |
| --- | --- | --- | --- | --- |
| P (d) | 4.0552843 +- 36e-7 | prior (7e-4) | 4.0552770 +- 9e-7 | 4.0552770 +- 9e-7 |
| t0 - 2459791.612 (d) | 68.4e-6 +- 9 | 63e-6 +- 14 | 62e-6 +- 14 | 61e-6 +- 13 |
| a/R* | 11.390 +- 0.012 | 11.411 +- 0.019 | 11.411 +- 0.019 | 11.410 +- 0.019 |
| b | 0.4498 +- 0.0022 | 0.4466 +- 0.0038 | 0.4467 +- 0.0037 | 0.4468 +- 0.0037 |
| Rp/R* NRS1 | 0.14613 +- 0.00010 | 0.14620 +- 0.00013 | 0.14621 +- 0.00012 | 0.14621 +- 0.00013 |
| Rp/R* NRS2 | 0.14678 +- 0.00011 | 0.14683 +- 0.00014 | 0.14683 +- 0.00014 | 0.14684 +- 0.00014 |
| Rp/R* TESS | 0.14364 +- 0.00071 | | 0.14363 +- 0.00072 | 0.14365 +- 0.00071 |
| K (m/s) | 36.2 +- 4.4 | | | 36.8 +- 5.4 |

- Geometry, depths, limb darkening and RV terms agree within 1 sigma. TESS
  and the RVs do not change the geometry (A = B = C): JWST sets it; TESS adds P
  and its own depth.
- **K error**: ours is larger than ERS's for the same RVs; an independent
  emcee fit of the RVs alone gives 36.6 +- 5.3. Why ERS's is smaller is not known.
- **P is 2.0 sigma below ERS**, and 3.6 sigma below Kokori et al. (2023;
  4.05528043 +- 3.2e-7, consistent with their T0 and our t0). It comes from the
  sector 91 transits. `check_tess_times.py` (per-transit times on a grid, linear
  baseline, no GP) gives the same: 4.0552768 +- 9e-7, so it is in the data as
  we used them and not from the joint fit. The sector 91 times scatter more than
  their errors (chi2 14.4 for 4; epoch 246 is 283 s early against the ERS
  ephemeris, epoch 247 78 s), although transits injected out of transit scatter
  as the errors say (36 vs 38 s). Not explained; the quoted error on P is
  probably too small by about the square root of 14.4 / 4 (1.9).
- The GP timescale differs from ERS (0.51 vs 1.56 d, 2 sigma): other data
  (sector 91, only the windows around transits).

### 5.10 JAX models and samplers, Phase 0: the model (2026-11-27)

**Decisions (2026-11-27).** An optional JAX backend next to juliet (juliet stays
the default). It is chosen by the sampler name: `'blackjax-nuts'`,
`'blackjax-nss'` (nested slice sampling, gives `lnZ`), `'numpyro-nuts'`
(underscores accepted), so more samplers can be added later; the rest of the
front end is unchanged. GPs with tinygp (pure JAX: jit, grad and vmap; its
quasiseparable `SHO` equals celerite1's SHO log-likelihood exactly with
sigma^2 = S0 w0 Q, and juliet's approximate Matern, celerite1 `eps=0.01`, is
reproduced exactly by `quasisep.Celerite(a=s^2, b=s^2 w/eps, c=w, d=eps)`,
w = sqrt(3)/rho; celerite2 0.3.3 works under jit and grad but has no vmap
batching rule). **No juliet post-processing of JAX fits**: results and plots
of a JAX fit use the JAX model. Result methods and plots stay one code base;
the few calls that evaluate a model go through a small model interface with a
juliet and a JAX implementation. CPU speed tests are reported but do not
decide anything (JAX aims at GPUs); a GPU benchmark script is to be written
for later. Phases: 0 model; 1 samplers and `BroadbandFit`; 2 GPs, RVs, NSS,
light-travel delay; 3 plots; 4 `SpectroscopicFit` with vmap; 5 real data;
6 docs and GPU script. Other limb-darkening laws matter less than the
quadratic one (user).

**Environment** (`jaxoplanet`): jax 0.12.1, jaxoplanet 0.2.0, numpyro 0.22.0,
blackjax 1.7.1, tinygp 0.3.1, celerite2 0.3.3; jstark installed editable
(`pip install -e . --no-deps`). pytest there needs `-o addopts=""`
(pytest-doctestplus is not installed).

**Code** (`jstark/fitting/jaxfit/`; importing it switches on float64):
- `priors.py`, `PriorSet`: juliet's distributions as its nested samplers draw
  them (the prior transforms; juliet's MCMC log-prior of `beta` has a wrong
  bounds check, its `exponential` is a gamma with shape `a`). Logistic maps
  for bounded priors, `exp` for positive ones, `mu + sigma z` for normal
  ones; log-densities, Jacobians, unit draws, starting points.
  Hyperparameters are an `(ndim, 4)` array, so batched fits can have
  different priors.
- `orbit.py`: batman's geometry: `t0` = conjunction, periastron time from
  the conjunction phase, `jaxoplanet.core.kepler` (batman's circular branch
  below e = 1e-5), separation 100 on the half-orbit without an event,
  `t_secondary` <-> conjunction, `a` from `rho` (juliet's G), r1/r2.
- `planet.py`: quadratic/linear with `jaxoplanet.core.limb_dark` (order 20:
  <= 1e-4 ppm from order 60; 30% faster than the default 40); other laws
  (square-root, logarithmic, exponential, power-2, 4-parameter) by
  Gauss-Legendre over the stellar radius (32 nodes after a cosine map, <=
  4e-4 ppm from 200 nodes); uniform-disk eclipse `1 + fp * visible
  fraction`; juliet's Cowan & Agol and sine phase curves.
- `model.py`, `JaxModel(.from_fit(BroadbandFit))`: juliet's name resolution
  (`*_iname`: LD, theta, sigma_w, mdilution, mflux, p, fp, C1-D2,
  phaseoffset; shared and group names), model flags (transit / eclipse /
  joint, phase curves), the joint transit + eclipse branch (`transit *
  (eclipse - fp)`, or `transit * (1 + pc * visible)` with a phase curve; the
  sine phase curve uses the conjunction implied by `t_secondary`, as batman
  leaves it), dilution and `mflux`, linear terms, ramps, your non-linear
  functions (written with `jax.numpy`; group aliases), supersampling,
  `sigma_w`. juliet's rejected draws (b > 1 + p, cos i >= 1, e > ecclim,
  periods not increasing in planet order) are -inf. Raises
  `NotImplementedError` for GPs, RVs and `light_travel_delay` (Phase 2) and
  for TTVs, catwoman, Lambert/kelp and global models (not used by jstark).
  Data are a pytree argument of the log-density (for vmap later).

**Checks** (`jstark/tests/test_jaxfit.py`, 13 tests, ~30 s): log-likelihood,
full model and variances against juliet's own `model` object at 40 prior
draws per configuration (same draws rejected; log-likelihood within the bound
the model difference implies, since prior draws sit hundreds of sigma from
the data): two transit visits (linear terms, ramp); rho, free e/omega,
square-root and power-2 laws; eclipses with secosomega and supersampling;
joint transit + eclipse with Cowan & Agol and sine phase curves (and without);
two planets with r1/r2 and ecosomega; non-linear functions (multiplicative
with a 2-term ramp, additive with two regressors), groups and shared
parameters; linear, logarithmic and 4-parameter laws; priors vs scipy;
gradients vs finite differences.

| Comparison | max model difference |
| --- | --- |
| quadratic and linear vs juliet (all configurations) | < 0.1 ppm (0.05 ppm with free e, omega: batman's Kepler tolerance 1e-7) |
| eclipses, phase curves vs juliet | < 0.1 ppm |
| other laws vs juliet's batman | 2.8-7.9 ppm |
| other laws vs batman with `max_err=0.01` | <= 0.15 ppm |

- **juliet's batman is the less accurate one for non-polynomial laws**: batman
  fixes its integration step (`fac`) once, at juliet's initial dummy
  parameters (u = 0.1, 0.3, rp = 0.1); its 1 ppm `max_err` does not hold at
  other coefficients. With a fresh step both agree to 0.15 ppm.
- **batman's exponential law** is off by up to 30 ppm near the limb, where the
  profile diverges (batman stops the integral short of r = 1); the JAX value
  equals scipy `quad` to 1e-11. Same profile and normalisation otherwise.
- batman failures at isolated points: the quadratic model at z = p exactly
  (`ellpic_bulirsch` does not converge, 21-25 ppm), the eclipse at z = 1 - p
  exactly (NaN); the JAX models are finite and correct there.

**Speed on this CPU (M4 Pro, for the record only).** Transit with linear
terms and jitter, 12 free parameters: one gradient inside a compiled loop
0.12 ms (465 points) and 2.0 ms (21,500 points); per chain with 8 chains
vmapped 0.07 / 1.5 ms. One juliet log-likelihood (no gradient): 0.02 ms /
0.47 ms (with changing parameters). The transit model dominates (value 0.36
ms, value + gradient 1.7 ms at 21,500 points); with a ramp and 5-fold
supersampling 9.7 ms per gradient. So on CPU a NUTS gradient costs 3-5x a
juliet evaluation; whether a fit is faster depends on the number of
evaluations (dynesty vs NUTS), the dimension, vmap, and on a GPU.

### 5.11 JAX backend, Phase 1: samplers and `BroadbandFit` (2026-11-27)

**Front end.** `BroadbandFit.fit(sampler='blackjax-nuts')` or
`'numpyro-nuts'` (underscores work; `jax_sampler_name`), everything else as
before; every other sampler name goes to juliet, which never imports jax.
JAX options: `nchains` 4, `nwarmup` 1000, `nsamples` 1000 per chain, `seed`
0, `target_accept` 0.9, `max_tree_depth` 10, `dense_mass` True,
`chain_method` (`'vectorized'`: all chains in one vmapped computation, or
`'sequential'`), `init` (`'map'`, `'prior'`, `{name: value}`), `nstart` 8,
`progress`; juliet's model options `pl`, `pu`, `ecclim` apply. Other juliet
options (`nthreads`, `n_live_points`, dynesty keywords) are ignored with a
warning, so switching only the sampler name works. Unknown JAX options,
`blackjax-nss` (Phase 2) and features the JAX model lacks (GPs, RVs,
light-travel delay) raise before `overwrite=True` empties the folder.
`SpectroscopicFit.run(sampler='blackjax-nuts')` works with `nproc=1` (bins
one after the other; `nproc > 1` raises, each JAX process already uses all
cores; vmap over bins is Phase 4). `plots()` raises for JAX fits (Phase 3).

**Code.**
- `jaxfit/samplers.py`, `run_sampler(model, sampler, **options)`: L-BFGS
  (scipy, JAX gradients) from the prior centre and `nstart - 1` prior draws;
  the Laplace covariance at the best maximum (Hessian eigenvalues made
  positive and floored at 1/10^2, so it always exists); chains start at
  draws from it. Sampling is in whitened coordinates `u = L^-1 (z - z_map)`
  (`L L^T` the Laplace covariance; an affine map, so the posterior is
  unchanged). blackjax: `window_adaptation` + `nuts` in `lax.scan`, chains
  vmapped; numpyro: `NUTS(potential_fn=...)`, `chain_method` passed on.
  blackjax's progress bar needs its `progress` extra (jaxtap, not installed
  here); without it one line per stage is printed.
- `jaxfit/results.py`, `JaxResults`: chains `(nchains, ndraws, ndim)`, the
  log-likelihood of every draw, sampler statistics; `posteriors` is juliet's
  dict (`posterior_samples` + `loglike`), so `posteriors`, `values`,
  `summary`, `channel_priors`, `phase_curve` are unchanged code;
  `diagnostics()` (rank-normalised split R-hat, bulk and tail ESS from
  `blackjax.diagnostics`; divergences, tree-depth hits, acceptance, step
  sizes, times) and `report()` (warnings for R-hat >= 1.01, ESS < 400,
  divergences, depth hits; `fit` turns them into Python warnings).
- The model interface (`fitting/evaluators.py`: `data`, `lc(name,
  nsamples, components)`, `rv_data`, `rv`, `index`): `JulietEvaluator`
  (juliet's `results.lc.evaluate`) and `jaxfit.JaxEvaluator` (the JAX model
  at `nsamples` random posterior draws, fixed seed, vmapped in chunks of <=
  2e6 model elements; median model and `planet` = juliet's `transit`
  component, `1 + sum(planet_i - 1)` before dilution, checked in juliet's
  code). `BroadbandFit.evaluator` picks one; `residuals` and `rv_residuals`
  use it, and with them `common_mode`, `channel_common_mode`,
  `save_residuals`, `diagnostics`.
- Folder of a JAX fit: `jstark_fit.json` (as before), `jstark_jax_inputs.npz`
  (t, y, e, X, ramp_x, N, index per dataset), `jstark_jax_priors.json`
  (complete priors in order), `jstark_jax_posteriors.npz`,
  `jstark_jax_diagnostics.ecsv`, and the usual summary/diagnostics/residual
  files. `overwrite` works as for juliet (same fingerprint: loaded, also with
  other sampler settings; a run of the other JAX sampler is replaced).
  `BroadbandFit.load` rebuilds the JAX model from these (non-linear
  functions passed again, as for juliet).

**Warmup needs whitening (found here).** The first runs took > 10 min for 600
points: every draw hit the maximum tree depth (1023 steps) at acceptance 0.94
with step sizes ~0.003. A Gaussian with the exact posterior covariance did the
same, so the cause is the adaptation, not the model: blackjax's (and numpyro's)
Stan-style mass-matrix regularisation, `(n/(n+5)) var + 1e-3 * 5/(n+5)`,
assumes parameters of unit scale. Unconstrained widths here go down to 2e-5
(the linear coefficients, `mflux`), so their adapted variances came out up to
8e4x too large. Sampling in the whitened coordinates fixes it: 31 steps per
draw, 28 s instead of > 10 min.

**Sampler settings.** 600-point transit with a ramp and a linear term (11
free parameters), blackjax, 4 x 1000 draws:

| settings | time | divergences | ESS bulk (b, a) | max R-hat |
| --- | --- | --- | --- | --- |
| target_accept 0.8, diagonal mass | 54 s | 39 | ~120 | 1.037 |
| target_accept 0.9 | 83 s | 0 | ~440 | 1.012 |
| target_accept 0.95 | 101 s | 0 | ~500 | 1.007 |
| target_accept 0.9, dense mass (default) | 84 s | 0 | ~1200 | 1.006 |

The divergences were spread over the bulk of the posterior (b 0.2-0.48), not
at a prior edge: transit models curve sharply where points cross the contacts.

**Against dynesty** (same data, juliet `dynamic_dynesty`, `nthreads=1`):
dynesty 718 s, blackjax 97 s, numpyro 85 s. Medians agree within 0.16 sigma
for all 11 parameters (both samplers; R-hat <= 1.007, 0 divergences, min ESS
720 / 378). The NUTS widths of b and a were 7-8% (68% intervals) and 12-20%
(1-99%) smaller than dynesty's, so a third, gradient-free sampler decided:
emcee (64 walkers x 60,000 steps, 10,000 discarded, autocorrelation time
~750) on the same JAX log-posterior in parameter space, once started from
NUTS draws and once from dynesty draws. The two emcee runs agree to <= 2.3%
in every 68% width, so their pooled draws are the reference:

| ratio to emcee (68% / 95% width) | p | b | a | q1 | q2 | t0 |
| --- | --- | --- | --- | --- | --- | --- |
| blackjax-nuts | 0.97 / 0.97 | 1.00 / 0.96 | 1.00 / 0.98 | 0.98 / 0.97 | 0.99 / 0.97 | 1.01 / 1.02 |
| numpyro-nuts | 1.01 / 1.02 | 1.00 / 0.99 | 0.99 / 1.00 | 1.02 / 1.06 | 1.02 / 1.05 | 0.98 / 0.97 |
| juliet dynamic_dynesty | 1.03 / 1.05 | 1.08 / 1.13 | 1.07 / 1.12 | 0.96 / 1.01 | 0.95 / 0.96 | 0.97 / 1.00 |

The NUTS posteriors are right to the 2-3% their ESS allows; this dynesty run
(juliet's defaults) is the one off, too wide along the b-a degeneracy. One
run only, so not yet a statement about dynesty in general.

**Tests** (`jstark/tests/test_jaxfit.py`, now 19, ~2.3 min): sampler names;
both samplers recover a batman transit (p, b, a within 4 sigma) with short
chains; summary, residuals (planet component, detrended, index),
diagnostics; `load` gives identical posteriors and models; the same fit is
loaded, not re-run (another seed), `overwrite=True` runs again; juliet
options warn; unknown options, `blackjax-nss` and a GP fit raise (the
earlier fit's files stay); `init='prior'` / given values with vectorized and
sequential chains; a NumPy operation on a parameter in a non-linear function
raises a `TypeError` that says to use `jax.numpy` (NumPy on the data alone
is fine); `SpectroscopicFit` with `blackjax-nuts` over two bins (and
`nproc=2` refused). `test_fitting.py` (juliet) still passes with the
refactored `residuals` / `rv_residuals` (35 tests).

### 5.12 JAX backend, Phase 2: GPs, RVs, light-travel delay, nested sampling (2026-11-27)

**Code.**
- `jaxfit/gp.py`: juliet's light-curve kernels in tinygp, chosen by the
  names of the `GP_` priors as juliet chooses them: celerite `Exp`,
  `Matern`, `MaternExp`, `SHO`, `QP` (juliet's `RotationTerm`), `DoubleSHO`,
  `MaternSHO` (quasiseparable, O(N), sorted 1D input), george `SE` and `M32`
  (multi-dimensional) and `ExpSineSquaredSE` (dense, exact). The jitter is
  on the diagonal with the errors, as juliet's `JitterTerm` / george
  white noise. The residual model is juliet's `GP.predict(residuals, X)`:
  computed as `r - D (K + D)^-1 r` because tinygp 0.3.1 cannot `condition`
  sums of quasiseparable kernels (an assertion in its solver).
- `JaxModel`: GP likelihood per dataset; radial velocities (offsets
  `mu_<inst>`, jitters, Keplerians with radvel's conventions and juliet's
  sorted `K_p<n>` numbering, `rv_slope` / `rv_quad` about `ta`, juliet's
  rejections); `light_travel_delay` with `stellar_radius` (juliet:
  `t_secondary` from `t0` with batman, eclipse and phase curves at `t - dt`,
  dt the line-of-sight position relative to the transit over c, Bell's
  `KeplerOrbit` with Omega = 270 deg; the transit at `t`).
- Fixed from Phase 0: juliet **sorts** the planet numbers
  (`numbering.sort()`); the JAX model took them in the order of the priors
  (only different with `p2` priors before `p1`, and only for the period
  ordering check).
- `blackjax-nss`: nested slice sampling (Yallup, Kroupa & Handley 2026) on
  the unconstrained z with the z-space prior density (Jacobian included,
  normalised, so lnZ is that of juliet's priors); prior draws as live
  points; stops at dynesty's criterion `log(Z + Z_live) - log Z < dlogz`;
  lnZ and its error from 200 simulated volume sequences; posterior
  resampled from the weighted dead points (default size = ESS). Options:
  `nlive` 500 (juliet's `n_live_points` is accepted), `num_delete` nlive/10,
  `num_inner_steps` 5 x dimension (blackjax warns that ~dimension biases
  lnZ upward above 10 dimensions), `dlogz` 0.01, `max_steps`, `nsamples`.
  `summary()` carries lnZ as for dynesty; `diagnostics()` / `report()` give
  ESS, dead points, steps and a warning if `max_steps` stopped it.
- `JaxEvaluator`: light-curve models include the GP prediction (`gp`
  component too); `rv_data` / `rv` for `rv_residuals` and `rv_diagnostics`.
  `load` restores the RVs (stored in `jstark_jax_inputs.npz`).

**Checks against juliet** (`test_jaxfit.py`, now 35 tests, ~4 min): every
kernel at juliet's own residuals (20 prior draws each):

| kernel | lnL, relative | GP prediction / std(r) |
| --- | --- | --- |
| Exp, SHO (Q 0.2-5), QP, DoubleSHO | <= 1e-14 | <= 1e-14 |
| SE (2D), M32, ExpSineSquaredSE vs george's exact solve | <= 2e-15 | <= 3e-14 |
| Matern, MaternExp, MaternSHO | <= 2e-5 | <= 3e-6 |

RVs (two planets, one RV-only with secosomega; quadratic trend; two
instruments) and the light-travel delay (circular and e = 0.3, joint transit
+ eclipse with a Cowan & Agol phase curve and supersampling) agree with
juliet to < 1e-8 m/s and < 1e-7 relative flux.

**juliet / package behaviour found here.**
- **radvel 1.5.0's Kepler solver stops after one iteration** (NumPy path:
  `nd = np.sum(convd is True)` is always 0; the C extension is not built in
  this environment, so juliet uses this path). Residual of Kepler's
  equation: 6e-10 (e = 0.05), 4e-5 (e = 0.35), 3e-3 (e = 0.7), so juliet's
  eccentric RVs are off by up to ~3e-3 K (0.3 m/s for K = 100 m/s at e =
  0.7). jaxoplanet's solver is good to 2e-15. The test gives radvel a
  converged solver; then juliet and JAX agree to 1e-8 m/s.
- **juliet's george GPs use the HODLR solver by default** (approximate). At
  prior draws its lnL was off by factors of 2-3 (e.g. -2.6e7 vs -6.8e7
  exact); near the posterior the error is not measured. JAX solves exactly.
- **george's constant kernel in n dimensions is n exp(log_constant)**, so
  juliet's multi-dimensional SE / M32 amplitude is n (1e-6 sigma)^2;
  reproduced.
- **juliet's Matern is celerite's eps = 0.01 approximation.** Its celerite
  form loses precision in tinygp for rho near the cadence (-inf), so JAX
  uses the exact Matern-3/2: |dlnL| vs celerite <= 1e-5 (300 points, 0.24 d),
  1e-3 (2000-20,000 points, <= 1 d), but up to 1.2 over a 27-day span
  (TESS sector), where celerite's approximation is off, not JAX.
- **juliet's exponential kernel is `GP_sigma exp(-GP_timescale tau)`**
  (`log sigma`, `log timescale` go into celerite's `RealTerm(log_a,
  log_c)`): `GP_sigma` is a variance and `GP_timescale` a rate (1/day).
  jstark's default priors for `gp_kernel='exp'` and `'matern-exp'` treat
  them as an amplitude (1e-6-1e-2) and a timescale (cadence to 10 x span),
  so the default exp GP has amplitudes 1e-3-0.1 and correlation times from
  1/(10 span) to 1/cadence. Not changed (it changes juliet fits too);
  for the user to decide.
- **Only kernels that contain a full jstark kernel set can be set up
  through `BroadbandFit`**: `build_priors` adds the default priors of
  `gp_kernel` (default matern) whenever not all of its variables are
  given, so george kernels and DoubleSHO always get an extra prior and
  juliet then finds no kernel. The JAX models support them (tested by
  removing the extra prior).
- `_check_eclipse_timing` still refuses a fixed `t_secondary` with a varying
  orbit (a juliet + batman bug, 5.3); the JAX models would place it
  correctly, but the check stays so both backends accept the same fits.

**Evidence.** 600-point transit with a ramp (11 parameters, an estimated ~50 nats of
compression; the Phase 1 problem):

| method | lnZ | time |
| --- | --- | --- |
| importance sampling, multivariate t from the NUTS posterior (3 proposals) | 3984.12 +- 0.01 | |
| blackjax-nss (500 live) | 3983.90 +- 0.37 | 421 s |
| juliet dynesty (static, 500 live, defaults) | 3981.68 +- 0.54 | 98 s |
| juliet dynesty (static, `sample='rslice'`) | 3979.57 +- 0.52 | 103 s |
| juliet dynamic_dynesty (defaults) | 3975.65 +- 0.23 | 718 s |

NSS agrees with the reference; the dynesty runs are 2.4-8.5 nats low and
disagree with each other by more than their errors. NSS posteriors vs the
emcee reference: medians within 0.06 sigma, 68% widths 0.98-1.04. On a
5-parameter transit, NSS and static dynesty agree (test).

**GP fit** (600-point transit plus a Matern-3/2 red-noise draw of 300 ppm,
rho 0.02 d; Matern GP on time; 10 free parameters):

| | time | lnZ | medians vs dynesty | 68% widths / dynesty |
| --- | --- | --- | --- | --- |
| blackjax-nuts | 183 s | | within 0.19 sigma | 0.92-1.14 (GP_sigma 0.61) |
| blackjax-nss | 327 s | 3995.06 +- 0.24 | within 0.17 sigma | 0.91-1.13 (GP_sigma 0.61) |
| juliet dynesty (static, 500 live) | 72 s | 3991.22 +- 0.43 | | |
| importance sampling (reference) | | 3995.10 +- 0.07 | | |

The GP posterior is broad (rho 0.4 +- 0.8 d on a 0.3-day visit, degenerate
with GP_sigma), so NUTS mixes more slowly there: ESS 270-350 for p, b, a in
4 x 1000 draws (R-hat <= 1.009, 1 divergence) and the ESS warning asks for
more draws. NUTS and NSS agree with each other on GP_sigma; dynesty is wider.
dynesty's lnZ is again low (3.9 nats).

### 5.13 JAX backend, Phase 3: plots through the model interface (2026-11-27)

`BroadbandFit.plots()` works for JAX fits. `julietPlots` (name kept) no
longer calls juliet: every model comes from the fit's evaluator, so the
plot code is the same for both backends.

**Code**
- `fitting/evaluators.py`: the interface grows from the residual methods'
  needs to the plots' (`posteriors`, `priors`, `regressors`, `fit_type`,
  `lc_models` (draws, median, 68% band, linear model, GP prediction and
  band), `planet_models(name, nsamples, t=None)` (planet only, at the data
  or any times), `non_linear`, `orbit(planet, name)` (medians of `P`, `t0`,
  `p`, `a`, `b`, `ecc`, `w`, `eclipse_phase`), `gp_predict(name, x, resid,
  err, x_pred, params)`, `keplerian`, `rv_ta`). `JulietEvaluator` holds the
  juliet code moved out of `plots.py` unchanged: `lc.evaluate`, the
  planet-only draws, the regressor interpolation and the juliet re-fit
  workaround for high-resolution models, juliet's GP object, radvel.
- `jaxfit/model.py`: the planet part of `components` is `_planet(pv, inst,
  t, tt)`; `planet_model(pv, inst, t)` evaluates it at any times
  (supersampled like the dataset; the light-travel delay included);
  `components` also returns `linear`; `geometry(pv, inst, n)` (derived
  `a`, `b`, `inc`, `ecc`, `w`, `t_secondary`); `gp_predict(pv, inst, X, r,
  variance, X_pred)`: mean `K*(K+D)^-1 r` and variance `k** - |L^-1 K*|^2`
  with the dense cross-covariance and tinygp's triangular solver (works for
  sums and products of quasiseparable kernels and for the george kernels).
- `jaxfit/evaluate.py`: the JAX implementations. Quantiles are the 16/50/84
  percentiles over the draws; the GP for `plot_gp(highres=True)` and
  `compute_gp_model` is at the posterior medians (`parameter_vector`:
  `{prior name: value}` replacing medians; for juliet fits it is still the
  GP object's vector). Predictions are chunked (2e6 elements).
- `plot_rv`: the Keplerians come from the evaluator (radvel for juliet, the
  converged JAX solver for JAX fits).
- `BroadbandFit.plots(**kwargs)`: JAX fits take no juliet.fit keywords
  (TypeError).

**Checks**

| check | result |
| --- | --- |
| juliet path before vs after (eclipse fit with ramp + linear + GP; transit with linear + GP; every array the plots compute, seeded) | identical (eclipse); 1e-11 (transit, = run-to-run) |
| JAX plots vs juliet plots with the JAX draws swapped into juliet (all 1000 draws, compared per point after sorting) | full models 2e-9 (GP, exact vs eps Matern) / 3e-14; planet models at the data and at 10000 times 3e-13 (eclipse), 9e-9 (transit); linear models 1e-14; orbit identical |
| medians of the same draws | ~1e-6: juliet's even-count median is one order statistic off `np.median` |
| `gp_predict` vs celerite (Matern, eps 0.01) | 5e-8 relative (mean and std) |
| `gp_predict` vs dense solve (Matern; Exp + Celerite; Exp x Matern) | 1e-13 |
| `planet_model` at permuted data times vs at the data (supersampled phase curve, light-travel delay, secosomega) | 1e-14 |
| `keplerian` JAX vs radvel (converged solver), shared and per-draw times | < 1e-8 |

Tests (`test_jaxfit.py`): planet models at any times; Keplerians; every plot
method on a JAX eclipse fit with a ramp, linear models and a GP, and the
models against juliet's with the same draws; `plot_rv` on the JAX RV fit.

**Fixed in jstark's juliet plot path** (2026-11-27, after review; these
are jstark bugs in `julietPlots`/`JulietEvaluator`, not juliet's)
- `plot_gp` / `compute_gp_model` counted `sigma_w` twice: juliet's GP object
  has its own jitter term and the plots pass errors that already include
  `sigma_w` (with `sigma_w` = the photometric error, the GP mean moved by 9%
  of its scatter and the band was 16% too wide), and the GP parameters were
  those of the last draw juliet evaluated. `JulietEvaluator.gp_predict` now
  sets the GP object to the posterior medians (`params`: `{prior name:
  value}` replacing some, as for JAX fits; a raw celerite/george vector is
  still accepted) with its jitter term off. It equals the JAX prediction to
  5e-8 (celerite's eps Matern). The reference fits' GP predictions moved by
  2-4e-5; every other plot array is identical.
- The planet-only draws set a fitted `mdilution` to 0, which made juliet's
  planet-only model flat (`(M D + 1 - D) / (1 + D mflux) = 1`); now 1.
- juliet's `lc.evaluate(t=...)` leaves the dataset's model arrays (`ones`,
  `M`, ...) at the new length, so evaluating at the data afterwards failed
  (e.g. `phase_folded_lc(highres=True)` then `highres=False`).
  `JulietEvaluator` restores the state after such calls.
- The same-draws test now has a free `mdilution` and compares the GP
  predictions; it failed before the fixes.

**Other findings**
- For a transit-only juliet fit, batman's `t_secondary` gives an eclipse
  phase of 0.83 instead of 0.5 (unused in transit plots).
- `plot_fake_allan_deviation` warns "invalid value encountered in scalar
  divide" (`utils.fake_allan_deviation`), for both backends.

### 5.14 JAX backend, Phase 4: `SpectroscopicFit` with batched NUTS (2026-11-27)

`SpectroscopicFit.run(sampler='blackjax-nuts')` samples all bins in one
compiled program (`jaxfit/batch.py`, `run_batch`); `numpyro-nuts` and
`blackjax-nss` still fit the bins one after the other.

**How**
- Every bin is prepared as before (`BroadbandFit`, `JaxModel`, folder and
  input files: `BroadbandFit._jax_prepare`), and saved as before
  (`_jax_finish`, diagnostics, residuals), so bin folders, `load`,
  `spectrum()` and resuming are unchanged. Finished bins are loaded.
- Bins are grouped by `structure(model)`: everything the compiled function
  bakes in (datasets, model configuration, free parameter names and prior
  types, fixed parameter names, model options, array keys and trailing
  shapes). One log density takes a bin's data, prior hyperparameters
  (`PriorSet.hyper`, already an array since Phase 0) and fixed values
  (`JaxModel._log_likelihood(..., fixed=)`) as arguments.
- Bins of different lengths (NaN integrations dropped in some bins,
  clipping) are padded to the longest with a `mask`: white noise leaves the
  padded points out of the sums; GPs give them a zero residual and variance
  1e20, place the GP regressor of padded points far beyond the data, and
  take their constant terms off (`_PAD_VARIANCE`). Padded and unpadded
  log-likelihoods agree to 1e-7 nats (Exp, Matern, SHO, Matern x Exp).
- Starting points and the whitening (posterior maximum, Laplace covariance)
  per bin, with the shared compiled value-and-gradient and Hessian
  (`find_start(vg=, hess=)`).
- Warmup and sampling: blackjax window adaptation and NUTS, chains vmapped,
  mapped over the bins with `jax.lax.map(batch_size=batch)`: `batch` bins
  vmapped at a time, the rest one after the other inside the program. Every
  bin keeps its own step size and mass matrix. The log-likelihoods of the
  draws are computed in the batch too.
- Default `batch` (`default_batch`): all bins on a GPU, 1 on a CPU. NUTS
  trees differ in size, and vmapped lanes move in lockstep, each iteration
  costing the largest tree of all lanes; on a CPU the lanes are computed one
  after the other, so vmapping bins only adds that cost.

**Checks** (8 bins of a 300-point transit with a Matern GP and a linear
model per bin; NaN integrations in two bins, 298 and 299 points; per-bin
priors: fixed limb darkening and depth ranges; 4 chains x 500 + 500 draws;
CPU, M4 Pro)

| run | wall time | warmup | sampling |
| --- | --- | --- | --- |
| one bin at a time (`BroadbandFit.fit` per bin, before) | 360 s | 123 s | 215 s |
| batched, mapped (`batch=1`, CPU default) | 321 s | 136 s | 170 s |
| batched, all 8 vmapped | 687 s | 282 s | 390 s |

- The batched log-likelihood of every saved draw equals the bin's own
  (unpadded) model to 2e-12.
- Mapped vs one at a time: `p` medians within 0.12 sd, widths 0.92-1.03;
  `theta0` within 0.07 sd, widths 1.01-1.05; GP hyperparameters within
  0.03 sd (their widths scatter, 0.8-1.65: heavy tails, ESS 200-600).
- On a CPU the work is the same, so the gain is the compilations (one per
  model structure instead of one per bin): about 10% here. vmapping bins is
  for GPUs (Phase 6 benchmark script); on a GPU all bins at once may need
  `batch` smaller for memory (bins x chains x points per lane).

Tests: padding exact (white noise, ramp, supersampling, GP); batched runs
with shorter bins and per-bin priors, mapped and vmapped, saved
log-likelihoods against each bin's own model, truth recovered, finished
bins loaded on a second run.

### 5.15 JAX backend, Phase 5: real-data validation (2026-11-27/28)

Each fit repeats a finished dynesty fit with the same data, priors and model
(the fit fingerprint is checked) with `blackjax-nuts` (4 chains x 1000 +
1000), CPU (M4 Pro). Scripts: `jwst-ers/NRSG395H/Outputs/scripts/`
`jax_validate_g395h.py`, `jax_validate_joint.py`;
`jwst-ers/Data/MIRI/reduction/fits/jax_validate_wasp43b.py`;
`gj3929/Data/jax_validate_gj3929.py`, `fit_gj3929_rslice.py`.

| fit | free | NUTS time | medians (NUTS - dynesty) | widths (NUTS / dynesty) | R-hat, ESS, divergences |
| --- | --- | --- | --- | --- | --- |
| G395H NRS1 white light | 11 | 25 s | <= 0.05 sd | 0.96-1.04 | 1.002, 1236, 0 |
| G395H NRS2 white light | 11 | 25 s | <= 0.07 sd | 0.97-1.04 | 1.005, 1197, 0 |
| G395H NRS1 R100 (30 bins, batched) | 8 per bin | 1107 s | depth mean +0.1 ppm, rms 2.5, <= 0.05 sd | errors 0.97-1.08 (median 1.005) | 6 bins R-hat >= 1.01, min ESS 86 |
| G395H NRS2 R100 (29 bins, batched) | 8 per bin | 1018 s | depth mean -0.9 ppm, rms 4.0, <= 0.12 sd | errors 0.97-1.06 (median 1.000) | 6 bins R-hat >= 1.01, min ESS 88 |
| WASP-43 b MIRI LRS phase curve | 19 | 392 s | <= 0.08 sd (fp, nightside, offset included) | 0.96-1.03 | 1.006, 769, 0 |
| WASP-39 b joint C (NRS1+2, TESS 51+91 + GP, 2 RVs) | 34 | 887 s | <= 0.07 sd | 0.96-1.08 | 1.003, 1559, 1 |
| GJ 3929 b D/F/E (5 visits, ramps) | 39-42 | 1600-1840 s | see below | see below | R-hat up to 1.06, ESS 85 |

**G395H R100 bins that mix badly.** All in `sigma_w`: its default prior
(log-uniform 0.1-1e5 ppm) leaves a long prior-dominated tail below a narrow
peak (bin 7: peak at 250 ppm, NUTS 6.5% of the draws below 50 ppm, dynesty
~3%). `target_accept=0.95` with `nwarmup=2000` does not help (bin 7 ESS 30).
The depths are unaffected; a narrower default jitter prior would fix it (it
would change juliet fits and fingerprints too: open, user decision).

**GJ 3929 b: dynesty's random walk mis-weights the ramp timescale.**
- NUTS and dynesty disagree (F: fp 93 vs 111 ppm, -0.96 sd; E: fp of V3 10
  vs 77 ppm; ramp parameters up to 150 sd apart). The JAX log-likelihood
  at dynesty's draws equals dynesty's to 1e-10, so the models are the same.
- The ramp of each visit has a long-timescale solution (tau near the top of
  its prior, the visit length ~0.22 d) that trades off against the flux
  offset and the linear time term. In the 5-visit fits dynesty put V3 95%
  at short tau (< 0.01 d), NUTS 80-100% at long tau (all four chains).
- V3 alone (11 free parameters, variant F priors): both samplers put >= 95%
  at long tau, but inside it they disagree (tau 0.19 vs 0.14 d, fp 10 vs 30
  ppm). An arbiter that uses neither sampler, the Laplace marginal of tau
  (on a grid of the unconstrained tau, the other parameters maximised and
  their Hessian determinant added), agrees with NUTS:

  | tau (d) | < 0.004 | 0.004-0.13 | 0.13-0.18 | 0.18-0.21 | > 0.21 |
  | --- | --- | --- | --- | --- | --- |
  | Laplace | 0.005 | 0.095 | 0.37 | 0.41 | 0.12 |
  | NUTS (default; target_accept 0.97, 2000 + 2000) | 0 | 0.09 | 0.30-0.32 | 0.42 | 0.18-0.19 |
  | dynesty rwalk (500 live) | 0.043 | 0.20 | 0.58 | 0.15 | 0.02 |
  | dynesty rslice (1500 live) | 0.038 | 0.17 | 0.40 | 0.30 | 0.09 |

  fp: NUTS 10 (both settings), dynesty rwalk 30, rslice 20 ppm (+-40).
  Importance sampling with a Gaussian mixture fitted to both samplers'
  draws failed (ESS 1: the mixture misses the corner at the tau bound).
- So the published-style GJ 3929 depths (§5.3, `current_eclipses_*`, rwalk)
  are likely biased high. NUTS: D 129 (rwalk 134), F 93 (111), E V1 31 (65),
  V3 10 (77) ppm. The 5-visit NUTS runs are not fully converged (R-hat up to
  1.06, ESS 85) and cannot hop between eclipse-time modes (D: no draws in
  the +60 min mode, rwalk 3.4%). Variant D with dynesty rslice: below.
- The first rslice run of D crashed after 58 min, 27.7 M calls, near the
  end of the initial run ("Slice sampler has failed to find a valid
  point"); its lnZ was then 70710.6 +- 0.7 with dlogz 3.6 to go, already
  13 nats above rwalk's final 70697.3 (rwalk misses posterior volume).
  Cause: the batman 2.5.1 eclipse bug of §5.3 (fixed `t_secondary`) also
  hits a free `t_secondary` whenever a call repeats the previous call's
  value on the same batman object: `light_curve` moves `params.t0` to the
  conjunction of `t_secondary` only when `t_secondary` changed, and juliet
  resets `params.t0` to the transit epoch every call, so the repeated call
  puts the eclipse at phase 0.5 (GJ 3929 D: lnL 7.7 lower, every one of 200
  posterior draws). Slice steps below the float resolution of a BJD repeat
  `t_secondary`, every proposal is then rejected and the slice shrinks to
  zero. rwalk proposals move `t_secondary` every call, so rwalk results are
  not affected (300 draws re-evaluated: equal to dynesty's stored lnL).
  Workaround in `fit_gj3929_rslice.py` only (module level, so dynesty's
  spawned workers get it): `TransitModel.light_curve` wrapped to set
  `self.t_secondary = nan` for eclipse models (conjunction recomputed every
  call); repeated calls then agree and first calls are unchanged. A jstark
  fix (and an upstream report) is the user's decision.
- Not changed: refitting with rslice or NUTS, or a narrower ramp timescale
  prior (a ramp as long as the visit is degenerate with the linear time
  term), is the user's decision.

**blackjax-nss on GJ 3929 D: stopped.** 39 parameters, 500 live, 50
replaced per step, 195 slice steps per point: ~8.5 s per step on the CPU and
~2700 steps needed (prior-to-posterior information ~260 nats, 0.105 nats per
step), ~6.5 h, against dynesty rwalk's 8.3 M likelihood calls (58 per
iteration) on 8 processes. NSS runs its replaced points vectorised, which
only pays on a GPU (Phase 6). Stopped after 731 steps (user).

**CPU use and `nproc > 1` for JAX.** One JAX process keeps only ~2.7 cores
busy (NUTS ~130 CPU-s per R100 bin, dynesty ~70), so
`SpectroscopicFit.run(nproc=N)` now runs JAX samplers in N worker processes
on the CPU (user, 2026-11-28): `blackjax-nuts` in chunks of bins (2N
chunks, one batched compiled run each), the other JAX samplers one bin per
task; refused on a GPU (use `batch`). XLA's CPU thread pool cannot be
limited in jax 0.12 (`--xla_cpu_multi_thread_eigen=false` is accepted but
matmul and element-wise ops still use 8-10 cores), so workers are not
single-threaded; per-bin NUTS is mostly serial anyway. 12 G395H NRS1 R100
bins, with a 6-thread dynesty run on the machine:

| nproc | wall | per bin | CPU per bin | cores busy | depth vs batched validation run |
| --- | --- | --- | --- | --- | --- |
| 1 | 577 s | 48 s | 132 s | 2.7 | identical |
| 2 | 364 s | 30 s | 136 s | 4.5 | <= 0.06 sd |
| 4 | 239 s | 20 s | 144 s | 7.2 | <= 0.07 sd |

Each JAX worker needs ~1-1.5 GB. Draws differ from `nproc=1` because the
random keys follow a bin's position in its chunk.

**GJ 3929 D with dynesty rslice (patched batman), 6 processes, 3494 s:**
eclipse -8.5 min, +60 min mode 0.7% (rwalk 3.4%, NUTS 0); fp 137.7 +20.0
-19.6 ppm (rwalk 134.0 +17.3 -19.5, NUTS 129.4 +-21.9); lnZ 70708.74 +-
0.63 (rwalk 70697.33 +- 0.62: 11.4 nats low). Long-tau fractions V1, V2,
V3, V4a, V4b: rslice 1.00, 0.01, 0.21, 0.03, 0.26; rwalk 1.00, 0.04, 0.03,
0.04, 0.04; NUTS 1.00, 0.40, 0.86, 0.25, 0.28. So in D rslice and NUTS
disagree on V3 (the V3-alone arbiter used F priors, eclipse at -54 min,
and does not carry over). Pinned (user, 2026-11-28): the Laplace marginal
of tau_V3 in the full D fit (38 other parameters) would decide.

**Samplers vs the original dynesty runs.** dynesty: dynamic, rwalk, 500
live, 8 processes (wall time = posterior pickle minus `jstark_fit.json`
modification time); NUTS: one process, ~2.7 cores busy.

| fit | dynesty wall | dynesty calls | dynesty lnZ | NUTS wall | NUTS vs dynesty posteriors |
| --- | --- | --- | --- | --- | --- |
| G395H NRS1 / NRS2 white | 42 / 39 s | 1.4 / 1.3 M | 3331.39 / 3197.06 | 25 / 25 s | <= 0.07 sd, widths 0.96-1.04 |
| G395H NRS1 / NRS2 R100 | 63 s per bin; 255 / 250 s for all (parallel) | | | 1107 / 1018 s (`nproc=1`); ~20 s per bin with `nproc=4` | depths <= 0.12 sd, errors +-8% |
| WASP-43 b LRS phase curve | 373 s | 2.8 M | 54224.46 | 392 s | <= 0.08 sd |
| WASP-39 b joint C | 616 s | 5.9 M | 23467.15 | 887 s | <= 0.07 sd |
| GJ 3929 D / F / E | 1106 / 1101 / 1321 s | 8.3 / 8.1 / 9.4 M | 70697.33 / 70693.75 / 70668.43 | 1786 / 1838 / 1609 s (not converged) | ramp-mode weights differ; fp -0.25 / -0.96 / up to -2.2 sd |

**numpyro-nuts on real data (2026-11-28).** Same fits, same defaults (4
chains x 1000 + 1000, dense mass, Laplace whitening); outputs
`fits/jax_numpyro_nuts_white_<det>`, `..._w5.174-10.488_numpyro_nuts`,
`joint/jax_numpyro_nuts_C`; comparison `scratchpad compare_samplers.py`.

| fit | numpyro vs dynesty | numpyro vs blackjax | numpyro: time, R-hat, min ESS bulk/tail, div. | blackjax: time (warmup + sampling) |
| --- | --- | --- | --- | --- |
| G395H NRS1 white | <= 0.05 sd, widths 0.96-1.02 | <= 0.07 sd, 0.96-1.03 | 26 s, 1.0025, 1455/775, 0 | 25 s (11 + 5) |
| G395H NRS2 white | <= 0.06 sd, 0.95-1.03 | <= 0.05 sd, 0.94-1.02 | 18 s, 1.0022, 1688/1192, 0 | 25 s (11 + 10) |
| WASP-43 b phase curve | <= 0.08 sd, 0.97-1.05 | <= 0.09 sd, 0.96-1.06 | 296 s, 1.0042, 1810/1350, 0 | 392 s (211 + 108) |
| WASP-39 b joint C | <= 0.05 sd, 0.97-1.07 | <= 0.06 sd, 0.91-1.06 | 984 s, 1.0042, 1786/1106, 2 | 887 s (708 + 104) |

The two NUTS implementations agree with each other and with dynesty as
well as either does with dynesty; neither is consistently faster. Plots
(phase-folded, full model, TESS GP, RVs) through the evaluator are fine.
blackjax spends most of its time in warmup (joint C: 708 of 887 s; the
Laplace whitening may allow a shorter default `nwarmup`: not tested).

On this CPU NUTS is about as fast as 8-process dynesty for single fits
(faster for small ones, 1.4-1.6x slower for the 34-42 parameter fits) at
~1/3 of the CPU; R100 spectra are faster with dynesty's 8 processes. lnZ:
only dynesty here (NSS stopped; its lnZ biased on synthetic tests, 5.13);
rwalk lnZ is not exact either (GJ 3929 D: 11.4 nats below rslice).

**GJ 3929 b refit with ramptau capped at 0.1 d (2026-11-28).** Script
`gj3929/Data/fit_gj3929_taucap.py`: as `fit_gj3929_current.py`, but with the
new jitter default and `ramptau_<visit>` LU(2 cadences, 0.1 d) (was up to
the visit length); outputs `fits/taucap_eclipses_<D|F|E>_<sampler>`.
dynesty rwalk with 10 processes: 1195 / 1235 / 1274 s; blackjax-nuts at
the same time: 1877 / 1864 / 1654 s.

| fit | capped rwalk | capped NUTS | old rwalk | old NUTS |
| --- | --- | --- | --- | --- |
| F fp | 112.6 +17.3 -17.7 | 105.6 +-20.1 | 111.0 +-18.3 | 93.4 +-24.3 |
| E V1 / V2 / V3 / V4 fp | 57 / 149 / 65 / 69 | 56 / 160 / 57 / 68 | 65 / 160 / 77 / 89 | 31 / 159 / 10 / 73 |
| D fp, t_sec | **-146.0 at +65.8 min (98.6%)** | 137.8 +20.7 -22.3 at -8.4 | 134.0 at -8.5 | 129.4 at -8.3 |
| NUTS R-hat / min ESS | | 1.013 / 305 (D), 1.012 / 133 (F), 1.022 / 293 (E) | | 1.028 / 155, 1.059 / 85, 1.027 / 37 |
| lnZ (rwalk) D / F / E | 70719.09 / 70708.15 / 70702.20 | | 70697.33 / 70693.75 / 70668.43 | |

- With the cap, F and E agree between the samplers within 0.4 sd (they
  differed by up to 2.2 sd), and NUTS mixes better. It still flags R-hat
  1.01-1.02 on the short-tau ramps (V1, V3, V4a/V4b).
- **D, capped rwalk: 98.6% of the posterior is an inverted eclipse**
  (fp < 0 at +66 min, near the edge of the +-72 min prior). This mode is
  in every dynesty run of D: old rwalk 3.4%, rslice 0.7%. Its weight is
  nested-sampling noise between two separated modes. Direct optimisation
  of the JAX model with the capped priors (`scratchpad d_mode_opt.py`,
  starts from the best samples of all five D runs) gives:
  - eclipse mode: lnL 70973.43, log posterior 70916.18 (-6.8 min,
    +137.8 ppm);
  - inverted mode: lnL 70969.17 (4.3 lower), log posterior 70913.96
    (2.2 lower; Hessian not positive definite there).
  The eclipse is the higher mode. The capped rwalk run's best eclipse-mode
  point was 70957.9, 15.5 below the peak, so it never converged there.
  The odds against the inverted dip are only ~e^2-e^4. So with fp free in
  sign and t_sec free, D's eclipse time is modest evidence. An fp >= 0
  prior would remove the inverted mode (user decision).
- rwalk lnZ moved by +22 / +14 / +34 (D / F / E). The prior changes alone
  shift it by ~+2: the narrower jitter prior gives +0.2 per visit, the
  tau cap +0.15 per visit. So the rwalk lnZ of these 40-parameter fits
  scatters by tens of nats and cannot be used for model comparison here.

### 5.16 JAX backend, Phase 6: docs and GPU benchmark (2026-11-28)

- User guide (`docs/user_guide/fitting.rst`), new section "JAX samplers
  (optional)": install, the model and what it does not support, how NUTS
  starts, options, results and diagnostics, spectroscopic fits (`batch`,
  `nproc`), real-data tests, limitations, the benchmark. Also a note on
  long ramps in "Priors" (cap `ramptau` when it runs to the top of its
  prior; GJ 3929). API page `jstark.fitting.jaxfit`. Install line
  `jwstark[fitting,jax]` in the README and quick start. New `jax` extra
  in `pyproject.toml`: jax, jaxoplanet, blackjax, numpyro and tinygp, with
  the tested versions as lower bounds. Read the Docs installs it for the
  API pages.
- `python -m jstark.fitting.jaxfit.benchmark` (`jaxfit/benchmark.py`). It
  uses synthetic data made with batman: a transit with a ramp and two
  regressors. It times:
  - `gradient`: log density + gradient for 1e3-1e5 points, alone and
    vmapped over 16-1024 parameter vectors;
  - `fit`: a white-light fit per NUTS sampler (`--dynesty` adds juliet);
  - `spectrum`: 64 bins with `batch` 1, 8 and all.
  The output is JSON with the device and versions. `JAX_PLATFORMS=cpu`
  gives the same tests on the CPU. Smoke test:
  `test_benchmark_script_runs`.
- `--quick` on the M4 Pro CPU, 5 min:
  - gradient: 0.27 ms (1e3 points) and 1.07 ms (1e4); vmapped 87 /
    750-885 us per evaluation;
  - fit, 1000 points: blackjax-nuts 33 s, numpyro-nuts 29 s;
  - spectrum, 8 bins x 500 points: batch 1 109 s, batch 8 96 s. vmapping
    here is not slower, unlike the Phase 4 test (8 bins x 4 chains, 1.9x).
- Fixed along the way:
  - `SpectroscopicFit.run(nsamples=...)` (the draws of the saved median
    models) swallowed the JAX samplers' `nsamples`, so a bin's NUTS always
    drew 1000 per chain. Now `nsamples=None` (models: 500) and, with a JAX
    sampler, a given `nsamples` is also the sampler's draws. The test
    checks `ndraws`.
  - numpyro-nuts `timing['sampling']` stopped the clock before JAX
    finished (asynchronous dispatch; 4.2 s of a 29 s run). It now stops
    after the draws are fetched. The Phase 5 numpyro times were wall
    times, so they are unaffected.

## 6. Open issues and next steps

- [x] Phase 2: `SpectroscopicFit` (per bin, all datasets covering it,
      shared depth, per-dataset systematics; parallel over bins with
      juliet single-threaded; resumable), `Spectrum` table, spectrum and
      2D residual plots; `jstark.py` deleted. Validated on WASP-39 b PRISM
      (§5.4).
- [ ] Phase 3 (plan agreed 2026-11-15; review after each step):
  - [x] Step 1: `channel_priors(fix=...)` matches name prefixes ending at
        an underscore (`'GP_rho'`, `'ramptau'`, `'GP_rho_V1'`; plain
        types as before), and `planet=` no longer drops dataset
        parameters. Test: red noise shared by 4 channels, the white-light
        GP timescale fixed in the bins with a free amplitude per bin; the
        depths come back within 4 sigma.
  - [x] Step 2: common mode, `cm_mode='divide'` or `'regressor'`, off by
        default. Validated on WASP-39 b PRISM and a full WASP-39 b SOSS
        reduction vs the SOSS benchmark (R100), §5.5. The HAT-P-30 SOSS
        check was not run: WASP-39 b SOSS already shows that the per-bin
        excess is not common mode.
  - [x] Step 3: per-channel common modes: extraction writes the readout
        amplifier of each column (`TRACE_O<n>.amp`), `LightCurves.groups`
        and `split()`, `BroadbandFit.channel_common_mode`,
        `SpectroscopicFit(common_mode=<per-amplifier fit>)`; `bin()` now
        averages per-channel regressors (flux weighted). The phase-of-4
        regressor already existed (`phase4`). Full WASP-39 b F322W2
        reduction vs the NIRCam benchmark, §5.5: no gain over one
        white-light common mode on these data.
  - [ ] Step 4: phase curves with juliet's built-in models (Cowan & Agol
        2008 first; eclipse mapping later). Derived dayside/nightside
        flux, amplitude, offset; `Spectrum` columns; phase-curve plots;
        optional time binning (off by default). Validate on WASP-43 b MIRI
        LRS (`jwst-ers/Data/MIRI/`). Code done 2026-11-16 (§5.7); the
        earlier worry that juliet does not set the per-planet component in
        joint transit + eclipse fits was wrong: it does (checked on a
        synthetic phase curve, planet model within 12 ppm of the truth).
  - [ ] Saturation check of `saturated_box` on LP 791-18 c NIRSpec PRISM
        (GTO 1201, obs 401) in `/Volumes/Jayshil SSD/JWST/lp791-18c/`.
- [x] PRISM saturated columns (b)-(d): judged with spectroscopic fits
      against NIRISS SOSS (extraction_notes.md §5.15). `saturated_box`
      brings the saturated bins to -48 ppm from SOSS (default +391, ERS
      -219) at no cost in errors.
- [x] `SpectroscopicFit.run(bins=...)`: bins without data are now skipped
      with a message instead of raising.
- [x] Joint fits with other observatories and RVs (5.9).
- [ ] WASP-39 b sector 91: transit times scatter more than their errors (5.9).
- [ ] TTVs: `channel_priors` would need P and t0 derived from the `T_`
      parametrisation. GPs and linear regressors on RVs.
- [ ] Report the juliet `non_linear_functions` / `input_folder` bug.
- [ ] JAX backend (5.10-5.15): Phases 0-4 done 2026-11-27 (model;
      samplers, `BroadbandFit`, interface, folders, `load`; GPs, RVs,
      light-travel delay, `blackjax-nss`; plots through the model
      interface; batched NUTS over spectroscopic bins); Phase 5 (real-data
      validation, `nproc > 1` for JAX) 2026-11-28; Phase 6 (docs, GPU
      benchmark script, 5.16) 2026-11-28, in review. Next: run the benchmark on
      a GPU (user).
- [ ] GJ 3929 b depths (5.3) likely biased by dynesty rwalk on the ramp
      timescale degeneracy (5.15): refit (rslice / NUTS) or narrow the
      ramp tau prior; user decision. Pinned (2026-11-28): Laplace marginal
      of tau_V3 in the full D fit to decide between rslice and NUTS.
      Refit with ramptau <= 0.1 d done (2026-11-28, 5.15): F and E agree
      between the samplers (F 113 / 106 ppm). D rwalk fell into the
      inverted-eclipse mode (fp < 0 at +66 min; the eclipse mode is 4.3
      higher in lnL). Open: an fp >= 0 prior for D, and which run to
      adopt (user decision). Pinned (user, 2026-11-28): whether the cap or
      sampler noise moved D's rwalk run into the inverted mode (a second seed
      would tell; the best points of both modes are within the cap).
- [ ] batman 2.5.1 eclipse bug (fixed or repeated `t_secondary` -> eclipse
      at phase 0.5; 5.3, 5.15): fix in jstark (wrap
      `TransitModel.light_curve`) and report to batman; user decision.
- [x] Default jitter prior: log-uniform 0.1-1e5 -> 0.1-1e4 ppm (user,
      2026-11-28). Fits made with the old default no longer match their
      fingerprint (`overwrite=True` or a new folder to refit). The lower
      end is unchanged, so the low-sigma_w tail behind the poor NUTS mixing
      in some R100 bins (5.15) remains. `init='prior'` with a short warmup can
      leave a chain in another mode (R-hat flags it); `init='map'` is the
      default.
- [ ] juliet's batman models for non-polynomial laws are good to only a
      few ppm (integration step fixed at juliet's initial parameters, 5.10).
- [ ] Default priors of `gp_kernel='exp'` / `'matern-exp'` assume an
      amplitude and a timescale; juliet's kernel takes a variance and a
      rate (5.12). Decide whether to change them.
- [ ] blackjax-nss evidence biased low on the 5-parameter NSS test problem
      (found 2026-11-27 while testing Phase 3; the test against dynesty
      fails intermittently). Reference: importance sampling from the NUTS
      posterior, 1316.90 +- 0.01 (two proposals agree; the z-space prior
      integrates to 1 within 0.007). NSS (400 live): 1315.55 / 1315.45
      (seeds 0, 1), 1316.06 (100 slice steps), 1315.25 (1 deleted per
      step), 1316.38 (20 deleted, 60 steps), 1316.24 (1000 live): 0.5-1.65
      nats low, with quoted errors 0.13-0.23. dynesty: 1316.03, 1315.88,
      1317.86 (+-0.37). The NSS posterior equals NUTS's (medians within
      0.1 sd, widths 0.98-1.01). With `sigma_w` fixed or log-uniform
      [1, 1000] instead of the default [0.1, 1e5], NSS is within -0.6 to
      +0.06 nats of the reference, so the bias comes with the wide,
      prior-dominated `sigma_w` direction; the quoted errors understate the
      run-to-run scatter (~0.5 nats) in every case. The 11-parameter check of
      5.12 agreed. Pinned by the user (2026-11-27): investigate later; do not
      trust NSS evidences until then.
- [ ] radvel 1.5.0 NumPy Kepler solver stops after one iteration (juliet's
      eccentric RVs off by up to ~3e-3 K at e = 0.7; 5.12): report upstream.
- [ ] dynesty evidences 2.4-8.5 nats low on an 11-parameter transit with a
      ramp (5.12): check on real fits before using juliet's lnZ for model
      comparison; `blackjax-nss` matched an importance-sampling reference.
- [ ] george kernels and DoubleSHO cannot be set up through `BroadbandFit`
      (default priors of `gp_kernel` are added; 5.12).
- [ ] `phase_folded_lc` y-limits: include the binned data (low-S/N
      eclipses).
- [ ] GJ 3929: V1 depth vs systematics model (trim, ramp, PLD/PCA) if the
      per-visit depths matter.
