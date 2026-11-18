# jstark light-curve fitting: development notes

As of 2026-11-15. The record of what was discussed, decided, implemented
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
| `fit.py` | `BroadbandFit`, `ExpRamp`, `red_noise_beta` |
| `spectroscopic.py` | `SpectroscopicFit` (one `BroadbandFit` per wavelength bin, parallel, resumable), `Spectrum` (astropy `Table` subclass), `make_bins`, `restrict_priors` |
| `plots.py` | `julietPlots` (moved from `jstark.py`, fixed) |
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
   N(0, 0.1), `sigma_w` LU(0.1, 1e5) ppm, `theta<i>` U(-1, 1), GP
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
  fit is loaded instead of re-run.
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
- [ ] Report the juliet `non_linear_functions` / `input_folder` bug.
- [ ] `phase_folded_lc` y-limits: include the binned data (low-S/N
      eclipses).
- [ ] GJ 3929: V1 depth vs systematics model (trim, ramp, PLD/PCA) if the
      per-visit depths matter.
