# jstark spectral extraction and photometry: development notes

As of 2026-11-22. The record of what was discussed, decided, implemented
and measured while building the extraction stage of jstark
(`jstark/extraction/`), for anyone continuing the work, including future
work sessions. The code and its docstrings remain the reference for
details; `docs/extraction.rst` is the user-facing walk-through.
`docs/detector_processing_notes.md` covers the stage before this one.

---

## 1. Decisions agreed with the user

- **Port stark** (A. Brandeker & J. A. Patel, https://github.com/Jayshil/stark)
  into `jstark.extraction`, with credit, fixes and vectorisation. jstark does
  not depend on stark.
- **Per segment, never stitched.** Exposures can be tens of GB. Only the
  small products (spectra, light curves) are combined.
- **The PSF uses all integrations** (a sample of ~300 was rejected as too
  noisy). The least-squares sums are streamed segment by segment.
- **Knot numbers are user inputs**, with defaults from one-segment tests.
- **Smoothness penalties (P-splines)** against wavy PSFs, with strengths set
  by cross-validation.
- **One exposure-wide PSF** (no per-segment PSFs). Time variations are
  monitored through the per-integration regressors (`dy`, `ds`, `chi2`).
- **Column range:** automatic by default, user override.
- **Aperture / background optimisation** on blocks of consecutive integrations
  from several segments (`segments='all'` optional), with a noise metric that
  is robust to transits.
- **SOSS:** our joint multi-order PSF extraction is the default. ATOCA is
  **not** built in: the literature finds order contamination small for
  relative measurements (ATOCA paper: <0.1% below 2.6 µm and up to ~2% at the
  red end of order 1; transit features biased by ~1% of their amplitude). Our
  cross-check is the box extraction plus a saved contamination fraction.
- `ApPhoto` moved from `jstark/jstark.py` to `jstark/extraction/photometry.py`
  (no back-compatibility needed).
- **Test outputs:** Stage 1 and cleaning products of segment 1 of every test
  dataset are in `<data dir>/reduction/` (the script is in §8).

Agreed on 2026-11-13 (second review):
- **No pipeline flat field.** The CRDS flat (Stage 2) is known to increase
  light-curve scatter and is not used. The pixel-response map Q (§5.2) is
  measured from the exposure itself and never applied to the data. It is on
  by default for all spectroscopic modes, can be switched off with
  `set('psf', pixel_correction=False)`, and is saved for inspection (`QMAP`).
- **Field-star contamination (SOSS):** detect zeroth orders in the F277W
  exposure and mask them; exclude them from the background region; model and
  subtract field-star traces; report what remains in the aperture (§5.11).
- **MIRI LRS** default `wavelength_range=(5, 12)`, overridable.
- **Disk space:** options not to keep everything (§5.12); wavelengths saved
  as their own extension in the spectra files.
- **Deferred:** leftover SOSS 1/f should be removed at the group level in
  Stage 1 (not with `fit_background`); mitigation of the NIRCam profile-shaped
  1/f (covariance weighting gave no gain) is to be discussed; full
  multi-segment runs next time.

## 2. Architecture

```
<exposure>_cleaninfo.fits + *_cleaned.fits (per segment)
   │  ExposureSource: chunked reads (.section), canonical orientation
   │  (spatial, spectral); MIRI/LRS transposed
   ├─ SpectralExtractor
   │    find_trace ─► [find_contaminants / add_field_trace] ─► fit_psf (1 streaming pass)
   │    ─► [optimise] ─► run
   │    ─► <root>-segNNN_<det>_spec.fits (per segment)
   │    ─► <exposure>_spectra.fits (combined)  + <exposure>_extract_model.fits
   └─ Photometry (imaging)
        fit_psf (1 pass) ─► [optimise] ─► run
        ─► <root>-segNNN_<det>_phot.fits ─► <exposure>_photometry.fits
```

| Module | Contents |
| --- | --- |
| `source.py` | `ExposureSource`: cleaninfo + cleaned segments, `read(k, ints)`, `read_ints`, `chunks` (memory budget), canonical orientation |
| `bspline.py` | vectorised cubic B-spline basis (+ derivative, checked against scipy), `NormalSums` (per-column streamed normal equations, even/odd integrations), `BandedSystem` (block-banded Cholesky) |
| `trace.py` | `find_trace` (iterative centre of flux + clipped cubic spline), `trace_from_reference` (SOSS), `column_range`, `cof_offsets`, windows / edge weights |
| `background.py` | distance-from-trace regions, clipped median or polynomial per spectral pixel, with variance |
| `psf.py` | `SplinePSF`: tensor spline S(u, x), P-spline penalties, column-block / integration cross-validation, unit-integral normalisation per column |
| `extract.py` | `optimal`, `box`, `refine_shift` (Gauss-Newton dy and width ds), `covariance_weighted` |
| `covariance.py` | row-offset covariance per amplifier (NIRCam 1/f), GLS extraction in sub-chunks |
| `contamination.py` | detection image loader (F277W rateints → median), local noise map, compact-source finder (high-pass, 3×3 median, Gaussian matched filter, hysteresis, size limit) |
| `metrics.py` | `p2p_sigma`, `p2p_noise` (pooled blocks), `binned_noise`, `white_light` |
| `extractor.py` | `SpectralExtractor`, `Spectra` |
| `photometry.py` | `Photometry`, `PhotometryResult`, `ApPhoto` (moved), `aperture_weights` |
| `diagnostics.py` | trace, PSF, width, residual, optimisation, white light, regressors, spectra, photometry, contaminant plots |
| `config.py` | default parameters per mode |

Tests: `jstark/tests/test_extraction.py`, with synthetic data:
- B-spline basis vs scipy; streamed sums = single pass; banded solve = dense;
- PSF recovery; optimal variance with masked pixels; p2p metric with transits;
- end to end on 3 segments: products, trace, jitter, transit depth, error
  ratios, flat-field recovery; optimise;
- photometry end to end;
- covariance extraction;
- zeroth-order mask (detection in a separate image, NaN for fully masked
  columns, unbiased partly masked columns, mask saved with the model);
- field-star trace model (dy refinement, target bias 27% → 0.7%);
- disk options (`remove_inputs`, `keep_segments`) and `WAVE_O1`;
- saturated columns (saturation map → `SATMAP` in cleaninfo →
  `SATGRP_O1`; group-0-only columns NaN, `min_groups=1` keeps them);
- PSF epochs (`test_psf_breaks`, `test_psf_breaks_validation`, §5.17).
The whole suite (`jstark/tests`, all stages) has 99 tests, all passing
(~5.5 min, 2026-11-22).

## 3. What changed relative to stark (and why)

1. **Vectorised.** stark loops in Python over integrations × columns
   (~35 s per extraction pass per G395H segment); jstark extracts a whole
   chunk at once.
2. **Streamed PSF fit.** Per-column normal equations `A_x = Σ w B Bᵀ`,
   `b_x = Σ w B y` for the 1D basis in u. The 2D (u, x) system is assembled
   from them: memory is independent of the number of integrations, and the
   column knots can change without re-reading. The result equals
   `LSQBivariateSpline` on all pixels. The system is block-banded
   (column-major ordering) and is solved by banded Cholesky. Moving from
   dense solves made the knot_col = 150 test ~100× faster.
3. **Outlier rejection** on normalised residuals with a symmetric robust
   threshold. stark's `|r| < median + clip·std` on unweighted residuals
   preferentially rejects the bright core; the notebook needed `clip = 20`.
4. **Noise-free normalisation:** median box spectrum × per-integration
   white-light factor, instead of each integration's own sum.
5. **Unit-integral profile per column**, so the flux scale does not depend
   on the sub-pixel trace position or on the aperture.
6. **Correct optimal variance** `Σ e²P²/V / (Σ e P²/V)²`. stark/Horne's
   `Σ P / Σ P²/V` is too small when pixels are masked.
7. **Time-averaged per-pixel variance** as weights (no bias from noisy
   per-pixel errors).
8. **Fractional aperture edges** (`e` in [0, 1]), so the estimate is smooth
   in the trace position.
9. **Gauss-Newton trace offsets** dy(t), plus a width-change regressor ds(t).
10. **Pixel-response map Q** (§5.2).
11. **Smoothness penalties chosen by cross-validation between sub-pixel
    phases** (§5.3).
12. **Temporal rejection** instead of model-based rejection (§5.4).
13. **Bugs found in stark:**
    - `i1 >= shape → shape − 1` drops the last row (5×);
    - `polynomial_bkg_rows` uses `frame[idx, i]` instead of `frame[i, idx]`
      and evaluates over `shape[0]` (reduce.py:178, 180);
    - `trace_spectra` uses `diff1` for `diff2` (reduce.py:396).

## 4. ApPhoto changes

- Sky: 3σ-clipped median of good pixels (was an unclipped mean ignoring
  `badpix`), with the error × √(π/2).
- Cosmic rays: symmetric MAD threshold.
- `brightpix` selected `nos_brightest + 1` pixels; fixed.
- Noise metric defaults to robust p2p.
- `pld_correction(fit_mask=...)` to fit out of eclipse. PLD regressors should
  be fitted jointly with the eclipse model.

## 5. Results (segment 1 of each dataset)

### 5.1 First G395H NRS1 run, and two problems found

- **First version:** optimal 187.5 ppm (ratio 1.12) was *worse* than box
  177.2 ppm (1.04). Without the per-pixel rejection, optimal gave 178.3. The
  culprit: 26 pixels per integration were rejected at 6σ on already cleaned
  data, all bright core pixels where the smooth PSF did not fit. They switched
  between rejected and kept from integration to integration (the same
  switching problem as with saturation).
- **Cause:** the data are not flat-fielded. A static data/model ratio per
  pixel has **2.2% rms** on bright pixels, reproduced between even and odd
  integrations with correlation 0.9996. More knots barely help: reduced χ²
  went from 1.40 to 1.29 with 12× more coefficients.

### 5.2 Pixel-response map Q

- **The model** becomes `f · P(u, x) · Q(pixel)`, with Q estimated in the
  same streaming pass.
  - χ² per pixel: 1.51 → 0.77.
  - >6σ pixels per integration: 31 → 0.005.
  - G395H NRS1: optimal **176.7 ppm (ratio 1.058)** vs box 177.9.
- **Degeneracies:**
  - Q is defined only up to one factor per column (absorbed by that column's
    flux: harmless).
  - Any smooth function of (u, x) can move between P and Q. The local
    average of Q over ±25 columns at the same offset is divided out, so P
    (which follows the trace) keeps the smooth structure.
- **Synthetic check:** a 2% flat is recovered to 0.46% per pixel (noise),
  after the per-column factor.
- **How it works (as explained to the user).** Per pixel p (row r, column x)
  and integration t: `D_p(t) = f_x(t) · P(u_p(t), x) · Q_p`, with `f_x(t)` the
  flux of column x, `u_p(t) = r - y0(x) - dy(t)` the offset from the moving
  trace, `P` the smooth profile (sums to 1 per column, moves with the trace)
  and `Q_p` the static sensitivity of the pixel. Estimated by least squares
  over all integrations, `Q_p = Σ w D m / Σ w m²` with `m = f̂ P`, iterated
  with the PSF fit (P fitted to D/Q). Then the local average is divided out
  and each factor is shrunk by its precision (§5.11).
- **Why it matters with a moving trace:** without Q the optimal estimate is
  `f · Σ e P² Q / V / Σ e P² / V`, a profile-weighted mean of Q that changes
  when the trace moves (pointing leaks into the flux). With Q in the model
  the estimate is unbiased whatever the position.
- **Not a flat field:** the data are never divided by Q; it enters only the
  model `P · Q` (weights, outlier test, trace refinement). An error in Q
  makes the weights slightly suboptimal and rescales a column by a constant.
- **Storage and control:** `QMAP` in `<exposure>_extract_model.fits`
  (canonical orientation, i.e. transposed for MIRI LRS; Q = 1 where not
  estimated: outside the PSF windows, under masked contaminants, in SOSS
  overlap pixels). Default on for all spectroscopic modes;
  `set('psf', pixel_correction=False)` gives Q = 1. Not used by photometry.

### 5.3 PSF flexibility, waviness and the choice of smoothing

**Evolution of the cross-validation (CV).** G395H NRS1, white-light p2p
(ratio to the propagated error):

| oversample / knot_col | even/odd-integration CV | column-block CV | **sub-pixel-phase CV (default)** |
| --- | --- | --- | --- |
| 2 / 20 | 176.7 (1.058) | 176.3 (1.055) | **175.5 (1.051)** |
| 4 / 60 | 225.7 (1.072); unsmoothed 2101.6 (1.60) | — | **175.1 (1.048)** |
| 4 / 150 | — | — | **175.1 (1.048)** |

**Mechanism of the waviness (confirmed):**
- With knot_col = 60 each column basis function spans ~100 columns. Over
  that span the trace's sub-pixel phase covers only ~0.66 px, while the u
  knots are 0.25 px apart.
- Basis functions with almost no data let the profile develop **negative
  lobes (−0.6 of the peak-normalised profile)**. The extraction then
  falls apart: 1195 vs 167 ppm on the same 400 integrations.

**Why the first two CV schemes failed:**
- *Even/odd integrations* sample the same phases, so the weakest smoothing
  always looked best.
- *Column blocks* test predicting unseen columns, which extraction never
  needs. They over-smoothed along the dispersion and under-smoothed across
  the trace: PRISM 478 ppm (1.49) vs 350–369 with weak or phase-CV smoothing.

**Default: CV by sub-pixel phase.**
- Pixels are split by the fractional part of their offset from the trace
  (below or above one half). A model trained on one half of the phases must
  predict the other.
- This tests exactly the interpolation that matters when the trace moves,
  and does not penalise real changes of the profile along the dispersion.
- It is cheap: the sums are accumulated per fold, and a coordinate search
  over the two penalties replaces the 11 × 11 grid.
- With it, PSF knot settings matter only at the noise level (175 ± 1 ppm
  for every G395H setting).

### 5.4 Rejection policy

- **Model-based rejection** (Horne: reject |D − fP| > 6σ) made pixels with a
  *static* misfit switch between kept and rejected:
  - G395H before Q: 26 pixels per integration;
  - SOSS: ~900 per integration.
- **Default: `reject_mode='temporal'`.** A pixel is rejected only if its
  residual deviates from its own median residual over the chunk. Static
  misfit never causes rejections; transients still do. G395H and SOSS now
  have 0.0–0.2 rejections per integration.

### 5.5 Aperture / background optimisation (G395H NRS1, 3 × 300 integrations)

White-light p2p (ppm); rows = background inner radius, columns = aperture
half-width 2–7:

```
 6:  181.5 172.9 177.9 175.7 175.3 176.3
 8:  182.5 175.2 178.0 175.5 177.6 179.4
10:  180.4 174.8 176.4 178.4 181.3 180.9
```

- The statistical uncertainty of each value is ±7 ppm (1.17σ/√N), so all
  settings are equivalent here. `optimise()` reports this uncertainty.

### 5.6 All modes, final defaults (segment 1)

| Mode | Columns (λ, µm) | Optimal (ppm, ratio) | Box (ppm, ratio) | Per-column ratio | PSF fit (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| G395H NRS1 | 525–2042 (2.69–3.72) | **175.2, 1.049** | 178.1, 1.049 | 1.00 | 25 | Stage 1 summed pixels: 177, 1.08 |
| G395H NRS2 | 5–2043 (3.82–5.17) | **243.7, 1.043** | 261.6, 1.055 | 1.01 | 37 | Stage 1: 1.18 |
| G395M | 656–2042 (2.68–5.17) | **169.0, 1.132** | 170.5, 1.127 | 0.98 | 20 | Stage 1: 1.19 |
| PRISM | 14–483 (0.50–5.52) | **374.4, 1.148** | 377.8, 1.298 | 0.98 | 29 | 43% of columns have masked (saturated) core pixels; Stage 1: 1.19 |
| SOSS order 1 | 5–2042 (0.85–2.83) | **90.7, 2.08** | 94.3, 2.17 | 1.20 | 286 (both orders) | joint; Stage 1: 2.31–2.43 |
| SOSS order 2 | 5–1789 (0.59–1.41) | **164.7, 1.98** | 191.1, 2.28 | 1.44 | | contamination by order 1 up to 3× in the overlap columns |
| F444W | 675–2043 (3.75–5.09) | **316.7, 3.33** | 326.1, 3.47 | 0.95 | 24 | Stage 1: 4.0–4.8 |
| F322W2 | 4–1788 (2.40–4.14) | **306.2, 3.66** | 329.0, 3.98 | 0.95 | 29 | Stage 1: 5.3–6.1 |
| MIRI LRS | 21–394 (4.06–13.9) | **374.0, 1.36** | 450.4, 1.51 | 1.17 | 4 | auto range includes the faint < 5 and > 12 µm ends |

- **Optimal beats or equals box everywhere.** The largest gains are on LRS
  (17%), SOSS order 2 (14%), G395H NRS2 (7%) and F322W2 (7%).
- **The p2p values have ~3–7% statistical uncertainty** (fewer integrations
  give more). For example, LRS gave 347 ppm in an earlier run with different
  smoothing.
- **Extraction takes ~1 min per NIRSpec segment.**
- **The PSF-fit χ² of PRISM (12.8) is not a model failure.** It includes the
  grey normalisation: at PRISM's per-pixel S/N of ~1000, chromatic flux
  changes show up as χ². The per-integration extraction χ² (`REG_O1.chi2`,
  median 1.16) is the meaningful diagnostic. NIRCam: 0.93–0.99; NIRSpec:
  ~0.81.

### 5.7 SOSS joint extraction

- **Traces:** PASTASOSS reference traces shifted by a fitted linear
  correction (~0.9 px). Centroid rms 0.10 px (order 1) and 0.17 px (order 2).
- **Order 3 is bright** in this data set (y ≈ 150 → 190). It is excluded
  from the background region; no field contaminant was modelled.
- **Overlap problem and fix:**
  - The first version had χ² 5.9 per pixel, entirely in the overlap columns
    (< 512): order 1 at u = +4..+12, order 2 at u = −16..−8.
  - Wider PSF windows made it worse (χ² 22 at ±22 px, 45 at ±28 px).
  - The cause was **Q absorbing the two models' inconsistencies** (28% rms
    in the overlap).
  - Fix: Q only from pixels outside the other orders' PSF windows (Q = 1 in
    the overlap). χ² is now **1.0 (order 1) and 0.94 (order 2)**.
- **Settings tested (order 1 / order 2, ppm):**

  | Setting | Order 1 | Order 2 |
  | --- | --- | --- |
  | oversample 1, 30 col knots | 98 | 183 |
  | oversample 2, 100 col knots | 95 | 175 |
  | + polynomial background | 86–91 | 162–168 |

  Aperture 12 instead of 15: 109 for order 1.
- **Contamination (`CONTAM_O<n>`):**
  - order-2 light in the order-1 aperture: at most 4%;
  - order-1 light in the order-2 aperture: up to 3× order 2's own flux at
    its red end (overlap columns), so those columns are the least reliable
    for order 2.
- **Per-column excess (1.20 / 1.44):** a per-column fitted background removes
  it (1.05 / 1.10), so it is column-wise offsets (residual 1/f). But it
  ruins the white light (order 1: 255 vs 91 ppm), being degenerate with the
  wide profile. `fit_background` stays off; better 1/f removal belongs in
  Stage 1. On G395H, `fit_background=True` also costs 5% (184 vs 175 ppm).

### 5.8 NIRCam: covariance-weighted extraction

- **Row-offset covariance per amplifier** from PSF-fit residuals of 600
  integrations: rms 0.5–0.6 (F444W) and 1.1–1.4 DN (F322W2).
- **No gain:**

  | | Covariance / optimal white light |
  | --- | --- |
  | F444W | 1.002 (317.3 vs 316.7 ppm) |
  | F322W2 | 1.040 (318.5 vs 306.2 ppm) |

- **As expected from the analysis in the plan:** the part of the 1/f that
  dominates the white light (3.3–3.7× the errors) has the shape of the
  profile. It is absorbed into the flux, invisible in the residuals, and
  cannot be reduced by any weighting. Only group-level 1/f removal
  (Stage 1) or regressors in the light-curve fit can deal with it.
- The method was kept (`method='covariance'`) but was not the default.
  Superseded by the weights for summed columns (§5.13), now the NIRCam
  default.
- In synthetic tests the estimated K is positive semi-definite but below the
  injected variance: background subtraction and the profile fit remove part
  of the offsets.

### 5.9 MIRI F1500W photometry (GJ 3929, integrations 60+)

| Radius (px) | Aperture | PSF-weighted |
| --- | --- | --- |
| 3 | 783 (1.17) | |
| 4 | 779 (1.26) | |
| 5 | 799 (1.30) | 754 (1.17) |
| 6 | 789 (1.28) | 769 (1.20) |
| 8 | 793 (1.29) | **737 (1.18)** |
| 10 | | 769 (1.25) |
| 12 | | 781 (1.27) |

Values in ppm (ratio to errors), from `optimise` on 3 × 300 integrations.

- **Final (aperture 4, PSF radius 8):** aperture 773 ppm (1.25),
  PSF-weighted **745 ppm (1.19)**. ApPhoto previously gave 740 ppm (4 px,
  group 0 start).
- PSF weighting gains ~4%. The remaining excess (~1.2×) is not reduced by
  weighting.
- Centroids (PSF-fit Gauss-Newton): rms 1.8 (x) and 3.0 (y) mpx.
- The whole run takes 6 s.

### 5.10 Performance

- **The PSF fit was slow** (LRS 1288 s, NIRCam 1666 s): 99% was in
  `cholesky_banded`. At these sizes (~500–1200 coefficients) the band is a
  quarter of the matrix, and the CV grid was re-solved on every rejection
  iteration.
- **Fixes:** dense Cholesky below 3000 coefficients (banded above), CV once
  per stage, coordinate search over the penalties.
- **Now:** LRS 4 s, NIRCam 24–29 s, NIRSpec 20–37 s, SOSS 286 s (two orders,
  with decontamination).

### 5.11 Field-star contamination, MASKFRAC cutoff and Q shrinkage (2026-11-13)

**Zeroth orders (HAT-P-30 SOSS, F277W exposure 04103 through Stage 1).**
- Not visible on the traces in the CLEAR data: the column-dependent PSF and
  the per-column flux absorb them (Q ≈ 1 there; residual ≈ 0 even without Q).
  Only F277W shows them, as the literature does.
- `find_contaminants(f277w)` finds 4 sources: (769, 92) S/N 31, across
  order 2 at ~1.05 µm; (1226, 137) S/N 17; (874, 56) S/N 6 on order 1 at
  ~1.97 µm; (291, 109) S/N 6, faint, on order 2. The target's own F277W
  light (order 1, x < ~480) is excluded; with the first padding (10 px) its
  far wing gave a false detection, so the pad is now max(10, PSF half-width).
- Effect (masked / unmasked median flux): order 2 columns 755-785 lose
  6-32% (the zeroth-order light), 11 columns are fully covered and NaN;
  order 1 columns 872-877 NaN; (291, 109) removes 2-4%.
- White light (segment 1): order 1 92.0 vs 90.9 ppm, order 2 175.9 vs
  166.0 ppm (fewer and partly masked columns; within the ~8% uncertainty of
  the p2p). The mask removes a dilution bias, not noise.

**Fully masked columns.** Masking the whole profile left the flux to far-wing
pixels with P ~ 0.001 (synthetic: errors of 20-40%). `extract.max_maskfrac`
(0.98) now sets such columns to NaN. Existing products: PRISM saturated cores
reach 0.94, G395H 0.97, all other modes < 0.82.

**Q shrinkage.** Q of faint pixels was noisy (12% at P ~ 0.001 with 120
integrations; SOSS Q rms 9% over all window pixels). Each factor is now
shrunk towards the local average with weight s²/(s² + var_q), var_q = 1/den,
s² from the best-exposed quarter of the pixels. G395H NRS1: 175.4 ppm
(ratio 1.050), unchanged; PSF χ² 0.77 → 1.22 (Q no longer fits noise).
SOSS: 90.9 / 166.0 ppm (was 90.7 / 164.7), Q rms 2.2%.

**Field-star traces (synthetic).** A parallel trace 6 px from the target
(30% of its flux), declared 0.6 px off: without the model the target is
biased by 27%; with `add_field_trace` the bias is 0.7% and the field star's
spectrum is recovered to 1%, dy to < 0.05 px. What was needed:
- the target PSF and Q are not fitted where the field model exceeds 5% of the
  target model (first iteration: a band around the declared trace), otherwise
  the flexible PSF absorbs the field star (a degenerate decomposition);
- a weak ridge (1e-3 of the mean diagonal, only with field traces) so the PSF
  decays to zero where no column has data instead of being extrapolated by
  the difference penalty (peak 0.8 and lobes of -0.55 without it);
- the field geometry extends the target's trace linearly beyond its column
  range (the field star covers other columns);
- the field model and the PSF alternate in the in-memory first stage (4
  iterations), then one streaming pass on field-subtracted data.
No real data set with a crossing trace has been tested yet.

### 5.12 Second review with the user (2026-11-13): points clarified

**White-light estimator (user saw 902 ppm instead of 175).** The user's
script took an inverse-variance weighted mean of the raw column fluxes,
with weights recomputed every integration. On the same file:

| White light | mad_std (ppm) | p2p (ppm) |
| --- | --- | --- |
| 1/σ² weights on raw flux, changing in time (user) | 902 | 919 |
| same, weights fixed in time | 255 | 239 |
| columns normalised first, 1/σ² weights fixed in time | 192 | 177 |
| plain sum (`Spectra.white_light`) | 190 | **175** |

- Weights ~1/flux make the mean dominated by the faintest columns (the
  faintest 10% carry 57% of the weight, ~16% error each); the ~0.4% changes
  of the per-integration variances (masking, trace position) move the mean
  because column fluxes differ by a factor 179.
- The values I report are p2p; mad_std also contains slow trends.
- Documented in extraction.rst ("White light from the saved spectra").

**Temporal rejection (§5.4), as explained.** Still residual-based, but the
residual `D − fP` of each detector pixel is compared with its own median
residual over the chunk: a static misfit (e.g. a 2% sensitivity difference,
~6σ on a bright pixel) cancels and no longer makes the pixel switch between
kept and rejected; a cosmic ray, local in time, is still rejected.

**`fit_background` (§5.7), as clarified.** Not a second background
subtraction: a constant per column per integration fitted together with the
flux inside the aperture, on top of the per-integration background step. It
showed that the SOSS per-column excess is column-wise offsets (leftover 1/f),
but the constant is degenerate with the wide SOSS profile (±15 px aperture),
so the white light got 2.8× worse. To be fixed at the group level in Stage 1.

**Disk space.** rateints + cleaned of one G395H segment take ~2.9 GB.
- `Stage1.run(save_variances=False)` drops `VAR_POISSON`/`VAR_RNOISE` (40%
  of the rateints; unused by jstark). Checked on the SOSS F277W exposure.
- `ExposureCleaner.run(remove_inputs=True)` deletes each rateints after the
  next segment (which borrows integrations from it) is cleaned; `INT_TIMES`
  are cached first and the info file is rewritten after every segment.
- `SpectralExtractor.run(keep_segments=False)` / `Photometry.run(...)`
  delete the per-segment spectra after combining.
- The cleaned cubes are the extraction input; keep them while settings are
  explored.

**Files in `<data dir>/reduction/`.** rateints (Stage 1, per segment),
stage1diag.npz, saturation_map, wavelength (Stage 2), cleaned (per segment:
SCI, ERR, BADPIX, CRMASK, INT_TIMES), cleaninfo (exposure summary, the
extraction entry point), stage1/stage2/extract logs, extract_model (TRACE,
PSF, VARMEAN, QMAP, CONTAM, FIELD, PARAMS), per-segment spec, combined
spectra. `._*` files are macOS metadata on the external disk.

**Spectra files** now carry `WAVE_O<n>` (µm per column; previously only in
the `TRACE_O<n>` table) and `FIELD_O<n>` when field-star traces are
modelled. `VAR_O<n>` is a variance.

**MIRI LRS range.** The automatic range was 4.06–13.9 µm. Below ~4.5 µm the
prism dispersion folds back (several wavelengths on the same pixels), 5–6 µm
receives some shorter-wavelength light (≤7%), and beyond ~12 µm the
throughput is low; the default is now 5–12 µm.

### 5.13 NIRCam: weights for summed columns and per-amplifier common mode (2026-11-13)

**Why the covariance extraction gave no gain (§5.8).** It minimised the noise
of each column, with C = diag(V) + K. Per column, K is negligible next to V.
But a row offset is shared by the ~500 columns of an amplifier, so in a sum
of B columns it adds up as B² K, while pixel noise adds as B V. The weights
that minimise the noise of the sum are w ∝ (V + B K)⁻¹ P: in the faint
wings, where V is mostly read noise, B K dominates and the weight drops; in
the core, photon noise dominates and nothing changes.

**Implementation.** `set('extract', method='covariance', cov_scale=B)`. The
saved variance is that of the column for these weights,
`wᵀ (diag(V) + K) w` with `w = C_B⁻¹ P / Pᵀ C_B⁻¹ P` (for B = 1 this equals
`1 / Pᵀ C⁻¹ P`; checked by Monte Carlo and in `test_extraction.py`).

**Results** (segment 1, Stage 1 with `wing_max=150`, white light with the
phase-of-4 pattern removed):

| `cov_scale` | F444W ppm (ratio) | F322W2 ppm (ratio) |
| --- | --- | --- |
| optimal | 182 (1.92) | 218 (2.60) |
| 1 | 183 (1.94) | 218 (2.63) |
| 50 | 174 (1.84) | 206 (2.47) |
| **500 (NIRCam default)** | **166 (1.74)** | **196 (2.33)** |
| 1500 | 165 (1.71) | 195 (2.29) |

Together with the Stage 1 `wing_max` default: F444W 243 → 151 ppm (ratio
2.55 → 1.57), F322W2 289 → 186 ppm (3.45 → 2.21) on segment 1
(detector_processing_notes.md §6.4.1). The extraction takes ~1.5-2 min per
segment (was ~1).

**Per-amplifier common mode for spectral light curves (light-curve stage, to
build).** All bins in one amplifier carry nearly the same additive 1/f
error. A regressor in each bin's fit = the white-light residual of the same
amplifier with the bin's own columns left out. Measured on 50-column bins
(phase-of-4 removed): 1.06 → 1.04 (F444W), 1.08 → 1.02 (F322W2) with the new
defaults; 1.21 → 1.06 with the old ones. It moves the shared noise into an
offset common to the bins of one amplifier, which must be reported (blocks
of 512 columns, ~0.4-0.5 µm); bins must not straddle amplifier boundaries.

### 5.14 First full exposure: WASP-39 b NIRSpec PRISM (2026-11-14)

All 4 segments (21,500 integrations), every stage with defaults, run in one
script (`<data dir>/reduction/comparison/run_prism_full.py`; products in
`/Volumes/Jayshil SSD/JWST/jwst-ers/Data/NRSPR/reduction/`, 11 GB). The
earlier segment-1 products are in `.../Data/reduction__NRSPR/`.

**Run time (16 min in total):** Stage 1 533 s (173 / 142 / 144 / 73 s per segment,
saturation pre-pass included), Stage 2 3 s, cleaning 138 s, extraction 290 s
(PSF fit over all 21,500 integrations 78 s, extraction 212 s).

**Pipeline behaviour over the whole visit.**
- Saturation map: 270 pixels (the same as from segment 1); 18 have only group 0.
- Cleaning: 1.2–1.3 transient flags per integration. Segment 4 shows 3.0,
  because 5658 of its 9488 flags fall in integrations 20751–20761 (the
  high-gain antenna (HGA) move reported by Rustamkulov et al. 2023, which
  excluded integrations 20750–20765).
- Extraction: trace rms 0.007 px; dy rms 4.2–4.4 mpx in every segment;
  0.0–0.1 rejected px/int; extraction χ² median 1.15 (flat; spike at the
  HGA move). The HGA move shows in dx (0.18 px), dy (0.03 px) and ds.
- ds falls by ~0.0015 during the transit (the PSF narrows at lower flux:
  brighter-fatter); dx shows a small transit-shaped signal too.
- **Segment boundaries:** in the ratio jstark / published (2–5.35 µm), the
  steps are +2 ± 13, +22 ± 14 (mid-transit) and +13 ± 13 ppm. Nothing
  significant.
- **Early ramp:** the first 500 integrations are +520–560 ppm above the
  linear baseline, the same in the published light curves (they trimmed
  2000–3000 integrations). Linear slope −193 ppm/h (2–5.35 µm), published
  −189 (Rustamkulov et al. removed "200 ppm/h" for display).

**Comparison with the published light curves.** Reference: the ERS
benchmark (Carter & May et al. 2024, Zenodo 10.5281/zenodo.10161743),
Eureka! Stage 4 light curves from the FIREFLy reduction, 147 native-resolution
bins, unbinned, same time stamps. A copy is in `comparison/published/`.
No transit fit was done: our columns were summed into the same 147 bins;
baselines are linear in time, fitted out of transit after integration 3000
(benchmark ephemeris: T0 = 2459771.335647, a/Rs 11.390, b 0.4498 → T14
2.87 h, T23 1.98 h). "Depth proxy" = 1 − mean normalised flux in the
central half of the flat bottom. It is not a transit depth (limb darkening
makes it ~2000 ppm larger than (Rp/Rs)²), but the same estimator on the same
photons isolates differences between reductions. Script:
`comparison/compare_prism.py`; numbers in `comparison_summary.json`.

| Quantity | jstark | published |
| --- | --- | --- |
| White light 2–5.35 µm, out-of-transit scatter per integration | **471 ppm** (p2p/err 1.08) | 487 ppm |
| same, 10-integration bins | 166 ppm | 174 ppm |
| White light 2.3–5.35 µm (unsaturated) | 538 ppm (p2p/err 1.05) | 550 ppm |
| White light 0.52–5.35 µm | 385 ppm (p2p/err 1.17) | 410 ppm |
| White light red noise, β at 64 / 256 integrations (baseline per stretch) | 1.12 / 1.27 | 1.12 / 1.23 |
| Depth proxy, 2–5.35 µm white light | 22,350 ppm | 22,375 ppm |
| 147 bins: median scatter ratio jstark / published, > 2.3 µm | 0.997 | |
| 147 bins: median p2p / propagated error | 1.00 (0.98 > 2.3 µm) | |
| 147 bins: median β (64–512 integrations) | 0.99 | 0.99 |
| Depth proxy jstark − published, > 2.3 µm | median −37 ppm; scatter well below the white-noise error | |

- **Outside the saturated region the two reductions agree:** noise,
  red noise and depths are the same per bin, and the 2D light curve
  (`comparison/lc2d.png`) shows the same features (CO2 at 4.3 µm, H2O).
- The benchmark white light (`lc.dat`, 2–5.5 µm, 10 integrations per point,
  jitter 219 ppm) scatters by 301 ppm per point, against 166–174 ppm for both
  summed bin sets. It is apparently built differently; it is not used for
  the comparison.

**Saturated region (0.63–2.06 µm): jstark is less noisy but biased too
deep.** Bins grouped by the first saturated group of the brightest pixel
(from `saturation_map`):

| First saturated group | Columns (µm) | Depth proxy − published: optimal / box | Published dilution correction (benchmark) | Scatter optimal / box / published (ppm) |
| --- | --- | --- | --- | --- |
| 1 (only group 0) | 99–101, 107–109 (1.09–1.26) | +6700 / +6300 | not given | 3650 / 3650 / 2690 |
| 2 | 88–121 (0.93–1.50) | +1540 / +530 | +140 | 2320 / 2460 / 2710 |
| 3 | 69–87, 122–134 | +500 / +240 | +137 | 2220 / 2270 / 3380 |
| 4 | 63–68, 135–140 | +33 / +19 | +107 | 2040 / 2030 / 2105 |
| unsaturated | | −6 / 0 | +63 | equal |

- **Mechanism, seen in per-row depths:** the depth varies strongly across the
  profile even in unsaturated columns (col 60: core row 21,200 ppm,
  neighbours ~28,700 ppm), because the PSF narrows in transit (ds) and
  charge moves back from the wings into the core. A plain column sum is
  immune. Optimal weights are not uniform, so they pick up the deeper wings
  where the core is down-weighted (fewer groups → larger variance) or masked
  (group-0-only cores, maskfrac 0.88 < `max_maskfrac` 0.98: the flux comes
  from rows 13/17, depths 32,000–39,700 ppm).
- **Box sums** remove ~60% of the excess in the group-2/3 columns but are
  still above the published values by more than the benchmark's dilution
  correction. The published reduction flags whole columns at the first
  saturated group (same groups for every pixel of a column), which avoids
  mixing ramps of different length within a column but is noisier (+50% in
  group-3 bins).
- **The steps in the depth proxy fall exactly** where the first saturated
  group changes (col 88: 3 → 2, col 122: 2 → 3), so the bias is tied to the
  number of groups.
- Which saturated-region depths are right can only be settled against the
  unsaturated NIRISS SOSS spectrum after light-curve fitting (benchmark:
  the published PRISM is diluted by 64–140 ppm there).

**Conclusions.**
- The full-exposure machinery works: segment handling, the exposure-wide
  saturation map, cleaning padding, one PSF from 21,500 integrations, and
  combining. No boundary artefacts.
- Unsaturated wavelengths match the published reduction, at the same or
  slightly lower noise (white light −2 to −6%).
- Saturated PRISM columns need a different treatment (§6, open item).
  Depths at 0.7–1.8 µm are biased deep by up to ~1500 ppm (optimal).

**Implemented the same day (agreed with the user; options (a) and (e)).**
- Stage 1 writes `NGROUPS` into `<exposure>_saturation_map.fits`. The
  cleaner copies the map into `cleaninfo` as `SATMAP`; older info files
  fall back to the map file next to them.
- `SpectralExtractor.column_saturation(order)`: the fewest usable groups of
  any pixel within `y0 ± aperture`, per column. It is saved as `SATGRP_O<n>`
  (header `NGROUPS`, `MINGRP`) and read as `Spectra.satgrp` /
  `Spectra.ngroups`.
- New `extract` parameter `min_groups` (default 2): columns below it are NaN
  (flux, var, box, opt) in all integrations; `run()` lists them.
- WASP-39 b re-extraction: exactly columns 99–101 and 107–109
  (1.088–1.263 µm) became NaN; all other columns are bit-identical.
  `SATGRP` histogram: 1 group 6 columns, 2 groups 28 (0.93–1.50 µm),
  3 groups 32 (0.74–1.76 µm), 4 groups 12 (0.70–1.87 µm), unsaturated 392.
  Full white light (0.52–5.35 µm): 385 → 345 ppm (p2p/err 1.17 → 1.10).
  One of the 147 published bins (1.25 µm) is now empty.
- Options (b)–(d) wait for the light-curve fitting tools (checked against
  NIRISS SOSS).

### 5.15 PRISM saturated columns, part 2: options (b)-(d) judged with transit fits (2026-11-15)

Reference: the NIRISS SOSS spectrum of the ERS benchmark (Carter & May
2024, `Fit_LimbDarkening`, bins_scale1; order 2 below 0.85 um, order 1
above; downloaded into `comparison/published/ZENODO/4_TRANSMISSION_SPECTRA/`),
rebinned onto the 147 PRISM bins by inverse-variance weighting. Each PRISM
spectrum gets its own offset to SOSS from the unsaturated bins SOSS covers
(0.60-0.62 and 1.9-2.8 um). The reference is only good to ~±300 ppm per bin:
even at 1.9-2.1 um, where nothing saturates, every PRISM reduction sits
300-400 ppm below SOSS. Transit fits: `fits/fit_spec_variant.py` (same setup
as `fit_spec_prism.py`, bins 0-58); comparison `fits/compare_sat_variants.py`
-> `fits/compare_sat/` (`compare_sat.json`, `compare_sat.png`). Extraction
variants: `comparison/extract_variant.py`, `comparison/run_variant_colsat.py`,
products in `reduction/variants/`.

Mean PRISM - SOSS (ppm, after the offset) by the bin's fewest usable groups:

| Variant | 1 group (n=1) | 2 groups (9) | 3 groups (11) | 4 groups (4) | all saturated (33) | chi2_red, saturated |
| --- | --- | --- | --- | --- | --- | --- |
| ERS benchmark | -234 | -272 | -273 | -78 | -219 | 15.6 |
| jstark default (optimal) | +1714 | +1127 | +260 | -38 | +391 | 161 |
| (d) whole columns flagged at the first saturated group | -708 | -389 | -490 | -129 | -359 | 50 |
| **(b) `saturated_box`** | +49 | +147 | -83 | -95 | **-48** | **12.8** |

- **(b) uniform weights in saturated columns works.** The new `extract`
  option `saturated_box` uses the box sum in every column whose aperture
  holds a saturating pixel (72 columns, 0.70-1.87 um); the other columns stay
  optimal. The mean bias falls from +391 to -48 ppm, and chi2 is below the
  ERS spectrum's. The errors do not grow: the median depth error of the
  saturated bins is 1.01x the default and 1.03x ERS. The unsaturated bins
  are unchanged (|difference| <= 5 ppm, i.e. sampler noise). Bin-to-bin
  structure of about ±400 ppm remains at 0.9-1.5 um (+400 at 1.0 and
  1.2 um, -380 at 1.55 um). That is comparable to the ±300 ppm the
  reference shows in unsaturated bins.
- A wider box (aperture 7, background from 9 px) changes the depth proxy by
  <= 20 ppm against aperture 5, so the aperture-5 box already holds the
  light that moves.
- **(d) the published approach is biased shallow** by 360 ppm on average,
  with a saw-tooth reaching -700 to -950 ppm. Flagging whole columns at the
  first saturated group was done with a column-wide saturation map passed
  to `consistent_saturation(saturation_map=...)` (no code change). This
  reproduces, and exceeds, the dilution of the ERS spectrum (-219 ppm,
  their dilution correction +64-140 ppm). So the dilution comes from
  slopes over the first 1-3 groups of unsaturated pixels, not from the
  extraction. Not pursued.
- **(c) PSF width change (`ds`) in the profile was tried and dropped.** The
  per-integration ds from `refine_shift`, applied to first order
  (`P - ds (P + u P')`) with a second optimal pass, removed only 160-290 ppm
  of the saturated excess. It also moved the unsaturated bins by +186 ppm
  away from the box sum, which is immune to the profile there. The
  in-transit profile change is not a uniform width scaling. The code was
  reverted.
- The depth proxy of §5.14 (light-curve level, no fit;
  `comparison/proxy_variants.py`) and the transit fits agree on the shifts
  between variants to <= 50 ppm. Against the default, at 2 / 3 groups:
  `saturated_box` -997 / -367 (proxy) against -980 / -343 (fits); (d)
  -1568 / -740 against -1516 / -750. So the proxy is enough for screening
  (minutes instead of an hour per variant).

### 5.16 One PSF per visit: `VisitExtractor` (2026-11-16)

**Problem** (WASP-43 b MIRI/LRS phase curve, 3 exposures; details in
`fitting_notes.md` §5.7). The optimal white light stepped by -2.48 ppt
(exp 1->2) and +0.39 ppt (exp 2->3) at the exposure boundaries; the box flux
only -0.46 / -0.01 ppt. The background, trace (< 0.001 px), and pixel-response
map mean (0.08 ppt) were the same; the per-exposure PSF profiles differed by
~1% (|dP|/P ~ 10-11 ppt). The optimal flux scales with the profile, so a PSF
fitted per exposure rescales each exposure by its own PSF noise. A phase-curve
fit absorbed the steps (nightside 243 instead of ~1600 ppm).

**Test.** Exposures 2-3 re-extracted with exposure 1's model (`load_model`):
steps -0.48 / +0.11 ppt (= box).

**Implementation.** `SpectralExtractor.fit_psf(others=[...])`: the first fit
samples `init_ints` spread over all exposures (in proportion to their
integrations), the streaming pass reads every exposure (`ints` likewise),
each through its own `_prepare` (background, centre-of-flux offsets) and
`_psf_table` (own trace, own `shat` normalisation), all into one set of PSF
normal sums and one pair of Q sums; outlier scale shared. Each member gets
the same PSF (copies) and Q, its own `var_mean`, and `psf_exposures` (saved
in the extraction model and in the spectra `PARAMS`). Checks: same mode,
frame, orientation, orders, trace columns and PSF parameters; field-star
traces not supported (NotImplementedError). `VisitExtractor(infos,
shared_psf=True)` (default on) wraps one `SpectralExtractor` per exposure:
`set` for all, `find_trace` gives the others the first exposure's columns,
`fit_psf` joint, `run` per exposure. `shared_psf=False` = one PSF per
exposure. Single-exposure `SpectralExtractor` use is unchanged (the 15
extraction tests pass; 2 new tests).

**WASP-43 b** (`reduction/scripts/extract_visitpsf.py` ->
`reduction/visitpsf/`): joint PSF from 9219 integrations (31.6 M pixels,
3x one exposure) in 32 s, reduced chi2 1.495 (per exposure 1.486-1.492);
Q rms 0.038 (per exposure 0.028-0.029: less shrinkage with 3x the data).
Boundary steps -0.475 / +0.118 ppt (box -0.448 / -0.008), the same as with
exposure 1's PSF. Phase-curve results: `fitting_notes.md` §5.7.

### 5.17 Full G395H exposure: WASP-39 b and the mirror tilt (2026-11-20)

**Data.** ERS 1366 obs 3, NRS1 + NRS2, SUB2048, NRSRAPID, 70 groups x 465
integrations (63 s each), 3 segments of 155. Products in
`/Volumes/Jayshil SSD/JWST/jwst-ers/NRSG395H/Outputs/{NRS1,NRS2}/`; scripts
in `Outputs/scripts/` (`run_g395h_full.py`, `redo_satmap_extract.py`,
`compare_g395h.py`, `tilt_test_noreject.py`, `calibrate_breaks.py`); logs in
`Outputs/logs/`. Whole pipeline per detector 9.5 min (Stage 1 ~8.5 min,
cleaning 13 s, extraction 30 s), peak RSS 13.5 GB with `ramp_fit
maximum_cores='4'`. Detector-level results: detector_processing_notes.md
§6.11.

**Mirror tilt at integration 269** (Alderson et al. 2023 report it). Between
integrations 268 and 270 the PSF width regressor `ds` jumps from -0.0147 to
+0.0086 (NRS1), the trace moves by +12 mpx; integration 269 is intermediate
(the tilt happened during it). Box white light jumps by -1900 / -1960 ppm
(NRS1 / NRS2). With the default extraction (one exposure-wide PSF, temporal
rejection over the chunk):
- **The temporal rejection breaks** in segment 2, which holds 115 pre-tilt
  and 40 post-tilt integrations: each pixel's median residual over the
  chunk is the pre-tilt one, so 2,500 pixels per integration (most of the
  aperture) were rejected in integrations 270-309 (NRS1; NRS2 98).
  Optimal/box white light there: -450 +- 300 ppm. Segment 3 (all post-tilt)
  had no rejections.
- **One PSF fits neither side**: PSF chi2 5.5 (NRS1), extraction chi2 4.3
  before / 7.0 after; the optimal/box ratio steps by -215 / -250 ppm after
  the tilt, varying with wavelength (-95 to -427 ppm per 0.17 um band).
- Effect on depths (vs the ERS light curves, below): optimal +150 to +340
  ppm per band (weighted means; band medians up to +1170) on NRS1, +170 ppm at
  3.8-4.2 um on NRS2. Box depths were unaffected.

**New: PSF epochs** (`set('psf', breaks=[...])`, global 0-based integration
numbers at which a new epoch starts; user decision 2026-11-20: manual breaks
plus a warning, not automatic). `fit_psf` runs the fit separately per epoch
on that epoch's integrations (`ExposureSource.chunks/sample_ints` take
`lo`/`hi`; `_fit_range`); each epoch keeps its PSF, Q and var_mean
(`ex.epochs`), activated per chunk (`_use_epoch`); chunks never span a
break (`chunks(breaks=...)`, also in `_block_chunks` for `optimise` and the
covariance estimate). Spectra get an `epoch` column in `REG_O<n>`; the
extraction model saves epoch 0 under the old names and later epochs as
`PSF_O<n>_E<k>`, `QMAP_E<k>`, `VARMEAN_E<k>`; `load_model` restores them
and fills parameters missing from older model files with the defaults.
Not supported (NotImplementedError): breaks with a PSF shared between
exposures, and with field-star traces.

**Break warning.** `metrics.find_psf_breaks`: local step of the median over
w = 5 integrations on either side, in units of its noise (p2p), threshold
20, and the step must persist over 25 integrations (half of it at least).
Calibrated on nine full exposures (`calibrate_breaks.py`): the tilt scores
91 / 63 in `ds` (11 / 10 in `dy`); everything else stays below 10 (PRISM
HGA move 9.8 in dy, TOI-561 G395H 8.2, G395M 7.8, MIRI LRS 5.2, NIRCam 3.2,
SOSS 3.6). `run()` warns with the suggested `set('psf', breaks=[...])` when
a step is found that no declared break covers. On the G395H run without
breaks it suggests 270 (the local maximum next to the intermediate 269);
269 was used.

**With `breaks=[269]`:** PSF chi2 2.21 / 2.24 (NRS1 epochs), 1.54 / 1.58
(NRS2); extraction chi2 1.05 on both sides; rejections back to 1-2 px/int
except integration 269 itself (2083 / 120 px; exclude it in fits);
optimal/box flat within each epoch (rms 11-14 ppm) with a constant +1050 /
+1290 ppm step at the tilt. The optimal white light now jumps by -885 /
-702 ppm against the box's -1903 / -1960: the per-epoch profile (unit
integral over the +-7 px window) keeps the light the wider post-tilt PSF
moves beyond the +-5 px box. Either way a step regressor at 269 is needed
in the light-curve fits. Synthetic test `test_psf_breaks` (PSF 12% wider and
0.05 px lower from integration 150 of 240, segments of 80): warning raised;
with the break opt/box flat to < 1e-3, both epochs chi2 < 2, epochs survive
save/load.

**Comparison with published light curves** (`compare_g395h.py`,
`tools/ratio_depth.py`; our / reference with a fixed transit shape, ERS
orbit and ERS exotic-ld LD per bin, linear in time plus a step at 269;
integrations < 15 and 269 not used; `Outputs/comparison/compare_lcs.json`).
The ERS benchmark G395H light curves are Eureka! Stage 4 (465 ints); the
Alderson et al. 2023 Zenodo (7185300) has the ExoTiC-JEDI 10-px light curves
(Grant; 447 ints) and 11 reductions' spectra. Columns: the ERS `x` is our
column - 5 (from the shared noise: correlation 0.90 / 0.96 at that shift,
~0 at the neighbours); for the same photons the wavelength solutions agree
to < 0.1 nm.

| Detector, band (um) | ours - ERS native bins (optimal) | ours - ExoTiC-JEDI 10 px (optimal) |
| --- | --- | --- |
| NRS1 2.70-3.00 | -9.5 +- 5.2 | -17.0 +- 7.2 |
| NRS1 3.00-3.30 | -1.3 +- 3.6 | -6.0 +- 5.2 |
| NRS1 3.30-3.72 | +4.8 +- 3.2 | +4.0 +- 4.4 |
| NRS2 3.82-4.20 | +3.8 +- 4.3 | +3.7 +- 4.9 |
| NRS2 4.20-4.60 | -10.6 +- 6.5 | -17.4 +- 7.6 |
| NRS2 4.60-5.18 | -7.5 +- 8.7 | -1.9 +- 10.0 |

- Depths agree with both reductions to <= 17 ppm per band (reduced chi2
  0.8-1.4 per bin), much closer than PRISM (fitting_notes 5.6).
- Noise per bin ours / reference 1.00-1.01 (5-95%: 0.92-1.08); residual
  correlation 0.96-0.98. Box and optimal give the same depths and noise
  here (photon-limited 70-group ramps; white-light p2p box 124 vs optimal
  128 ppm on NRS1).
- Our p2p / propagated error: 1.04-1.06 in native bins, 1.10-1.13 at R100
  (some correlated noise in wider bins).

### 5.18 `overwrite`: finished steps are loaded (2026-11-22)

**Request.** Re-opening a notebook should not re-run everything. Before, all
`run` methods re-ran; `overwrite=False` of `Stage1.run` only made the final
save fail. **User decisions:** `overwrite=False` everywhere by default; an
error if settings or inputs differ (as in the fits); `overwrite=True` re-runs
identical fits too; products without a record load with a notice.

**Implementation** (`jstark/reduction/products.py`). One
`<exposure>_products.json` per output folder: per product its settings, a
content hash (`uid`), the `uid`s of its inputs, size, and status (`writing` /
`done`, so interrupted files are redone). Content hashes mean that making an
identical file again does not invalidate what follows. User guide:
reduction.rst, "Running again".
- **Stage 1:** restores masks, wing threshold and group range from the first
  finished segment (record + stage1diag.npz; npz + log for old products), so a
  resumed run equals an uninterrupted one. The saturation map is described by
  the segments it was made from.
- **Cleaner:** info file written after pass 1 and after each segment
  (atomically); segments of earlier sessions stay listed (before, `run
  (segments=[1])` in a new session dropped the others, and the same for
  spectra). Constructors work when `remove_inputs` deleted the rateints.
- **Extraction:** the model file stores `psf_record` (identifier, settings at
  the time of the fit, inputs) and `fit_info`; QMAP / VARMEAN now in double
  precision, so extraction from the reloaded PSF is bit-identical. Spectra
  depend on the PSF identifier. `fit_psf` / `optimise` recompute on changed
  settings; a model that existing spectra were made from is not replaced
  without `overwrite=True`. New files: `<exposure>_optimise.json`,
  `<exposure>_phot_model.fits`, `<exposure>_phot_optimise.json`.
- **Not detected:** changes of the jstark code or the CRDS context. Memory
  limits are ignored, although the chunk size enters the temporal rejection.
- **Bug fixed on the way:** `optimise` restored the background mask but not
  `shat` / `ref` (those of the last background tested stayed); effect on
  G395H fluxes 1e-11.

**Checks.** `jstark/tests/test_products.py` (27 tests). G395H NRS1 segment 2:
first session 3.5 min, second 0 s, PSF fitted / reloaded / refitted give
identical fluxes; spectra equal the tutorial's to 1e-9. A copy of the full
NRS1 exposure of 2026-11-20 (no records) loads at every stage.

## 6. Open issues and next steps

- [x] PRISM saturated columns, part 1 (§5.14): (a) columns with
      < `min_groups` (2) usable groups in the aperture are NaN;
      (e) `SATGRP_O<n>` saved.
- [x] PRISM saturated columns, part 2 (§5.15): (b) `saturated_box` adopted
      (option, default off until decided); (c) dropped; (d) biased
      shallow, not pursued.
- [ ] `saturated_box` stays an option, default off (user decision
      2026-11-15). Test it on another published saturated data set first;
      make it the default only if it holds there. The alternative with one
      method everywhere is the box sum for all columns: +5% noise per bin
      (median over unsaturated bins, 5-95%: 0-18%), +2.5% in the
      white light.

- [ ] Full exposures (all segments): red noise, PSF stability over the visit
      (`ds`, `chi2` regressors), the optimiser on several blocks. **Done for
      PRISM (§5.14) and G395H (§5.17)**; the other modes remain.
- [x] Abrupt PSF changes (mirror tilt): `psf breaks` epochs + warning
      (§5.17).
- [x] G395H: light-curve fits with a step at the tilt; spectrum vs the ERS
      benchmark (R100 and native bins) and the 11 Alderson et al.
      reductions: agree, fitting_notes §5.8.
- [ ] SOSS: column-wise residual 1/f (per-column ratio 1.2–1.4). Revisit the
      group-level `diff` / `median` choice of Stage 1 on a full exposure;
      a scaled zodiacal background model.
- [x] SOSS field-star contaminants: zeroth orders masked from the F277W
      exposure, field-star traces modelled (§5.11).
- [ ] SOSS: the PSF wings beyond the ±16 px window reach the background
      region (residual image, §5.11); check whether this matters on a full
      exposure (it is proportional to the stellar flux, so it mostly rescales
      each column).
- [x] NIRCam white-light excess: Stage 1 offsets measured on the trace
      wings plus weights for summed columns (§5.13): ratio 2.55 → 1.57
      (F444W), 3.45 → 2.21 (F322W2), phase-of-4 removed.
- [x] `_blocks` (covariance estimate, `optimise`): blocks overlapped when the
      exposure is shorter than nblocks × block_len (55 Cnc, 742 integrations:
      "900 integrations"); now shrunk to nint // nblocks.
- [ ] NIRCam: phase-of-4 regressor and per-amplifier common mode (§5.13) in
      the light-curve stage; rerun the remaining excess analysis on a full
      exposure.
- [ ] Field-star traces: tested only on synthetic data; validate on a real
      SOSS data set with a crossing trace (e.g. WASP-96).
- [x] MIRI LRS: default `wavelength_range=(5, 12)` (fold-over of the prism
      dispersion below ~4.5 µm, short-wavelength light at 5–6 µm, low
      throughput beyond 12 µm).
- [ ] G235H / G235M once data exist.
- [ ] Light-curve fitting stage: dilution correction from `CONTAM_O<n>`,
      regressors (`dy`, `ds`, `dx`, NIRCam phase-of-4), PLD jointly with the
      eclipse model.
- [ ] Report the stark bugs (§3.13) upstream.

## 7. Quick usage

```python
from jstark.extraction import SpectralExtractor, Photometry

ex = SpectralExtractor('/data/NRS1/reduction/<exposure>_cleaninfo.fits')
ex.show(); ex.find_trace(); ex.fit_psf()
ex.optimise(apertures=[3, 4, 5, 6], bkg_inner=[6, 8, 10])
spec = ex.run()                       # all segments
wl, wl_err = spec.white_light(); wave, lc, lc_err = spec.bin(nbins=30)

ph = Photometry('/data/F1500W/reduction/<exposure>_cleaninfo.fits')
res = ph.run(); res.flux['psf'], res.flux['aperture']

# NIRISS/SOSS with field stars (F277W exposure through Stage 1 first)
ex = SpectralExtractor(info); ex.find_trace()
ex.find_contaminants(f277w_rateints)      # zeroth orders -> masked
ex.add_field_trace(order=1, dx=..., dy=...)  # only if a field trace crosses
spec = ex.run(keep_segments=False)
```

## 8. Test data products

The script used to re-run Stage 1, Stage 2 and cleaning on segment 1 of every
dataset (outputs in `<data dir>/reduction/`): per mode,
`Stage1(d, outdir=d + '/reduction', exposure=...)`, then
`compute_saturation_map(segments=[0])`, `run(segments=[0])`,
`Stage2(out).run()` and `ExposureCleaner(out, wavelength=...).run()`.
Exposure selections: G395M `exposure='nrs1'`, SOSS `exposure='04102'`.
Run times (Stage 1, s): NRS1 152, NRS2 138, PRISM 151, G395M 61, F444W 390,
F322W2 217, SOSS 132, LRS 222, F1500W 251.

The SOSS F277W exposure (04103) was processed with
`Stage1(<uncal>, outdir=...).run(save_variances=False)` into the scratch
area for the contamination tests (§5.11); scripts that call `Stage1.run`
need `if __name__ == '__main__':` (ramp fitting uses multiprocessing).
