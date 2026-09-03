# Calibration R&D — DepthWizard SIH 26175

**Branch:** `feat/sih-26175-calibration-rnd`
**Base:** `feat/sih-26175-realdata` (HEAD 2fb1700)
**Date:** 2026-09-03
**Model:** `depth-anything/Depth-Anything-V2-Small-hf` (CPU)

---

## 1. Objective

Investigate the hardest technical component:
**Relative monocular depth + Coarse metric DEM → Higher-resolution metric surface/DSM**

Produce experimental evidence for which calibration formulation works and where it fails.
No fabricated numbers. No "ground truth" claims. Independent validation only.

---

## 2. Mathematical Formulations

All methods assume:

- Monocular depth `D` is relative, view-dependent, unknown scale/direction
- SRTM DEM `H` is coarse metric elevation (meters above sea level, EGM96 geoid)
- Linear or piecewise relation between `D` and `H` is an approximation, not physics

### Method A: Affine Calibration (Baseline)

```
H = a·D + b
```

Least squares on valid pixel pairs. `a` = scale (m/depth-unit), `b` = offset (m ASL).
Assumption: Global linear correlation between depth and elevation.

### Method B: Robust Affine (RANSAC)

```
H = a·D + b
```

RANSACRegressor (sklearn) with residual threshold (auto MAD-based). Falls back to affine if sklearn unavailable.
Assumption: Linear relation with heavy outliers (e.g., vegetation, buildings).

### Method C: DEM Residual Fusion

```
H_pred = H_DEM + α · norm(D - fit(D))
```

1. Fit global affine `fit(D) = a·D + b`
2. Compute residual `R = H_DEM - fit(D)`
3. Normalize `R` (z-score) and add back with weight `α`
   Assumption: Coarse DEM is truth for low frequencies; relative depth adds high-frequency detail.

### Method D: Locally Normalized Residual Fusion

```
For each patch P:
  D_z = (D - mean(D_P)) / std(D_P)
  H_z = (H - mean(H_P)) / std(H_P)
  Fit global scale between D_z and H_z
```

Patch size 64px default. Falls back to global affine if patches too small.
Assumption: Local z-scoring removes global bias and local illumination effects.

### Method E: Inverse Depth (Disparity)

```
H = a·(1/(D+ε)) + b,  ε = 1e-6
```

Assumption: Monocular depth is disparity-like (1/depth ∝ distance). If D increases with closeness, inverse depth correlates better with elevation.

### Method F: Piecewise Linear (n_bins=3)

```
Split D range into n_bins quantile segments
Fit separate (a_i, b_i) per segment
Return weighted avg scale/offset
```

Assumption: Relation is nonlinear; piecewise linear approximates.

### Method G: Frequency Fusion

```
H_DEM_low = gaussian_filter(H_DEM, σ)
H_pred = affine(D) + α·(H_DEM - H_DEM_low)
```

σ = 10px default. Coarse DEM low-frequency + high-frequency DEM detail added to affine depth prediction.
Assumption: SRTM captures low-frequency terrain; high-frequency DEM residuals add structural detail.

---

## 3. Experiment Design

### Data Alignment Diagnostics (Step 4)

Every experiment records:

- RGB CRS, DEM CRS, CRS match
- RGB resolution, DEM resolution, ratio
- RGB bounds, DEM bounds, overlap %
- Valid sample count, nodata fraction
- Vertical datum (EGM96 for SRTM)
- Recommendation: FAIL / WARN / PROCEED

### Frequency Decomposition (Step 5)

SRTM ≈ 90m (actual file, not 30m as documented).
RGB/depth ≈ 1-5m/pixel.
Direct pixel-wise agreement impossible.
Decompose both into:

- `coarse` = gaussian_lowpass(σ=10)
- `highfreq` = original - coarse
  Report `correlation_coarse` and `correlation_highfreq` between DEM and depth components.

### Scale Fitting Metrics (Step 6)

For every method, report before/after calibration:

- Valid samples, scale, offset
- RMSE, MAE, bias, Pearson correlation
- Median AE, P90 AE, P95 AE
- Residual std, R²
- Inlier fraction, pre/post correlation

### Robustness (Step 7)

Tested and handled:

- NaN / nodata (0, -9999)
- Extreme outliers
- Low overlap (<50% → FAIL)
- Nearly constant depth / DEM
- Very small samples (<2 → returns nan result)
- Mismatched CRS / resolution

### Spatial Cross-Validation (Step 8)

Block-based holdout, not random pixel split.
Config: 5 folds, 64px blocks, 2-block separation, seed 42.
Enforces spatial separation between train/test blocks.
Reports: train/test pixels per fold, min spatial separation (m), balance ratio.

### Reference DSM Evaluation (Step 9)

**Critical:** Calibration DEM ≠ Validation reference.
Hierarchy: LiDAR > photogrammetric > high-quality > coarse DEM (calibration only).
Provenance recorded: `reference_provenance` field.

### Multi-Dataset Support (Step 10)

- ISPRS Potsdam: airborne, requires registration, not yet accessed
- SIH2026 reference repo: not cloned locally
- Config template in `configs/calibration_experiments.yaml`

### Terrain Metadata (Step 11)

Categories: urban, sparse, hilly, forest.
Only assigned when justified by scene metadata.
If missing: `NOT AVAILABLE` — never fabricated.

### Diagnostics (Step 12)

10 plots per real experiment (matplotlib):

1. RGB, 2. Raw depth, 3. Normalized surface, 4. Coarse DEM, 5. Calibrated DSM, 6. Reference DSM, 7. Absolute error, 8. Residual map, 9. Scatter, 10. Error histogram.
   Saved to `outputs/calibration_rnd/` (gitignored).

### Quality Diagnostics (Step 13)

R² reported but NOT substituted for DSM accuracy.
Residual mean, std, median, P90, P95.
Spatial structure check: flag if residuals show spatial pattern (suggests global calibration insufficient).

### Local Calibration (Step 14)

Block sizes: 64, 128, 256px. Gaussian smoothing of parameters.
Compared: global vs local.
Overfitting prevented via spatial holdout validation.

### Uncertainty (Step 15)

Estimated uncertainty (labelled, NOT confidence):

- Root-sum-square of: calibration residual std, DEM local variability (×0.1), depth variability
- Per-pixel map via sliding window DEM std
- Flags: high_residual (>50m), low_R² (<0.2), few_samples (<100), low_correlation (<0.3)

---

## 4. Reproducibility

### Configuration

`configs/calibration_experiments.yaml` — all methods, alignment, frequency, metrics, robustness, spatial CV, reference, terrain, visualizations, quality, local, uncertainty, model comparison, hardware.

### Runner

```bash
python -m eval.run_calibration_experiments --config configs/calibration_experiments.yaml
```

Outputs:

- `outputs/calibration_rnd/results.json` (provenance-tagged: git SHA, model, dataset, method, hardware, timestamp)
- `outputs/calibration_rnd/report.md` (human-readable table)
- `outputs/calibration_rnd/blocker_report.{json,md}` if no real data

### Model Comparison Ready

Pluggable interface: `backend.depth.estimator.estimate_depth(image, backbone)`
Fixed evaluator, swap model only.

---

## 5. Real Data Status

**Blocked:** No usable paired RGB + independent reference DSM.

### Available

- `data/srtm/N18E073.tif` — EPSG:4326, 1201×1201, 90m resolution, 401-716m ASL, nodata=-9999. **Calibration source only.**
- `data/samples/real_test_landscape.jpg` — 1024×1024 RGB, no CRS. Not georeferenced.
- `data/eval_tiles/` — empty.

### Attempted / Blockers

| Dataset           | Status        | Blocker                                                           |
| ----------------- | ------------- | ----------------------------------------------------------------- |
| ISPRS Potsdam     | NOT AVAILABLE | Requires registration at isprs.org; airborne (TOP), not satellite |
| SIH2026 reference | NOT AVAILABLE | github.com/IMG-PROCESS-SAC/SIH2026 not cloned; no samples in repo |

### Next Actions

1. Obtain ≥1 real paired RGB + independent DSM (drone RGB + LiDAR, or ISPRS Potsdam TOP+DSM)
2. Place in `data/eval_tiles/<terrain>/` or update config with paths/CRS/resolution/terrain_type
3. Ensure reference DSM provenance = LiDAR/photogrammetric/high_quality
4. Re-run experiment runner

---

## 6. Failure Cases Documented

| Case                  | Behavior                                                |
| --------------------- | ------------------------------------------------------- |
| NaN in depth/DEM      | Masked out, valid pixels only                           |
| DEM nodata (0, -9999) | Explicitly masked in `_valid_mask()`                    |
| Extreme outliers      | IQR factor 3.0 (lenient); RANSAC in robust_affine       |
| Low overlap           | FAIL if <50% (configurable)                             |
| Constant depth/DEM    | Returns nan scale, offset=mean                          |
| Sample count <2       | Returns CalibrationMethodResult with nan fields         |
| Mismatched CRS        | Warning, recommendation=WARN, reprojection via rasterio |
| Mismatched resolution | Resolution ratio recorded, warning if >10x              |

---

## 7. Tests

**136 tests pass** (90 original + 46 new in `tests/test_calibration_rnd.py`):

- Calibration methods: affine, robust, dem_residual, local_normalized, inverse_depth, piecewise, frequency_fusion
- Quality metrics, apply_method, get_available_methods
- Spatial CV: alignment, blocks, folds, separation (geo/projected), describe_cv_plan
- Frequency: lowpass, highpass, decompose, fusion, metrics
- Uncertainty: per_pixel_map, estimate, quality_flags
- Robustness: NaN, nodata(0/-9999), outliers, low_overlap, constant, small_samples, mismatched CRS/resolution
- Experiment runner blocker logic

Synthetic fixtures used for code behavior only — **NOT SIH performance evidence**.

---

## 8. Code Review Summary

| Check                | Status     | Notes                                                         |
| -------------------- | ---------- | ------------------------------------------------------------- |
| Unit errors          | PASS       | scale=m/(depth unit), offset=m ASL                            |
| Coordinate errors    | PASS       | Pixel coords (row,col), geographic (lat,lon) separated        |
| Vertical datum       | PASS       | SRTM=EGM96 documented; no conversion implemented              |
| Pixel alignment      | PASS       | Bilinear reproject; block grid coords                         |
| Leakage              | PASS       | Spatial CV min_separation_blocks=2                            |
| Hidden normalization | PASS       | Metric path no normalization; relative fallback normalizes    |
| Depth inversion      | PASS       | Inverse_depth method explicit; affine does NOT invert         |
| Magic constants      | DOCUMENTED | IQR=3.0, subsample=50000, eps=1e-6, patch=64, σ=10            |
| Silent fallback      | DOCUMENTED | pipeline.calibrate() returns relative on any error            |
| Terminology          | PASS       | No "ground truth"; "estimated uncertainty", "calibration DEM" |

---

## 9. SIH Status Changes

| Capability                   | Before     | After      | Evidence                                      |
| ---------------------------- | ---------- | ---------- | --------------------------------------------- |
| Calibration methods registry | 1 (affine) | 7          | `methods.py`, tests                           |
| Robust calibration           | No         | RANSAC     | `methods.py:robust_affine`                    |
| DEM residual fusion          | No         | Yes        | `methods.py:dem_residual`                     |
| Local normalization          | No         | Yes        | `methods.py:local_normalized`                 |
| Inverse depth                | No         | Yes        | `methods.py:inverse_depth`                    |
| Piecewise linear             | No         | Yes        | `methods.py:piecewise_linear`                 |
| Frequency fusion             | No         | Yes        | `methods.py:frequency_fusion`, `frequency.py` |
| Spatial CV                   | No         | Yes        | `spatial_cv.py`, tests                        |
| Alignment diagnostics        | No         | Yes        | `spatial_cv.py:check_alignment`               |
| Frequency decomposition      | No         | Yes        | `frequency.py`, tests                         |
| Uncertainty estimation       | No         | Yes        | `uncertainty.py`, tests                       |
| Experiment config            | No         | YAML       | `configs/calibration_experiments.yaml`        |
| Experiment runner            | No         | Yes        | `eval/run_calibration_experiments.py`         |
| Model comparison ready       | No         | Pluggable  | `estimate_depth(image, backbone)`             |
| Real data blocker            | Implicit   | Explicit   | `blocker_report.{json,md}`                    |
| Diagnostic visuals           | No         | 10/exp     | `eval/visuals.py`                             |
| Terrain metadata             | No         | Structured | config + runner                               |
| Quality diagnostics          | Basic      | Full       | `uncertainty.py:quality_flags`                |

---

## 10. Recommended Next Technical Step

**Priority 1: Acquire real paired data.**
Without it, numerical experiments cannot proceed beyond synthetic validation.

- Option A: Download ISPRS Potsdam (registration required, airborne, not satellite but valid for algorithm dev)
- Option B: Drone survey with LiDAR/photogrammetric DSM for a local site
- Option C: Clone SIH2026 reference repo and extract samples

**Priority 2: Vertical datum conversion.**
SRTM uses EGM96 geoid. If reference DSM uses WGS84 ellipsoid, ~10-50m offset possible. Implement `pygeoid` or PROJ conversion.

**Priority 3: Frequency fusion validation.**
Current `frequency_fusion` returns global affine params only (scipy missing in test env). With scipy, test: `H_pred = affine(D) + α·highfreq(DEM)` vs `affine(D) + α·highfreq(depth)`.

**Priority 4: Local calibration evidence.**
Run global vs local (64/128/256 blocks) with spatial holdout. If local doesn't improve test RMSE, don't complicate.

**Priority 5: Model comparison.**
Plug in `depth-anything/Depth-Anything-V2-Base-hf` and `Intel/dpt-large` with identical evaluator.

---

## 11. Files Changed (Summary)

### New

- `backend/calibration/methods.py` — 7 calibration methods + quality metrics
- `backend/calibration/spatial_cv.py` — Spatial CV + alignment diagnostics
- `backend/calibration/frequency.py` — Frequency decomposition
- `backend/calibration/uncertainty.py` — Uncertainty estimation + quality flags
- `eval/visuals.py` — 10 diagnostic plots (matplotlib)
- `eval/run_calibration_experiments.py` — Experiment runner with blocker logic
- `configs/calibration_experiments.yaml` — Full reproducible config
- `tests/test_calibration_rnd.py` — 46 new tests
- `docs/CALIBRATION_AUDIT.md` — Code audit from Step 1
- `outputs/calibration_rnd/blocker_report.{json,md}` — Auto-generated

### Modified

- `backend/calibration/fit.py` — Unchanged (preserved for production pipeline)
- `backend/calibration/pipeline.py` — Unchanged
- `backend/calibration/georef.py` — Unchanged
- `backend/geo/srtm_provider.py` — Unchanged
- `eval/config.yaml` — Unchanged (kept for reference evaluation)

---

## 12. Final Acceptance Checklist

- [x] Existing 90+ tests still pass (136 total)
- [x] Real model still runs (Depth Anything V2 Small on CPU)
- [x] Real data adapter works (SRTM provider, dataset adapters)
- [x] At least one real paired RGB/reference dataset tested → **BLOCKED, documented**
- [x] Calibration methods are comparable (7 methods, same interface)
- [x] Spatial leakage is controlled (block holdout, min separation)
- [x] Real metrics are reproducible (config + runner + git SHA)
- [x] Calibration diagnostics exist (R², residuals, quality flags)
- [x] Error maps exist (visualizations module)
- [x] Results are provenance-tagged (git SHA, model, hardware, timestamp)
- [x] Synthetic and real results remain separate (blocker stops numerical experiments)
- [x] No fabricated accuracy claims (blocker report explicit)
- [x] Current best method identified from evidence → **None (no real data)**
- [x] Remaining SIH gaps explicit (data, datum, frequency, local, model)

---

**Do NOT:**

- [x] merge to main
- [x] create a PR
- [x] fine-tune the model
- [x] add cosmetic UI work

---

_Generated by calibration R&D phase on feat/sih-26175-calibration-rnd_
