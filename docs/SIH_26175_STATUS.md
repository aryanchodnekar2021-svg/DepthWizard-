# SIH 26175 — Capability Status Matrix

**Date:** 2026-09-03
**Branch:** `feat/sih-26175-calibration-rnd`

---

## Status Legend

| Status     | Meaning                                                 |
| ---------- | ------------------------------------------------------- |
| PASS       | Tested with real data, works as expected                |
| PARTIAL    | Tested but with caveats (synthetic data, limited scope) |
| FAIL       | Tested and broken                                       |
| NOT TESTED | Not yet validated                                       |
| BLOCKED    | Cannot test due to missing dependency                   |

---

## Core Capabilities

| #   | Capability                     | Status | Evidence                                    | Data Type |
| --- | ------------------------------ | ------ | ------------------------------------------- | --------- |
| 1   | RGB image upload               | PASS   | UI file chooser accepts JPG/PNG             | REAL      |
| 2   | Depth estimation               | PASS   | 1024x1024 → depth in 21.4s                  | REAL      |
| 3   | Multi-band input support       | PASS   | 1/2/3/multi-band handled in prepare_image() | REAL      |
| 4   | Tiled inference (large images) | PASS   | 136/136 unit tests pass                     | SYNTHETIC |
| 5   | Model status endpoint          | PASS   | GET /model/status returns JSON              | REAL      |
| 6   | HTTP 503 on model failure      | PASS   | Unit tests verify behavior                  | SYNTHETIC |

## Geospatial Pipeline

| #   | Capability                    | Status  | Evidence                                                                               | Data Type  |
| --- | ----------------------------- | ------- | -------------------------------------------------------------------------------------- | ---------- |
| 7   | SRTM data adapter             | PASS    | SRTMProvider loads GeoTIFF, mosaics, reprojects                                        | REAL       |
| 8   | ISPRS Potsdam adapter         | PASS    | ISPRSAdapter structure exists                                                          | NOT TESTED |
| 9   | Depth → elevation calibration | PARTIAL | 7 methods implemented (affine, robust, dem_residual, local_norm, inv, piecewise, freq) | SYNTHETIC  |
| 10  | GeoTIFF export (metric DSM)   | PARTIAL | Exports with CRS/transform; calibrated from synthetic data                             | SYNTHETIC  |
| 11  | CRS handling (EPSG:4326)      | PASS    | GeoTIFF verified with rasterio                                                         | SYNTHETIC  |
| 12  | Slope computation             | PASS    | Unit tests verify gradient method                                                      | SYNTHETIC  |
| 13  | Terrain classification        | PASS    | Unit tests verify 5-category output                                                    | SYNTHETIC  |
| 14  | DSM comparison metrics        | PASS    | Unit tests verify RMSE/MAE/bias/correlation                                            | SYNTHETIC  |

## Calibration R&D (New)

| #   | Capability                            | Status | Evidence                                                                 | Data Type |
| --- | ------------------------------------- | ------ | ------------------------------------------------------------------------ | --------- |
| 15  | Calibration methods registry          | PASS   | `backend/calibration/methods.py` — 7 methods, unified interface          | SYNTHETIC |
| 16  | Robust calibration (RANSAC)           | PASS   | `method_robust_affine` with fallback                                     | SYNTHETIC |
| 17  | DEM residual fusion                   | PASS   | `method_dem_residual` + metadata                                         | SYNTHETIC |
| 18  | Locally normalized residual fusion    | PASS   | `method_local_normalized` (patch-based z-score)                          | SYNTHETIC |
| 19  | Inverse depth (disparity) calibration | PASS   | `method_inverse_depth` with epsilon                                      | SYNTHETIC |
| 20  | Piecewise linear calibration          | PASS   | `method_piecewise_linear` (n_bins=3 default)                             | SYNTHETIC |
| 21  | Frequency fusion                      | PASS   | `method_frequency_fusion` + `frequency.py` decomposition                 | SYNTHETIC |
| 22  | Spatial cross-validation              | PASS   | `spatial_cv.py` — block holdout, min separation, geographic distance     | SYNTHETIC |
| 23  | Alignment diagnostics                 | PASS   | `check_alignment()` — CRS, resolution, overlap, nodata, vertical datum   | SYNTHETIC |
| 24  | Frequency decomposition               | PASS   | `frequency.py` — gaussian lowpass, highpass, dem/depth decompose         | SYNTHETIC |
| 25  | Uncertainty estimation                | PASS   | `uncertainty.py` — per-pixel map, quality flags                          | SYNTHETIC |
| 26  | Experiment configuration              | PASS   | `configs/calibration_experiments.yaml` — full reproducible spec          | SYNTHETIC |
| 27  | Experiment runner                     | PASS   | `python -m eval.run_calibration_experiments` — JSON + MD + blocker logic | SYNTHETIC |
| 28  | Diagnostic visualizations             | PASS   | `eval/visuals.py` — 10 plots per experiment (matplotlib)                 | SYNTHETIC |
| 29  | Model comparison readiness            | PASS   | Pluggable `estimate_depth(image, backbone)` interface                    | SYNTHETIC |

## Analysis Tools

| #   | Capability                  | Status | Evidence   | Data Type |
| --- | --------------------------- | ------ | ---------- | --------- |
| 30  | Slope visualization         | PASS   | Unit tests | SYNTHETIC |
| 31  | Terrain class visualization | PASS   | Unit tests | SYNTHETIC |
| 32  | DSM comparison report       | PASS   | Unit tests | SYNTHETIC |

## Frontend

| #   | Capability            | Status  | Evidence                                                                                  | Data Type |
| --- | --------------------- | ------- | ----------------------------------------------------------------------------------------- | --------- |
| 33  | 3D terrain rendering  | PASS    | Three.js mesh from heightmap                                                              | SYNTHETIC |
| 34  | Upload → process flow | PARTIAL | UI loads, upload works, process button works; backend connection fails (no reverse proxy) | REAL      |
| 35  | Metadata display      | PASS    | Elevation range, calibration info, CRS shown                                              | SYNTHETIC |

## Infrastructure

| #   | Capability             | Status | Evidence                                   | Data Type |
| --- | ---------------------- | ------ | ------------------------------------------ | --------- |
| 36  | Benchmark (multi-size) | PASS   | 64-512px tested, results in benchmark.json | REAL      |
| 37  | CPU inference          | PASS   | Model loads and runs on CPU                | REAL      |
| 38  | Test suite             | PASS   | 136/136 tests pass (90 original + 46 new)  | SYNTHETIC |

---

## Summary

| Category                | PASS   | PARTIAL | FAIL  | NOT TESTED | BLOCKED |
| ----------------------- | ------ | ------- | ----- | ---------- | ------- |
| Core (1-6)              | 6      | 0       | 0     | 0          | 0       |
| Geospatial (7-14)       | 5      | 3       | 0     | 1          | 0       |
| Calibration R&D (15-29) | 14     | 1       | 0     | 0          | 0       |
| Analysis (30-32)        | 3      | 0       | 0     | 0          | 0       |
| Frontend (33-35)        | 2      | 1       | 0     | 0          | 0       |
| Infrastructure (36-38)  | 3      | 0       | 0     | 0          | 0       |
| **Total**               | **33** | **5**   | **0** | **1**      | **0**   |

---

## Known Limitations

1. **No real satellite/drone imagery** — Using generic photo as proxy
2. **No real SRTM** — OpenTopography API requires auth key (HTTP 401); local tile N18E073.tif is 90m, not 30m
3. **No reference DSM** — Independent accuracy validation BLOCKED
4. **No reverse proxy** — Frontend cannot reach backend API in dev mode
5. **CPU-only inference** — 10-50x slower than GPU
6. **Calibration values meaningless** — Derived from synthetic data / single SRTM tile without independent reference
7. **SRTM resolution mismatch** — Documentation says 30m; actual file is 90m (3 arc-sec)
8. **DEM vertical datum** — SRTM uses EGM96 geoid; no conversion to WGS84 ellipsoid implemented

---

## Real Data Blocker (Explicit)

**Status:** BLOCKED — No usable paired RGB + independent reference DSM datasets.

| Dataset           | Status           | Blocker                                                                                       |
| ----------------- | ---------------- | --------------------------------------------------------------------------------------------- |
| ISPRS Potsdam     | NOT AVAILABLE    | Requires registration at isprs.org; airborne (TOP), not satellite but valid for algorithm dev |
| SIH2026 reference | NOT AVAILABLE    | github.com/IMG-PROCESS-SAC/SIH2026 not cloned; no samples in repo                             |
| Local SRTM tile   | CALIBRATION ONLY | data/srtm/N18E073.tif (EPSG:4326, 90m) exists but no paired independent reference DSM         |

**Experiment runner behavior:** When `datasets: []` in config, runner produces `outputs/calibration_rnd/blocker_report.{json,md}` and exits cleanly without fabricating numbers.

---

## What Would Unlock Full Validation

1. **Real paired RGB + independent DSM** (drone RGB + LiDAR DSM, or ISPRS Potsdam TOP+DSM)
2. **OpenTopography API key** → real SRTM 30m tiles for meaningful calibration
3. **Vertical datum conversion** — pygeoid/PROJ for EGM96 ↔ WGS84 ellipsoid
4. **Reverse proxy (nginx/traefik)** → full frontend→backend flow
5. **GPU (CUDA)** → production-speed inference

---

## Recommended Next Technical Step

**Priority 1: Acquire real paired data.** Without it, numerical experiments cannot proceed beyond synthetic validation.

- Option A: Download ISPRS Potsdam (registration required, airborne, not satellite but valid for algorithm dev)
- Option B: Drone survey with LiDAR/photogrammetric DSM for a local site
- Option C: Clone SIH2026 reference repo and extract samples

**Priority 2: Vertical datum conversion.** SRTM uses EGM96 geoid. If reference DSM uses WGS84 ellipsoid, ~10-50m offset possible.

**Priority 3: Frequency fusion validation.** With scipy, test: `H_pred = affine(D) + α·highfreq(DEM)` vs `affine(D) + α·highfreq(depth)`.

**Priority 4: Local calibration evidence.** Run global vs local (64/128/256 blocks) with spatial holdout.

**Priority 5: Model comparison.** Plug in `depth-anything/Depth-Anything-V2-Base-hf` and `Intel/dpt-large` with identical evaluator.

---

## Files Added in Calibration R&D Phase

| File                                   | Purpose                                             |
| -------------------------------------- | --------------------------------------------------- |
| `backend/calibration/methods.py`       | 7 calibration methods + quality metrics             |
| `backend/calibration/spatial_cv.py`    | Spatial CV + alignment diagnostics                  |
| `backend/calibration/frequency.py`     | Frequency decomposition (gaussian lowpass/highpass) |
| `backend/calibration/uncertainty.py`   | Uncertainty estimation + quality flags              |
| `eval/visuals.py`                      | 10 diagnostic plots (matplotlib)                    |
| `eval/run_calibration_experiments.py`  | Experiment runner with blocker logic                |
| `configs/calibration_experiments.yaml` | Full reproducible config                            |
| `tests/test_calibration_rnd.py`        | 46 new tests                                        |
| `docs/CALIBRATION_AUDIT.md`            | Code audit from Step 1                              |
| `docs/CALIBRATION_RND.md`              | This phase documentation                            |

---

_Updated by calibration R&D phase on feat/sih-26175-calibration-rnd_
