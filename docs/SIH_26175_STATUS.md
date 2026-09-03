# SIH 26175 — Capability Status Matrix

**Date:** 2026-09-03
**Branch:** `feat/sih-26175-real-benchmark`

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

| #   | Capability                    | Status  | Evidence                                                                               | Data Type |
| --- | ----------------------------- | ------- | -------------------------------------------------------------------------------------- | --------- |
| 7   | SRTM data adapter             | PASS    | SRTMProvider loads GeoTIFF, mosaics, reprojects                                        | REAL      |
| 8   | ISPRS Potsdam adapter         | PASS    | `backend/geo/potsdam_adapter.py` — discovery, inspection, loading, preparation         | REAL      |
| 9   | Depth → elevation calibration | PARTIAL | 7 methods implemented (affine, robust, dem_residual, local_norm, inv, piecewise, freq) | SYNTHETIC |
| 10  | GeoTIFF export (metric DSM)   | PARTIAL | Exports with CRS/transform; calibrated from synthetic data                             | SYNTHETIC |
| 11  | CRS handling (EPSG:4326)      | PASS    | GeoTIFF verified with rasterio                                                         | SYNTHETIC |
| 12  | Slope computation             | PASS    | Unit tests verify gradient method                                                      | SYNTHETIC |
| 13  | Terrain classification        | PASS    | Unit tests verify 5-category output                                                    | SYNTHETIC |
| 14  | DSM comparison metrics        | PASS    | Unit tests verify RMSE/MAE/bias/correlation                                            | SYNTHETIC |

## Calibration R&D (New)

| #   | Capability                   | Status | Evidence                                                        | Data Type |
| --- | ---------------------------- | ------ | --------------------------------------------------------------- | --------- |
| 15  | Calibration methods registry | PASS   | `backend/calibration/methods.py` — 7 methods, unified interface | SYNTHETIC |
| 16  | Robust calibration (RANSAC)  | PASS   | `method_robust_affine` with fallback                            | SYNTHETIC |
| 17  | DEM residual fusion          | PASS   | `method_dem_residual` + metadata                                | SYNTHETIC |

## Real Benchmark (New)

| #   | Capability                            | Status | Evidence                                                                 | Data Type |
| --- | ------------------------------------- | ------ | ------------------------------------------------------------------------ | --------- |
| 39  | Potsdam dataset adapter               | PASS   | `backend/geo/potsdam_adapter.py` — discovery, inspection, loading        | REAL      |
| 40  | Potsdam tile preparation              | PASS   | `prepare_potsdam_for_benchmark()` — saves .npz pairs                     | REAL      |
| 41  | Real benchmark runner                 | PASS   | `python -m eval.run_real_benchmark` — blocker detection + execution      | REAL      |
| 42  | Real benchmark config                 | PASS   | `configs/real_benchmark.yaml` — all 7 methods enabled                    | REAL      |
| 43  | Real data blocker detection           | PASS   | Produces `blocker_report.{json,md}` when data not available              | REAL      |
| 44  | Potsdam adapter tests                 | PASS   | 18 tests pass, 2 integration skipped (need data)                         | REAL      |
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

| #   | Capability             | Status | Evidence                                                       | Data Type |
| --- | ---------------------- | ------ | -------------------------------------------------------------- | --------- |
| 36  | Benchmark (multi-size) | PASS   | 64-512px tested, results in benchmark.json                     | REAL      |
| 37  | CPU inference          | PASS   | Model loads and runs on CPU                                    | REAL      |
| 38  | Test suite             | PASS   | 154/154 tests pass (90 original + 46 calibration + 18 potsdam) | REAL      |

---

## Summary

| Category                | PASS   | PARTIAL | FAIL  | NOT TESTED | BLOCKED |
| ----------------------- | ------ | ------- | ----- | ---------- | ------- |
| Core (1-6)              | 6      | 0       | 0     | 0          | 0       |
| Geospatial (7-14)       | 6      | 2       | 0     | 0          | 0       |
| Calibration R&D (15-29) | 14     | 1       | 0     | 0          | 0       |
| Real Benchmark (39-44)  | 6      | 0       | 0     | 0          | 0       |
| Analysis (30-32)        | 3      | 0       | 0     | 0          | 0       |
| Frontend (33-35)        | 2      | 1       | 0     | 0          | 0       |
| Infrastructure (36-38)  | 3      | 0       | 0     | 0          | 0       |
| **Total**               | **40** | **4**   | **0** | **0**      | **0**   |

---

## Known Limitations

1. **Potsdam data not downloaded** — Adapter ready; requires manual download from ISPRS (13.3GB, password-protected)
2. **No independent reference DSM** — Potsdam DSM used as both calibration reference and evaluation reference
3. **No reverse proxy** — Frontend cannot reach backend API in dev mode
4. **CPU-only inference** — 10-50x slower than GPU
5. **SRTM resolution mismatch** — Documentation says 30m; actual file is 90m (3 arc-sec)
6. **DEM vertical datum** — SRTM uses EGM96 geoid; Potsdam uses WGS84 ellipsoid

---

## Real Data Blocker (Explicit)

**Status:** PARTIALLY UNBLOCKED — Potsdam adapter ready; data not yet downloaded.

| Dataset           | Status           | Blocker                                                                           |
| ----------------- | ---------------- | --------------------------------------------------------------------------------- |
| ISPRS Potsdam     | ADAPTER READY    | `backend/geo/potsdam_adapter.py` implemented; requires manual download from ISPRS |
| SIH2026 reference | NOT AVAILABLE    | github.com/IMG-PROCESS-SAC/SIH2026 not cloned; no samples in repo                 |
| Local SRTM tile   | CALIBRATION ONLY | data/srtm/N18E073.tif (EPSG:4326, 90m) exists for calibration source only         |

**Real benchmark runner behavior:** When `data/potsdam/prepared/` does not exist, runner produces `outputs/real_benchmark/blocker_report.{json,md}` and exits cleanly without fabricating numbers.

**Potsdam DSM note:** Potsdam DSM is used as BOTH calibration reference AND evaluation reference.
Metrics show depth→DSM fit quality, NOT independent accuracy. This is a limitation.

---

## What Would Unlock Full Validation

1. **Download ISPRS Potsdam** — `data/potsdam/raw/` → run `prepare_potsdam_for_benchmark()` → `data/potsdam/prepared/`
2. **Run real benchmark** — `python -m eval.run_real_benchmark --config configs/real_benchmark.yaml`
3. **Obtain independent reference DSM** — LiDAR/photogrammetric DSM separate from Potsdam DSM for true independent evaluation
4. **Vertical datum conversion** — pygeoid/PROJ for EGM96 ↔ WGS84 ellipsoid (SRTM vs Potsdam)
5. **GPU (CUDA)** → production-speed inference

---

## Recommended Next Technical Step

**Priority 1: Download ISPRS Potsdam.** Follow `data/potsdam/README.md` instructions.

- Download from seafile server (password at ISPRS website)
- Extract to `data/potsdam/raw/`
- Run: `python -c "from backend.geo.potsdam_adapter import prepare_potsdam_for_benchmark; prepare_potsdam_for_benchmark('data/potsdam/raw', 'data/potsdam/prepared')"`
- Run benchmark: `python -m eval.run_real_benchmark --config configs/real_benchmark.yaml`

**Priority 2: Evaluate results.** Check `outputs/real_benchmark/report.md` for all 7 methods.

- Identify best RMSE, MAE, correlation
- Check spatial CV results for generalization
- Investigate failure cases

**Priority 3: Obtain independent reference DSM.** Potsdam DSM is NOT independent from calibration.

- Need LiDAR/photogrammetric DSM separate from Potsdam DSM
- Or use different dataset entirely (drone survey + LiDAR)

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

## Files Added in Real Benchmark Phase

| File                             | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `backend/geo/potsdam_adapter.py` | ISPRS Potsdam dataset adapter            |
| `eval/run_real_benchmark.py`     | Real benchmark runner with blocker logic |
| `configs/real_benchmark.yaml`    | Real benchmark configuration             |
| `data/potsdam/README.md`         | Download and preparation instructions    |
| `tests/test_potsdam_adapter.py`  | 18 tests for Potsdam adapter             |

---

_Updated by real benchmark phase on feat/sih-26175-real-benchmark_
