# SIH 26175 — Capability Status Matrix

**Date:** 2026-09-03
**Branch:** `feat/sih-26175-realdata`

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
| 4   | Tiled inference (large images) | PASS   | 90/90 unit tests pass                       | SYNTHETIC |
| 5   | Model status endpoint          | PASS   | GET /model/status returns JSON              | REAL      |
| 6   | HTTP 503 on model failure      | PASS   | Unit tests verify behavior                  | SYNTHETIC |

## Geospatial Pipeline

| #   | Capability                    | Status  | Evidence                                                   | Data Type  |
| --- | ----------------------------- | ------- | ---------------------------------------------------------- | ---------- |
| 7   | SRTM data adapter             | PASS    | SRTMAdapter loads GeoTIFF, resamples                       | REAL       |
| 8   | ISPRS Potsdam adapter         | PASS    | ISPRSAdapter reads RGB+DSM                                 | NOT TESTED |
| 9   | Depth → elevation calibration | PARTIAL | Linear fit works; uses synthetic SRTM                      | SYNTHETIC  |
| 10  | GeoTIFF export (metric DSM)   | PARTIAL | Exports with CRS/transform; calibrated from synthetic data | SYNTHETIC  |
| 11  | CRS handling (EPSG:4326)      | PASS    | GeoTIFF verified with rasterio                             | SYNTHETIC  |
| 12  | Slope computation             | PASS    | Unit tests verify gradient method                          | SYNTHETIC  |
| 13  | Terrain classification        | PASS    | Unit tests verify 5-category output                        | SYNTHETIC  |
| 14  | DSM comparison metrics        | PASS    | Unit tests verify RMSE/MAE/bias/correlation                | SYNTHETIC  |

## Analysis Tools

| #   | Capability                  | Status | Evidence   | Data Type |
| --- | --------------------------- | ------ | ---------- | --------- |
| 15  | Slope visualization         | PASS   | Unit tests | SYNTHETIC |
| 16  | Terrain class visualization | PASS   | Unit tests | SYNTHETIC |
| 17  | DSM comparison report       | PASS   | Unit tests | SYNTHETIC |

## Frontend

| #   | Capability            | Status  | Evidence                                                                                  | Data Type |
| --- | --------------------- | ------- | ----------------------------------------------------------------------------------------- | --------- |
| 18  | 3D terrain rendering  | PASS    | Three.js mesh from heightmap                                                              | SYNTHETIC |
| 19  | Upload → process flow | PARTIAL | UI loads, upload works, process button works; backend connection fails (no reverse proxy) | REAL      |
| 20  | Metadata display      | PASS    | Elevation range, calibration info, CRS shown                                              | SYNTHETIC |

## Infrastructure

| #   | Capability             | Status | Evidence                                   | Data Type |
| --- | ---------------------- | ------ | ------------------------------------------ | --------- |
| 21  | Benchmark (multi-size) | PASS   | 64-512px tested, results in benchmark.json | REAL      |
| 22  | CPU inference          | PASS   | Model loads and runs on CPU                | REAL      |

---

## Summary

| Category               | PASS   | PARTIAL | FAIL  | NOT TESTED | BLOCKED |
| ---------------------- | ------ | ------- | ----- | ---------- | ------- |
| Core (1-6)             | 6      | 0       | 0     | 0          | 0       |
| Geospatial (7-14)      | 5      | 2       | 0     | 1          | 0       |
| Analysis (15-17)       | 3      | 0       | 0     | 0          | 0       |
| Frontend (18-20)       | 2      | 1       | 0     | 0          | 0       |
| Infrastructure (21-22) | 2      | 0       | 0     | 0          | 0       |
| **Total**              | **18** | **3**   | **0** | **1**      | **0**   |

---

## Known Limitations

1. **No real satellite/drone imagery** — Using generic photo as proxy
2. **No real SRTM** — OpenTopography API requires auth key (HTTP 401)
3. **No reference DSM** — Independent accuracy validation blocked
4. **No reverse proxy** — Frontend cannot reach backend API in dev mode
5. **CPU-only inference** — 10-50x slower than GPU
6. **Calibration values meaningless** — Derived from synthetic data

---

## What Would Unlock Full Validation

1. OpenTopography API key → real SRTM data → meaningful calibration
2. LiDAR/photogrammetric reference DSM → independent accuracy assessment
3. Real satellite/drone imagery → real-world performance metrics
4. Reverse proxy (nginx/traefik) → full frontend→backend flow
5. GPU (CUDA) → production-speed inference
