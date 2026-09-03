# Real Data Validation Report

**Date:** 2026-09-03
**Branch:** `feat/sih-26175-realdata`
**Status:** Partial — synthetic SRTM only, no reference DSM available

---

## Executive Summary

This report documents the real-data validation performed on DepthWizard using:

- A **real RGB photograph** (1024x1024, generic landscape photo) as proxy for satellite/drone imagery
- A **synthetic SRTM placeholder** (OpenTopography API returned HTTP 401 — no API key configured)
- **No reference DSM** or LiDAR ground truth available

**Key finding:** The pipeline works mechanically end-to-end. However, calibration values are meaningless without real elevation data, and independent DSM validation is blocked without a reference surface.

---

## What Was Tested

### 1. Model Smoke Test (REAL)

| Metric              | Value                                       |
| ------------------- | ------------------------------------------- |
| Model               | `depth-anything/Depth-Anything-V2-Small-hf` |
| Device              | CPU (no CUDA despite RTX 3050 present)      |
| Cold load           | 54.3s                                       |
| 256x256 inference   | 2.8s                                        |
| 1024x1024 inference | 21.4s                                       |
| Output shape        | (1024, 1024) float32                        |
| Output range        | [0, 255]                                    |

**Verdict:** Model loads and infers correctly. Output is relative depth (not metric elevation).

### 2. Real Inference (REAL INPUT, SYNTHETIC CALIBRATION)

| Metric         | Value                                                        |
| -------------- | ------------------------------------------------------------ |
| Input          | `data/samples/real_test_landscape.jpg` (1024x1024 RGB photo) |
| Output         | `outputs/real_depth_output.npy` (1024x1024 float32)          |
| Inference time | 21.4s                                                        |
| Depth range    | [0.00, 255.00]                                               |
| Status         | COMPLETED                                                    |

**Verdict:** Real monocular depth estimation produces plausible relative structure.

### 3. SRTM Calibration (SYNTHETIC SRTM ONLY)

| Metric              | Value                                           |
| ------------------- | ----------------------------------------------- |
| SRTM source         | `data/srtm/N18E073.tif` (SYNTHETIC placeholder) |
| SRTM shape          | 1201x1201                                       |
| SRTM range          | [401.5, 717.0]                                  |
| Calibration method  | linear_fit                                      |
| Scale (H = a*D + b) | 0.017491                                        |
| Offset              | 558.19                                          |
| Valid samples       | 1,048,576                                       |
| RMSE                | 69.74m                                          |
| MAE                 | 59.05m                                          |

**Verdict:** Pipeline works mechanically. Calibration values are **meaningless** — synthetic SRTM is random elevation data unrelated to the actual image content.

### 4. Metric DSM Export (SYNTHETIC)

| Metric      | Value                                   |
| ----------- | --------------------------------------- |
| Output      | `outputs/real_metric_dsm.tif` (GeoTIFF) |
| CRS         | EPSG:4326                               |
| Bounds      | 73.36-74.36 E, 18.02-19.02 N            |
| Shape       | 1024x1024                               |
| Value range | [558.19, 562.65]                        |
| Mean        | 560.69m                                 |

**Verdict:** GeoTIFF export works. Values are calibrated from synthetic SRTM — not real elevation.

### 5. Reference Evaluation (BLOCKED)

**Cannot perform.** No independent reference DSM/LiDAR/photogrammetric surface available on disk or from external source.

Required for independent validation:

- LiDAR-derived DSM of the same geographic area
- Or photogrammetric DSM from high-resolution stereo imagery
- Or high-res DEM (better than 30m SRTM)

### 6. UI Test (PARTIAL)

| Check              | Result                                                                 |
| ------------------ | ---------------------------------------------------------------------- |
| Page loads         | PASS — title "DepthWizard"                                             |
| Upload button      | PASS — file chooser opens                                              |
| File upload        | PASS — 1024x1024 JPG accepted                                          |
| Process button     | PASS — clickable                                                       |
| Backend connection | FAIL — API_BASE resolves to localhost:8080 (frontend), backend on 8001 |
| 3D rendering       | NOT TESTED — blocked by connection failure                             |

**Root cause:** Frontend `main.js` line 7 sets `API_BASE = window.location.origin`. Without a reverse proxy, the frontend POSTs to itself (port 8080) instead of the backend (port 8001). This is a **dev config issue**, not a pipeline bug.

### 7. Benchmark (REAL)

| Size    | Mean (ms) | Pixels/sec |
| ------- | --------- | ---------- |
| 64x64   | 1249.8    | 3,277      |
| 128x128 | 1266.1    | 12,941     |
| 256x256 | 1293.1    | 50,680     |
| 512x512 | 1194.4    | 219,482    |

**Note:** CPU-only inference. GPU would be 10-50x faster.

---

## Validation Type Separation

| Validation Type            | Status    | Data Type                                  |
| -------------------------- | --------- | ------------------------------------------ |
| Model load/inference       | COMPLETED | REAL                                       |
| Monocular depth output     | COMPLETED | REAL (input) / RELATIVE (output)           |
| SRTM calibration           | COMPLETED | SYNTHETIC (meaningless values)             |
| GeoTIFF export             | COMPLETED | SYNTHETIC (calibrated from synthetic SRTM) |
| Independent DSM validation | BLOCKED   | N/A (no reference available)               |
| UI upload/process          | PARTIAL   | REAL (connection issue in dev mode)        |
| Benchmark                  | COMPLETED | REAL (model timing)                        |

---

## Blockers for Full Validation

1. **No real SRTM data** — OpenTopography API requires API key (HTTP 401)
2. **No reference DSM** — Cannot validate metric accuracy against ground truth
3. **No real satellite/drone imagery** — Using generic photo as proxy
4. **No reverse proxy** — Frontend cannot reach backend API in dev mode

---

## Files Produced

| File                                        | Description                                    |
| ------------------------------------------- | ---------------------------------------------- |
| `outputs/real_depth_output.npy`             | 1024x1024 float32 depth from Depth Anything V2 |
| `outputs/real_depth_output.png`             | Depth visualization (grayscale)                |
| `outputs/real_inference_diagnostics.json`   | Inference timing and shape info                |
| `outputs/real_metric_dsm.npy`               | Calibrated metric DSM (float32)                |
| `outputs/real_metric_dsm.tif`               | GeoTIFF export (EPSG:4326)                     |
| `outputs/real_calibration_diagnostics.json` | Calibration fit results                        |
| `outputs/real_validation_report.json`       | Structured validation report                   |
| `outputs/benchmark.json`                    | Multi-size benchmark results                   |
| `data/srtm/N18E073.tif`                     | SYNTHETIC SRTM placeholder                     |

---

## Honest Assessment

The pipeline **works mechanically**:

- Model loads and infers on CPU
- Depth output is plausible relative structure
- Calibration applies linear fit to map depth to elevation
- GeoTIFF exports with correct CRS/transform
- Frontend renders 3D terrain from heightmap

The pipeline **has NOT been validated**:

- No real satellite/drone input tested
- No real SRTM calibration tested
- No independent accuracy assessment possible
- All calibration values are from synthetic data

**Do NOT present synthetic results as real-world performance.**
