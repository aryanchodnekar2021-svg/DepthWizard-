# DepthWizard — Baseline Audit Report

## Phase 0: Repository Safety Check

| Item              | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| Working Directory | `D:\SIH MAPDEPTH`                                                     |
| Git Remote        | `origin → https://github.com/aryanchodnekar2021-svg/DepthWizard-.git` |
| Default Branch    | `main`                                                                |
| Current Branch    | `main`                                                                |
| Working Tree      | Clean                                                                 |
| Total Commits     | 3                                                                     |
| Branches          | `main`, `remotes/origin/shravan-jadhav`                               |

### Recent Commits

```
7b6728f Fix API port conflict by changing port from 8000 to 8001
4a54432 Remove phase_prompts.md
af5b71e Initial commit
```

### Runtime Environment

- **Python**: 3.12.10 (via `C:\Users\Shivam\AppData\Local\Programs\Python\Python312\python.exe`)
- **Node.js**: v22.20.0
- **npm**: 10.9.3
- **No Dockerfile, no CI/CD, no Makefile, no pyproject.toml, no root .gitignore**

---

## Phase 1: Baseline Execution Status

### Project Structure

```
DepthWizard-/
├── backend/
│   ├── main.py                    # FastAPI entry point (port 8001)
│   ├── requirements.txt           # Python dependencies
│   ├── depth/
│   │   ├── estimator.py           # Depth Anything V2 inference
│   │   ├── run_sample.py          # CLI depth testing
│   │   ├── generate_dummy_assets.py
│   │   ├── NOTES.md               # Manual inference benchmarks
│   │   └── __pycache__/
│   └── calibration/
│       ├── pipeline.py            # Calibrate orchestrator
│       ├── fit.py                 # Linear regression fitting
│       ├── georef.py              # GeoTIFF metadata + SRTM alignment
│       ├── run_sample.py          # CLI calibration testing
│       └── __pycache__/
├── frontend/
│   ├── index.html                 # Upload + Three.js viewer
│   ├── main.js                    # Upload flow + fetch API
│   ├── scene.js                   # Three.js scene + PointerLock/Orbit
│   ├── mesh.js                    # Heightmap → PlaneGeometry
│   └── demo_cache/                # Pre-baked forest demo
├── eval/
│   ├── metrics.py                 # RMSE, MAE, Pearson correlation
│   ├── run_eval.py                # Simulated evaluation harness
│   ├── results.md                 # Simulated results (random data)
│   └── mesh_qa.md                 # Manual QA checklist
├── data/
│   ├── eval_tiles/                # Empty (only .gitkeep)
│   ├── README.md                  # Instructions for sourcing data
│   └── .gitignore                 # Ignores samples/, srtm/, images
├── outputs/                       # 24 pre-generated output files
├── docs/
│   ├── TECHNICAL_SUMMARY.md
│   └── DEMO_SCRIPT.md
├── README.md
├── PRD.docx
└── TRD.docx
```

### Frontend Architecture

- **Framework**: Vanilla JS + Three.js 0.160.0 (ES modules via importmap)
- **Scene**: PointerLockControls (WASD flythrough) + OrbitControls (toggle via 'C')
- **Mesh**: PlaneGeometry displaced by 16-bit grayscale heightmap PNG
- **Texture**: Original input image UV-mapped onto displaced geometry
- **API Call**: Hardcoded to `http://localhost:8001/process`
- **Demo Mode**: Loads pre-cached `demo_forest_heightmap.png` + `demo_forest_texture.png`
- **Stub Buttons**: Minimap, Height Ramp, Slope Shading (all disabled)

### Backend Architecture

- **Framework**: FastAPI on port 8001
- **Depth Model**: `depth-anything/Depth-Anything-V2-Small-hf` via HuggingFace `transformers.pipeline`
- **Calibration**: GeoTIFF detection → SRTM tile fetch → rasterio reprojection → sklearn LinearRegression
- **Output**: Metric DSM → GeoTIFF (rasterio); Relative DSM → 16-bit PNG
- **Dependencies**: fastapi, uvicorn, torch, transformers, rasterio, numpy, scipy, scikit-learn, pillow, python-multipart

### Known Issues from Code Inspection

| Issue                                     | Severity     | Details                                                                                                                                                                                                            |
| ----------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Missing `__init__.py` files               | **CRITICAL** | `backend/depth/` and `backend/calibration/` have NO `__init__.py`. The `calibration/pipeline.py` uses relative imports (`from .georef import ...`) which require package `__init__.py`. This will crash on import. |
| `fetch_srtm_tile` is a stub               | **CRITICAL** | Just returns the first `.tif` in `data/srtm/` directory — no actual bbox matching. Without SRTM data in the repo, metric calibration path always falls back to relative mode.                                      |
| No SRTM data in repo                      | **CRITICAL** | `data/srtm/` doesn't exist. GeoTIFF path always falls back to relative.                                                                                                                                            |
| No real test data                         | **HIGH**     | `data/eval_tiles/` is empty. `data/samples/` doesn't exist. No real satellite imagery.                                                                                                                             |
| Eval harness uses random data             | **HIGH**     | `run_eval.py` generates random DSMs — results in `results.md` are fake (random reference + noise). Not real accuracy measurements.                                                                                 |
| Hardcoded API URL                         | **MEDIUM**   | `main.js` line 66: `http://localhost:8001/process` — not configurable.                                                                                                                                             |
| Port mismatch README                      | **LOW**      | README says port 8000, code uses 8001 (fixed in latest commit).                                                                                                                                                    |
| No validation for input types             | **MEDIUM**   | `accept="image/*"` allows any file. No GeoTIFF-specific upload path or metadata display.                                                                                                                           |
| No relative vs absolute distinction in UI | **HIGH**     | User cannot tell if they're viewing relative or absolute DSM. No "meters" vs "relative" labeling.                                                                                                                  |
| No height readout / inspection            | **MEDIUM**   | No way to query elevation at a cursor position.                                                                                                                                                                    |
| No slope analysis                         | **MEDIUM**   | Slope Shading button is a stub.                                                                                                                                                                                    |
| No error map / reference comparison       | **MEDIUM**   | No validation visualization in the UI.                                                                                                                                                                             |
| Heightmap read only uses 8-bit            | **MEDIUM**   | `mesh.js` reads `imgData[pixelIndex]` (red channel 0-255) but `main.py` saves 16-bit PNG — only 8 bits are used in Three.js, losing precision.                                                                     |
| No `pyproject.toml` or `setup.py`         | **LOW**      | No standard Python project packaging.                                                                                                                                                                              |
| No tests                                  | **HIGH**     | Zero automated test files.                                                                                                                                                                                         |

### Pre-Generated Outputs Analysis

The `outputs/` directory contains 24 files that appear to be from previous manual runs:

- 7 input images (1.jpg through 4.png, sample, dummy_texture, test_landscape)
- 7 DSM PNGs (relative, 8-bit)
- 7 heightmap PNGs (identical to DSMs — same file sizes)
- 1 depth visualization
- 1 mock metric DSM GeoTIFF (65KB, created by `calibration/run_sample.py`)
- 1 mock relative DSM (401 bytes — nearly empty)

---

## Phase 2: SIH 26175 Gap Analysis

### Requirements Matrix

#### A. INPUT SUPPORT

| Requirement                 | Status     | Details                                                                                                                          |
| --------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| PNG input                   | ✅ WORKING | Saves to outputs/, processes via PIL                                                                                             |
| JPG/JPEG input              | ✅ WORKING | Same pipeline as PNG                                                                                                             |
| TIFF input                  | ⚠️ PARTIAL | Only checked by file extension `.tif/.tiff` — no actual rasterio read for non-GeoTIFF TIFF                                       |
| GeoTIFF input               | ⚠️ PARTIAL | Extension-based detection only. If CRS is present, attempts SRTM calibration. But SRTM data doesn't exist, so always falls back. |
| Non-georeferenced detection | ✅ WORKING | Falls through to relative mode                                                                                                   |
| Georeferenced detection     | ⚠️ PARTIAL | Detects via extension + rasterio CRS check, but calibration is non-functional without SRTM data                                  |

#### B. ELEVATION EXTRACTION

| Requirement                      | Status     | Details                                                                                |
| -------------------------------- | ---------- | -------------------------------------------------------------------------------------- |
| Pretrained monocular depth model | ✅ WORKING | Depth Anything V2 Small (HuggingFace)                                                  |
| Relative depth generation        | ✅ WORKING | Produces float32 depth map                                                             |
| Preservation of image structure  | ✅ WORKING | Resizes output to match input dimensions                                               |
| Usable resolution                | ⚠️ PARTIAL | Model outputs at model's native resolution, then resized. No tiling for large images.  |
| CPU fallback                     | ✅ WORKING | Auto-detects CUDA, falls back to CPU                                                   |
| GPU acceleration                 | ✅ WORKING | Uses CUDA if available                                                                 |
| Deterministic/reproducible       | ⚠️ PARTIAL | Same image gives same result (deterministic inference), but no seed control documented |

#### C. NON-GEOREFERENCED PATH

| Requirement                                  | Status     | Details                                                  |
| -------------------------------------------- | ---------- | -------------------------------------------------------- |
| Input: PNG/JPG/TIFF without spatial metadata | ✅ WORKING | Detected as non-georeferenced                            |
| Output: Relative DSM (rDSM)                  | ✅ WORKING | Normalized 0-1 float → 16-bit PNG                        |
| Clear distinction from metric                | ❌ FAIL    | No UI labeling; no output metadata indicating "relative" |

#### D. GEOREFERENCED PATH

| Requirement                          | Status                   | Details                                           |
| ------------------------------------ | ------------------------ | ------------------------------------------------- |
| Input: GeoTIFF with CRS/geotransform | ⚠️ PARTIAL               | Reads metadata correctly via rasterio             |
| Output: Absolute metric DSM          | ❌ FAIL                  | SRTM data missing → always falls back to relative |
| Preserve CRS                         | ✅ WORKING (when metric) | Saved via rasterio with CRS+transform             |
| Preserve affine transform            | ✅ WORKING (when metric) | Same                                              |
| Raster dimensions                    | ✅ WORKING               |                                                   |
| Valid pixel alignment                | ✅ WORKING               | Uses rasterio reprojection                        |

#### E. SCALE CALIBRATION

| Requirement                      | Status             | Details                                                        |
| -------------------------------- | ------------------ | -------------------------------------------------------------- |
| Calibration method               | ⚠️ STUB            | Linear regression exists but no SRTM data to calibrate against |
| SRTM/DEM source                  | ❌ FAIL            | `fetch_srtm_tile` is a stub that returns any .tif in directory |
| Ground Control Points            | ❌ NOT IMPLEMENTED | `refine_with_gcps` is a stub                                   |
| Scene-level statistics           | ❌ NOT IMPLEMENTED | `patch_stats` method falls back to linear                      |
| Scale/offset parameters reported | ⚠️ PARTIAL         | Printed to console, not returned in API response               |
| Confidence/uncertainty           | ❌ NOT IMPLEMENTED |                                                                |
| Limitations reported             | ❌ NOT IMPLEMENTED |                                                                |

#### F. DSM OUTPUT

| Requirement                          | Status                   | Details               |
| ------------------------------------ | ------------------------ | --------------------- |
| Standard geospatial output (GeoTIFF) | ✅ WORKING (when metric) | Via rasterio          |
| Preserve spatial metadata            | ✅ WORKING (when metric) | CRS + transform saved |
| Relative DSM/raster output           | ✅ WORKING               | 16-bit PNG            |
| Visualization                        | ✅ WORKING               | 3D mesh with texture  |

#### G. VALIDATION

| Requirement                         | Status             | Details                           |
| ----------------------------------- | ------------------ | --------------------------------- |
| RMSE                                | ✅ IMPLEMENTED     | `eval/metrics.py`                 |
| MAE                                 | ✅ IMPLEMENTED     | `eval/metrics.py`                 |
| Correlation (Pearson)               | ✅ IMPLEMENTED     | `eval/metrics.py`                 |
| Bias/mean error                     | ❌ NOT IMPLEMENTED |                                   |
| Median absolute error               | ❌ NOT IMPLEMENTED |                                   |
| Percentile error                    | ❌ NOT IMPLEMENTED |                                   |
| Valid-pixel coverage                | ✅ IMPLEMENTED     | Reported in metrics               |
| Alignment/resampling support        | ❌ NOT IMPLEMENTED | Metrics assume pre-aligned arrays |
| No reference data available warning | ❌ NOT IMPLEMENTED |                                   |

#### H. LANDSCAPE STABILITY

| Requirement                  | Status       | Details                           |
| ---------------------------- | ------------ | --------------------------------- |
| Urban validation             | ❌ SIMULATED | Random data, not real imagery     |
| Sparse/bare validation       | ❌ SIMULATED | Same                              |
| Hilly validation             | ❌ SIMULATED | Same                              |
| Forested validation          | ❌ SIMULATED | Same                              |
| Per-scene/category reporting | ⚠️ PARTIAL   | Structure exists but data is fake |

#### I. 3D VISUALIZATION

| Requirement                     | Status             | Details                                   |
| ------------------------------- | ------------------ | ----------------------------------------- |
| Optical texture projection      | ✅ WORKING         | Original image UV-mapped onto mesh        |
| Terrain mesh generation         | ✅ WORKING         | PlaneGeometry with heightmap displacement |
| Camera movement                 | ✅ WORKING         | WASD + mouse look                         |
| First-person navigation         | ✅ WORKING         | PointerLockControls                       |
| Aerial viewing                  | ✅ WORKING         | OrbitControls via 'C' toggle              |
| Arbitrary camera perspectives   | ✅ WORKING         | Orbit + free camera                       |
| Elevation inspection            | ❌ NOT IMPLEMENTED | No cursor height readout                  |
| Slope inspection                | ❌ NOT IMPLEMENTED | Stub button only                          |
| Legend/scale                    | ❌ NOT IMPLEMENTED |                                           |
| Loading/error states            | ⚠️ PARTIAL         | Basic loading text, no progress           |
| Actual generated elevation data | ✅ WORKING         | Uses real heightmap from backend          |

#### J. USER EXPERIENCE

| Requirement                 | Status             | Details                               |
| --------------------------- | ------------------ | ------------------------------------- |
| Upload image                | ✅ WORKING         | File input + Process button           |
| Determine georeferenced     | ❌ NOT SHOWN       | No UI indication                      |
| Run reconstruction          | ✅ WORKING         | Process button → API call             |
| View depth/elevation output | ⚠️ PARTIAL         | Only 3D view, no 2D depth map display |
| View 3D reconstruction      | ✅ WORKING         | Three.js flythrough                   |
| Inspect height values       | ❌ NOT IMPLEMENTED |                                       |
| Inspect slope/terrain       | ❌ NOT IMPLEMENTED |                                       |
| Compare with reference      | ❌ NOT IMPLEMENTED |                                       |

#### K. DEPLOYMENT

| Requirement               | Status            | Details                                                 |
| ------------------------- | ----------------- | ------------------------------------------------------- |
| Installation documented   | ⚠️ PARTIAL        | README has basic setup                                  |
| Model download            | ⚠️ PARTIAL        | Auto-downloads via HuggingFace (may be slow/unreliable) |
| Environment setup         | ⚠️ PARTIAL        | requirements.txt exists but no venv lock                |
| CPU/GPU requirements      | ❌ NOT DOCUMENTED |                                                         |
| Starting frontend/backend | ✅ DOCUMENTED     |                                                         |
| Sample input              | ❌ NOT PROVIDED   | No real sample images in repo                           |
| Output locations          | ✅ DOCUMENTED     | `outputs/` directory                                    |
| Troubleshooting           | ❌ NOT DOCUMENTED |                                                         |

---

## Summary Scorecard

| Category                | Status                                         |
| ----------------------- | ---------------------------------------------- |
| Core Depth Estimation   | ✅ Functional (Depth Anything V2)              |
| Relative DSM Path       | ✅ Functional                                  |
| Absolute DSM Path       | ❌ Non-functional (no SRTM data, stub fetcher) |
| GeoTIFF Handling        | ⚠️ Partial (reads OK, calibration broken)      |
| Scale Calibration       | ❌ Stub only                                   |
| Validation Metrics      | ⚠️ Implemented but runs on fake data           |
| 3D Visualization        | ✅ Functional (mesh + texture + flythrough)    |
| Height/Slope Inspection | ❌ Not implemented                             |
| UI Polish               | ⚠️ Basic, no relative/absolute distinction     |
| Tests                   | ❌ Zero automated tests                        |
| Documentation           | ⚠️ Basic README, needs expansion               |
| Deployment              | ❌ No Docker, no CI, no lock files             |

---

## Architecture Assessment

**Classification**: **Transitional** — clean separation between depth/calibration/visualization, but missing critical package structure (`__init__.py`), no tests, stub implementations masked as complete, and simulated evaluation results presented as real.

**Existing Architecture** (preservable):

1. `depth/estimator.py` — Clean depth estimation abstraction with model registry
2. `calibration/pipeline.py` — Good orchestration pattern (detect → fetch → align → fit)
3. `calibration/georef.py` — Correct rasterio usage for GeoTIFF metadata and reprojection
4. `calibration/fit.py` — Linear regression calibration (needs real data to be meaningful)
5. `eval/metrics.py` — Solid metric computation with NaN masking
6. `frontend/scene.js` — Well-structured Three.js scene with dual control modes
7. `frontend/mesh.js` — Clean heightmap-to-mesh conversion

**Architecture that needs work**:

1. Missing `__init__.py` files (will crash)
2. No API endpoint for health/metadata inspection
3. Hardcoded URLs
4. No relative/absolute mode distinction in output or UI
5. No GeoTIFF-specific upload flow or metadata display
6. No height readout, slope analysis, or inspection tools
7. Simulated evaluation with no real data
