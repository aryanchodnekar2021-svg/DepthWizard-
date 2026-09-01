# Phase-Wise Build Prompts
### Satellite Image → 3D Terrain Reconstruction & Flythrough (SIH 2026)

Copy-paste one phase at a time into your AI coding assistant (Claude Code, Cursor, etc.). Each prompt assumes the previous phase's output already exists in the repo — paste them in order, and fill in the `[bracketed]` details before running.

---

## Phase 0 — Setup & Team Split

```
Set up a monorepo for a hackathon project: "RGB image → 3D terrain reconstruction & flythrough."

Structure:
- /backend — Python 3.10+, FastAPI, Uvicorn, Pydantic
- /frontend — Three.js + vanilla JS/TS, simple upload UI
- /data — sample images and DEM tiles (gitignored, with a README on where to source them)
- /eval — accuracy evaluation harness (added later)

Do the following:
1. Create backend/requirements.txt with: fastapi, uvicorn, pydantic, torch, transformers, rasterio, numpy, scipy, scikit-learn, pillow.
2. Create a minimal FastAPI app in backend/main.py with a GET /health endpoint returning {"status": "ok"}.
3. Create frontend/index.html + frontend/main.js with an empty Three.js scene (just a rotating cube placeholder) served via a simple dev server, to confirm the rendering pipeline works before real data arrives.
4. Add a data/README.md explaining how to pull sample images from github.com/IMG-PROCESS-SAC/SIH2026 and SRTM 30m DEM tiles from OpenTopography or USGS EarthExplorer for [my test region].
5. Add a root README.md describing the repo layout and how the 3 sub-teams (Depth/ML, Calibration/Geospatial, Visualization/Frontend) plus 1 integrator should work in parallel branches.

Keep everything runnable with minimal setup — no Docker yet.
```

---

## Phase 1 — Elevation Extraction Module

```
In backend/, build a depth estimation module.

Requirements:
1. Create backend/depth/estimator.py with a function:
   estimate_depth(image: np.ndarray) -> np.ndarray
   that loads a pretrained Depth Anything V2 model from HuggingFace (use the smallest/fastest checkpoint suitable for CPU/limited-GPU inference) and returns a relative depth map as a float32 numpy array, same H×W as the input.
2. Make the backbone swappable — add a `backbone` parameter ("depth_anything_v2" default, "zoedepth", "midas" as fallbacks) so we can swap models if compute is tight, without changing the calling code.
3. Write backend/depth/run_sample.py: a CLI script that loads an image from data/samples/[filename], runs estimate_depth, and saves a normalized grayscale visualization of the depth map to outputs/ so I can visually sanity-check it.
4. Add a short markdown note (backend/depth/NOTES.md) documenting: model used, inference time on a sample image, and any visible domain-gap issues — flag specifically whether flat urban blocks or forest canopy areas look flattened/noisy in the output, since that will matter for calibration.
5. Wrap this as a clean, testable module with no FastAPI dependencies — it should be importable standalone.

Test it against images in data/samples/ covering [urban / sparse / hilly / forested] terrain if available, and summarize what you observe per terrain type.
```

---

## Phase 2 — Scale Calibration

```
Build the scale-calibration module that converts relative depth into metric elevation using SRTM reference data, in backend/calibration/.

Requirements:
1. backend/calibration/georef.py:
   - read_geotiff_metadata(path) -> {bbox, crs, resolution} using rasterio
   - fetch_srtm_tile(bbox) -> path to a matching SRTM 30m tile from data/srtm/ (assume tiles are pre-downloaded there; just handle lookup/matching by bbox)
   - align_to_grid(srtm_path, target_shape, target_transform) -> resampled SRTM elevation array matching the input image's pixel grid, using rasterio.warp

2. backend/calibration/fit.py:
   - fit_scale(relative_depth: np.ndarray, srtm_elevation: np.ndarray, method="linear") -> calibrated metric elevation array
   - Start with the simplest version: global linear regression (scale + offset) between relative depth values and resampled SRTM elevation at sparse sample points.
   - Add a second method="patch_stats" option that fits per-local-patch using mean/gradient/percentile statistics, for later use if time allows — but make "linear" the default and make sure it always works first.
   - Add an optional refine_with_gcps(elevation, gcp_points) function for lightweight bias correction using a few known-elevation flat regions — stub this simply for now if time is short.

3. backend/calibration/pipeline.py:
   - calibrate(image_path, depth_map) -> returns EITHER a metric DSM (if the image is georeferenced GeoTIFF) OR a normalized relative DSM (if it's plain PNG/JPG) — the plain-image path must never raise/fail, it should always succeed and just skip calibration.
   - Output the metric DSM as a GeoTIFF using rasterio; output the normalized rDSM as a standard PNG/array.

4. Write a test script backend/calibration/run_sample.py that runs the full calibrate() on one georeferenced and one non-georeferenced sample, prints whether metric or relative mode was used, and saves both outputs.

Prioritize getting a crude global-linear version working end-to-end first — this is the highest-risk part of the whole project.
```

---

## Phase 3 — Visualization Layer

```
Build the Three.js visualization layer in frontend/.

Requirements:
1. frontend/mesh.js:
   - buildTerrainMesh(heightmapUrl, textureUrl, options) -> returns a Three.js Mesh
   - Load the heightmap (grayscale PNG), use it to displace the vertices of a PlaneGeometry (subdivided enough for smooth relief, e.g. 256x256 segments — make this configurable).
   - Load the original RGB image as a texture and UV-map it 1:1 onto the same grid as the heightmap.

2. frontend/scene.js:
   - Set up a Three.js scene, camera, renderer.
   - Add a DirectionalLight + AmbientLight so terrain relief is clearly visible (not flat-lit).
   - Add first-person navigation using PointerLockControls (primary) with OrbitControls as a fallback/inspection mode toggle — this is a judged criterion, so movement should feel smooth and intuitive (WASD + mouse look, reasonable move speed, no jitter).

3. frontend/main.js:
   - Wire buildTerrainMesh + scene together with hardcoded sample heightmap/texture URLs from data/samples/ for now (before backend integration).
   - Add basic UI: an on-screen hint for controls (WASD to move, mouse to look, ESC to unlock).

4. Stub out (but don't fully build yet) three optional overlays behind simple toggle buttons: a minimap, a height color-ramp overlay, and slope shading — just enough scaffolding to fill in later if time allows.

Test with a placeholder heightmap/texture pair first to confirm the mesh renders and navigation feels right, before wiring to real depth output.
```

---

## Phase 4 — Integration

```
Wire the backend and frontend together end-to-end.

Backend (backend/main.py):
1. Add POST /process accepting multipart/form-data: `image` (file, PNG/JPG or GeoTIFF) and optional `geotag`.
2. Inside the handler: run estimate_depth() from Phase 1, then calibrate() from Phase 2, then convert the resulting DSM/rDSM into:
   - a compressed PNG heightmap (normalized 0-255, or 16-bit if precision matters)
   - a texture URL (the original image, saved/served statically)
3. Return JSON matching this contract:
   {
     "mode": "metric" | "relative",
     "dsm_url": string,
     "heightmap_url": string,
     "texture_url": string
   }
4. Serve outputs via a static files mount (e.g. /outputs/) so the frontend can fetch them directly.
5. Add CORS middleware so the frontend dev server can call the API locally.

Frontend (frontend/main.js):
1. Replace the hardcoded sample URLs with a real upload flow: a file input + "Process" button that POSTs to /process, shows a loading state, then calls buildTerrainMesh() with the returned heightmap_url/texture_url once the response arrives.
2. Handle and surface errors gracefully (e.g. if calibration fails, still render the relative fallback rather than showing a blank screen).

Get this working end-to-end on ONE sample image first, even if the mesh looks rough — confirm the full loop (upload → process → render → fly through) works before polishing anything else. Report back what breaks.
```

---

## Phase 5 — Evaluation Readiness

```
Build the accuracy evaluation harness in /eval.

Requirements:
1. eval/metrics.py:
   - compute_metrics(predicted_dsm: np.ndarray, reference_dsm: np.ndarray) -> {rmse, mae, correlation}
   - Handle NaN/nodata masking and resolution mismatches (resample reference to match predicted if needed, using rasterio).

2. eval/run_eval.py:
   - Loop over held-out test tiles in data/eval_tiles/, each tagged with a terrain_type ("urban", "sparse", "hilly", "forested").
   - For each tile: run the full pipeline (depth → calibration) from Phases 1–2, compute metrics against the matching SRTM/LiDAR reference, and print/save a results table broken out by terrain_type.
   - Output a summary eval/results.md with a markdown table: terrain_type | RMSE | MAE | correlation | notes.

3. Also write a short manual QA checklist (eval/mesh_qa.md) for the visualization half: navigability check, mesh artifact check (spikes/holes — describe what to look for and how to reproduce), and a standalone-deployment stability check (run a full 5-minute session without a crash).

Run this against whatever sample/held-out data currently exists, report the actual numbers, and flag any terrain type where accuracy looks especially weak so we know where to focus remaining time.
```

---

## Phase 6 — Polish & Demo Prep

```
Prepare the project for live demo and judging.

1. Pick/confirm 2-3 canned demo regions from data/ (ideally covering different terrain types) and pre-process them through /process now, caching the results in data/demo_cache/ so the live demo doesn't depend on a fresh upload working perfectly under time pressure. Add a "Load demo region" dropdown in the frontend UI that loads these cached results instantly instead of re-running inference.

2. Write docs/TECHNICAL_SUMMARY.md covering:
   - A short pipeline diagram (ASCII or mermaid) matching the actual implemented flow.
   - Model choices and why (depth backbone, calibration method used).
   - Calibration method actually implemented (linear vs. patch-stats vs. GCP-refined).
   - Final accuracy numbers per terrain type, pulled from eval/results.md.

3. Write docs/DEMO_SCRIPT.md: a rehearsed narration script for a 2-minute demo that explicitly hits BOTH judged criteria — e.g. "Here's our RMSE/MAE/correlation against SRTM on [terrain type]... and here's the live navigable reconstruction" — with a fallback line ready in case live upload fails (switch to a cached demo region without breaking flow).

4. Do a final pass: confirm the app starts cleanly from a fresh clone (README steps actually work), remove dead code/debug prints, and note any known limitations at the bottom of TECHNICAL_SUMMARY.md so judges hear it from us first, not discover it live.

Run through the full demo script once end-to-end and report any rough edges.
```

---

**Tip:** if your AI coding assistant supports it, keep a running summary of what each phase actually produced (file paths, function signatures) and prepend it to the next phase's prompt — that keeps later phases grounded in what was really built instead of what the plan assumed.
