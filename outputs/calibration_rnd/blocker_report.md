# Calibration R&D — Real Data Blocker

**Timestamp:** 2026-09-03T03:59:56.444649+00:00

**Git SHA:** 2fb17009c1cb88f883215b272bb491593f492ee0

**Model:** depth-anything/Depth-Anything-V2-Small-hf

**Status:** blocked

**Reason:** No real paired RGB+reference DSM datasets configured

## Attempted datasets
ISPRS Potsdam (airborne, not satellite) and SIH2026 reference repository were inspected; no files present in repo.

## Required files
- RGB image (GeoTIFF with CRS or plain JPG with known bounds)
- Calibration DEM (SRTM tile, e.g., data/srtm/N18E073.tif exists but is single tile, not paired with reference)
- Independent reference DSM (LiDAR/photogrammetric, NOT the same as calibration DEM)

## Available local
- **srtm_tile:** data/srtm/N18E073.tif (EPSG:4326, 1201x1201, 90m, 401-716m) exists for calibration source only
- **sample_rgb:** data/samples/real_test_landscape.jpg (1024x1024, no CRS) exists but not georeferenced
- **eval_tiles:** data/eval_tiles empty
- **isprs_potsdam:** NOT FOUND locally; requires download from https://www2.isprs.org/commissions/comm2/wg4/benchmark/2D-sem-label-potsdam/ (requires registration)
- **sih2026_repo:** github.com/IMG-PROCESS-SAC/SIH2026 — not cloned locally; sample images not present

## Next actions
1. Obtain at least one real paired RGB + independent DSM tile (e.g., ISPRS Potsdam TOP image + DSM, or drone RGB + LiDAR DSM)
2. Place RGB and reference DSM in data/eval_tiles/<terrain>/ or update configs/calibration_experiments.yaml datasets with correct paths, CRS, resolution, terrain_type
3. Re-run: python -m eval.run_calibration_experiments --config configs/calibration_experiments.yaml
4. Ensure reference DSM is independent from calibration DEM; provenance must be LiDAR/photogrammetric/high_quality
