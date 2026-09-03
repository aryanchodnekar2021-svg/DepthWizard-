# Calibration Code Audit — DepthWizard feat/sih-26175-calibration-rnd

Date: 2026-05-13
Branch: feat/sih-26175-calibration-rnd (from feat/sih-26175-realdata, HEAD 2fb1700)
Auditor: Sisyphus (opencode/nemotron-3-ultra-free)

## Scope

Inspected: backend/calibration/{fit.py,georef.py,pipeline.py}, backend/geo/{raster_inspect.py,srtm_provider.py}, backend/depth/estimator.py, backend/analysis/{comparison.py,slope.py,classification.py}, eval/{metrics.py,run_eval.py,config.yaml}, backend/models.py

## 1. What "depth" means

- Source: Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf) via HF pipeline(task="depth-estimation").
- Output: relative disparity-like structure, NOT metric depth, NOT height above ground, NOT DSM. Values are float32 per-pixel relative depth. Pipeline returns PIL "depth" image; estimator converts to np.float32 array resized to input (H,W) via bilinear.
- prepare_image() handles grayscale→RGB, multi-band→RGB, float→uint8 scaling (max>1 → clip 0-255, else *255). No normalization of depth output.
- No documentation of depth direction (closer = larger or smaller?) in code. Must verify empirically: HF Depth Anything V2 typically outputs larger = closer (disparity). This is UNVERIFIED assumption in fit.py.
- Tiled inference: estimate_depth_tiled() splits with overlap 64, stride = tile_size-overlap, linear ramp weighting. No global scale consistency guarantee across tiles.

## 2. Depth direction

- ASSUMED (not verified): larger D → closer/higher? Linear fit H = aD + b will absorb sign via negative scale if inverted. Current code does not invert. If depth is inverse (closer=larger) and elevation has no such inversion, scale may be negative. No sign check.
- CRITICAL: Do not assume D correlates positively with elevation. Document pre_correlation and post_correlation always.

## 3. Normalization

- NO normalization before fit.fit.py uses raw depth values directly in np.polyfit.
- pipeline._relative_fallback() DOES normalize to 0-1 for non-georeferenced case: (D-min)/(max-min). But metric path does NOT.
- Hidden scaling: subsampling random choice (50000 limit) without seed documentation in audit (now: np.random.choice without seed). Non-reproducible.

## 4. Calibration sample selection

- valid_mask = isfinite(depth) & isfinite(dem) & (dem != 0)
- ASSUMPTION: dem==0 is nodata (SRTM alternative nodata). But real SRTM tile data/srtm/N18E073.tif has nodata=-9999, min 401m, so 0-mask is redundant but harmless. For that tile, no zeros exist.
- Threshold: n_valid <10 → ValueError. Else proceed.
- No masking for depth==0, no check for DEM nodata=-9999 explicitly (relies on isfinite? -9999 is finite, so not masked! BUG: DEM nodata -9999 would be treated as valid elevation -9999m, skewing fit. srtm_provider mosaic uses NaN for nodata, but single tile read via read_srtm_tile returns raw float32 with -9999 nodata preserved? Actually mosaic reprojects with NaN init but read_srtm_tile returns raw -9999 if file nodata. For N18E073.tif, nodata=-9999, but array contains no -9999 values, so currently not hit. For other tiles with voids, this WOULD be bug.)
- No spatial stratification, no terrain weighting.

## 5. DEM alignment

- read_geotiff_metadata(): extracts CRS, transform, bounds, resolution, nodata, epsg, geographic_extent. Returns None if no valid CRS.
- align_dem_to_image_grid(): uses rasterio.warp.reproject with bilinear interpolation from DEM grid to image grid. DST is NaN-filled float32. No nodata mask propagation beyond NaN.
- srtm_provider.SRTMProvider.get_dem(): finds tiles by lat/lon floor, mosaics via mosaic_tiles() with bilinear reprojection onto target bounds/shape derived from image metadata (bounds left/bottom/right/top, shape HxW, CRS). Valid pixel pct computed; >50% nodata → error, >10% → warning.
- ASSUMPTION: DEM CRS is EPSG:4326. Tile file for N18E073 is EPSG:4326, res 0.00083 deg (~92m at 18N, not 30m). SRTM 1-arcsec should be 1201x1201 per degree (~30m at equator, ~92m lon at 18N? Actually 1/3600 deg ~30m, but file reports 0.00083 deg = ~3 arcsec ~90m, suggests 3-arcsec SRTM, not 1-arcsec. Documentation says 30m but data is 90m. Mismatch.
- Overlap: not explicitly checked beyond valid pixel count. No diagnostic recording of overlap pct, resolution ratio.

## 6. Outlier rejection

- _remove_outliers(x,y,method=iqr,factor=3.0): fits initial polyfit, computes residuals, IQR on residuals, lower=Q1-factor*IQR, upper=Q3+factor*IQR. Returns inlier mask.
- Factor 3.0 is lenient (typical 1.5). Retains most points.
- Alternative zscore path exists but not used by default.
- len<4 → no rejection. std==0 → no rejection.
- No RANSAC, no robust estimator, no spatial coherence.

## 7. Scale/offset estimation

- After outlier removal + optional subsampling (50000 random without seed), does np.polyfit(x_fit,y_fit,1): returns scale=a, offset=b minimizing squared error.
- Residuals computed on full inlier set (not just subsample): y_pred = a*x + b, rmse=sqrt(mean((y-y_pred)^2)), mae=mean(abs(...)).
- No R², no correlation reported in CalibrationResult (only rmse/mae). No bias, median, p90.
- apply_calibration(): H = scale*D + offset, cast to float32.

## 8. DEM assumptions

- DEM is treated as coarse metric reference in meters above sea level (WGS84, SRTM EGM96 geoid). No vertical datum conversion. Assumes DEM and target DSM share vertical datum.
- SRTM 30m (actually 90m in sample) is much coarser than RGB resolution (e.g., 1024x1024 over ~0.01 deg → ~1m/pixel vs 90m DEM). Direct pixel-wise fit will be dominated by low-frequency trend, high-frequency depth detail has no DEM counterpart.
- DEM is used BOTH as training (to fit a,b) and implicitly as desired output baseline. No independent validation DSM distinction in current pipeline.

## 9. Output quantity: surface elevation or other?

- Current fit produces: calibrated DSM = linear map of relative depth → elevation. This is NOT physically surface elevation; it is an affine rescaling of relative structure to match coarse DEM statistics.
- Since depth ≠ height above ground, H = aD+b is mathematically convenient but physically unjustified. For oblique or landscape photos, depth is view-dependent distance, not nadir height. Only for nadir aerial with flat terrain does depth correlate with elevation.
- Result is labeled "metric DSM" but without independent LiDAR/photogrammetric reference, no accuracy claim is valid. Must tag provenance: calibration source SRTM, not ground truth.

## 10. Additional findings

- models.CalibrationMetadata: applied, source, scale, offset, fit_method, valid_samples, residual_rmse/mae, dem_tile_ids.
- 90/90 tests pass (claimed in prior phase, not yet re-verified this branch; synthetic fixtures only).
- No uncertainty estimation.
- No spatial cross-validation; leakage risk if random pixel split.
- No frequency decomposition experiment.
- No terrain-specific evaluation.
- Magic constants: subsample_limit 50000, outlier factor 3.0, min valid 10, dem==0 nodata, no seed.
- Silent fallback: pipeline.calibrate() returns relative fallback on any DEM error without raising.

## 11. Documented assumptions that must be stated per method

- Depth is relative, unknown scale, unknown offset, unknown direction.
- Linear affine H=aD+b assumes perfect correlation between depth and elevation (R² ≈1). If correlation low, method fails.
- DEM is coarse truth for low-frequency component only.
- Outlier rejection assumes Gaussian-like residuals.
- Alignment assumes correct CRS/transform and bilinear is appropriate for elevation (should be bilinear, not nearest).
- No lens distortion correction, no camera intrinsics.

## 12. Required corrections before R&D experiments

- Mask DEM nodata -9999 explicitly (or use src.nodata).
- Record overlap pct, resolution ratio, CRS match for every experiment (Step 4).
- Add pre/post correlation, R², bias, median/P90/P95 to calibration result.
- Make subsampling reproducible via seed.
- Verify depth direction empirically with synthetic ramp test.
- Never use calibration DEM as validation reference; require independent DSM.
- Treat N18E073.tif as calibration source only; note its actual 90m resolution, not 30m.

## 13. ISPRS Potsdam / SIH2026 reference

- No dataset present in repo. data/eval_tiles empty, eval/config.yaml datasets: []. Must attempt access and document blocker if unavailable (Step 19).
