# Evaluation Results

**⚠️ PREVIOUS RESULTS REMOVED — THEY WERE FABRICATED**

The previous version of this file contained accuracy numbers generated from
random arrays, not real model evaluation. Those numbers were never measurements
of actual model performance.

## Current Status

No real evaluation has been performed yet. To run real evaluation:

1. Place predicted DSMs and reference DSMs in `data/eval_tiles/<terrain_type>/`
2. Expected structure:
   ```
   data/eval_tiles/
       urban/
           predicted.tif
           reference.tif
       hilly/
           predicted.tif
           reference.tif
       sparse/
           predicted.tif
           reference.tif
       forested/
           predicted.tif
           reference.tif
   ```
3. Run: `python -m eval.run_eval`

## Required Data Sources

- **Reference DSMs**: LiDAR-derived or high-quality photogrammetric DSMs
- **Predicted DSMs**: Output from DepthWizard pipeline
- Both must be spatially aligned (same CRS, transform, resolution)

## Metrics Reported

When real evaluation is performed:

- RMSE (Root Mean Square Error) in meters
- MAE (Mean Absolute Error) in meters
- Pearson correlation coefficient
- Bias (Mean Error)
- Median Absolute Error
- 90th and 95th percentile errors
- Valid-pixel coverage percentage

Per-terrain-type breakdown is required to identify weak cases.
