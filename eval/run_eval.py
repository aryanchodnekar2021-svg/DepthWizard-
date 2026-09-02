"""
Evaluation harness for DepthWizard DSM accuracy.

THIS MODULE DOES NOT GENERATE REAL EVALUATION RESULTS.
It provides the infrastructure for evaluating against real datasets.

To run real evaluation:
1. Place predicted DSM and reference DSM in data/eval_tiles/
2. Ensure both are spatially aligned (same CRS, transform, shape)
3. Call compute_metrics() from eval/metrics.py

The previous version of this file generated random arrays and
presented them as evaluation results. That is no longer possible.
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import compute_metrics, compute_metrics_by_category

logger = logging.getLogger(__name__)


def evaluate_pair(
    predicted_path: str,
    reference_path: str,
    terrain_type: str = "unknown",
) -> dict:
    """
    Evaluate a single predicted/reference DSM pair.

    Both files must be spatially aligned GeoTIFFs or compatible rasters.

    Args:
        predicted_path: Path to predicted DSM
        reference_path: Path to reference DSM
        terrain_type: Label for the terrain category

    Returns:
        Dictionary with metrics and metadata.
    """
    import rasterio

    # Read predicted
    with rasterio.open(predicted_path) as src:
        pred = src.read(1).astype(np.float64)
        pred_nodata = src.nodata

    # Read reference
    with rasterio.open(reference_path) as src:
        ref = src.read(1).astype(np.float64)
        ref_nodata = src.nodata

    # Handle shape mismatch
    if pred.shape != ref.shape:
        logger.warning(
            f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}. "
            f"Resampling not implemented — skipping."
        )
        return {
            "terrain_type": terrain_type,
            "status": "skipped",
            "reason": "Shape mismatch",
        }

    # Compute metrics
    metrics = compute_metrics(
        pred,
        ref,
        nodata_pred=pred_nodata if pred_nodata is not None else np.nan,
        nodata_ref=ref_nodata if ref_nodata is not None else np.nan,
    )

    return {
        "terrain_type": terrain_type,
        "status": "ok",
        "predicted": predicted_path,
        "reference": reference_path,
        **metrics,
    }


def main():
    """
    Run evaluation on all available tile pairs in data/eval_tiles/.

    Expected directory structure:
        data/eval_tiles/
            urban/
                predicted.tif
                reference.tif
            hilly/
                predicted.tif
                reference.tif
            ...

    If no real data is available, prints a message and exits.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(base_dir, "data", "eval_tiles")

    if not os.path.isdir(eval_dir):
        print("No eval_tiles directory found. Cannot run evaluation.")
        print("Place predicted and reference DSMs in data/eval_tiles/<terrain_type>/")
        return

    # Find terrain subdirectories with both predicted.tif and reference.tif
    terrain_dirs = []
    for entry in os.listdir(eval_dir):
        entry_path = os.path.join(eval_dir, entry)
        if os.path.isdir(entry_path):
            pred = os.path.join(entry_path, "predicted.tif")
            ref = os.path.join(entry_path, "reference.tif")
            if os.path.isfile(pred) and os.path.isfile(ref):
                terrain_dirs.append((entry, pred, ref))

    if not terrain_dirs:
        print("No valid tile pairs found.")
        print(
            "Expected structure: data/eval_tiles/<terrain>/predicted.tif + reference.tif"
        )
        return

    print(f"Found {len(terrain_dirs)} terrain categories to evaluate.")

    results = []
    for terrain, pred_path, ref_path in terrain_dirs:
        print(f"\nEvaluating: {terrain}")
        result = evaluate_pair(pred_path, ref_path, terrain_type=terrain)
        results.append(result)

        if result["status"] == "ok":
            print(f"  RMSE: {result['rmse']:.2f}m")
            print(f"  MAE:  {result['mae']:.2f}m")
            print(f"  Corr: {result['correlation']:.3f}")
            print(f"  Coverage: {result['coverage_pct']:.1f}%")
        else:
            print(f"  Skipped: {result.get('reason', 'unknown')}")

    # Save results
    results_path = os.path.join(base_dir, "eval", "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "note": "REAL evaluation results from actual DSM pairs",
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
