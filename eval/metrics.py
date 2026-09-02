"""
DSM evaluation metrics.

Computes quantitative accuracy metrics between predicted and reference DSMs.

Required metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Pearson correlation

Additional metrics:
- Bias (Mean Error)
- Median Absolute Error
- Percentile errors
- Valid-pixel coverage

IMPORTANT: Both arrays MUST be spatially aligned before calling this.
Do NOT compare mismatched coordinate systems or unaligned rasters.
"""

import numpy as np
from scipy.stats import pearsonr


def compute_metrics(
    predicted_dsm: np.ndarray,
    reference_dsm: np.ndarray,
    nodata_pred: float = np.nan,
    nodata_ref: float = np.nan,
) -> dict:
    """
    Compute accuracy metrics between predicted and reference DSMs.

    Args:
        predicted_dsm: 2D array of predicted elevation values
        reference_dsm: 2D array of reference elevation values (same shape)
        nodata_pred: Value treated as nodata in prediction (default: NaN)
        nodata_ref: Value treated as nodata in reference (default: NaN)

    Returns:
        Dictionary with metrics and metadata.

    Raises:
        ValueError: If array shapes don't match or insufficient valid data.
    """
    if predicted_dsm.shape != reference_dsm.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_dsm.shape} "
            f"vs reference {reference_dsm.shape}. "
            f"Resample to matching grid before comparing."
        )

    # Build valid mask
    valid_mask = np.isfinite(predicted_dsm) & np.isfinite(reference_dsm)

    # Also mask explicit nodata values
    if np.isfinite(nodata_pred):
        valid_mask &= predicted_dsm != nodata_pred
    if np.isfinite(nodata_ref):
        valid_mask &= reference_dsm != nodata_ref

    n_valid = int(np.sum(valid_mask))
    n_total = predicted_dsm.size
    coverage_pct = (n_valid / n_total) * 100 if n_total > 0 else 0.0

    if n_valid < 2:
        return {
            "rmse": None,
            "mae": None,
            "correlation": None,
            "bias": None,
            "median_ae": None,
            "percentile_90": None,
            "percentile_95": None,
            "valid_pixels": n_valid,
            "total_pixels": n_total,
            "coverage_pct": coverage_pct,
            "notes": "Insufficient valid pixels for metric computation.",
        }

    pred_valid = predicted_dsm[valid_mask].astype(np.float64)
    ref_valid = reference_dsm[valid_mask].astype(np.float64)

    # Errors
    errors = pred_valid - ref_valid
    abs_errors = np.abs(errors)

    # Core metrics
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_errors))
    bias = float(np.mean(errors))  # Mean Error
    median_ae = float(np.median(abs_errors))

    # Percentile errors
    percentile_90 = float(np.percentile(abs_errors, 90))
    percentile_95 = float(np.percentile(abs_errors, 95))

    # Pearson correlation
    try:
        if np.std(pred_valid) == 0 or np.std(ref_valid) == 0:
            corr = np.nan
        else:
            corr, _ = pearsonr(pred_valid, ref_valid)
            corr = float(corr)
    except Exception:
        corr = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": corr,
        "bias": bias,
        "median_ae": median_ae,
        "percentile_90": percentile_90,
        "percentile_95": percentile_95,
        "valid_pixels": n_valid,
        "total_pixels": n_total,
        "coverage_pct": coverage_pct,
    }


def compute_metrics_by_category(
    predicted_dsm: np.ndarray,
    reference_dsm: np.ndarray,
    category_mask: np.ndarray,
    categories: dict,
) -> dict:
    """
    Compute metrics per terrain category.

    Args:
        predicted_dsm: 2D predicted elevation
        reference_dsm: 2D reference elevation
        category_mask: 2D integer array where each value is a category ID
        categories: {category_id: category_name} mapping

    Returns:
        {category_name: metrics_dict} for each category.
    """
    results = {}
    for cat_id, cat_name in categories.items():
        mask = category_mask == cat_id
        if np.sum(mask) == 0:
            results[cat_name] = {"notes": "No pixels in this category"}
            continue

        pred_cat = predicted_dsm[mask]
        ref_cat = reference_dsm[mask]

        # Reshape to 2D for compute_metrics (needs matching shapes)
        # Use flat comparison instead
        valid = np.isfinite(pred_cat) & np.isfinite(ref_cat)
        if np.sum(valid) < 2:
            results[cat_name] = {"notes": "Insufficient valid pixels"}
            continue

        p = pred_cat[valid].astype(np.float64)
        r = ref_cat[valid].astype(np.float64)
        errors = p - r
        abs_errors = np.abs(errors)

        results[cat_name] = {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(abs_errors)),
            "bias": float(np.mean(errors)),
            "median_ae": float(np.median(abs_errors)),
            "valid_pixels": int(np.sum(valid)),
        }

    return results
