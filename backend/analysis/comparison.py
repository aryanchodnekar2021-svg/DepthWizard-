"""
Reference DSM comparison — accuracy metrics and error maps.

Accepts two aligned 2D arrays (predicted + reference), computes pixel-wise
error, and returns comprehensive accuracy metrics. Handles NaN masking and
degenerate cases (insufficient samples, constant arrays).
"""

import numpy as np
from scipy.stats import pearsonr


def compute_error_map(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> dict:
    """
    Compute pixel-wise error between predicted and reference DSMs.

    Both arrays must already be spatially aligned (same shape, same CRS,
    same resolution). Reprojection and resampling are NOT done here —
    the caller is responsible for alignment.

    Args:
        predicted: 2D float array of predicted elevation.
        reference: 2D float array of reference elevation (same shape).

    Returns:
        dict with keys:
            error_map: np.ndarray (pred - ref), NaN where either input is NaN
            abs_error_map: np.ndarray of absolute errors
            rmse: float
            mae: float
            bias: float (mean error, positive = overprediction)
            correlation: float (Pearson r) or NaN
            median_abs_error: float
            p90_abs_error: float
            p95_abs_error: float
            valid_pixels: int
            total_pixels: int
            coverage_pct: float
            max_error: float

    Raises:
        ValueError: If shapes don't match or no valid pixels.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if predicted.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs "
            f"reference {reference.shape}"
        )

    # Build valid mask
    valid_mask = np.isfinite(predicted) & np.isfinite(reference)
    n_valid = int(np.sum(valid_mask))
    n_total = predicted.size
    coverage_pct = (n_valid / n_total) * 100.0 if n_total > 0 else 0.0

    # Error maps (NaN where either input is NaN)
    error_map = np.full_like(predicted, np.nan)
    abs_error_map = np.full_like(predicted, np.nan)

    if n_valid < 2:
        return {
            "error_map": error_map,
            "abs_error_map": abs_error_map,
            "rmse": None,
            "mae": None,
            "bias": None,
            "correlation": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "p95_abs_error": None,
            "valid_pixels": n_valid,
            "total_pixels": n_total,
            "coverage_pct": coverage_pct,
            "max_error": None,
            "notes": "Insufficient valid pixels for metric computation.",
        }

    pred_valid = predicted[valid_mask]
    ref_valid = reference[valid_mask]

    # Compute error on valid pixels, write back
    errors = pred_valid - ref_valid
    abs_errors = np.abs(errors)

    error_map[valid_mask] = errors
    abs_error_map[valid_mask] = abs_errors

    # Core metrics
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_errors))
    bias = float(np.mean(errors))
    median_abs_error = float(np.median(abs_errors))
    p90_abs_error = float(np.percentile(abs_errors, 90))
    p95_abs_error = float(np.percentile(abs_errors, 95))
    max_error = float(np.max(abs_errors))

    # Pearson correlation — handle degenerate cases
    correlation = np.nan
    try:
        if np.std(pred_valid) > 0 and np.std(ref_valid) > 0:
            corr, _ = pearsonr(pred_valid, ref_valid)
            correlation = float(corr)
    except Exception:
        pass

    return {
        "error_map": error_map,
        "abs_error_map": abs_error_map,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": correlation,
        "median_abs_error": median_abs_error,
        "p90_abs_error": p90_abs_error,
        "p95_abs_error": p95_abs_error,
        "valid_pixels": n_valid,
        "total_pixels": n_total,
        "coverage_pct": coverage_pct,
        "max_error": max_error,
    }
