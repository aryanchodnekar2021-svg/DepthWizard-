"""
Uncertainty estimation — basic, labeled estimated, not confidence, not statistically complete.
Sources: calibration residual, DEM variability, depth variation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UncertaintyResult:
    per_pixel_uncertainty: np.ndarray
    mean_uncertainty: float
    residual_std: float
    dem_std: float
    depth_variability: float
    method: str


def per_pixel_uncertainty_map(
    residual_rmse: float, dem: np.ndarray, depth: np.ndarray, window: int = 5
) -> np.ndarray:
    """Local std based per-pixel uncertainty."""
    # local dem std via sliding window approx using uniform filter if available
    try:
        from scipy.ndimage import uniform_filter

        dem_f = dem.astype(np.float64)
        # nan handling: fill with mean
        valid = np.isfinite(dem_f)
        mean_val = float(np.mean(dem_f[valid])) if np.sum(valid) > 0 else 0.0
        filled = np.where(valid, dem_f, mean_val)
        mean = uniform_filter(filled, size=window)
        sq_mean = uniform_filter(filled**2, size=window)
        local_var = np.maximum(sq_mean - mean**2, 0)
        local_std = np.sqrt(local_var)
        # combine with residual
        unc = np.sqrt(residual_rmse**2 + local_std**2)
        unc[~valid] = np.nan
        return unc.astype(np.float32)
    except ImportError:
        return np.full_like(
            dem,
            float(residual_rmse) if np.isfinite(residual_rmse) else np.nan,
            dtype=np.float32,
        )


def estimate_uncertainty_from_residual(
    residuals: np.ndarray,
    dem: np.ndarray,
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> UncertaintyResult:
    """Combine residual_std, DEM variability, depth variation."""
    if mask is not None:
        valid = mask.astype(bool) & np.isfinite(dem) & np.isfinite(depth)
        if residuals is not None and residuals.shape == dem.shape:
            valid &= np.isfinite(residuals)
    else:
        valid = np.isfinite(dem) & np.isfinite(depth)
        if residuals is not None and residuals.shape == dem.shape:
            valid &= np.isfinite(residuals)
    residual_std = (
        float(np.std(residuals[np.isfinite(residuals)]))
        if residuals is not None and np.sum(np.isfinite(residuals)) > 1
        else float("nan")
    )
    dem_std = float(np.std(dem[valid])) if np.sum(valid) > 1 else float("nan")
    depth_var = float(np.std(depth[valid])) if np.sum(valid) > 1 else float("nan")
    # mean uncertainty as root sum square (illustrative, not rigorous)
    comps = [c for c in [residual_std, dem_std * 0.1, depth_var] if np.isfinite(c)]
    mean_unc = float(np.sqrt(sum(c**2 for c in comps))) if comps else float("nan")
    per_pixel = per_pixel_uncertainty_map(
        residual_std if np.isfinite(residual_std) else 10.0, dem, depth
    )
    return UncertaintyResult(
        per_pixel,
        mean_unc,
        residual_std,
        dem_std,
        depth_var,
        "root_sum_square_illustrative",
    )


def calibration_quality_flags(result) -> dict:
    """Warnings if residual large, R² low, etc."""
    flags = {}
    try:
        rmse = result.residual_rmse
        r2 = result.r_squared
        n = result.n_samples
        pre = result.pre_correlation
        if np.isfinite(rmse) and rmse > 50:
            flags["high_residual"] = f"RMSE {rmse:.1f}m suggests poor fit"
        if np.isfinite(r2) and r2 < 0.2:
            flags["low_r2"] = f"R² {r2:.2f} indicates weak linear relation"
        if n < 100:
            flags["few_samples"] = f"Only {n} samples"
        if np.isfinite(pre) and abs(pre) < 0.3:
            flags["low_correlation"] = f"pre-correlation {pre:.2f} weak"
    except Exception:
        pass
    return flags
