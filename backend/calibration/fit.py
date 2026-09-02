"""
Metric calibration — converts relative depth to absolute elevation.

Implements: H_metric = a * D_relative + b

Where:
- H_metric = metric elevation (meters above sea level)
- D_relative = relative depth from monocular model
- a = scale factor (slope of depth-to-elevation relationship)
- b = offset (elevation intercept)

The monocular depth model provides relative disparity-like information.
SRTM provides coarse metric elevation as anchor.
The calibration finds the best linear mapping between them.

IMPORTANT: This treats SRTM as a coarse metric reference.
SRTM does NOT provide fine building-height ground truth.
The monocular model contributes higher-frequency structural detail.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from backend.models import CalibrationMetadata, CalibrationSource

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration fitting process."""

    scale: float  # a in H = a*D + b
    offset: float  # b in H = a*D + b
    valid_samples: int
    residual_rmse: float
    residual_mae: float
    method: str
    inlier_mask: Optional[np.ndarray] = None


def _remove_outliers(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "iqr",
    factor: float = 3.0,
) -> np.ndarray:
    """
    Remove outliers from paired data.

    Args:
        x, y: 1D arrays of paired values
        method: "iqr" for interquartile range, "zscore" for z-score
        factor: outlier threshold multiplier

    Returns:
        Boolean mask of inlier positions
    """
    if method == "iqr":
        # IQR method on residuals from initial fit
        if len(x) < 4:
            return np.ones(len(x), dtype=bool)

        # Quick initial fit
        coeffs = np.polyfit(x, y, 1)
        residuals = y - np.polyval(coeffs, x)

        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        return (residuals >= lower) & (residuals <= upper)

    elif method == "zscore":
        if len(x) < 3:
            return np.ones(len(x), dtype=bool)

        coeffs = np.polyfit(x, y, 1)
        residuals = y - np.polyval(coeffs, x)
        std = np.std(residuals)
        if std == 0:
            return np.ones(len(x), dtype=bool)

        z = np.abs(residuals / std)
        return z < factor

    else:
        return np.ones(len(x), dtype=bool)


def calibrate_depth_to_elevation(
    relative_depth: np.ndarray,
    dem_elevation: np.ndarray,
    method: str = "linear",
    outlier_method: str = "iqr",
    outlier_factor: float = 3.0,
    subsample_limit: int = 50000,
) -> CalibrationResult:
    """
    Fit relative depth to metric DEM elevation.

    Model: H_metric = a * D_relative + b

    Args:
        relative_depth: 2D array of relative depth values
        dem_elevation: 2D array of DEM elevation values (same shape)
        method: Fitting method ("linear", "ransac" reserved for future)
        outlier_method: Outlier detection method
        outlier_factor: Outlier threshold
        subsample_limit: Max samples for fitting (for speed)

    Returns:
        CalibrationResult with fit parameters and statistics.
    """
    if relative_depth.shape != dem_elevation.shape:
        raise ValueError(
            f"Shape mismatch: depth {relative_depth.shape} vs DEM {dem_elevation.shape}"
        )

    # Create valid mask: both must be finite and DEM must not be nodata
    valid_mask = (
        np.isfinite(relative_depth)
        & np.isfinite(dem_elevation)
        & (dem_elevation != 0)  # SRTM uses 0 as nodata in some products
    )

    n_valid = np.sum(valid_mask)
    if n_valid < 10:
        raise ValueError(
            f"Not enough valid pixels for calibration: {n_valid} (need at least 10)"
        )

    x = relative_depth[valid_mask].astype(np.float64)
    y = dem_elevation[valid_mask].astype(np.float64)

    # Remove outliers
    inlier_mask = _remove_outliers(x, y, method=outlier_method, factor=outlier_factor)
    x = x[inlier_mask]
    y = y[inlier_mask]
    n_inliers = len(x)

    logger.info(
        f"Calibration: {n_valid} valid pixels, {n_inliers} after outlier removal"
    )

    # Subsample for speed if needed
    if n_inliers > subsample_limit:
        indices = np.random.choice(n_inliers, subsample_limit, replace=False)
        x_fit = x[indices]
        y_fit = y[indices]
    else:
        x_fit = x
        y_fit = y

    # Fit linear model: y = a*x + b
    # Using least squares via numpy
    coeffs = np.polyfit(x_fit, y_fit, 1)
    scale = coeffs[0]  # a
    offset = coeffs[1]  # b

    # Compute residuals on full inlier set
    y_pred = scale * x + offset
    residuals = y - y_pred

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    logger.info(
        f"Calibration result: H = {scale:.4f} * D + {offset:.4f}, "
        f"RMSE={rmse:.2f}m, MAE={mae:.2f}m"
    )

    return CalibrationResult(
        scale=scale,
        offset=offset,
        valid_samples=n_inliers,
        residual_rmse=rmse,
        residual_mae=mae,
        method=method,
    )


def apply_calibration(
    relative_depth: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    """
    Apply calibration to convert relative depth to metric elevation.

    H_metric = scale * D_relative + offset

    Args:
        relative_depth: 2D array of relative depth
        scale: Calibration scale factor (a)
        offset: Calibration offset (b)

    Returns:
        2D array of metric elevation in meters.
    """
    return (relative_depth.astype(np.float64) * scale + offset).astype(np.float32)


def create_calibration_metadata(
    result: CalibrationResult,
    dem_tile_ids: list,
    source: CalibrationSource = CalibrationSource.SRTM,
) -> CalibrationMetadata:
    """Convert a CalibrationResult to CalibrationMetadata for the result model."""
    return CalibrationMetadata(
        applied=True,
        source=source,
        scale=result.scale,
        offset=result.offset,
        fit_method=result.method,
        valid_samples=result.valid_samples,
        residual_rmse=result.residual_rmse,
        residual_mae=result.residual_mae,
        dem_tile_ids=dem_tile_ids,
    )
