"""
Slope computation from elevation arrays.

Computes terrain slope in degrees using numpy gradient.
Handles nodata (NaN), boundary effects, and provides deterministic results
for analytical verification (e.g., planar surfaces with known slope).
"""

import numpy as np


def compute_slope(
    elevation: np.ndarray,
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
) -> np.ndarray:
    """
    Compute slope in degrees from a 2D elevation array.

    Uses numpy gradient to compute partial derivatives, then:
        slope_degrees = arctan(sqrt(dz_dx^2 + dz_dy^2))

    For a planar surface z = a*x + b*y, the expected slope is:
        arctan(sqrt(a^2 + b^2))

    Args:
        elevation: 2D float array of elevation values. NaN treated as nodata.
        cell_size_x: Pixel width in map units. For projected CRS, this is
            the real pixel size in meters. For relative DSMs, use 1.0.
        cell_size_y: Pixel height in map units. May be negative for north-up
            rasters, but the slope formula squares the gradient so sign
            does not affect the result.

    Returns:
        2D float64 array of slope in degrees [0, 90].
        Pixels at nodata locations are NaN.
    """
    elevation = np.asarray(elevation, dtype=np.float64)

    if elevation.ndim != 2:
        raise ValueError(f"Expected 2D array, got {elevation.ndim}D")
    if elevation.shape[0] < 2 or elevation.shape[1] < 2:
        raise ValueError(
            f"Array too small for gradient: {elevation.shape}. "
            "Need at least 2 elements in each dimension."
        )

    # Replace NaN with 0 for gradient computation, then restore NaN in output
    nan_mask = ~np.isfinite(elevation)
    elevation_clean = np.where(nan_mask, 0.0, elevation)

    # Compute gradients along rows (axis 0, dz/dy) and cols (axis 1, dz/dx)
    # np.gradient returns [dz_dy, dz_dx] when given a 2D array
    dz_dy, dz_dx = np.gradient(elevation_clean, cell_size_y, cell_size_x)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Where input was NaN, output should be NaN
    slope_deg[nan_mask] = np.nan

    return slope_deg


def compute_slope_percent(
    elevation: np.ndarray,
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
) -> np.ndarray:
    """
    Compute slope as percent grade from a 2D elevation array.

    Percent slope = 100 * tan(slope_rad) = 100 * sqrt(dz_dx^2 + dz_dy^2)

    Args:
        elevation: 2D float array of elevation values.
        cell_size_x: Pixel width in map units.
        cell_size_y: Pixel height in map units.

    Returns:
        2D float64 array of slope in percent [0, inf).
    """
    elevation = np.asarray(elevation, dtype=np.float64)

    if elevation.ndim != 2:
        raise ValueError(f"Expected 2D array, got {elevation.ndim}D")
    if elevation.shape[0] < 2 or elevation.shape[1] < 2:
        raise ValueError(
            f"Array too small for gradient: {elevation.shape}. "
            "Need at least 2 elements in each dimension."
        )

    nan_mask = ~np.isfinite(elevation)
    elevation_clean = np.where(nan_mask, 0.0, elevation)

    dz_dy, dz_dx = np.gradient(elevation_clean, cell_size_y, cell_size_x)
    slope_pct = 100.0 * np.sqrt(dz_dx**2 + dz_dy**2)

    slope_pct[nan_mask] = np.nan

    return slope_pct
