"""
Terrain classification from slope and elevation.

Classifies terrain into categories based on slope angle.
Categories are designed for SIH 26175 terrain-type grouping
but do NOT fabricate results — they require actual slope data.
"""

import numpy as np
from typing import Dict


# Terrain category IDs and names
TERRAIN_CATEGORIES: Dict[int, str] = {
    0: "flat",
    1: "gentle",
    2: "moderate",
    3: "steep",
    4: "very_steep",
}

# Slope thresholds in degrees
THRESHOLDS = {
    "flat_max": 2.0,
    "gentle_max": 8.0,
    "moderate_max": 15.0,
    "steep_max": 30.0,
    # >30 = very steep
}


def classify_terrain(slope_degrees: np.ndarray) -> np.ndarray:
    """
    Classify terrain into categories based on slope angle.

    Categories (uint8):
        0: flat      — slope < 2 degrees
        1: gentle    — 2-8 degrees
        2: moderate  — 8-15 degrees
        3: steep     — 15-30 degrees
        4: very_steep — > 30 degrees

    Args:
        slope_degrees: 2D float array of slope in degrees.
            NaN pixels are classified as 255 (nodata).

    Returns:
        2D uint8 array with category IDs.
        255 = nodata (where slope was NaN).
    """
    slope_degrees = np.asarray(slope_degrees, dtype=np.float64)

    if slope_degrees.ndim != 2:
        raise ValueError(f"Expected 2D array, got {slope_degrees.ndim}D")

    classification = np.full(slope_degrees.shape, 255, dtype=np.uint8)

    valid = np.isfinite(slope_degrees)

    classification[valid & (slope_degrees < THRESHOLDS["flat_max"])] = 0
    classification[
        valid
        & (slope_degrees >= THRESHOLDS["flat_max"])
        & (slope_degrees < THRESHOLDS["gentle_max"])
    ] = 1
    classification[
        valid
        & (slope_degrees >= THRESHOLDS["gentle_max"])
        & (slope_degrees < THRESHOLDS["moderate_max"])
    ] = 2
    classification[
        valid
        & (slope_degrees >= THRESHOLDS["moderate_max"])
        & (slope_degrees < THRESHOLDS["steep_max"])
    ] = 3
    classification[valid & (slope_degrees >= THRESHOLDS["steep_max"])] = 4

    return classification


def get_category_name(category_id: int) -> str:
    """Return the human-readable name for a category ID."""
    return TERRAIN_CATEGORIES.get(category_id, "unknown")


def get_category_legend() -> Dict[str, dict]:
    """Return the full legend for UI rendering."""
    legend = {}
    for cat_id, cat_name in TERRAIN_CATEGORIES.items():
        if cat_id == 0:
            desc = "slope < 2 degrees"
        elif cat_id == 4:
            desc = "slope > 30 degrees"
        else:
            thresholds = sorted(THRESHOLDS.values())
            desc = f"slope {thresholds[cat_id - 1]}-{thresholds[cat_id]} degrees"
        legend[cat_name] = {"id": cat_id, "description": desc}
    return legend
