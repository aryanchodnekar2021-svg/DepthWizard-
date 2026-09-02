"""
Calibration pipeline — orchestrates the full reconstruction flow.

Detects georeferencing → fetches DEM → aligns → calibrates → exports.

This module is the main entry point for the geospatial pipeline.
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np

from backend.models import (
    ElevationMode,
    CalibrationSource,
    CalibrationMetadata,
    ReconstructionResult,
    RasterMetadata,
)
from backend.geo.raster_inspect import inspect_raster
from backend.geo.srtm_provider import SRTMProvider
from backend.calibration.georef import (
    read_geotiff_metadata,
    align_dem_to_image_grid,
)
from backend.calibration.fit import (
    calibrate_depth_to_elevation,
    apply_calibration,
    create_calibration_metadata,
)

logger = logging.getLogger(__name__)


def calibrate(
    image_path: str,
    relative_depth: np.ndarray,
    srtm_dir: Optional[str] = None,
    force_mode: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[dict], str]:
    """
    Calibrate relative depth to metric elevation (if possible).

    This is the main pipeline entry point, preserving the existing API
    while implementing real geospatial logic.

    Args:
        image_path: Path to the input image/GeoTIFF
        relative_depth: 2D float array from depth estimation
        srtm_dir: Path to SRTM tile directory (default: data/srtm/)
        force_mode: "relative" or "absolute" to override auto-detection

    Returns:
        (result_array, metadata_dict_or_None, mode_string)
        mode_string is "metric" or "relative"
    """
    if srtm_dir is None:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        srtm_dir = os.path.join(base_dir, "data", "srtm")

    # Step 1: Inspect the input raster
    raster_meta = inspect_raster(image_path)

    if not raster_meta.is_georeferenced:
        logger.info(
            f"Input {os.path.basename(image_path)} is not georeferenced. "
            f"Operating in relative mode."
        )
        return _relative_fallback(relative_depth)

    # Step 2: We have a georeferenced input — attempt metric calibration
    if force_mode == "relative":
        return _relative_fallback(relative_depth)

    logger.info(
        f"Input {os.path.basename(image_path)} is georeferenced "
        f"(CRS: {raster_meta.crs}). Attempting metric calibration."
    )

    # Step 3: Get DEM data
    provider = SRTMProvider(srtm_dir=srtm_dir)

    # Get AOI from raster metadata
    aoi_bounds = raster_meta.bounds
    if aoi_bounds is None:
        logger.warning("No bounds in raster metadata. Falling back to relative.")
        return _relative_fallback(relative_depth)

    target_shape = (raster_meta.height, raster_meta.width)
    target_crs = raster_meta.crs or "EPSG:4326"

    dem, dem_meta, error = provider.get_dem(
        aoi_bounds=aoi_bounds,
        target_shape=target_shape,
        target_crs=target_crs,
    )

    if error:
        logger.warning(f"DEM unavailable: {error}")
        return _relative_fallback(relative_depth, reason=error)

    # Step 4: Calibrate
    try:
        cal_result = calibrate_depth_to_elevation(
            relative_depth=relative_depth,
            dem_elevation=dem,
        )
    except ValueError as e:
        logger.warning(f"Calibration failed: {e}")
        return _relative_fallback(relative_depth, reason=str(e))

    # Step 5: Apply calibration
    metric_dsm = apply_calibration(relative_depth, cal_result.scale, cal_result.offset)

    # Step 6: Build metadata
    metadata = {
        "crs": raster_meta.crs,
        "epsg": raster_meta.epsg,
        "transform": raster_meta.transform,
        "bounds": raster_meta.bounds,
        "resolution": raster_meta.resolution,
        "shape": (raster_meta.height, raster_meta.width),
        "nodata": raster_meta.nodata,
        "calibration": {
            "applied": True,
            "source": "srtm",
            "scale": cal_result.scale,
            "offset": cal_result.offset,
            "fit_method": cal_result.method,
            "valid_samples": cal_result.valid_samples,
            "residual_rmse": cal_result.residual_rmse,
            "residual_mae": cal_result.residual_mae,
            "dem_tile_ids": dem_meta.get("tile_ids", []),
        },
    }

    return metric_dsm, metadata, "metric"


def _relative_fallback(
    depth_map: np.ndarray,
    reason: Optional[str] = None,
) -> Tuple[np.ndarray, None, str]:
    """
    Produce a normalized relative DSM.

    Values are normalized to 0-1 range.
    These are NOT meters and must never be labeled as such.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max > d_min:
        rdsm = (depth_map - d_min) / (d_max - d_min)
    else:
        rdsm = np.zeros_like(depth_map)

    if reason:
        logger.info(f"Relative fallback: {reason}")

    return rdsm, None, "relative"


def save_output(
    output_path: str,
    array: np.ndarray,
    metadata: Optional[dict] = None,
    mode: str = "relative",
):
    """
    Save the output DSM to disk.

    For metric mode: saves as float32 GeoTIFF with CRS/transform.
    For relative mode: saves as 16-bit PNG (normalized 0-1).

    Never exports relative values as a GeoTIFF with fake CRS.
    """
    if mode == "metric" and metadata and metadata.get("crs"):
        # Save as float32 GeoTIFF with proper metadata
        _save_geotiff(output_path, array, metadata)
    else:
        # Save as 16-bit PNG for visualization
        # Ensure values are in 0-1 range for PNG
        if array.max() > 1.0 or array.min() < 0.0:
            a_min, a_max = array.min(), array.max()
            if a_max > a_min:
                array = (array - a_min) / (a_max - a_min)
            else:
                array = np.zeros_like(array)

        array_16bit = (np.clip(array, 0, 1) * 65535).astype(np.uint16)
        from PIL import Image

        Image.fromarray(array_16bit).save(output_path)
        logger.info(f"Saved relative DSM PNG to {output_path}")


def _save_geotiff(output_path: str, array: np.ndarray, metadata: dict):
    """Save a metric DSM as a proper GeoTIFF."""
    import rasterio
    from rasterio.transform import Affine

    crs_string = metadata["crs"]
    transform_list = metadata["transform"]
    transform = Affine(*transform_list[:6])

    # Ensure output directory exists
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype if array.dtype in ("float32", "float64") else "float32",
        crs=crs_string,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        write_array = array.astype(np.float32) if array.dtype != "float32" else array
        dst.write(write_array, 1)

    logger.info(f"Saved metric DSM GeoTIFF to {output_path}")
