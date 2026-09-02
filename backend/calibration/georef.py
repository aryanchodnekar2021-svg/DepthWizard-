"""
Georeferencing utilities — GeoTIFF metadata extraction and raster alignment.

This module handles:
- Reading geospatial metadata from GeoTIFF files
- Aligning DEM data to an image's pixel grid
- CRS validation and reprojection
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def read_geotiff_metadata(path: str) -> Optional[dict]:
    """
    Extract full geospatial metadata from a GeoTIFF file.

    Returns None if the file is not a valid GeoTIFF with CRS.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF operations")

    try:
        with rasterio.open(path) as src:
            crs = src.crs
            if crs is None or not crs.is_valid:
                return None

            transform = src.transform
            bounds = src.bounds

            # Extract EPSG
            epsg = None
            try:
                epsg = crs.to_epsg()
            except Exception:
                pass

            # Geographic extent in WGS84
            geographic_extent = None
            try:
                if not crs.is_geographic:
                    from rasterio.warp import transform_bounds

                    wgs84_bounds = transform_bounds(
                        crs,
                        "EPSG:4326",
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )
                    geographic_extent = {
                        "west": wgs84_bounds[0],
                        "south": wgs84_bounds[1],
                        "east": wgs84_bounds[2],
                        "north": wgs84_bounds[3],
                    }
                else:
                    geographic_extent = {
                        "west": bounds.left,
                        "south": bounds.bottom,
                        "east": bounds.right,
                        "north": bounds.top,
                    }
            except Exception:
                pass

            return {
                "bbox": bounds,
                "crs": crs,
                "crs_string": str(crs),
                "epsg": epsg,
                "resolution": src.res,
                "transform": transform,
                "shape": (src.height, src.width),
                "nodata": src.nodata,
                "dtype": src.dtypes[0] if src.dtypes else None,
                "bounds_dict": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "geographic_extent": geographic_extent,
            }
    except Exception as e:
        logger.warning(f"Error reading GeoTIFF metadata from {path}: {e}")
        return None


def align_dem_to_image_grid(
    dem_data: np.ndarray,
    dem_transform: object,
    dem_crs: str,
    image_shape: Tuple[int, int],
    image_transform: object,
    image_crs: str,
) -> np.ndarray:
    """
    Reproject and resample DEM data to match an image's pixel grid.

    This ensures that each pixel in the aligned DEM corresponds to the
    same geographic location as the same pixel in the image.

    Args:
        dem_data: 2D DEM elevation array
        dem_transform: Affine transform of the DEM
        dem_crs: CRS string of the DEM
        image_shape: (height, width) of the target image grid
        image_transform: Affine transform of the target image
        image_crs: CRS string of the target image

    Returns:
        2D array of DEM values aligned to the image grid.
        NaN where DEM data is unavailable.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    target_height, target_width = image_shape
    destination = np.full((target_height, target_width), np.nan, dtype=np.float32)

    reproject(
        source=dem_data,
        destination=destination,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=image_transform,
        dst_crs=image_crs,
        resampling=Resampling.bilinear,
    )

    return destination


def validate_crs_consistency(
    crs1: str,
    crs2: str,
    context: str = "",
) -> bool:
    """
    Check if two CRS strings refer to the same coordinate system.

    Args:
        crs1, crs2: CRS strings to compare
        context: Description for logging

    Returns:
        True if CRS are equivalent, False otherwise.
    """
    try:
        from rasterio.crs import CRS

        c1 = CRS.from_string(crs1)
        c2 = CRS.from_string(crs2)
        result = c1 == c2
        if not result:
            logger.warning(
                f"CRS mismatch{f' ({context})' if context else ''}: {crs1} vs {crs2}"
            )
        return result
    except Exception as e:
        logger.warning(f"CRS comparison failed: {e}")
        return False
