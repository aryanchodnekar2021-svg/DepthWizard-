"""
Raster input inspection module.

Determines whether a file is a plain image or a georeferenced raster.
Extracts full geospatial metadata from GeoTIFFs.
Does NOT determine georeferencing from filename extension alone.
"""

import os
from typing import Optional

from backend.models import RasterMetadata

# Extensions we accept
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RASTER_EXTENSIONS = {".tif", ".tiff"}  # Could potentially be GeoTIFF
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # Never georeferenced


def inspect_raster(file_path: str) -> RasterMetadata:
    """
    Inspect a raster file and extract metadata.

    For plain images (PNG/JPG): returns basic dimensions, is_georeferenced=False.
    For TIF/TIFF: attempts to read as GeoTIFF via rasterio.
    If rasterio reads CRS successfully: returns full geospatial metadata.
    If rasterio fails or no CRS: returns basic info, is_georeferenced=False.

    Args:
        file_path: Path to the raster file.

    Returns:
        RasterMetadata with all available information.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # For plain images, we know they're not georeferenced
    if ext in IMAGE_EXTENSIONS:
        return _inspect_plain_image(file_path)

    # For TIF/TIFF, try rasterio first
    return _inspect_geotiff(file_path)


def _inspect_plain_image(file_path: str) -> RasterMetadata:
    """Inspect a plain image file (PNG/JPG) with no geospatial data."""
    from PIL import Image
    import numpy as np

    with Image.open(file_path) as img:
        width, height = img.size
        # Determine band count from mode
        mode_to_bands = {
            "L": 1,
            "P": 1,
            "RGB": 3,
            "RGBA": 4,
            "CMYK": 4,
            "YCbCr": 3,
            "I": 1,
            "F": 1,
        }
        band_count = mode_to_bands.get(img.mode, len(img.getbands()))
        mode_to_dtype = {
            "L": "uint8",
            "P": "uint8",
            "RGB": "uint8",
            "RGBA": "uint8",
            "CMYK": "uint8",
            "YCbCr": "uint8",
            "I": "int32",
            "F": "float32",
        }
        dtype = np.dtype(mode_to_dtype.get(img.mode, "uint8")).name

    return RasterMetadata(
        is_georeferenced=False,
        file_path=file_path,
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
    )


def _inspect_geotiff(file_path: str) -> RasterMetadata:
    """
    Attempt to inspect a TIF/TIFF file as a GeoTIFF using rasterio.

    If the file has valid CRS and transform metadata, returns full
    geospatial metadata. Otherwise returns basic raster info with
    is_georeferenced=False.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import array_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF inspection. "
            "Install with: pip install rasterio"
        )

    try:
        with rasterio.open(file_path) as src:
            # Basic metadata (always available)
            width = src.width
            height = src.height
            band_count = src.count
            dtype = src.dtypes[0] if src.dtypes else "unknown"

            # Check if file has valid geospatial reference
            crs = src.crs
            transform = src.transform
            has_crs = crs is not None and crs.is_valid
            has_transform = transform is not None

            if has_crs and has_transform:
                # Full geospatial metadata
                bounds = src.bounds  # Left, Bottom, Right, Top
                resolution = src.res  # (x, y) pixel size
                nodata = src.nodata

                # Extract EPSG code
                epsg = None
                try:
                    epsg = crs.to_epsg()
                except Exception:
                    pass

                # Compute geographic extent in WGS84 if not already
                geographic_extent = None
                try:
                    if crs and not crs.is_geographic:
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

                return RasterMetadata(
                    is_georeferenced=True,
                    file_path=file_path,
                    width=width,
                    height=height,
                    band_count=band_count,
                    dtype=dtype,
                    crs=str(crs),
                    epsg=epsg,
                    transform=[
                        transform.a,
                        transform.b,
                        transform.c,
                        transform.d,
                        transform.e,
                        transform.f,
                    ],
                    bounds={
                        "left": bounds.left,
                        "bottom": bounds.bottom,
                        "right": bounds.right,
                        "top": bounds.top,
                    },
                    resolution={
                        "x": resolution[0],
                        "y": resolution[1],
                    },
                    nodata=nodata,
                    geographic_extent=geographic_extent,
                )
            else:
                # TIFF without valid geospatial reference
                return RasterMetadata(
                    is_georeferenced=False,
                    file_path=file_path,
                    width=width,
                    height=height,
                    band_count=band_count,
                    dtype=dtype,
                )

    except rasterio.errors.RasterioIOError:
        # Not a valid rasterio-readable file
        # Fall back to PIL to get basic info
        return _inspect_plain_image(file_path)
    except Exception as e:
        # Malformed file or other error
        raise ValueError(f"Error reading raster metadata from {file_path}: {e}")


def validate_input(file_path: str) -> RasterMetadata:
    """
    Validate input and return metadata.

    Convenience wrapper that combines inspection with validation.
    Raises clear errors for unsupported or malformed inputs.
    """
    metadata = inspect_raster(file_path)
    return metadata
