"""
SRTM DEM Provider — real tile lookup, mosaicking, and alignment.

Treats SRTM 1 arc-second Global (~30m) data.
Respects WGS84 coordinate system and tile naming conventions.

This module does NOT download data — it works with locally supplied SRTM tiles.
Place tiles in data/srtm/ following standard naming: N00E000.hgt or .tif
"""

import os
import math
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SRTMTileInfo:
    """Information about a discovered SRTM tile."""

    tile_id: str  # e.g., "N28E077"
    file_path: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def lat_lon_to_tile_id(lat: float, lon: float) -> str:
    """
    Convert latitude/longitude to SRTM tile name.

    SRTM tiles are named by their SW corner:
    - Latitude: N if >= 0, S if < 0
    - Longitude: E if >= 0, W if < 0

    SRTM 1 arc-second covers 1° x 1° per tile.

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)

    Returns:
        Tile ID string, e.g., "N28E077"
    """
    # Tile covers the 1-degree square from (floor(lat), floor(lon))
    tile_lat = int(math.floor(lat))
    tile_lon = int(math.floor(lon))

    lat_dir = "N" if tile_lat >= 0 else "S"
    lon_dir = "E" if tile_lon >= 0 else "W"

    return f"{lat_dir}{abs(tile_lat):02d}{lon_dir}{abs(tile_lon):03d}"


def tile_id_to_bounds(tile_id: str) -> Tuple[float, float, float, float]:
    """
    Convert SRTM tile ID to its geographic bounds (WGS84).

    Args:
        tile_id: e.g., "N28E077"

    Returns:
        (min_lat, max_lat, min_lon, max_lon)
    """
    lat_dir = tile_id[0]
    lon_dir = tile_id[3]  # After lat digits + lon direction

    # Parse latitude
    lat_str = tile_id[1:3]
    lat_val = int(lat_str)
    min_lat = lat_val if lat_dir == "N" else -lat_val
    max_lat = min_lat + 1.0

    # Parse longitude
    lon_str = tile_id[4:7]
    lon_val = int(lon_str)
    min_lon = lon_val if lon_dir == "E" else -lon_val
    max_lon = min_lon + 1.0

    return (min_lat, max_lat, min_lon, max_lon)


def find_tiles_for_aoi(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float, srtm_dir: str
) -> List[SRTMTileInfo]:
    """
    Find all SRTM tiles that overlap with the given AOI.

    Args:
        min_lat, max_lat: Latitude bounds of AOI
        min_lon, max_lon: Longitude bounds of AOI
        srtm_dir: Directory containing SRTM tiles

    Returns:
        List of SRTMTileInfo for tiles that overlap the AOI.
    """
    if not os.path.isdir(srtm_dir):
        logger.warning(f"SRTM directory does not exist: {srtm_dir}")
        return []

    # Determine which tiles the AOI spans
    # A tile covers [floor(lat), floor(lat)+1) x [floor(lon), floor(lon)+1)
    min_tile_lat = int(math.floor(min_lat))
    max_tile_lat = int(math.floor(max_lat))
    min_tile_lon = int(math.floor(min_lon))
    max_tile_lon = int(math.floor(max_lon))

    tiles = []
    for tile_lat in range(min_tile_lat, max_tile_lat + 1):
        for tile_lon in range(min_tile_lon, max_tile_lon + 1):
            lat_dir = "N" if tile_lat >= 0 else "S"
            lon_dir = "E" if tile_lon >= 0 else "W"
            tile_id = f"{lat_dir}{abs(tile_lat):02d}{lon_dir}{abs(tile_lon):03d}"

            # Check for .tif or .hgt files
            for ext in [".tif", ".tiff", ".hgt"]:
                file_path = os.path.join(srtm_dir, f"{tile_id}{ext}")
                if os.path.isfile(file_path):
                    tile_bounds = tile_id_to_bounds(tile_id)
                    tiles.append(
                        SRTMTileInfo(
                            tile_id=tile_id,
                            file_path=file_path,
                            min_lat=tile_bounds[0],
                            max_lat=tile_bounds[1],
                            min_lon=tile_bounds[2],
                            max_lon=tile_bounds[3],
                        )
                    )
                    break

    return tiles


def read_srtm_tile(file_path: str) -> Tuple[np.ndarray, object, str]:
    """
    Read a single SRTM tile file.

    Args:
        file_path: Path to SRTM tile (.tif or .hgt)

    Returns:
        (data_array, transform, crs_string)
    """
    import rasterio

    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = str(src.crs)

    return data, transform, crs


def mosaic_tiles(
    tile_infos: List[SRTMTileInfo],
    target_bounds: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
    target_crs: str = "EPSG:4326",
) -> Tuple[np.ndarray, dict]:
    """
    Mosaic multiple SRTM tiles into a single aligned array.

    Handles:
    - CRS reprojection
    - Different resolutions
    - Nodata masking
    - Cropping to target bounds

    Args:
        tile_infos: List of SRTMTileInfo to mosaic
        target_bounds: (left, bottom, right, top) in target CRS
        target_shape: (height, width) of output array
        target_crs: Target coordinate reference system

    Returns:
        (mosaic_array, metadata_dict)
        mosaic_array is float32 with NaN for nodata.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import array_bounds

    left, bottom, right, top = target_bounds
    target_height, target_width = target_shape

    # Calculate target transform
    xres = (right - left) / target_width
    yres = (top - bottom) / target_height
    from rasterio.transform import Affine

    target_transform = Affine(xres, 0.0, left, 0.0, -yres, top)

    # Initialize output with NaN
    mosaic = np.full((target_height, target_width), np.nan, dtype=np.float32)

    for tile_info in tile_infos:
        try:
            tile_data, tile_transform, tile_crs = read_srtm_tile(tile_info.file_path)

            # Create destination array for this tile
            tile_reprojected = np.full(
                (target_height, target_width), np.nan, dtype=np.float32
            )

            # Reproject tile to target grid
            reproject(
                source=tile_data,
                destination=tile_reprojected,
                src_transform=tile_transform,
                src_crs=tile_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

            # Merge: use valid values from this tile where mosaic is still NaN
            valid_mask = ~np.isnan(tile_reprojected) & (tile_reprojected != 0)
            # SRTM uses 0 as nodata in some products
            mosaic[valid_mask] = tile_reprojected[valid_mask]

            logger.info(
                f"Added tile {tile_info.tile_id}: {np.sum(valid_mask)} valid pixels"
            )

        except Exception as e:
            logger.warning(f"Failed to read tile {tile_info.tile_id}: {e}")
            continue

    metadata = {
        "bounds": {"left": left, "bottom": bottom, "right": right, "top": top},
        "transform": [
            target_transform.a,
            target_transform.b,
            target_transform.c,
            target_transform.d,
            target_transform.e,
            target_transform.f,
        ],
        "crs": target_crs,
        "shape": (target_height, target_width),
        "tile_ids": [t.tile_id for t in tile_infos],
    }

    return mosaic, metadata


def crop_to_aoi(
    data: np.ndarray,
    full_transform: object,
    aoi_bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, object]:
    """
    Crop a raster array to a specific AOI.

    Args:
        data: 2D numpy array
        full_transform: Affine transform of the full array
        aoi_bounds: (left, bottom, right, top) to crop to

    Returns:
        (cropped_array, crop_transform)
    """
    from rasterio.transform import rowcol

    left, bottom, right, top = aoi_bounds

    # Convert bounds to pixel coordinates
    # rowcol returns (row, col) for a given (x, y) point
    try:
        # Top-left corner of AOI
        row_start, col_start = rowcol(full_transform, left, top)
        # Bottom-right corner of AOI
        row_end, col_end = rowcol(full_transform, right, bottom)
    except Exception:
        # Fallback: compute from transform
        inv_transform = ~full_transform
        col_start, row_start = inv_transform * (left, top)
        col_end, row_end = inv_transform * (right, bottom)
        row_start, col_start = int(row_start), int(col_start)
        row_end, col_end = int(row_end), int(col_end)

    # Clamp to array bounds
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(data.shape[0], row_end)
    col_end = min(data.shape[1], col_end)

    cropped = data[row_start:row_end, col_start:col_end].copy()

    # Compute transform for cropped region
    from rasterio.transform import Affine

    crop_transform = full_transform * Affine.translation(col_start, row_start)

    return cropped, crop_transform


class SRTMProvider:
    """
    DEM provider using locally stored SRTM tiles.

    Usage:
        provider = SRTMProvider(srtm_dir="data/srtm")
        dem, metadata = provider.get_dem(aoi_bounds, target_shape, target_crs)
    """

    def __init__(self, srtm_dir: str = "data/srtm"):
        self.srtm_dir = srtm_dir

    def get_dem(
        self,
        aoi_bounds: dict,
        target_shape: Tuple[int, int],
        target_crs: str = "EPSG:4326",
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
        """
        Get DEM data covering the given AOI.

        Args:
            aoi_bounds: {'left', 'bottom', 'right', 'top'} in WGS84
            target_shape: (height, width) for output
            target_crs: Target CRS for reprojection

        Returns:
            (dem_array, metadata, error_message)
            If successful, error_message is None.
            If failed, dem_array and metadata are None.
        """
        left = aoi_bounds["left"]
        bottom = aoi_bounds["bottom"]
        right = aoi_bounds["right"]
        top = aoi_bounds["top"]

        # Find overlapping tiles
        tiles = find_tiles_for_aoi(
            min_lat=bottom,
            max_lat=top,
            min_lon=left,
            max_lon=right,
            srtm_dir=self.srtm_dir,
        )

        if not tiles:
            return (
                None,
                None,
                (
                    f"No SRTM tiles found in {self.srtm_dir} covering "
                    f"bounds [{left:.4f}, {bottom:.4f}, {right:.4f}, {top:.4f}]. "
                    f"Download SRTM tiles from https://opentopography.org/ "
                    f"and place them in {self.srtm_dir}/"
                ),
            )

        tile_ids = [t.tile_id for t in tiles]
        logger.info(f"Found {len(tiles)} SRTM tiles: {tile_ids}")

        # Mosaic tiles
        dem, metadata = mosaic_tiles(
            tile_infos=tiles,
            target_bounds=(left, bottom, right, top),
            target_shape=target_shape,
            target_crs=target_crs,
        )

        # Report nodata coverage
        total_pixels = dem.shape[0] * dem.shape[1]
        valid_pixels = np.sum(~np.isnan(dem))
        nodata_pct = (1 - valid_pixels / total_pixels) * 100

        if nodata_pct > 50:
            return (
                None,
                None,
                (
                    f"SRTM coverage is only {100 - nodata_pct:.1f}% for this AOI. "
                    f"Tiles: {tile_ids}"
                ),
            )

        if nodata_pct > 10:
            logger.warning(f"SRTM has {nodata_pct:.1f}% nodata coverage for this AOI")

        metadata["valid_pixel_pct"] = 100 - nodata_pct
        return dem, metadata, None

    def get_available_tiles(self) -> List[str]:
        """List all available SRTM tile IDs in the configured directory."""
        if not os.path.isdir(self.srtm_dir):
            return []

        tiles = []
        for f in os.listdir(self.srtm_dir):
            base, ext = os.path.splitext(f)
            if ext.lower() in (".tif", ".tiff", ".hgt"):
                # Validate tile name format
                if len(base) == 7 and base[0] in "NS" and base[3] in "EW":
                    tiles.append(base)
        return sorted(tiles)
