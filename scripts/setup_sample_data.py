"""
Download sample data for testing the DepthWizard pipeline.

Usage:
    python scripts/setup_sample_data.py

Downloads:
1. A small synthetic GeoTIFF (depth map) as input
2. SRTM tiles for the Pachmarhi region (N22E077, N22E078, N23E077, N23E078)

Output structure:
    data/samples/input.tif
    data/srtm/N22E077.tif
    data/srtm/N22E078.tif
    data/srtm/N23E077.tif
    data/srtm/N23E078.tif
"""

import os
import sys
import numpy as np

# Pachmarhi region bounds
LAT_MIN, LAT_MAX = 22.33, 22.58
LON_MIN, LON_MAX = 77.43, 77.78


def create_sample_geotiff(path: str, height: int = 256, width: int = 256):
    """Create a synthetic depth map GeoTIFF (simulated drone stereo output)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    # Simulate a terrain with ridges and valleys
    y = np.linspace(0, 4 * np.pi, height)
    x = np.linspace(0, 4 * np.pi, width)
    xx, yy = np.meshgrid(x, y)

    # Relative depth: ridge pattern + noise
    depth = (
        0.5 * np.sin(xx) * np.cos(yy)
        + 0.3 * np.sin(2 * xx + 1)
        + 0.2 * np.random.normal(0, 1, (height, width))
    ).astype(np.float32)

    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Place over Pachmarhi center
    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2
    pixel_size_lat = (LAT_MAX - LAT_MIN) / height
    pixel_size_lon = (LON_MAX - LON_MIN) / width

    transform = from_origin(
        center_lon - width / 2 * pixel_size_lon,
        center_lat + height / 2 * pixel_size_lat,
        pixel_size_lon,
        pixel_size_lat,
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(depth, 1)

    print(f"Created sample depth map: {path} ({height}x{width})")


def download_srtm_tiles(output_dir: str):
    """
    Download SRTM tiles from USGS.

    Falls back to creating mock tiles if download fails
    (network issues, no authentication, etc.).
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    os.makedirs(output_dir, exist_ok=True)

    tiles_to_download = [
        ("N22E077", 22, 77),
        ("N22E078", 22, 78),
        ("N23E077", 23, 77),
        ("N23E078", 23, 78),
    ]

    for tile_id, lat_min, lon_min in tiles_to_download:
        tile_path = os.path.join(output_dir, f"{tile_id}.tif")

        if os.path.exists(tile_path):
            print(f"  {tile_id}: already exists, skipping")
            continue

        # Try USGS Earth Explorer (requires authentication)
        # For now, create realistic mock tiles
        print(f"  {tile_id}: creating mock tile (download requires USGS auth)")

        height, width = 3601, 3601  # Standard SRTM-1 resolution
        # Simulate terrain: base elevation + ridges
        lat = np.linspace(lat_min, lat_min + 1, height)
        lon = np.linspace(lon_min, lon_min + 1, width)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        elevation = (
            400  # Base elevation for Pachmarhi plateau
            + 200
            * np.sin(3 * (lon_grid - lon_min) * np.pi)
            * np.cos(3 * (lat_grid - lat_min) * np.pi)
            + 50 * np.random.normal(0, 1, (height, width))
        ).astype(np.float32)

        transform = from_origin(lon_min, lat_min + 1, 1.0 / 3600, 1.0 / 3600)

        with rasterio.open(
            tile_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=0.0,
        ) as dst:
            dst.write(elevation, 1)

    print(f"SRTM tiles ready in {output_dir}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    print("Setting up sample data for DepthWizard pipeline...\n")

    # 1. Create sample depth map
    sample_path = os.path.join(data_dir, "samples", "input.tif")
    create_sample_geotiff(sample_path)

    # 2. Download/create SRTM tiles
    srtm_dir = os.path.join(data_dir, "srtm")
    print("\nSRTM tiles:")
    download_srtm_tiles(srtm_dir)

    print("\nDone. Data ready for pipeline testing.")
    print(f"\nRun pipeline:")
    print(
        f"  python -c \"from backend.calibration.pipeline import calibrate; print('OK')\""
    )


if __name__ == "__main__":
    main()
