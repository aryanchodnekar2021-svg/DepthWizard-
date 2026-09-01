import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def read_geotiff_metadata(path):
    """
    Extracts bounding box, CRS, and resolution from a GeoTIFF.
    """
    try:
        with rasterio.open(path) as src:
            return {
                "bbox": src.bounds,
                "crs": src.crs,
                "resolution": src.res,
                "transform": src.transform,
                "shape": (src.height, src.width)
            }
    except Exception as e:
        print(f"Error reading GeoTIFF metadata from {path}: {e}")
        return None

def fetch_srtm_tile(bbox, data_dir="data/srtm/"):
    """
    Finds a matching SRTM 30m tile covering the bbox.
    For this stub, we just return the first .tif file found in the directory,
    or None if none exist (since we don't have a real tile database).
    """
    if not os.path.exists(data_dir):
        return None
        
    for file in os.listdir(data_dir):
        if file.endswith(".tif") or file.endswith(".tiff"):
            return os.path.join(data_dir, file)
            
    return None

def align_to_grid(srtm_path, target_shape, target_transform, target_crs):
    """
    Resamples the SRTM elevation array to match the input image's pixel grid.
    """
    with rasterio.open(srtm_path) as src:
        srtm_data = src.read(1)
        srtm_transform = src.transform
        srtm_crs = src.crs

        # Prepare destination array
        destination = np.empty(target_shape, dtype=np.float32)

        reproject(
            source=srtm_data,
            destination=destination,
            src_transform=srtm_transform,
            src_crs=srtm_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        return destination
