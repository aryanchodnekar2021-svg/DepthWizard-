import os
import numpy as np
from PIL import Image
import rasterio

from .georef import read_geotiff_metadata, fetch_srtm_tile, align_to_grid
from .fit import fit_scale

def calibrate(image_path: str, depth_map: np.ndarray):
    """
    Orchestrates the calibration process.
    Returns:
        tuple (result_array, metadata_dict, mode)
        mode is either 'metric' or 'relative'
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    srtm_dir = os.path.join(base_dir, "data", "srtm")
    
    # Check if image is a GeoTIFF
    metadata = None
    if image_path.lower().endswith(('.tif', '.tiff')):
        metadata = read_geotiff_metadata(image_path)
        
    if metadata and metadata.get("crs") is not None:
        # We have a valid georeferenced image
        print(f"Image {os.path.basename(image_path)} is georeferenced. Attempting metric calibration.")
        
        srtm_path = fetch_srtm_tile(metadata["bbox"], data_dir=srtm_dir)
        
        if srtm_path:
            print(f"Found matching SRTM tile: {os.path.basename(srtm_path)}")
            # Align SRTM to the input image's grid
            srtm_aligned = align_to_grid(
                srtm_path, 
                metadata["shape"], 
                metadata["transform"], 
                metadata["crs"]
            )
            
            # Perform calibration
            metric_dsm = fit_scale(depth_map, srtm_aligned, method="linear")
            return metric_dsm, metadata, "metric"
        else:
            print(f"Warning: No SRTM tile found covering the bbox in {srtm_dir}. Falling back to relative mode.")
    else:
        print(f"Image {os.path.basename(image_path)} is not georeferenced (or no CRS found). Operating in relative mode.")
        
    # Fallback: Normalized Relative DSM
    # Normalize to 0-1
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max > d_min:
        rdsm = (depth_map - d_min) / (d_max - d_min)
    else:
        rdsm = np.zeros_like(depth_map)
        
    return rdsm, None, "relative"

def save_output(output_path, array, metadata=None, mode="relative"):
    """
    Saves the output array as a GeoTIFF or standard PNG.
    """
    if mode == "metric" and metadata:
        # Save as Float32 GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=metadata['crs'],
            transform=metadata['transform'],
        ) as dst:
            dst.write(array, 1)
        print(f"Saved metric DSM GeoTIFF to {output_path}")
    else:
        # Save as 16-bit grayscale PNG for better precision than 8-bit
        # Assuming array is 0-1 normalized
        array_16bit = (array * 65535).astype(np.uint16)
        Image.fromarray(array_16bit).save(output_path)
        print(f"Saved normalized relative DSM PNG to {output_path}")
