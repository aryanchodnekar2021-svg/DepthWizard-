import os
import sys
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

# Add the parent directory to the path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.pipeline import calibrate, save_output

def create_mock_geotiff(path, data, crs="EPSG:4326", transform=None):
    if transform is None:
        # Just some arbitrary origin and pixel size for testing
        transform = from_origin(-120.0, 35.0, 0.0001, 0.0001)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1 if len(data.shape) == 2 else data.shape[2],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if len(data.shape) == 2:
            dst.write(data, 1)
        else:
            for i in range(data.shape[2]):
                dst.write(data[:, :, i], i + 1)
    return transform

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_dir = os.path.join(base_dir, "data", "samples")
    srtm_dir = os.path.join(base_dir, "data", "srtm")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(srtm_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("--- Preparing Mock Data ---")
    
    # 1. Create a mock SRTM tile
    srtm_path = os.path.join(srtm_dir, "mock_srtm.tif")
    srtm_data = np.linspace(100, 500, 256*256).reshape((256, 256)).astype(np.float32)
    # Give it a specific transform
    srtm_transform = create_mock_geotiff(srtm_path, srtm_data)
    print(f"Created mock SRTM at {srtm_path}")
    
    # 2. Create a mock GeoTIFF image (aligned inside the SRTM bounds roughly)
    mock_geo_img_path = os.path.join(samples_dir, "mock_georeferenced.tif")
    geo_img_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    # Use same origin but different resolution to test resampling
    img_transform = from_origin(-120.0, 35.0, 0.00005, 0.00005)
    create_mock_geotiff(mock_geo_img_path, geo_img_data, transform=img_transform)
    print(f"Created mock GeoTIFF image at {mock_geo_img_path}")
    
    # 3. Create a mock PNG image
    mock_png_path = os.path.join(samples_dir, "mock_plain.png")
    png_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(png_data).save(mock_png_path)
    print(f"Created mock PNG image at {mock_png_path}")
    
    # 4. Mock depth map (relative depth output from Phase 1)
    mock_depth = np.linspace(0.1, 0.9, 128*128).reshape((128, 128)).astype(np.float32)
    
    print("\n--- Testing Pipeline ---")
    
    # Test 1: Georeferenced Image (Metric Mode)
    print("\nTest 1: Georeferenced GeoTIFF")
    metric_array, metric_metadata, mode1 = calibrate(mock_geo_img_path, mock_depth)
    print(f"Mode returned: {mode1}")
    print(f"Output stats: min={metric_array.min():.2f}, max={metric_array.max():.2f}")
    
    out_metric_path = os.path.join(outputs_dir, "mock_metric_dsm.tif")
    save_output(out_metric_path, metric_array, metric_metadata, mode1)
    
    # Test 2: Plain PNG Image (Relative Mode)
    print("\nTest 2: Plain PNG Image")
    relative_array, relative_metadata, mode2 = calibrate(mock_png_path, mock_depth)
    print(f"Mode returned: {mode2}")
    print(f"Output stats: min={relative_array.min():.2f}, max={relative_array.max():.2f}")
    
    out_relative_path = os.path.join(outputs_dir, "mock_relative_dsm.png")
    save_output(out_relative_path, relative_array, relative_metadata, mode2)
    
    print("\nCalibration tests completed successfully.")

if __name__ == "__main__":
    main()
