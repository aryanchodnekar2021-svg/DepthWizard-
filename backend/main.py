import os
import shutil
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from depth.estimator import estimate_depth
from calibration.pipeline import calibrate, save_output

app = FastAPI(title="DepthWizard API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup outputs directory
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
outputs_dir = os.path.join(project_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    # 1. Save uploaded file to outputs/
    input_path = os.path.join(outputs_dir, f"input_{image.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    try:
        # 2. Run depth estimation (Phase 1)
        # Load image into numpy array
        img_pil = Image.open(input_path).convert("RGB")
        img_np = np.array(img_pil)
        
        # Estimate depth
        relative_depth = estimate_depth(img_np, backbone="depth_anything_v2")
        
        # 3. Run calibration (Phase 2)
        # Calibrate will return metric (if GeoTIFF) or relative DSM
        dsm_array, metadata, mode = calibrate(input_path, relative_depth)
        
        # 4. Save Outputs
        # Texture URL (the saved input image)
        texture_url = f"/outputs/input_{image.filename}"
        
        # DSM URL
        dsm_filename = f"dsm_{image.filename}"
        if mode == "metric":
            dsm_filename += ".tif"
        else:
            dsm_filename += ".png"
            
        dsm_path = os.path.join(outputs_dir, dsm_filename)
        save_output(dsm_path, dsm_array, metadata, mode)
        dsm_url = f"/outputs/{dsm_filename}"
        
        # Heightmap URL (always a PNG for Three.js)
        # If we have metric, we should normalize it for the visualizer to handle easily as 16-bit PNG
        # If relative, it's already normalized 0-1
        d_min, d_max = dsm_array.min(), dsm_array.max()
        if d_max > d_min:
            heightmap_norm = (dsm_array - d_min) / (d_max - d_min)
        else:
            heightmap_norm = np.zeros_like(dsm_array)
            
        heightmap_16bit = (heightmap_norm * 65535).astype(np.uint16)
        heightmap_filename = f"heightmap_{image.filename}.png"
        heightmap_path = os.path.join(outputs_dir, heightmap_filename)
        Image.fromarray(heightmap_16bit).save(heightmap_path)
        heightmap_url = f"/outputs/{heightmap_filename}"
        
        return {
            "mode": mode,
            "dsm_url": dsm_url,
            "heightmap_url": heightmap_url,
            "texture_url": texture_url
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
