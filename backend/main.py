"""
DepthWizard API — FastAPI backend for depth estimation and reconstruction.

Exposes structured metadata about every reconstruction result,
including whether metric calibration succeeded or failed.
"""

import os
import shutil
import logging
import traceback
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.depth.estimator import estimate_depth
from backend.calibration.pipeline import calibrate, save_output
from backend.geo.raster_inspect import inspect_raster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "depthwizard-api"}


@app.get("/inspect")
async def inspect_upload(image_path: str):
    """
    Inspect a raster file and return metadata without processing.

    Useful for the frontend to determine if an upload is georeferenced
    before running the full pipeline.
    """
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        metadata = inspect_raster(image_path)
        return {
            "status": "ok",
            "is_georeferenced": metadata.is_georeferenced,
            "width": metadata.width,
            "height": metadata.height,
            "band_count": metadata.band_count,
            "dtype": metadata.dtype,
            "crs": metadata.crs,
            "epsg": metadata.epsg,
            "bounds": metadata.bounds,
            "resolution": metadata.resolution,
            "nodata": metadata.nodata,
            "geographic_extent": metadata.geographic_extent,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    """
    Process an uploaded image through the depth estimation pipeline.

    Returns structured metadata including:
    - mode: "relative" or "absolute"
    - Whether calibration was applied
    - Calibration parameters (if applied)
    - DSM and heightmap URLs
    """
    # Validate file extension
    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    _, ext = os.path.splitext(image.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. "
            f"Supported: {sorted(allowed_extensions)}",
        )

    # Save uploaded file
    input_path = os.path.join(outputs_dir, f"input_{image.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # Step 1: Inspect input
        raster_meta = inspect_raster(input_path)
        logger.info(
            f"Input: {image.filename}, "
            f"georeferenced={raster_meta.is_georeferenced}, "
            f"CRS={raster_meta.crs}"
        )

        # Step 2: Load image for depth estimation
        img_pil = Image.open(input_path).convert("RGB")
        img_np = np.array(img_pil)

        # Step 3: Estimate depth
        relative_depth = estimate_depth(img_np, backbone="depth_anything_v2")

        # Step 4: Calibrate (handles both relative and absolute paths)
        srtm_dir = os.path.join(base_dir, "data", "srtm")
        dsm_array, metadata, mode = calibrate(
            input_path, relative_depth, srtm_dir=srtm_dir
        )

        # Step 5: Save outputs
        # Texture URL (the saved input image)
        texture_url = f"/outputs/input_{image.filename}"

        # DSM output
        if mode == "metric":
            dsm_filename = f"dsm_{image.filename}"
            if not dsm_filename.lower().endswith((".tif", ".tiff")):
                dsm_filename += ".tif"
        else:
            dsm_filename = f"dsm_{image.filename}"
            if not dsm_filename.lower().endswith(".png"):
                dsm_filename += ".png"

        dsm_path = os.path.join(outputs_dir, dsm_filename)
        save_output(dsm_path, dsm_array, metadata, mode)
        dsm_url = f"/outputs/{dsm_filename}"

        # Heightmap (always PNG for Three.js visualization)
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

        # Build structured response
        response = {
            "status": "ok",
            "mode": mode,
            "units": (
                "meters above sea level"
                if mode == "metric"
                else "relative (normalized 0-1, no metric meaning)"
            ),
            "is_georeferenced": raster_meta.is_georeferenced,
            "input": {
                "width": raster_meta.width,
                "height": raster_meta.height,
                "band_count": raster_meta.band_count,
                "dtype": raster_meta.dtype,
                "crs": raster_meta.crs,
                "epsg": raster_meta.epsg,
                "bounds": raster_meta.bounds,
                "resolution": raster_meta.resolution,
            },
            "calibration": (
                metadata.get(
                    "calibration",
                    {
                        "applied": False,
                        "source": "none",
                        "reason": (
                            "Input is not georeferenced"
                            if not raster_meta.is_georeferenced
                            else "DEM data not available"
                        ),
                    },
                )
                if metadata
                else {
                    "applied": False,
                    "source": "none",
                    "reason": "Operating in relative mode",
                }
            ),
            "dsm_url": dsm_url,
            "heightmap_url": heightmap_url,
            "texture_url": texture_url,
            "warnings": [],
        }

        # Add warnings for important conditions
        if mode == "relative" and raster_meta.is_georeferenced:
            response["warnings"].append(
                "Input is georeferenced but metric calibration failed. "
                "Output is relative, not metric."
            )

        return response

    except Exception as e:
        logger.error(f"Processing failed: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


@app.get("/srtm/status")
async def srtm_status():
    """Check SRTM data availability."""
    srtm_dir = os.path.join(base_dir, "data", "srtm")
    if not os.path.isdir(srtm_dir):
        return {
            "status": "unavailable",
            "message": f"SRTM directory not found: {srtm_dir}",
            "tiles": [],
        }

    from backend.geo.srtm_provider import SRTMProvider

    provider = SRTMProvider(srtm_dir=srtm_dir)
    tiles = provider.get_available_tiles()

    return {
        "status": "available" if tiles else "empty",
        "srtm_dir": srtm_dir,
        "tiles": tiles,
        "tile_count": len(tiles),
    }
