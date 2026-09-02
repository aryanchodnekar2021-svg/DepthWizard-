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

from backend.depth.estimator import (
    estimate_depth,
    get_model_status,
    estimate_depth_tiled,
)
from backend.calibration.pipeline import calibrate, save_output
from backend.geo.raster_inspect import inspect_raster
from backend.analysis.slope import compute_slope
from backend.analysis.classification import classify_terrain, get_category_legend
from backend.analysis.comparison import compute_error_map

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


@app.get("/model/status")
def model_status():
    """Return model health, device, and inference statistics."""
    status = get_model_status()
    return {
        "status": "ok",
        "model": status,
    }


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

        # Heightmap (always 16-bit PNG for Three.js visualization)
        d_min, d_max = float(dsm_array.min()), float(dsm_array.max())
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
            "elevation_min": d_min,
            "elevation_max": d_max,
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

        # Add inference timing from model stats
        model_stats = get_model_status()
        response["inference_ms"] = model_stats.get("last_inference_ms")

        return response

    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")
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


@app.post("/slope")
async def compute_slope_endpoint(image: UploadFile = File(...)):
    """
    Compute slope from an uploaded image's depth map.

    Runs depth estimation, computes slope in degrees, and returns
    a 16-bit slope raster plus metadata.
    """
    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    _, ext = os.path.splitext(image.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {sorted(allowed_extensions)}",
        )

    input_path = os.path.join(outputs_dir, f"slope_input_{image.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        img_pil = Image.open(input_path).convert("RGB")
        img_np = np.array(img_pil)

        # Estimate depth (relative, 0-1 range)
        relative_depth = estimate_depth(img_np, backbone="depth_anything_v2")

        # Compute slope — for relative depth, cell sizes are 1.0 (shape-only)
        slope_deg = compute_slope(relative_depth, cell_size_x=1.0, cell_size_y=1.0)

        # Save slope as 16-bit PNG: map 0-90 degrees → 0-65535
        slope_min = 0.0
        slope_max = 90.0
        slope_norm = np.clip(slope_deg / slope_max, 0.0, 1.0)
        slope_16bit = (slope_norm * 65535).astype(np.uint16)

        slope_filename = f"slope_{image.filename}"
        if not slope_filename.lower().endswith(".png"):
            slope_filename += ".png"
        slope_path = os.path.join(outputs_dir, slope_filename)
        Image.fromarray(slope_16bit).save(slope_path)

        # Classification
        classification = classify_terrain(slope_deg)
        class_filename = f"classification_{image.filename}"
        if not class_filename.lower().endswith(".png"):
            class_filename += ".png"
        class_path = os.path.join(outputs_dir, class_filename)
        Image.fromarray(classification).save(class_path)

        return {
            "status": "ok",
            "analysis": "slope",
            "units": "degrees",
            "slope_min": slope_min,
            "slope_max": slope_max,
            "slope_url": f"/outputs/{slope_filename}",
            "classification_url": f"/outputs/{class_filename}",
            "legend": get_category_legend(),
            "warnings": [
                "Slope computed from relative depth (shape only). "
                "For metric slope, process a georeferenced input with calibration."
            ],
        }
    except Exception as e:
        logger.error(f"Slope computation failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_terrain_endpoint(image: UploadFile = File(...)):
    """
    Classify terrain from an uploaded image.

    Computes depth → slope → terrain categories.
    """
    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    _, ext = os.path.splitext(image.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {sorted(allowed_extensions)}",
        )

    input_path = os.path.join(outputs_dir, f"classify_input_{image.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        img_pil = Image.open(input_path).convert("RGB")
        img_np = np.array(img_pil)

        relative_depth = estimate_depth(img_np, backbone="depth_anything_v2")
        slope_deg = compute_slope(relative_depth)
        classification = classify_terrain(slope_deg)

        class_filename = f"classification_{image.filename}"
        if not class_filename.lower().endswith(".png"):
            class_filename += ".png"
        class_path = os.path.join(outputs_dir, class_filename)
        Image.fromarray(classification).save(class_path)

        # Count pixels per category
        from collections import Counter

        counts = Counter(classification.flatten().tolist())

        return {
            "status": "ok",
            "classification_url": f"/outputs/{class_filename}",
            "legend": get_category_legend(),
            "pixel_counts": {str(k): int(v) for k, v in counts.items()},
            "total_pixels": int(classification.size),
        }
    except Exception as e:
        logger.error(f"Classification failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_endpoint(
    predicted_path: str,
    reference_path: str,
):
    """
    Compare a predicted DSM against a reference DSM.

    Both files must exist in the outputs/ directory.
    Returns accuracy metrics and error map.
    """
    pred_full = os.path.join(outputs_dir, os.path.basename(predicted_path))
    ref_full = os.path.join(outputs_dir, os.path.basename(reference_path))

    if not os.path.exists(pred_full):
        raise HTTPException(
            status_code=404, detail=f"Predicted file not found: {predicted_path}"
        )
    if not os.path.exists(ref_full):
        raise HTTPException(
            status_code=404, detail=f"Reference file not found: {reference_path}"
        )

    try:
        import rasterio

        # Load predicted DSM
        with rasterio.open(pred_full) as src:
            pred_array = src.read(1).astype(np.float64)
            pred_nodata = src.nodata
            pred_transform = src.transform
            pred_crs = str(src.crs) if src.crs else None

        # Load reference DSM
        with rasterio.open(ref_full) as src:
            ref_array = src.read(1).astype(np.float64)
            ref_nodata = src.nodata
            ref_transform = src.transform
            ref_crs = str(src.crs) if src.crs else None

        # Basic alignment check
        alignment = {
            "predicted_shape": list(pred_array.shape),
            "reference_shape": list(ref_array.shape),
            "predicted_crs": pred_crs,
            "reference_crs": ref_crs,
            "shape_match": pred_array.shape == ref_array.shape,
            "crs_match": pred_crs == ref_crs,
        }

        if pred_array.shape != ref_array.shape:
            return {
                "status": "error",
                "error": (
                    f"Shape mismatch: predicted {pred_array.shape} vs "
                    f"reference {ref_array.shape}. Resample to matching grid first."
                ),
                "alignment": alignment,
            }

        # Compute error map and metrics
        result = compute_error_map(pred_array, ref_array)

        # Save error map as raster
        error_map = result.pop("error_map")
        abs_error_map = result.pop("abs_error_map")

        error_filename = "error_map.tif"
        error_path = os.path.join(outputs_dir, error_filename)

        # Write error map preserving geospatial metadata from predicted
        profile = {
            "driver": "GTiff",
            "dtype": "float64",
            "width": error_map.shape[1],
            "height": error_map.shape[0],
            "count": 1,
            "crs": None,
            "transform": None,
        }

        try:
            with rasterio.open(pred_full) as src:
                profile.update(crs=src.crs, transform=src.transform)
        except Exception:
            pass

        with rasterio.open(error_path, "w", **profile) as dst:
            dst.write(error_map, 1)

        return {
            "status": "ok",
            "alignment": alignment,
            "metrics": {k: v for k, v in result.items()},
            "error_map_url": f"/outputs/{error_filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
