"""
Depth estimation module — wraps HuggingFace depth models with error handling,
inference timing, tiled processing, and multi-band input support.
"""

import time
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model mapping — pinned model IDs
MODELS = {
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "midas": "Intel/dpt-large",
    "zoedepth": "Intel/dpt-large",  # Placeholder for ZoeDepth
}

# Cache for loaded pipelines
_pipelines = {}

# Module-level stats tracking
_model_stats = {
    "model_id": None,
    "device": "cpu",
    "loaded": False,
    "last_inference_ms": None,
    "total_inferences": 0,
}


def get_model_status() -> dict:
    """
    Return current model status without forcing a load.

    Returns:
        Dict with model_id, device, loaded, last_inference_ms, total_inferences
    """
    return {
        "model_id": _model_stats["model_id"],
        "device": _model_stats["device"],
        "loaded": _model_stats["loaded"],
        "last_inference_ms": _model_stats["last_inference_ms"],
        "total_inferences": _model_stats["total_inferences"],
    }


def get_pipeline(backbone: str):
    """
    Get or create a cached pipeline for the given backbone.

    Raises ValueError for unknown backbone, RuntimeError for load failures.
    """
    if backbone not in MODELS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available options: {list(MODELS.keys())}"
        )

    model_id = MODELS[backbone]

    if backbone not in _pipelines:
        logger.info(f"Loading model {model_id} for backbone {backbone}...")
        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = -1
            device_name = "cpu"

        try:
            from transformers import pipeline as hf_pipeline

            _pipelines[backbone] = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=device,
            )
            _model_stats["model_id"] = model_id
            _model_stats["device"] = device_name
            _model_stats["loaded"] = True
            logger.info(f"Model {model_id} loaded on {device_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model load failed: {e}") from e

    return _pipelines[backbone]


def prepare_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare an input image for depth estimation.

    Handles:
    - Grayscale (2D) → RGB
    - Multi-band (>3 channels) → first 3 channels as RGB
    - Float arrays → uint8

    Args:
        image: Input array of any shape/dtype

    Returns:
        uint8 RGB array of shape (H, W, 3)
    """
    if image.ndim == 2:
        # Grayscale → RGB
        return np.stack([image, image, image], axis=-1).astype(np.uint8)

    if image.ndim == 3:
        bands = image.shape[2]
        if bands == 1:
            # Single-band → RGB
            image = np.concatenate([image, image, image], axis=-1)
        elif bands == 2:
            # 2-band → duplicate third channel
            logger.info("2-band input, duplicating to 3 channels")
            image = np.concatenate([image, image[:, :, :1]], axis=-1)
        elif bands > 3:
            # Multi-band → take first 3 channels
            logger.info(f"Multi-band input ({bands} bands), using first 3 as RGB")
            image = image[:, :, :3]

    if image.dtype != np.uint8:
        # Float → uint8
        if image.max() > 1.0:
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return image


def estimate_depth(
    image: np.ndarray, backbone: str = "depth_anything_v2"
) -> np.ndarray:
    """
    Estimate relative depth from an RGB image.

    Args:
        image: np.ndarray of shape (H, W, 3) and dtype uint8 (or convertible)
        backbone: Model to use for depth estimation

    Returns:
        np.ndarray of shape (H, W) and dtype float32 representing relative depth.

    Raises:
        RuntimeError: If model loading or inference fails
        ValueError: If input is empty or invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    # Prepare image (handle grayscale, multi-band, dtype)
    image = prepare_image(image)

    try:
        pil_image = Image.fromarray(image)
        pipe = get_pipeline(backbone)

        t0 = time.time()
        result = pipe(pil_image)
        inference_ms = (time.time() - t0) * 1000

        # Update stats
        _model_stats["last_inference_ms"] = round(inference_ms, 1)
        _model_stats["total_inferences"] += 1

        # Extract depth output
        depth_output = result["depth"]
        depth_np = np.array(depth_output).astype(np.float32)

        # Resize back to original shape if model changed dimensions
        if depth_np.shape != image.shape[:2]:
            depth_pil = Image.fromarray(depth_np).resize(
                (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
            )
            depth_np = np.array(depth_pil).astype(np.float32)

        logger.info(f"Inference completed in {inference_ms:.1f}ms")
        return depth_np

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Depth estimation failed: {e}") from e


def estimate_depth_tiled(
    image: np.ndarray,
    backbone: str = "depth_anything_v2",
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Estimate depth using tiled processing for large images.

    Splits the image into overlapping tiles, runs each through the model,
    and blends the results with linear ramp weighting.

    Args:
        image: Input image (H, W, 3) uint8
        backbone: Model to use
        tile_size: Size of each tile (pixels)
        overlap: Overlap between tiles (pixels)

    Returns:
        Depth map of shape (H, W) float32
    """
    image = prepare_image(image)
    h, w = image.shape[:2]

    # If image fits in one tile, use standard inference
    if h <= tile_size and w <= tile_size:
        return estimate_depth(image, backbone)

    # Calculate tile positions
    stride = tile_size - overlap
    y_positions = list(range(0, max(1, h - tile_size + 1), stride))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)

    x_positions = list(range(0, max(1, w - tile_size + 1), stride))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    # Accumulator for blending
    depth_accum = np.zeros((h, w), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    # Linear ramp weight function
    def make_weight(size: int) -> np.ndarray:
        """Create a 2D weight map that peaks in center, ramps at edges."""
        ramp = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float64)
        ramp = np.maximum(ramp, 1.0)  # Avoid zero division
        ramp = ramp / ramp.max()
        return np.outer(ramp, ramp)

    total_tiles = len(y_positions) * len(x_positions)
    tile_idx = 0

    for y in y_positions:
        for x in x_positions:
            tile_idx += 1
            tile = image[y : y + tile_size, x : x + tile_size]

            # Run inference on tile
            depth_tile = estimate_depth(tile, backbone)

            # Create weight mask for this tile
            th, tw = depth_tile.shape
            weight = make_weight(th)

            # Accumulate weighted depth
            depth_accum[y : y + th, x : x + tw] += depth_tile * weight
            weight_accum[y : y + th, x : x + tw] += weight

    # Normalize by weights
    depth_accum = np.where(weight_accum > 0, depth_accum / weight_accum, 0)

    return depth_accum.astype(np.float32)


if __name__ == "__main__":
    # Quick sanity check
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = estimate_depth(dummy_img, "depth_anything_v2")
    print("Depth shape:", depth.shape)
    print("Depth range:", depth.min(), "-", depth.max())
    print("Model status:", get_model_status())
