import numpy as np
from PIL import Image
from transformers import pipeline

# Model mapping
MODELS = {
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "midas": "Intel/dpt-large",
    "zoedepth": "Intel/dpt-large", # Placeholder for ZoeDepth for now
}

# Cache for loaded pipelines to avoid reloading on multiple calls
_pipelines = {}

def get_pipeline(backbone: str):
    if backbone not in MODELS:
        raise ValueError(f"Unknown backbone '{backbone}'. Available options: {list(MODELS.keys())}")
        
    model_id = MODELS[backbone]
    
    if backbone not in _pipelines:
        print(f"Loading model {model_id} for backbone {backbone}...")
        # device="cpu" is default, will use GPU if device=0 is passed and available
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1
            
        _pipelines[backbone] = pipeline(
            task="depth-estimation",
            model=model_id,
            device=device
        )
    return _pipelines[backbone]

def estimate_depth(image: np.ndarray, backbone: str = "depth_anything_v2") -> np.ndarray:
    """
    Estimates relative depth from an RGB image.
    
    Args:
        image: np.ndarray of shape (H, W, 3) and dtype uint8
        backbone: Model to use for depth estimation
        
    Returns:
        np.ndarray of shape (H, W) and dtype float32 representing relative depth.
    """
    # Ensure image is in RGB uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # The pipeline expects a PIL Image
    pil_image = Image.fromarray(image)
    
    pipe = get_pipeline(backbone)
    result = pipe(pil_image)
    
    # The pipeline returns a dictionary with 'depth' containing a PIL Image or tensor
    # depending on the model. For DPT/DepthAnything, it usually returns a PIL Image.
    depth_output = result["depth"]
    
    # Convert back to numpy array and ensure it matches the original shape
    depth_np = np.array(depth_output).astype(np.float32)
    
    # Sometimes the model resizes the output. We should resize it back to original HxW if needed.
    if depth_np.shape != image.shape[:2]:
        depth_pil = Image.fromarray(depth_np).resize((image.shape[1], image.shape[0]), Image.Resampling.BILINEAR)
        depth_np = np.array(depth_pil).astype(np.float32)
        
    return depth_np

if __name__ == "__main__":
    # Quick sanity check
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = estimate_depth(dummy_img, "depth_anything_v2")
    print("Depth shape:", depth.shape)
    print("Depth range:", depth.min(), "-", depth.max())
