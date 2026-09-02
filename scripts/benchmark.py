"""
Benchmark utility for depth estimation.

Measures inference time, memory usage, and output quality metrics
for different input sizes and models.
"""

import os
import sys
import time
import json
import logging
from typing import Optional, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.depth.estimator import estimate_depth, get_model_status, MODELS

logger = logging.getLogger(__name__)


def benchmark_inference(
    image: np.ndarray,
    backbone: str = "depth_anything_v2",
    runs: int = 3,
) -> Dict:
    """
    Benchmark depth inference on a single image.

    Args:
        image: Input image (H, W, 3) uint8
        backbone: Model to use
        runs: Number of inference runs to average

    Returns:
        Dict with timing stats, output shape, and model info
    """
    from backend.depth.estimator import prepare_image

    image = prepare_image(image)
    h, w = image.shape[:2]

    # Warmup run
    _ = estimate_depth(image, backbone)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.time()
        depth = estimate_depth(image, backbone)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)

    stats = {
        "input_shape": [h, w],
        "input_pixels": h * w,
        "output_shape": list(depth.shape),
        "output_dtype": str(depth.dtype),
        "output_range": [float(depth.min()), float(depth.max())],
        "backbone": backbone,
        "model_id": MODELS.get(backbone, "unknown"),
        "runs": runs,
        "time_ms_mean": round(np.mean(times), 1),
        "time_ms_min": round(min(times), 1),
        "time_ms_max": round(max(times), 1),
        "time_ms_std": round(np.std(times), 1),
        "pixels_per_second": round(h * w / (np.mean(times) / 1000), 0),
    }

    stats["model_status"] = get_model_status()

    return stats


def benchmark_sizes(
    backbone: str = "depth_anything_v2",
    sizes: Optional[List[int]] = None,
    runs: int = 2,
) -> List[Dict]:
    """
    Benchmark inference across multiple image sizes.

    Args:
        backbone: Model to use
        sizes: List of square sizes to test (default: [64, 128, 256, 512])
        runs: Number of runs per size

    Returns:
        List of benchmark results
    """
    if sizes is None:
        sizes = [64, 128, 256, 512]

    results = []
    for size in sizes:
        logger.info(f"Benchmarking {size}x{size}...")
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        result = benchmark_inference(img, backbone=backbone, runs=runs)
        results.append(result)

    return results


def save_benchmark(results: List[Dict], output_path: str):
    """Save benchmark results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Running benchmark...")
    results = benchmark_sizes(runs=2)

    for r in results:
        print(
            f"  {r['input_shape'][0]}x{r['input_shape'][1]}: "
            f"{r['time_ms_mean']:.1f}ms "
            f"({r['pixels_per_second']:.0f} px/s)"
        )

    output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "benchmark.json",
    )
    save_benchmark(results, output)
