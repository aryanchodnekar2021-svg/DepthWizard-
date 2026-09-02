"""
SRTM dataset adapter.

Reads SRTM 1 arc-second (.hgt or .tif) elevation tiles from data/srtm/.
Each tile covers 1° x 1° of latitude/longitude.
"""

import os
import math
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class SRTMAdapter:
    """Adapter for reading SRTM elevation tiles."""

    def __init__(self, srtm_dir: str = None):
        """
        Args:
            srtm_dir: Path to directory containing SRTM tiles.
                      Defaults to data/srtm/ relative to project root.
        """
        if srtm_dir is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            srtm_dir = os.path.join(base_dir, "data", "srtm")
        self.srtm_dir = srtm_dir

    def list_samples(self) -> List[Dict]:
        """List available SRTM tiles."""
        if not os.path.isdir(self.srtm_dir):
            return []

        samples = []
        for f in sorted(os.listdir(self.srtm_dir)):
            base, ext = os.path.splitext(f)
            if ext.lower() in (".tif", ".tiff", ".hgt") and len(base) == 7:
                if base[0] in "NS" and base[3] in "EW":
                    samples.append(
                        {
                            "id": base,
                            "description": f"SRTM tile {base}",
                            "file": f,
                            "format": ext.lower(),
                        }
                    )
        return samples

    def load_sample(self, sample_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a SRTM tile as a pseudo-RGB image for depth estimation.

        The DEM is normalized to uint8 and replicated across 3 channels.
        """
        # Find the file
        file_path = None
        for ext in [".tif", ".tiff", ".hgt"]:
            candidate = os.path.join(self.srtm_dir, f"{sample_id}{ext}")
            if os.path.isfile(candidate):
                file_path = candidate
                break

        if file_path is None:
            raise FileNotFoundError(
                f"SRTM tile not found: {sample_id} in {self.srtm_dir}"
            )

        # Read with rasterio
        import rasterio

        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = str(src.crs) if src.crs else None
            bounds = src.bounds

        metadata = {
            "tile_id": sample_id,
            "crs": crs,
            "transform": list(transform) if transform else None,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            }
            if bounds
            else None,
            "shape": data.shape,
            "dtype": str(data.dtype),
        }

        # Normalize to 0-255 uint8 for pseudo-RGB
        valid = data[~np.isnan(data)]
        if valid.size > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                normalized = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255)
            else:
                normalized = np.zeros_like(data)
        else:
            normalized = np.zeros_like(data)

        # Handle NaN → 0
        normalized = np.nan_to_num(normalized, nan=0.0).astype(np.uint8)

        # Replicate to 3 channels (RGB)
        rgb = np.stack([normalized, normalized, normalized], axis=-1)

        return rgb, metadata

    def get_reference_dsm(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Get the raw DEM elevation array for a tile.

        Returns float32 meters, with NaN for nodata.
        """
        file_path = None
        for ext in [".tif", ".tiff", ".hgt"]:
            candidate = os.path.join(self.srtm_dir, f"{sample_id}{ext}")
            if os.path.isfile(candidate):
                file_path = candidate
                break

        if file_path is None:
            return None

        import rasterio

        with rasterio.open(file_path) as src:
            return src.read(1).astype(np.float32)
