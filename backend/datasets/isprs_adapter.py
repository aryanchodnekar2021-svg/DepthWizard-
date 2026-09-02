"""
ISPRS Potsdam / Vaihingen dataset adapter.

Expected directory structure:
    data/isprs_potsdam/
        <tile_id>_RGB.tif    — RGB orthophoto
        <tile_id>_dsm.tif    — DSM reference (if available)

The adapter gracefully handles missing reference DSMs.
"""

import os
import glob
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class ISPRSAdapter:
    """Adapter for ISPRS Potsdam / Vaihingen datasets."""

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: Path to dataset directory.
                      Defaults to data/isprs_potsdam/ relative to project root.
        """
        if data_dir is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            data_dir = os.path.join(base_dir, "data", "isprs_potsdam")
        self.data_dir = data_dir

    def _discover_tiles(self) -> Dict[str, Dict[str, str]]:
        """
        Discover RGB/DSM tile pairs in the data directory.

        Returns:
            Dict mapping tile_id → {"rgb": path, "dsm": path_or_None}
        """
        if not os.path.isdir(self.data_dir):
            return {}

        tiles = {}

        # Find all RGB files
        for pattern in ["*_RGB.tif", "*_RGB.tiff", "*_rgb.tif", "*_rgb.tiff"]:
            for f in glob.glob(os.path.join(self.data_dir, pattern)):
                basename = os.path.basename(f)
                # Extract tile ID (everything before _RGB)
                for suffix in ["_RGB.tif", "_RGB.tiff", "_rgb.tif", "_rgb.tiff"]:
                    if basename.lower().endswith(suffix.lower()):
                        tile_id = basename[: -len(suffix)]
                        break
                else:
                    continue

                tiles.setdefault(tile_id, {"rgb": None, "dsm": None})
                tiles[tile_id]["rgb"] = f

        # Find matching DSM files
        for pattern in ["*_dsm.tif", "*_dsm.tiff", "*_DSM.tif", "*_DSM.tiff"]:
            for f in glob.glob(os.path.join(self.data_dir, pattern)):
                basename = os.path.basename(f)
                for suffix in ["_dsm.tif", "_dsm.tiff", "_DSM.tif", "_DSM.tiff"]:
                    if basename.lower().endswith(suffix.lower()):
                        tile_id = basename[: -len(suffix)]
                        break
                else:
                    continue

                if tile_id not in tiles:
                    tiles[tile_id] = {"rgb": None, "dsm": None}
                tiles[tile_id]["dsm"] = f

        return tiles

    def list_samples(self) -> List[Dict]:
        """List available ISPRS Potsdam tiles."""
        tiles = self._discover_tiles()
        samples = []

        for tile_id, paths in sorted(tiles.items()):
            if paths["rgb"] is None:
                continue  # Skip tiles without RGB

            samples.append(
                {
                    "id": tile_id,
                    "description": f"ISPRS Potsdam tile {tile_id}",
                    "has_reference_dsm": paths["dsm"] is not None,
                    "rgb_file": os.path.basename(paths["rgb"]),
                    "dsm_file": os.path.basename(paths["dsm"])
                    if paths["dsm"]
                    else None,
                }
            )

        return samples

    def load_sample(self, sample_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load an ISPRS Potsdam RGB tile.

        Returns uint8 RGB array and metadata dict.
        """
        tiles = self._discover_tiles()
        tile = tiles.get(sample_id)

        if tile is None or tile["rgb"] is None:
            raise FileNotFoundError(
                f"ISPRS tile {sample_id} not found in {self.data_dir}"
            )

        import rasterio

        with rasterio.open(tile["rgb"]) as src:
            data = src.read()  # (bands, H, W)
            transform = src.transform
            crs = str(src.crs) if src.crs else None
            bounds = src.bounds

        # Convert bands-first to HWC
        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            # Grayscale → RGB
            band = data[0]
            rgb = np.stack([band, band, band], axis=-1)

        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

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
            "shape": rgb.shape,
        }

        return rgb, metadata

    def get_reference_dsm(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Get the reference DSM for a tile.

        Returns float32 elevation in meters, or None if not available.
        """
        tiles = self._discover_tiles()
        tile = tiles.get(sample_id)

        if tile is None or tile["dsm"] is None:
            return None

        import rasterio

        with rasterio.open(tile["dsm"]) as src:
            return src.read(1).astype(np.float32)
