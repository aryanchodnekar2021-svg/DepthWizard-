"""
ISPRS Potsdam 2D Semantic Labeling Dataset Adapter.

Handles the Potsdam dataset structure:
- 38 tiles, each 6000x6000 pixels, 5cm/pixel
- CRS: UTM WGS84 (EPSG:32633)
- TOP (True Orthophoto) + DSM (Digital Surface Model) pairs
- RGB, IRRG, RGBIR channel compositions

Dataset provenance:
- Source: ISPRS WG III/4, 2D Semantic Labeling Contest - Potsdam
- URL: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
- Download: https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/
- Size: ~13.3GB total
- License: ISPRS benchmark terms of use

IMPORTANT: Raw dataset files are NOT committed to the repository.
This adapter works with locally downloaded data only.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Potsdam tile naming convention
# Files: top_potsdam_{row}_{col}_{type}.tif
# Types: RGB, IRRG, RGBIR, DSM, label
# Row/col numbering: row 2-7 (top to bottom), col 7-15 (left to right)

# Training tiles (with ground truth labels available)
TRAIN_TILES = [
    "2_10",
    "2_11",
    "2_12",
    "3_10",
    "3_11",
    "3_12",
    "4_10",
    "4_11",
    "4_12",
    "5_10",
    "5_11",
    "5_12",
    "6_7",
    "6_8",
    "6_9",
    "6_10",
    "6_11",
    "6_12",
    "7_7",
    "7_8",
    "7_9",
    "7_10",
    "7_11",
    "7_12",
]

# Test tiles (reference labels held by organizers)
TEST_TILES = [
    "2_13",
    "2_14",
    "3_13",
    "3_14",
    "4_13",
    "4_14",
    "4_15",
    "5_13",
    "5_14",
    "5_15",
    "6_13",
    "6_14",
    "6_15",
    "7_13",
]

# Channel ordering for different compositions
CHANNEL_ORDER = {
    "RGB": ["R", "G", "B"],
    "IRRG": ["IR", "R", "G"],
    "RGBIR": ["R", "G", "B", "IR"],
}


@dataclass
class PotsdamTileInfo:
    """Metadata for a single Potsdam tile."""

    tile_id: str  # e.g., "2_10"
    row: int
    col: int
    split: str  # "train" or "test"

    # File paths (None if not found)
    rgb_path: Optional[str] = None
    irrg_path: Optional[str] = None
    rgbir_path: Optional[str] = None
    dsm_path: Optional[str] = None
    label_path: Optional[str] = None
    tfw_path: Optional[str] = None  # world file

    # Metadata (populated on inspect)
    rgb_shape: Optional[Tuple[int, ...]] = None
    rgb_dtype: Optional[str] = None
    rgb_crs: Optional[str] = None
    rgb_transform: Optional[list] = None
    rgb_bounds: Optional[dict] = None
    rgb_resolution: Optional[dict] = None

    dsm_shape: Optional[Tuple[int, ...]] = None
    dsm_dtype: Optional[str] = None
    dsm_crs: Optional[str] = None
    dsm_transform: Optional[list] = None
    dsm_bounds: Optional[dict] = None
    dsm_resolution: Optional[dict] = None
    dsm_nodata: Optional[float] = None
    dsm_elevation_min: Optional[float] = None
    dsm_elevation_max: Optional[float] = None
    dsm_elevation_median: Optional[float] = None
    dsm_nodata_pct: Optional[float] = None

    # Validation
    rgb_dsm_match: Optional[bool] = None
    crs_match: Optional[bool] = None
    resolution_match: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PotsdamDatasetInfo:
    """Summary of the Potsdam dataset found at a root directory."""

    root: str
    exists: bool
    total_tiles: int = 0
    train_tiles: int = 0
    test_tiles: int = 0
    tiles_with_rgb: int = 0
    tiles_with_dsm: int = 0
    tiles_with_both: int = 0
    tiles: List[PotsdamTileInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _parse_tile_id(filename: str) -> Optional[Tuple[int, int]]:
    """
    Parse tile row/col from Potsdam filename.

    Examples:
        "top_potsdam_2_10_RGB.tif" -> (2, 10)
        "2_10" -> (2, 10)
    """
    name = Path(filename).stem
    # Try pattern: top_potsdam_{row}_{col}_...
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "top" and parts[1] == "potsdam":
        try:
            row = int(parts[2])
            col = int(parts[3])
            return (row, col)
        except (ValueError, IndexError):
            pass
    # Try pattern: {row}_{col}
    parts = name.split("_")
    if len(parts) == 2:
        try:
            row = int(parts[0])
            col = int(parts[1])
            return (row, col)
        except (ValueError, IndexError):
            pass
    return None


def discover_potsdam_tiles(root: str) -> PotsdamDatasetInfo:
    """
    Discover Potsdam dataset tiles at a root directory.

    Expected layout:
        root/
            4_Ortho_RGBIR/  (or 1_TOP/ or similar)
                top_potsdam_2_10_RGBIR.tif
                top_potsdam_2_10_RGB.tif
                ...
            5_DSM/  (or 2_DSM/)
                top_potsdam_2_10_DSM.tif
                ...
            5_Labels_all/
                top_potsdam_2_10_label.tif
                ...

    The adapter is flexible about directory names and searches recursively.
    """
    info = PotsdamDatasetInfo(root=root, exists=os.path.isdir(root))

    if not info.exists:
        info.errors.append(f"Root directory does not exist: {root}")
        return info

    root_path = Path(root)

    # Find all TIFF files recursively
    tiff_files = list(root_path.rglob("*.tif")) + list(root_path.rglob("*.tiff"))

    # Categorize files by type
    rgb_files: Dict[str, str] = {}
    irrg_files: Dict[str, str] = {}
    rgbir_files: Dict[str, str] = {}
    dsm_files: Dict[str, str] = {}
    label_files: Dict[str, str] = {}
    tfw_files: Dict[str, str] = {}

    for f in tiff_files:
        name = f.stem.lower()
        tile = _parse_tile_id(str(f))
        if tile is None:
            continue

        tile_id = f"{tile[0]}_{tile[1]}"

        if "_rgbir" in name:
            rgbir_files[tile_id] = str(f)
        elif "_irrg" in name:
            irrg_files[tile_id] = str(f)
        elif "_rgb" in name:
            rgb_files[tile_id] = str(f)
        elif "_dsm" in name:
            dsm_files[tile_id] = str(f)
        elif "_label" in name:
            label_files[tile_id] = str(f)

    # Find .tfw world files
    for f in root_path.rglob("*.tfw"):
        tile = _parse_tile_id(str(f))
        if tile:
            tile_id = f"{tile[0]}_{tile[1]}"
            tfw_files[tile_id] = str(f)

    # Build tile info objects
    all_tile_ids = (
        set(rgb_files.keys())
        | set(irrg_files.keys())
        | set(rgbir_files.keys())
        | set(dsm_files.keys())
        | set(label_files.keys())
    )

    for tile_id in sorted(all_tile_ids):
        row, col = _parse_tile_id(tile_id)
        if row is None:
            # Try from filename
            parts = tile_id.split("_")
            if len(parts) == 2:
                row, col = int(parts[0]), int(parts[1])
            else:
                continue

        split = "train" if tile_id in TRAIN_TILES else "test"

        tile = PotsdamTileInfo(
            tile_id=tile_id,
            row=row,
            col=col,
            split=split,
            rgb_path=rgb_files.get(tile_id),
            irrg_path=irrg_files.get(tile_id),
            rgbir_path=rgbir_files.get(tile_id),
            dsm_path=dsm_files.get(tile_id),
            label_path=label_files.get(tile_id),
            tfw_path=tfw_files.get(tile_id),
        )

        info.tiles.append(tile)
        info.total_tiles += 1
        if split == "train":
            info.train_tiles += 1
        else:
            info.test_tiles += 1
        if tile.rgb_path or tile.irrg_path or tile.rgbir_path:
            info.tiles_with_rgb += 1
        if tile.dsm_path:
            info.tiles_with_dsm += 1
        if (tile.rgb_path or tile.irrg_path or tile.rgbir_path) and tile.dsm_path:
            info.tiles_with_both += 1

    if info.total_tiles == 0:
        info.warnings.append(
            "No Potsdam tiles found. Expected structure: root/top_potsdam_{row}_{col}_{type}.tif"
        )

    return info


def inspect_tile(tile: PotsdamTileInfo) -> PotsdamTileInfo:
    """
    Inspect a single Potsdam tile and populate metadata.

    Reads CRS, transform, bounds, resolution, dtype, nodata for RGB and DSM.
    """
    try:
        import rasterio
    except ImportError:
        tile.errors.append("rasterio required for tile inspection")
        return tile

    # Inspect RGB
    rgb_path = tile.rgb_path or tile.rgbir_path or tile.irrg_path
    if rgb_path and os.path.isfile(rgb_path):
        try:
            with rasterio.open(rgb_path) as src:
                tile.rgb_shape = (src.height, src.width, src.count)
                tile.rgb_dtype = str(src.dtypes[0]) if src.dtypes else None
                tile.rgb_crs = str(src.crs) if src.crs else None
                tile.rgb_transform = list(src.transform)[:6] if src.transform else None
                tile.rgb_bounds = {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                }
                tile.rgb_resolution = {
                    "x": abs(src.transform.a),
                    "y": abs(src.transform.e),
                }
        except Exception as e:
            tile.errors.append(f"Failed to read RGB: {e}")

    # Inspect DSM
    if tile.dsm_path and os.path.isfile(tile.dsm_path):
        try:
            with rasterio.open(tile.dsm_path) as src:
                tile.dsm_shape = (src.height, src.width, src.count)
                tile.dsm_dtype = str(src.dtypes[0]) if src.dtypes else None
                tile.dsm_crs = str(src.crs) if src.crs else None
                tile.dsm_transform = list(src.transform)[:6] if src.transform else None
                tile.dsm_bounds = {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                }
                tile.dsm_resolution = {
                    "x": abs(src.transform.a),
                    "y": abs(src.transform.e),
                }
                tile.dsm_nodata = src.nodata

                # Read DSM values for statistics
                dsm = src.read(1).astype(np.float64)
                valid = np.isfinite(dsm)
                if tile.dsm_nodata is not None:
                    valid &= dsm != tile.dsm_nodata
                if np.any(valid):
                    tile.dsm_elevation_min = float(np.min(dsm[valid]))
                    tile.dsm_elevation_max = float(np.max(dsm[valid]))
                    tile.dsm_elevation_median = float(np.median(dsm[valid]))
                    tile.dsm_nodata_pct = float(100.0 * np.mean(~valid))
                else:
                    tile.dsm_nodata_pct = 100.0
        except Exception as e:
            tile.errors.append(f"Failed to read DSM: {e}")

    # Validate alignment
    if tile.rgb_crs and tile.dsm_crs:
        tile.crs_match = tile.rgb_crs == tile.dsm_crs
        if not tile.crs_match:
            tile.warnings.append(
                f"CRS mismatch: RGB={tile.rgb_crs}, DSM={tile.dsm_crs}"
            )

    if tile.rgb_resolution and tile.dsm_resolution:
        ratio = (
            tile.dsm_resolution["x"] / tile.rgb_resolution["x"]
            if tile.rgb_resolution["x"] > 0
            else None
        )
        tile.resolution_match = ratio is not None and abs(ratio - 1.0) < 0.01
        if not tile.resolution_match:
            tile.warnings.append(
                f"Resolution mismatch: RGB={tile.rgb_resolution}, DSM={tile.dsm_resolution}"
            )

    if tile.rgb_bounds and tile.dsm_bounds:
        # Check overlap
        overlap_left = max(tile.rgb_bounds["left"], tile.dsm_bounds["left"])
        overlap_right = min(tile.rgb_bounds["right"], tile.dsm_bounds["right"])
        overlap_bottom = max(tile.rgb_bounds["bottom"], tile.dsm_bounds["bottom"])
        overlap_top = min(tile.rgb_bounds["top"], tile.dsm_bounds["top"])

        if overlap_left < overlap_right and overlap_bottom < overlap_top:
            tile.rgb_dsm_match = True
        else:
            tile.rgb_dsm_match = False
            tile.errors.append("RGB and DSM do not overlap spatially")

    return tile


def load_tile_rgb(
    tile: PotsdamTileInfo,
    band_mode: str = "RGB",
    max_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load RGB data from a Potsdam tile.

    Args:
        tile: PotsdamTileInfo with valid path
        band_mode: "RGB", "IRRG", or "RGBIR"
        max_size: Optional maximum dimension for resize

    Returns:
        uint8 RGB array (H, W, 3) or None on failure
    """
    try:
        import rasterio
    except ImportError:
        logger.error("rasterio required for tile loading")
        return None

    # Select path based on band_mode
    if band_mode == "RGBIR" and tile.rgbir_path:
        path = tile.rgbir_path
        bands = [1, 2, 3]  # R, G, B (skip IR)
    elif band_mode == "IRRG" and tile.irrg_path:
        path = tile.irrg_path
        bands = [1, 2, 3]  # IR, R, G -> treat as RGB for display
    elif tile.rgb_path:
        path = tile.rgb_path
        bands = [1, 2, 3]
    elif tile.rgbir_path:
        path = tile.rgbir_path
        bands = [1, 2, 3]
    elif tile.irrg_path:
        path = tile.irrg_path
        bands = [1, 2, 3]
    else:
        logger.warning(f"No RGB image found for tile {tile.tile_id}")
        return None

    try:
        with rasterio.open(path) as src:
            if src.count >= 3:
                rgb = src.read(bands).transpose(1, 2, 0)  # CHW -> HWC
            else:
                # Grayscale fallback
                gray = src.read(1)
                rgb = np.stack([gray, gray, gray], axis=-1)

            # Ensure uint8
            if rgb.dtype != np.uint8:
                if rgb.max() > 255:
                    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

            # Optional resize
            if max_size and max(rgb.shape[:2]) > max_size:
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(rgb)
                ratio = max_size / max(rgb.shape[:2])
                new_size = (int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio))
                pil_img = pil_img.resize(new_size, PILImage.BILINEAR)
                rgb = np.array(pil_img)

            return rgb

    except Exception as e:
        logger.error(f"Failed to load RGB for tile {tile.tile_id}: {e}")
        return None


def load_tile_dsm(
    tile: PotsdamTileInfo,
    max_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load DSM data from a Potsdam tile.

    Returns:
        float64 DSM array (H, W) in meters, or None on failure
    """
    try:
        import rasterio
    except ImportError:
        logger.error("rasterio required for tile loading")
        return None

    if not tile.dsm_path or not os.path.isfile(tile.dsm_path):
        logger.warning(f"No DSM found for tile {tile.tile_id}")
        return None

    try:
        with rasterio.open(tile.dsm_path) as src:
            dsm = src.read(1).astype(np.float64)

            # Handle nodata
            if tile.dsm_nodata is not None:
                dsm[dsm == tile.dsm_nodata] = np.nan

            # Optional resize
            if max_size and max(dsm.shape) > max_size:
                from PIL import Image as PILImage

                # Use nanmean for fill
                fill_val = (
                    float(np.nanmean(dsm[np.isfinite(dsm)]))
                    if np.any(np.isfinite(dsm))
                    else 0.0
                )
                dsm_filled = np.nan_to_num(dsm, nan=fill_val)
                pil_img = PILImage.fromarray(dsm_filled.astype(np.float32))
                ratio = max_size / max(dsm.shape)
                new_size = (int(dsm.shape[1] * ratio), int(dsm.shape[0] * ratio))
                pil_img = pil_img.resize(new_size, PILImage.BILINEAR)
                dsm = np.array(pil_img, dtype=np.float64)

            return dsm

    except Exception as e:
        logger.error(f"Failed to load DSM for tile {tile.tile_id}: {e}")
        return None


def prepare_potsdam_for_benchmark(
    root: str,
    output_dir: str = "data/eval_tiles/potsdam",
    max_tiles: Optional[int] = None,
    split: str = "train",
    max_size: Optional[int] = None,
) -> Dict:
    """
    Prepare Potsdam tiles for the calibration benchmark.

    For each tile with both RGB and DSM:
    1. Load RGB and DSM
    2. Validate alignment
    3. Save as standardized pairs in output_dir
    4. Generate metadata manifest

    Args:
        root: Path to extracted Potsdam dataset
        output_dir: Where to save prepared pairs
        max_tiles: Limit number of tiles (for testing)
        split: "train", "test", or "all"
        max_size: Maximum dimension for resize (None = full resolution)

    Returns:
        Dict with manifest and statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    info = discover_potsdam_tiles(root)
    if not info.exists:
        return {"status": "failed", "error": f"Root not found: {root}"}

    # Filter tiles
    tiles = [
        t
        for t in info.tiles
        if (t.rgb_path or t.rgbir_path or t.irrg_path) and t.dsm_path
    ]
    if split != "all":
        tiles = [t for t in tiles if t.split == split]

    if max_tiles:
        tiles = tiles[:max_tiles]

    if not tiles:
        return {
            "status": "failed",
            "error": "No valid RGB+DSM tile pairs found",
            "discovered": info.total_tiles,
            "with_rgb": info.tiles_with_rgb,
            "with_dsm": info.tiles_with_dsm,
        }

    manifest = {
        "source": "ISPRS Potsdam 2D Semantic Labeling",
        "source_url": "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx",
        "root": root,
        "output_dir": output_dir,
        "crs": "EPSG:32633",
        "resolution_m": 0.05,
        "tile_size_px": 6000,
        "split": split,
        "tiles_prepared": [],
        "tiles_failed": [],
    }

    for i, tile in enumerate(tiles):
        print(f"Preparing tile {i + 1}/{len(tiles)}: {tile.tile_id} ({tile.split})")

        # Inspect tile
        tile = inspect_tile(tile)

        if tile.errors:
            manifest["tiles_failed"].append(
                {
                    "tile_id": tile.tile_id,
                    "errors": tile.errors,
                }
            )
            continue

        # Load data
        rgb = load_tile_rgb(tile, max_size=max_size)
        dsm = load_tile_dsm(tile, max_size=max_size)

        if rgb is None or dsm is None:
            manifest["tiles_failed"].append(
                {
                    "tile_id": tile.tile_id,
                    "errors": ["Failed to load RGB or DSM"],
                }
            )
            continue

        # Handle shape mismatch
        if rgb.shape[:2] != dsm.shape:
            # Resize DSM to match RGB
            from PIL import Image as PILImage

            fill_val = (
                float(np.nanmean(dsm[np.isfinite(dsm)]))
                if np.any(np.isfinite(dsm))
                else 0.0
            )
            dsm_filled = np.nan_to_num(dsm, nan=fill_val)
            pil_img = PILImage.fromarray(dsm_filled.astype(np.float32))
            pil_img = pil_img.resize((rgb.shape[1], rgb.shape[0]), PILImage.BILINEAR)
            dsm = np.array(pil_img, dtype=np.float64)
            logger.info(
                f"Resized DSM from {dsm.shape} to match RGB {rgb.shape[:2]} for tile {tile.tile_id}"
            )

        # Save as npz for fast loading
        out_path = os.path.join(output_dir, f"tile_{tile.tile_id}.npz")
        np.savez_compressed(
            out_path,
            rgb=rgb,
            dsm=dsm,
            tile_id=tile.tile_id,
            row=tile.row,
            col=tile.col,
            split=tile.split,
            crs=tile.dsm_crs or "EPSG:32633",
            resolution_m=0.05,
        )

        manifest["tiles_prepared"].append(
            {
                "tile_id": tile.tile_id,
                "split": tile.split,
                "rgb_shape": list(rgb.shape),
                "dsm_shape": list(dsm.shape),
                "dsm_elevation_min": tile.dsm_elevation_min,
                "dsm_elevation_max": tile.dsm_elevation_max,
                "dsm_elevation_median": tile.dsm_elevation_median,
                "dsm_nodata_pct": tile.dsm_nodata_pct,
                "output_path": out_path,
            }
        )

    # Save manifest
    import json

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\nPrepared {len(manifest['tiles_prepared'])} tiles, {len(manifest['tiles_failed'])} failed"
    )
    print(f"Manifest: {manifest_path}")

    return manifest
