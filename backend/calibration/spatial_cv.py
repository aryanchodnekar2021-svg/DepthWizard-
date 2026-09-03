"""
Spatial cross-validation and alignment diagnostics.

Prevents spatial leakage by block holdout; random pixel split would mix neighbors.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

import numpy as np


@dataclass
class AlignmentDiagnostics:
    crs_match: bool
    resolution_ratio: float
    overlap_pct: float
    nodata_fraction: float
    vertical_datum: str
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class SpatialBlock:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    block_id: int
    n_valid: int
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


@dataclass
class SpatialCVResult:
    fold_id: int
    train_blocks: List[SpatialBlock]
    test_blocks: List[SpatialBlock]
    n_train_pixels: int
    n_test_pixels: int
    spatial_separation_m: float


@dataclass
class SpatialCVConfig:
    n_folds: int = 5
    block_size_px: int = 64
    min_separation_blocks: int = 2
    seed: int = 42
    min_valid_fraction: float = 0.1


def _parse_res(transform) -> Tuple[float, float]:
    if transform is None:
        return 1.0, 1.0
    try:
        # rasterio Affine
        return float(abs(transform.a)), float(abs(transform.e))
    except Exception:
        try:
            return float(abs(transform[0])), float(abs(transform[4]))
        except Exception:
            return 1.0, 1.0


def _crs_is_geographic(crs_str: Optional[str]) -> bool:
    if not crs_str:
        return True
    s = str(crs_str).lower()
    return "4326" in s or "geographic" in s or "wgs84" in s


def check_alignment(
    depth_shape, depth_transform, depth_crs, dem_shape, dem_transform, dem_crs
) -> AlignmentDiagnostics:
    warnings: List[str] = []
    # CRS match
    c1 = str(depth_crs) if depth_crs else ""
    c2 = str(dem_crs) if dem_crs else ""
    crs_match = (c1 == c2) if c1 and c2 else False
    if not crs_match:
        warnings.append(f"CRS mismatch: {c1} vs {c2}")
    # resolution ratio
    dx1, dy1 = _parse_res(depth_transform)
    dx2, dy2 = _parse_res(dem_transform)
    # DEM is usually coarser, so ratio >1
    res1 = (dx1 + dy1) / 2 if (dx1 and dy1) else 1.0
    res2 = (dx2 + dy2) / 2 if (dx2 and dy2) else 1.0
    ratio = float(res2 / res1) if res1 != 0 else float("nan")
    if ratio > 5:
        warnings.append(f"DEM much coarser than depth: ratio {ratio:.1f}")
    elif ratio < 0.2:
        warnings.append(f"DEM much finer than depth: ratio {ratio:.1f}")
    # overlap: approximate by shape ratio if transforms missing
    # If same shape, assume 100% overlap
    if depth_shape == dem_shape:
        overlap = 100.0
    else:
        # overlap approx = min area / max area
        a1 = depth_shape[0] * depth_shape[1] if depth_shape else 0
        a2 = dem_shape[0] * dem_shape[1] if dem_shape else 0
        overlap = (min(a1, a2) / max(a1, a2) * 100.0) if max(a1, a2) > 0 else 0.0
        if overlap < 100:
            warnings.append(f"Shape mismatch implies partial overlap {overlap:.1f}%")
    nodata_fraction = float("nan")  # caller should compute from actual arrays
    vertical_datum = "WGS84/EGM96" if _crs_is_geographic(c2) else "unknown"
    if overlap < 50:
        rec = "FAIL: overlap insufficient for calibration"
    elif not crs_match:
        rec = "WARN: reprojection required; verify alignment"
    elif ratio > 10:
        rec = "WARN: DEM too coarse; high-freq fusion may be limited"
    else:
        rec = "Proceed with calibration"
    return AlignmentDiagnostics(
        crs_match, ratio, overlap, nodata_fraction, vertical_datum, warnings, rec
    )


def compute_spatial_separation(
    block_a: SpatialBlock,
    block_b: SpatialBlock,
    pixel_size_x: float,
    pixel_size_y: float,
    is_geographic: bool = True,
) -> float:
    # centroid in pixel coords
    ca_r = (block_a.row_start + block_a.row_end) / 2.0
    ca_c = (block_a.col_start + block_a.col_end) / 2.0
    cb_r = (block_b.row_start + block_b.row_end) / 2.0
    cb_c = (block_b.col_start + block_b.col_end) / 2.0
    dr = (cb_r - ca_r) * pixel_size_y
    dc = (cb_c - ca_c) * pixel_size_x
    if is_geographic:
        # approximate meters: 1 deg lat ~111320m, lon ~111320*cos(lat)
        # use center lat if available
        lat = block_a.centroid_lat if block_a.centroid_lat is not None else 0.0
        lat_rad = math.radians(lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * max(0.1, math.cos(lat_rad))
        # dr,dc are in degrees
        dy_m = dr * m_per_deg_lat
        dx_m = dc * m_per_deg_lon
        return float(math.hypot(dx_m, dy_m))
    else:
        return float(math.hypot(dc, dr))


def create_spatial_blocks(
    shape: Tuple[int, int], config: SpatialCVConfig
) -> List[SpatialBlock]:
    h, w = shape
    if h <= 0 or w <= 0:
        return []
    bs = max(1, config.block_size_px)
    blocks: List[SpatialBlock] = []
    bid = 0
    for r in range(0, h, bs):
        for c in range(0, w, bs):
            r1 = min(r + bs, h)
            c1 = min(c + bs, w)
            n_pix = (r1 - r) * (c1 - c)
            # n_valid not known here (no array), set to n_pix, filtered later by caller if needed
            # Use min_valid_fraction to filter tiny edge blocks? Keep all, filter externally
            if n_pix / (bs * bs) < config.min_valid_fraction and (h > bs or w > bs):
                # still keep but could be filtered
                pass
            blocks.append(SpatialBlock(r, r1, c, c1, bid, n_pix, None, None))
            bid += 1
    return blocks


def create_spatial_cv_folds(
    blocks: List[SpatialBlock], config: SpatialCVConfig
) -> List[SpatialCVResult]:
    if not blocks:
        return []
    n = len(blocks)
    n_folds = min(config.n_folds, n)
    if n_folds < 2:
        return []
    # assign blocks to folds round-robin after shuffling with seed
    rng = np.random.RandomState(config.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds: List[List[SpatialBlock]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(indices):
        folds[i % n_folds].append(blocks[idx])
    results: List[SpatialCVResult] = []
    for fid in range(n_folds):
        test = folds[fid]
        train = [b for fi, fl in enumerate(folds) if fi != fid for b in fl]

        # enforce min separation: remove train blocks within min_separation_blocks of any test block
        # block distance in grid units: compare block indices proximity via row/col
        # compute block grid positions
        def block_grid(b: SpatialBlock):
            return (
                b.row_start // config.block_size_px,
                b.col_start // config.block_size_px,
            )

        test_grids = [block_grid(b) for b in test]
        filtered_train = []
        for b in train:
            gr, gc = block_grid(b)
            min_dist = (
                min(max(abs(gr - tr), abs(gc - tc)) for tr, tc in test_grids)
                if test_grids
                else 999
            )
            if min_dist >= config.min_separation_blocks:
                filtered_train.append(b)
            # else drop to enforce separation
        n_train = sum(b.n_valid for b in filtered_train)
        n_test = sum(b.n_valid for b in test)
        # separation distance approx via first train/test pair or nan if no train left
        sep = float("nan")
        if filtered_train and test:
            # minimal separation among kept
            sep = min(
                compute_spatial_separation(tb, te, 1.0, 1.0, True)
                for tb in filtered_train
                for te in test
            )
        results.append(SpatialCVResult(fid, filtered_train, test, n_train, n_test, sep))
    return results


def describe_cv_plan(cv_results: List[SpatialCVResult]) -> dict:
    if not cv_results:
        return {"n_folds": 0, "notes": "no folds"}
    train_counts = [r.n_train_pixels for r in cv_results]
    test_counts = [r.n_test_pixels for r in cv_results]
    seps = [
        r.spatial_separation_m
        for r in cv_results
        if np.isfinite(r.spatial_separation_m)
    ]
    return {
        "n_folds": len(cv_results),
        "avg_train_pixels": float(np.mean(train_counts)) if train_counts else 0,
        "avg_test_pixels": float(np.mean(test_counts)) if test_counts else 0,
        "min_train": int(np.min(train_counts)) if train_counts else 0,
        "max_train": int(np.max(train_counts)) if train_counts else 0,
        "min_test": int(np.min(test_counts)) if test_counts else 0,
        "max_test": int(np.max(test_counts)) if test_counts else 0,
        "min_spatial_separation_m": float(np.min(seps)) if seps else float("nan"),
        "balance_ratio": float(np.mean(train_counts) / np.mean(test_counts))
        if test_counts and np.mean(test_counts) > 0
        else float("nan"),
    }
