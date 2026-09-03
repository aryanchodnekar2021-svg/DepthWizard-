"""
Real benchmark runner: python -m eval.run_real_benchmark

Runs calibration experiments on real Potsdam data (or other paired datasets).
Produces results.json, report.md, blocker_report if no data.
"""

import os
import sys
import json
import time
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError as e:
    raise SystemExit(f"yaml required: pip install pyyaml ({e})")


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def describe_hardware():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def run_single_potsdam_experiment(tile_info, method_name, method_fn, config):
    """Run one Potsdam tile + method experiment."""
    from PIL import Image
    from backend.calibration.methods import compute_quality_metrics, apply_method
    from backend.calibration.spatial_cv import check_alignment, AlignmentDiagnostics
    from backend.calibration.frequency import compute_frequency_metrics
    from backend.calibration.uncertainty import estimate_uncertainty_from_residual

    tile_path = tile_info["npz_path"]
    tile_id = tile_info["tile_id"]

    # Load tile data
    try:
        data = np.load(tile_path, allow_pickle=True)
        rgb = data["rgb"]
        dsm = data["dsm"].astype(np.float64)
        crs = str(data.get("crs", "EPSG:32633"))
        resolution_m = float(data.get("resolution_m", 0.05))
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Failed to load tile {tile_id}: {e}",
            "method": method_name,
            "tile_id": tile_id,
        }

    # Run depth estimation
    try:
        from backend.depth.estimator import estimate_depth

        depth = estimate_depth(rgb)
        inference_time_ms = None  # would need timing wrapper
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Depth estimation failed for tile {tile_id}: {e}",
            "method": method_name,
            "tile_id": tile_id,
        }

    # Depth and DSM should be same shape after loading
    if depth.shape != dsm.shape:
        # Resize depth to match DSM
        from PIL import Image as PILImage

        pil_img = Image.fromarray(
            (depth * 255).astype(np.uint8)
            if depth.max() <= 1
            else depth.astype(np.uint8)
        )
        pil_img = pil_img.resize((dsm.shape[1], dsm.shape[0]), PILImage.BILINEAR)
        depth = np.array(pil_img, dtype=np.float32)
        if depth.max() > 1.0:
            depth = depth / 255.0  # normalize to 0-1

    # Valid mask
    valid_mask = np.isfinite(depth) & np.isfinite(dsm) & (dsm != 0) & (dsm != -9999)
    overlap_pct = float(np.mean(valid_mask) * 100) if valid_mask.size > 0 else 0.0

    if overlap_pct < config.get("alignment", {}).get("required_overlap_pct", 50.0):
        return {
            "status": "failed",
            "error": f"Insufficient overlap {overlap_pct:.1f}%",
            "method": method_name,
            "tile_id": tile_id,
        }

    # Alignment diagnostics
    align_diag = AlignmentDiagnostics(
        crs_match=True,  # Potsdam: both EPSG:32633
        resolution_ratio=1.0,  # same grid
        overlap_pct=overlap_pct,
        nodata_fraction=100.0 - overlap_pct,
        vertical_datum="WGS84_ellipsoid",
        warnings=[],
        recommendation="PROCEED" if overlap_pct >= 50 else "FAIL",
    )

    # Run calibration method
    try:
        result = method_fn(depth, dsm, mask=None)
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Method {method_name} failed: {e}",
            "method": method_name,
            "tile_id": tile_id,
        }

    # Apply calibration
    calibrated = apply_method(result, depth)

    # Metrics before/after
    before_metrics = compute_quality_metrics(depth, dsm)
    after_metrics = compute_quality_metrics(calibrated, dsm)

    # Frequency
    freq_metrics = compute_frequency_metrics(
        dsm, depth, sigma=config.get("frequency", {}).get("sigma", 10.0)
    )

    # Residual for uncertainty
    valid = np.isfinite(calibrated) & np.isfinite(dsm) & (dsm != 0) & (dsm != -9999)
    residuals = (calibrated[valid] - dsm[valid]) if np.sum(valid) > 0 else np.array([])
    unc = estimate_uncertainty_from_residual(residuals, dsm, depth)

    # Spatial CV
    spatial = None
    if config.get("spatial_cv", {}).get("enabled", False):
        try:
            from backend.calibration.spatial_cv import (
                create_spatial_blocks,
                create_spatial_cv_folds,
                describe_cv_plan,
                SpatialCVConfig,
            )

            sc = config["spatial_cv"]
            cfg = SpatialCVConfig(
                n_folds=sc.get("n_folds", 5),
                block_size_px=sc.get("block_size_px", 64),
                min_separation_blocks=sc.get("min_separation_blocks", 2),
                seed=sc.get("seed", 42),
                min_valid_fraction=sc.get("min_valid_fraction", 0.1),
            )
            blocks = create_spatial_blocks(depth.shape, cfg)
            for b in blocks:
                b.n_valid = int(
                    np.sum(valid_mask[b.row_start : b.row_end, b.col_start : b.col_end])
                )
                block_area = (b.row_end - b.row_start) * (b.col_end - b.col_start)
                if b.n_valid / max(block_area, 1) < cfg.min_valid_fraction:
                    b.n_valid = 0
            blocks = [b for b in blocks if b.n_valid > 0]
            folds = create_spatial_cv_folds(blocks, cfg)
            spatial = describe_cv_plan(folds)
        except Exception as e:
            spatial = {"error": str(e)}

    # Build result
    res = {
        "tile_id": tile_id,
        "split": tile_info.get("split", "unknown"),
        "dataset": "potsdam",
        "terrain_type": "urban",
        "method": method_name,
        "status": "ok",
        "assumption": config.get("methods", {})
        .get(method_name, {})
        .get("assumption", ""),
        "scale": float(result.scale) if np.isfinite(result.scale) else None,
        "offset": float(result.offset) if np.isfinite(result.offset) else None,
        "residual_rmse": float(result.residual_rmse)
        if np.isfinite(result.residual_rmse)
        else None,
        "residual_mae": float(result.residual_mae)
        if np.isfinite(result.residual_mae)
        else None,
        "inlier_fraction": float(result.inlier_fraction)
        if np.isfinite(result.inlier_fraction)
        else None,
        "pre_correlation": float(result.pre_correlation)
        if np.isfinite(result.pre_correlation)
        else None,
        "post_correlation": float(result.post_correlation)
        if np.isfinite(result.post_correlation)
        else None,
        "r_squared": float(result.r_squared) if np.isfinite(result.r_squared) else None,
        "n_samples": int(result.n_samples),
        "metadata": result.metadata,
        "alignment": {
            "crs_match": align_diag.crs_match,
            "resolution_ratio": align_diag.resolution_ratio,
            "overlap_pct": align_diag.overlap_pct,
            "nodata_fraction": align_diag.nodata_fraction,
            "vertical_datum": align_diag.vertical_datum,
            "warnings": align_diag.warnings,
            "recommendation": align_diag.recommendation,
        },
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "frequency": freq_metrics,
        "uncertainty": {
            "mean_uncertainty": float(unc.mean_uncertainty)
            if np.isfinite(unc.mean_uncertainty)
            else None,
            "residual_std": float(unc.residual_std)
            if np.isfinite(unc.residual_std)
            else None,
            "dem_std": float(unc.dem_std) if np.isfinite(unc.dem_std) else None,
            "depth_variability": float(unc.depth_variability)
            if np.isfinite(unc.depth_variability)
            else None,
            "method": unc.method,
        },
        "spatial_cv": spatial,
        "reference_evaluation": {
            "notes": "Potsdam DSM is the reference; calibration uses same DSM (NOT independent). Metrics show depth→DSM fit quality."
        },
        "input_resolution": list(rgb.shape),
        "depth_resolution": list(depth.shape),
    }

    # Generate diagnostics
    try:
        from eval.visuals import generate_diagnostics

        d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
        norm = (
            (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
        )
        out_dir = config.get("visualizations", {}).get(
            "output_dir", "outputs/real_benchmark"
        )
        prefix = f"potsdam_{tile_id}_{method_name}"
        paths = generate_diagnostics(
            rgb, depth, norm, dsm, calibrated, dsm, out_dir, prefix
        )
        res["diagnostic_figures"] = paths
    except Exception as e:
        res["diagnostic_figures_error"] = str(e)

    return res


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run real benchmark on Potsdam data")
    parser.add_argument(
        "--config", default="configs/real_benchmark.yaml", help="Config path"
    )
    parser.add_argument("--output", default="outputs/real_benchmark", help="Output dir")
    parser.add_argument("--methods", nargs="*", help="Subset of methods")
    parser.add_argument(
        "--tiles", nargs="*", help="Subset of tile IDs (e.g., 2_10 2_11)"
    )
    parser.add_argument("--max-tiles", type=int, help="Max tiles to process")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")
    config = load_config(config_path)
    config["git_commit"] = get_git_sha()
    config["timestamp"] = datetime.now(timezone.utc).isoformat()
    config["hardware"] = describe_hardware()
    config["model"] = config.get("model", "depth-anything/Depth-Anything-V2-Small-hf")

    # Discover prepared tiles
    data_dir = config.get("data_dir", "data/potsdam/prepared")
    if not os.path.isdir(data_dir):
        blocker = {
            "status": "blocked",
            "reason": f"Prepared data directory not found: {data_dir}",
            "timestamp": config["timestamp"],
            "git_commit": config["git_commit"],
            "model": config["model"],
            "hardware": config["hardware"],
            "config_path": config_path,
            "required_files": [
                f"Prepared Potsdam tiles in {data_dir}/",
                "Run: python -c \"from backend.geo.potsdam_adapter import prepare_potsdam_for_benchmark; prepare_potsdam_for_benchmark('data/potsdam/raw', 'data/potsdam/prepared')\"",
            ],
            "download_instructions": "See data/potsdam/README.md for download and preparation steps",
        }
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "blocker_report.json"), "w") as f:
            json.dump(blocker, f, indent=2)
        with open(os.path.join(out_dir, "blocker_report.md"), "w") as f:
            f.write("# Real Benchmark — Data Blocker\n\n")
            f.write(f"**Status:** {blocker['status']}\n\n")
            f.write(f"**Reason:** {blocker['reason']}\n\n")
            f.write("## Required\n")
            for rf in blocker["required_files"]:
                f.write(f"- {rf}\n")
            f.write(f"\n## Download\n{blocker['download_instructions']}\n")
        print(f"BLOCKED: {blocker['reason']}")
        return

    # Find tile files
    tile_files = sorted(Path(data_dir).glob("tile_*.npz"))
    if not tile_files:
        print(f"No tile files found in {data_dir}")
        return

    # Build tile info list
    tiles = []
    for tf in tile_files:
        try:
            data = np.load(str(tf), allow_pickle=True)
            tile_id = str(data.get("tile_id", tf.stem.replace("tile_", "")))
            split = str(data.get("split", "unknown"))
            tiles.append(
                {
                    "npz_path": str(tf),
                    "tile_id": tile_id,
                    "split": split,
                }
            )
        except Exception as e:
            print(f"Warning: failed to load {tf}: {e}")

    # Filter
    if args.tiles:
        tiles = [t for t in tiles if t["tile_id"] in args.tiles]
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]

    if not tiles:
        print("No valid tiles found")
        return

    print(f"Found {len(tiles)} tiles: {[t['tile_id'] for t in tiles]}")

    # Methods
    from backend.calibration.methods import get_available_methods

    all_methods = get_available_methods()
    enabled_methods = [
        k for k, v in config.get("methods", {}).items() if v.get("enabled", True)
    ]
    if args.methods:
        enabled_methods = [m for m in args.methods if m in all_methods]

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for tile in tiles:
        for mname in enabled_methods:
            mfn = all_methods[mname]
            print(f"Running tile={tile['tile_id']} method={mname} ...")
            r = run_single_potsdam_experiment(tile, mname, mfn, config)
            r["git_commit"] = config["git_commit"]
            r["model"] = config["model"]
            r["hardware"] = config["hardware"]
            r["timestamp"] = config["timestamp"]
            results.append(r)
            status = r.get("status", "unknown")
            print(
                f"  -> {status} scale={r.get('scale')} rmse={r.get('residual_rmse')} pre_corr={r.get('pre_correlation')} post_corr={r.get('post_correlation')}"
            )

    # Save JSON
    out_json = os.path.join(out_dir, "results.json")
    payload = {
        "timestamp": config["timestamp"],
        "git_commit": config["git_commit"],
        "model": config["model"],
        "hardware": config["hardware"],
        "config_path": config_path,
        "dataset": "ISPRS Potsdam",
        "dataset_provenance": "ISPRS WG III/4, 2D Semantic Labeling Contest",
        "dataset_note": "Potsdam DSM is used as BOTH calibration reference AND evaluation reference. Metrics show depth→DSM fit quality, NOT independent accuracy.",
        "tiles_evaluated": len([r for r in results if r.get("status") == "ok"]),
        "tiles_failed": len([r for r in results if r.get("status") == "failed"]),
        "results": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    # Save human-readable report
    out_md = os.path.join(out_dir, "report.md")
    with open(out_md, "w") as f:
        f.write("# Real Benchmark — ISPRS Potsdam\n\n")
        f.write(f"**Timestamp:** {config['timestamp']}\n\n")
        f.write(f"**Git SHA:** {config['git_commit']}\n\n")
        f.write(f"**Model:** {config['model']}\n\n")
        f.write(f"**Dataset:** ISPRS Potsdam (5cm, UTM WGS84)\n\n")
        f.write(
            "**IMPORTANT:** Potsdam DSM is used as BOTH calibration reference AND evaluation reference.\n"
        )
        f.write("Metrics show depth→DSM fit quality, NOT independent accuracy.\n\n")
        f.write(
            f"**Tiles:** {len(tiles)}  **Methods:** {', '.join(enabled_methods)}\n\n"
        )

        # Summary table
        f.write(
            "| Tile | Split | Method | Status | Scale | Offset | RMSE | MAE | R² | PreCorr | PostCorr | Samples |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r.get('tile_id')} | {r.get('split')} | {r.get('method')} | {r.get('status')} | {r.get('scale')} | {r.get('offset')} | {r.get('residual_rmse')} | {r.get('residual_mae')} | {r.get('r_squared')} | {r.get('pre_correlation')} | {r.get('post_correlation')} | {r.get('n_samples')} |\n"
            )

        # Details
        f.write("\n## Details per experiment\n")
        for r in results:
            f.write(
                f"\n### Tile {r.get('tile_id')} — {r.get('method')} — {r.get('status')}\n"
            )
            f.write(f"- Assumption: {r.get('assumption')}\n")
            f.write(
                f"- Before metrics: {json.dumps(r.get('before_metrics'), indent=2)}\n"
            )
            f.write(
                f"- After metrics: {json.dumps(r.get('after_metrics'), indent=2)}\n"
            )
            f.write(f"- Frequency: {json.dumps(r.get('frequency'), indent=2)}\n")
            f.write(f"- Uncertainty: {json.dumps(r.get('uncertainty'), indent=2)}\n")
            f.write(f"- Spatial CV: {json.dumps(r.get('spatial_cv'), indent=2)}\n")
            if r.get("diagnostic_figures"):
                f.write(f"- Figures: {', '.join(r.get('diagnostic_figures'))}\n")

    print(f"\nSaved {out_json} and {out_md}")


if __name__ == "__main__":
    main()
