"""
Diagnostic visualizations — 10 plots per real experiment.
Outputs outside Git (outputs/calibration_rnd/).
"""

import os
from typing import Optional

import numpy as np


def generate_diagnostics(
    rgb: Optional[np.ndarray],
    raw_depth: np.ndarray,
    normalized_surface: np.ndarray,
    coarse_dem: np.ndarray,
    calibrated_dsm: np.ndarray,
    reference_dsm: Optional[np.ndarray],
    out_dir: str,
    prefix: str = "exp",
) -> list:
    """Generate 10 diagnostic figures. Returns list of saved paths. Uses matplotlib if available, else skips."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save_fig(arr, title, fname, cmap="viridis", vmin=None, vmax=None):
        try:
            plt.figure(figsize=(6, 5))
            if arr is not None and arr.ndim == 3:
                plt.imshow(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr)
            else:
                im = (
                    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    if arr is not None
                    else None
                )
                if im is not None:
                    plt.colorbar(im, shrink=0.7)
            plt.title(title)
            plt.axis("off")
            p = os.path.join(out_dir, f"{prefix}_{fname}.png")
            plt.savefig(p, dpi=120, bbox_inches="tight")
            plt.close()
            paths.append(p)
        except Exception:
            try:
                plt.close()
            except Exception:
                pass

    # 1 RGB
    if rgb is not None:
        save_fig(rgb, "1 RGB image", "01_rgb", cmap=None)
    else:
        save_fig(np.zeros((64, 64, 3), dtype=np.uint8), "1 RGB (missing)", "01_rgb")

    # 2 raw relative depth
    save_fig(raw_depth, "2 raw relative depth", "02_raw_depth")

    # 3 normalized relative surface
    save_fig(normalized_surface, "3 normalized relative surface", "03_normalized")

    # 4 coarse DEM
    save_fig(coarse_dem, "4 coarse DEM", "04_coarse_dem", cmap="terrain")

    # 5 calibrated DSM
    save_fig(calibrated_dsm, "5 calibrated DSM", "05_calibrated", cmap="terrain")

    # 6 reference DSM or placeholder
    if reference_dsm is not None:
        save_fig(reference_dsm, "6 reference DSM", "06_reference", cmap="terrain")
    else:
        save_fig(
            np.full_like(coarse_dem, np.nan)
            if coarse_dem is not None
            else np.zeros((64, 64)),
            "6 reference DSM (not available)",
            "06_reference",
            cmap="terrain",
        )

    # 7 absolute error (if reference)
    if (
        reference_dsm is not None
        and calibrated_dsm is not None
        and reference_dsm.shape == calibrated_dsm.shape
    ):
        try:
            abs_err = np.abs(
                calibrated_dsm.astype(np.float64) - reference_dsm.astype(np.float64)
            )
            abs_err[~np.isfinite(abs_err)] = np.nan
            save_fig(abs_err, "7 absolute error", "07_abs_error", cmap="inferno")
        except Exception:
            save_fig(
                np.zeros_like(calibrated_dsm),
                "7 absolute error (error)",
                "07_abs_error",
                cmap="inferno",
            )
    else:
        # residual map as proxy
        if (
            coarse_dem is not None
            and calibrated_dsm is not None
            and coarse_dem.shape == calibrated_dsm.shape
        ):
            try:
                resid = calibrated_dsm.astype(np.float64) - coarse_dem.astype(
                    np.float64
                )
                resid[~np.isfinite(resid)] = np.nan
                # diverging
                save_fig(
                    resid,
                    "7 residual (calibrated - coarse DEM)",
                    "07_abs_error",
                    cmap="RdBu",
                )
            except Exception:
                save_fig(
                    np.zeros((64, 64)), "7 abs error (no reference)", "07_abs_error"
                )
        else:
            save_fig(np.zeros((64, 64)), "7 abs error (no reference)", "07_abs_error")

    # 8 residual map (calibrated vs coarse)
    if (
        coarse_dem is not None
        and calibrated_dsm is not None
        and coarse_dem.shape == calibrated_dsm.shape
    ):
        try:
            resid = calibrated_dsm.astype(np.float64) - coarse_dem.astype(np.float64)
            resid[~np.isfinite(resid)] = np.nan
            save_fig(resid, "8 residual map", "08_residual", cmap="RdBu")
        except Exception:
            save_fig(np.zeros((64, 64)), "8 residual map", "08_residual")
    else:
        save_fig(np.zeros((64, 64)), "8 residual map", "08_residual")

    # 9 scatter plot: depth vs DEM and calibrated vs reference
    try:
        plt.figure(figsize=(6, 5))
        if (
            coarse_dem is not None
            and raw_depth is not None
            and coarse_dem.shape == raw_depth.shape
        ):
            valid = np.isfinite(raw_depth) & np.isfinite(coarse_dem)
            # subsample 5000 for speed
            idx = np.where(valid.ravel())[0]
            if len(idx) > 5000:
                idx = np.random.RandomState(42).choice(idx, 5000, replace=False)
                x = raw_depth.ravel()[idx]
                y = coarse_dem.ravel()[idx]
            else:
                x = raw_depth[valid]
                y = coarse_dem[valid]
            plt.scatter(x, y, s=1, alpha=0.3)
            plt.xlabel("raw depth")
            plt.ylabel("coarse DEM (m)")
            plt.title("9 scatter: raw depth vs DEM")
        else:
            plt.text(0.5, 0.5, "no aligned data", ha="center")
            plt.title("9 scatter")
        p = os.path.join(out_dir, f"{prefix}_09_scatter.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(p)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    # 10 histogram of errors
    try:
        plt.figure(figsize=(6, 4))
        if (
            reference_dsm is not None
            and calibrated_dsm is not None
            and reference_dsm.shape == calibrated_dsm.shape
        ):
            valid = np.isfinite(calibrated_dsm) & np.isfinite(reference_dsm)
            errors = (calibrated_dsm[valid] - reference_dsm[valid]).ravel()
            label = "calibrated - reference"
        elif (
            coarse_dem is not None
            and calibrated_dsm is not None
            and coarse_dem.shape == calibrated_dsm.shape
        ):
            valid = np.isfinite(calibrated_dsm) & np.isfinite(coarse_dem)
            errors = (calibrated_dsm[valid] - coarse_dem[valid]).ravel()
            label = "calibrated - coarse DEM (residual)"
        else:
            errors = np.array([0])
            label = "no data"
        # subsample large
        if len(errors) > 10000:
            errors = np.random.RandomState(42).choice(errors, 10000, replace=False)
        plt.hist(errors[np.isfinite(errors)], bins=50, color="steelblue", alpha=0.7)
        plt.xlabel(label)
        plt.ylabel("count")
        plt.title("10 histogram of errors")
        p = os.path.join(out_dir, f"{prefix}_10_hist.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(p)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    return paths
