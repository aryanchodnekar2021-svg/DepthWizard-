"""
Calibration methods registry — R&D comparison of formulations.

This module is for EXPERIMENTAL comparison, not production.
No method is "ground truth". All assume SRTM is coarse metric reference
and monocular depth is relative structure.

Methods:
- affine: H = a*D + b (least squares)
- robust_affine: H = a*D + b with RANSAC
- dem_residual: H = H_DEM + alpha * normalized residual
- local_normalized: locally z-scored fusion
- inverse_depth: H = a*(1/(D+eps)) + b
- piecewise_linear: n_bins piecewise affine
- frequency_fusion: coarse + high-freq decomposition
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Callable

import numpy as np
from scipy.stats import pearsonr


@dataclass
class CalibrationMethodResult:
    """Result of a calibration method for R&D comparison."""

    method_name: str
    scale: float
    offset: float
    residual_rmse: float
    residual_mae: float
    inlier_fraction: float
    pre_correlation: float
    post_correlation: float
    r_squared: float
    n_samples: int
    metadata: dict = field(default_factory=dict)


def _valid_mask(
    depth: np.ndarray, dem: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    m = np.isfinite(depth) & np.isfinite(dem)
    # DEM nodata: 0 and -9999 are common; also handle NaN already
    m &= (dem != 0) & (dem != -9999) & (dem != -9999.0)
    if mask is not None:
        m &= mask.astype(bool) & np.isfinite(mask)
    return m


def compute_quality_metrics(
    predicted: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> dict:
    """Compute quality metrics between predicted and reference."""
    if mask is not None:
        valid = mask.astype(bool) & np.isfinite(predicted) & np.isfinite(reference)
    else:
        valid = np.isfinite(predicted) & np.isfinite(reference)
    n_valid = int(np.sum(valid))
    n_total = predicted.size
    coverage = (n_valid / n_total * 100.0) if n_total > 0 else 0.0
    if n_valid < 2:
        return {
            "rmse": None,
            "mae": None,
            "bias": None,
            "r_squared": None,
            "correlation": None,
            "median_ae": None,
            "p90_ae": None,
            "p95_ae": None,
            "residual_std": None,
            "valid_pixels": n_valid,
            "total_pixels": n_total,
            "coverage_pct": coverage,
            "notes": "Insufficient valid pixels",
        }
    p = predicted[valid].astype(np.float64)
    r = reference[valid].astype(np.float64)
    errors = p - r
    abs_e = np.abs(errors)
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_e))
    bias = float(np.mean(errors))
    median_ae = float(np.median(abs_e))
    p90 = float(np.percentile(abs_e, 90))
    p95 = float(np.percentile(abs_e, 95))
    residual_std = float(np.std(errors))
    # R²
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # correlation
    corr = float("nan")
    try:
        if np.std(p) > 0 and np.std(r) > 0:
            corr, _ = pearsonr(p, r)
            corr = float(corr)
    except Exception:
        pass
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r_squared": r2,
        "correlation": corr,
        "median_ae": median_ae,
        "p90_ae": p90,
        "p95_ae": p95,
        "residual_std": residual_std,
        "valid_pixels": n_valid,
        "total_pixels": n_total,
        "coverage_pct": coverage,
    }


def _fit_linear(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.std(x) == 0:
        return 0.0, float(np.mean(y)) if len(y) > 0 else 0.0
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def method_affine(
    depth: np.ndarray, dem: np.ndarray, mask: Optional[np.ndarray] = None
) -> CalibrationMethodResult:
    """Affine H = a*D + b via least squares."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    # pre-correlation
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "affine",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"notes": "insufficient samples"},
        )
    x = depth[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)
    scale, offset = _fit_linear(x, y)
    pred = scale * x + offset
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) > 0 else float("nan")
    mae = float(np.mean(np.abs(resid))) if len(resid) > 0 else float("nan")
    # R² and post corr
    m = compute_quality_metrics(pred, y)
    post_corr = m["correlation"]
    r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
    return CalibrationMethodResult(
        "affine",
        scale,
        offset,
        rmse,
        mae,
        1.0,
        pre_corr,
        post_corr if post_corr is not None else float("nan"),
        r2 if r2 is not None else float("nan"),
        n_valid,
        {},
    )


def method_robust_affine(
    depth: np.ndarray,
    dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
    residual_threshold: Optional[float] = None,
) -> CalibrationMethodResult:
    """Robust affine via RANSAC, fallback to affine if sklearn unavailable."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "robust_affine",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"notes": "insufficient"},
        )
    x = depth[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)
    try:
        from sklearn.linear_model import RANSACRegressor

        X = x.reshape(-1, 1)
        # auto threshold if not provided: median absolute deviation
        if residual_threshold is None:
            # estimate MAD from initial fit
            a0, b0 = _fit_linear(x, y)
            resid0 = y - (a0 * x + b0)
            mad = (
                float(np.median(np.abs(resid0 - np.median(resid0))))
                if len(resid0) > 0
                else 1.0
            )
            residual_threshold = max(1.0, 1.4826 * mad * 2)  # ~2 sigma
        ransac = RANSACRegressor(
            min_samples=0.5, residual_threshold=residual_threshold, random_state=42
        )
        ransac.fit(X, y)
        scale = float(ransac.estimator_.coef_[0])
        offset = float(ransac.estimator_.intercept_)
        inlier_mask = ransac.inlier_mask_
        inlier_frac = float(np.mean(inlier_mask)) if len(inlier_mask) > 0 else 0.0
        # compute residuals on inliers
        y_pred = scale * x + offset
        resid = (
            y[inlier_mask] - y_pred[inlier_mask]
            if np.sum(inlier_mask) > 0
            else y - y_pred
        )
    except ImportError:
        # fallback
        scale, offset = _fit_linear(x, y)
        y_pred = scale * x + offset
        resid = y - y_pred
        inlier_frac = 1.0
    except Exception:
        scale, offset = _fit_linear(x, y)
        y_pred = scale * x + offset
        resid = y - y_pred
        inlier_frac = 1.0
    rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) > 0 else float("nan")
    mae = float(np.mean(np.abs(resid))) if len(resid) > 0 else float("nan")
    m = compute_quality_metrics(scale * x + offset, y)
    post_corr = m["correlation"]
    r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
    return CalibrationMethodResult(
        "robust_affine",
        scale,
        offset,
        rmse,
        mae,
        inlier_frac,
        pre_corr,
        post_corr if post_corr is not None else float("nan"),
        r2 if r2 is not None else float("nan"),
        n_valid,
        {"residual_threshold": residual_threshold},
    )


def method_dem_residual(
    depth: np.ndarray, dem: np.ndarray, mask: Optional[np.ndarray] = None
) -> CalibrationMethodResult:
    """DEM residual fusion: affine + normalized residual metadata."""
    base = method_affine(depth, dem, mask)
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    if n_valid < 2 or not np.isfinite(base.scale):
        return CalibrationMethodResult(
            "dem_residual",
            base.scale,
            base.offset,
            base.residual_rmse,
            base.residual_mae,
            base.inlier_fraction,
            base.pre_correlation,
            base.post_correlation,
            base.r_squared,
            n_valid,
            {"notes": "base failed"},
        )
    x = depth[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)
    y_pred = base.scale * x + base.offset
    residual = y - y_pred
    # normalized residual
    r_mean, r_std = float(np.mean(residual)), float(np.std(residual))
    # metadata about residual distribution
    meta = {
        "residual_mean": r_mean,
        "residual_std": r_std,
        "residual_median": float(np.median(residual)),
        "alpha": 1.0,
        "description": "H_pred = H_DEM + alpha*normalized_relative_residual (alpha=1)",
    }
    return CalibrationMethodResult(
        "dem_residual",
        base.scale,
        base.offset,
        base.residual_rmse,
        base.residual_mae,
        base.inlier_fraction,
        base.pre_correlation,
        base.post_correlation,
        base.r_squared,
        n_valid,
        meta,
    )


def method_local_normalized(
    depth: np.ndarray,
    dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 64,
) -> CalibrationMethodResult:
    """Locally normalized residual fusion via sliding window z-score."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "local_normalized",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"patch_size": patch_size},
        )
    # Compute local z-score maps for valid region only? For simplicity, compute global z then patch-wise re-estimation
    # Use valid pixels to compute per-patch normalization via block processing
    h, w = depth.shape
    # Build z-score arrays (fallback to global if patch too small)
    depth_z = np.full_like(depth, np.nan, dtype=np.float64)
    dem_z = np.full_like(dem, np.nan, dtype=np.float64)
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            r1, c1 = min(r + patch_size, h), min(c + patch_size, w)
            d_patch = depth[r:r1, c:c1]
            e_patch = dem[r:r1, c:c1]
            m_patch = valid[r:r1, c:c1]
            if np.sum(m_patch) < 2:
                continue
            d_vals = d_patch[m_patch]
            e_vals = e_patch[m_patch]
            d_mean, d_std = float(np.mean(d_vals)), float(np.std(d_vals))
            e_mean, e_std = float(np.mean(e_vals)), float(np.std(e_vals))
            if d_std > 1e-9:
                depth_z[r:r1, c:c1][m_patch] = (d_vals - d_mean) / d_std
            if e_std > 1e-9:
                dem_z[r:r1, c:c1][m_patch] = (e_vals - e_mean) / e_std
    # valid z pixels
    z_valid = np.isfinite(depth_z) & np.isfinite(dem_z) & valid
    n_z = int(np.sum(z_valid))
    if n_z >= 2 and np.std(depth_z[z_valid]) > 0:
        scale, offset = _fit_linear(
            depth_z[z_valid].astype(np.float64), dem_z[z_valid].astype(np.float64)
        )
        # also fit global affine for actual application
        global_res = method_affine(depth, dem, mask)
        pred = global_res.scale * depth[valid].astype(np.float64) + global_res.offset
        m = compute_quality_metrics(pred, dem[valid].astype(np.float64))
        post_corr = m["correlation"]
        r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
        rmse, mae = (
            m["rmse"] if m["rmse"] is not None else float("nan"),
            m["mae"] if m["mae"] is not None else float("nan"),
        )
        return CalibrationMethodResult(
            "local_normalized",
            global_res.scale,
            global_res.offset,
            rmse,
            mae,
            float(n_z / max(n_valid, 1)),
            pre_corr,
            post_corr if post_corr is not None else float("nan"),
            r2 if r2 is not None else float("nan"),
            n_valid,
            {
                "patch_size": patch_size,
                "z_scale": scale,
                "z_offset": offset,
                "n_z_samples": n_z,
            },
        )
    else:
        # fallback to global
        base = method_affine(depth, dem, mask)
        base.method_name = "local_normalized"
        base.metadata = {
            "patch_size": patch_size,
            "fallback": "global",
            "n_z_samples": n_z,
        }
        return base


def method_inverse_depth(
    depth: np.ndarray, dem: np.ndarray, mask: Optional[np.ndarray] = None
) -> CalibrationMethodResult:
    """Inverse depth: H = a*(1/(D+eps)) + b."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "inverse_depth",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"epsilon": 1e-6},
        )
    x = depth[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)
    x_inv = 1.0 / (x + 1e-6)
    # handle inf/nan from inversion
    finite = np.isfinite(x_inv)
    x_inv, y = x_inv[finite], y[finite]
    if len(x_inv) < 2:
        return CalibrationMethodResult(
            "inverse_depth",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"epsilon": 1e-6, "notes": "no finite inv"},
        )
    scale, offset = _fit_linear(x_inv, y)
    pred = scale * x_inv + offset
    m = compute_quality_metrics(pred, y)
    rmse = m["rmse"] if m["rmse"] is not None else float("nan")
    mae = m["mae"] if m["mae"] is not None else float("nan")
    post_corr = m["correlation"] if m["correlation"] is not None else float("nan")
    r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
    return CalibrationMethodResult(
        "inverse_depth",
        scale,
        offset,
        rmse,
        mae,
        float(len(x_inv) / max(n_valid, 1)),
        pre_corr,
        post_corr,
        r2,
        n_valid,
        {"epsilon": 1e-6, "n_inv_samples": int(len(x_inv))},
    )


def method_piecewise_linear(
    depth: np.ndarray,
    dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 3,
) -> CalibrationMethodResult:
    """Piecewise linear: n_bins segments, weighted avg scale/offset."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "piecewise_linear",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"n_bins": n_bins},
        )
    x = depth[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)
    # bin edges by depth quantiles
    try:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        # deduplicate edges
        edges = np.unique(edges)
        if len(edges) < n_bins + 1:
            # fallback to equal width
            edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    except Exception:
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    bin_params = []
    all_pred = np.full_like(y, np.nan)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin inclusive
        if i == n_bins - 1:
            bmask = (x >= lo) & (x <= hi)
        else:
            bmask = (x >= lo) & (x < hi)
        if np.sum(bmask) < 2:
            continue
        xb, yb = x[bmask], y[bmask]
        sc, off = _fit_linear(xb, yb)
        bin_params.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "scale": sc,
                "offset": off,
                "n": int(np.sum(bmask)),
            }
        )
        all_pred[bmask] = sc * xb + off
    valid_pred = np.isfinite(all_pred)
    if np.sum(valid_pred) < 2:
        base = method_affine(depth, dem, mask)
        return CalibrationMethodResult(
            "piecewise_linear",
            base.scale,
            base.offset,
            base.residual_rmse,
            base.residual_mae,
            base.inlier_fraction,
            pre_corr,
            base.post_correlation,
            base.r_squared,
            n_valid,
            {"n_bins": n_bins, "fallback": "global", "bin_params": bin_params},
        )
    m = compute_quality_metrics(all_pred[valid_pred], y[valid_pred])
    rmse = m["rmse"] if m["rmse"] is not None else float("nan")
    mae = m["mae"] if m["mae"] is not None else float("nan")
    post_corr = m["correlation"] if m["correlation"] is not None else float("nan")
    r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
    # weighted avg scale/offset
    total = sum(b["n"] for b in bin_params) if bin_params else 1
    avg_scale = (
        sum(b["scale"] * b["n"] for b in bin_params) / total
        if total > 0
        else float("nan")
    )
    avg_offset = (
        sum(b["offset"] * b["n"] for b in bin_params) / total
        if total > 0
        else float("nan")
    )
    return CalibrationMethodResult(
        "piecewise_linear",
        float(avg_scale),
        float(avg_offset),
        rmse,
        mae,
        float(np.sum(valid_pred) / max(n_valid, 1)),
        pre_corr,
        post_corr,
        r2,
        n_valid,
        {"n_bins": n_bins, "bin_params": bin_params},
    )


def method_frequency_fusion(
    depth: np.ndarray,
    dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lowpass_sigma: float = 10.0,
) -> CalibrationMethodResult:
    """Frequency fusion: Gaussian lowpass decomposition."""
    valid = _valid_mask(depth, dem, mask)
    n_valid = int(np.sum(valid))
    pre_corr = float("nan")
    try:
        if n_valid >= 2:
            d = depth[valid].astype(np.float64)
            e = dem[valid].astype(np.float64)
            if np.std(d) > 0 and np.std(e) > 0:
                pre_corr, _ = pearsonr(d, e)
                pre_corr = float(pre_corr)
    except Exception:
        pass
    if n_valid < 2:
        return CalibrationMethodResult(
            "frequency_fusion",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            pre_corr,
            float("nan"),
            float("nan"),
            n_valid,
            {"lowpass_sigma": lowpass_sigma},
        )
    # Try gaussian lowpass
    try:
        from scipy.ndimage import gaussian_filter

        # nan-aware: fill nan with nanmean for filtering, then mask
        dem_filled = np.where(valid, dem.astype(np.float64), np.nan)
        depth_filled = np.where(valid, depth.astype(np.float64), np.nan)
        # simple nan handling: replace nan with mean before filter
        dem_mean = float(np.nanmean(dem_filled)) if np.sum(valid) > 0 else 0.0
        depth_mean = float(np.nanmean(depth_filled)) if np.sum(valid) > 0 else 0.0
        dem_in = np.where(np.isfinite(dem_filled), dem_filled, dem_mean)
        depth_in = np.where(np.isfinite(depth_filled), depth_filled, depth_mean)
        dem_low = gaussian_filter(dem_in, sigma=lowpass_sigma)
        depth_low = gaussian_filter(depth_in, sigma=lowpass_sigma)
        dem_high = dem_in - dem_low
        depth_high = depth_in - depth_low
        # correlation of coarse components
        coarse_corr = float("nan")
        try:
            if np.std(dem_low[valid]) > 0 and np.std(depth_low[valid]) > 0:
                coarse_corr, _ = pearsonr(depth_low[valid], dem_low[valid])
                coarse_corr = float(coarse_corr)
        except Exception:
            pass
        # Global affine still primary
        base = method_affine(depth, dem, mask)
        m = compute_quality_metrics(
            base.scale * depth[valid].astype(np.float64) + base.offset,
            dem[valid].astype(np.float64),
        )
        post_corr = m["correlation"] if m["correlation"] is not None else float("nan")
        r2 = m["r_squared"] if m["r_squared"] is not None else float("nan")
        rmse = m["rmse"] if m["rmse"] is not None else float("nan")
        mae = m["mae"] if m["mae"] is not None else float("nan")
        meta = {
            "lowpass_sigma": lowpass_sigma,
            "coarse_correlation": coarse_corr,
            "dem_high_std": float(np.std(dem_high[valid])),
            "depth_high_std": float(np.std(depth_high[valid])),
            "description": "H_pred = global_affine(D) + highfreq(DEM) concept; current R&D returns global_affine params for comparison",
        }
        return CalibrationMethodResult(
            "frequency_fusion",
            base.scale,
            base.offset,
            rmse,
            mae,
            base.inlier_fraction,
            pre_corr,
            post_corr,
            r2,
            n_valid,
            meta,
        )
    except ImportError:
        base = method_affine(depth, dem, mask)
        base.method_name = "frequency_fusion"
        base.metadata = {
            "lowpass_sigma": lowpass_sigma,
            "fallback": "scipy missing",
            "description": "scipy.ndimage.gaussian_filter not available",
        }
        return base
    except Exception as e:
        base = method_affine(depth, dem, mask)
        base.method_name = "frequency_fusion"
        base.metadata = {"lowpass_sigma": lowpass_sigma, "error": str(e)}
        return base


def get_available_methods() -> Dict[str, Callable]:
    return {
        "affine": method_affine,
        "robust_affine": method_robust_affine,
        "dem_residual": method_dem_residual,
        "local_normalized": method_local_normalized,
        "inverse_depth": method_inverse_depth,
        "piecewise_linear": method_piecewise_linear,
        "frequency_fusion": method_frequency_fusion,
    }


def apply_method(
    method_result: CalibrationMethodResult, depth: np.ndarray
) -> np.ndarray:
    """Apply calibration to depth. For most methods: scale*depth+offset."""
    if not np.isfinite(method_result.scale) or not np.isfinite(method_result.offset):
        return np.full_like(depth, np.nan, dtype=np.float32)
    return (
        depth.astype(np.float64) * method_result.scale + method_result.offset
    ).astype(np.float32)
