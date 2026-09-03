"""
Frequency decomposition — coarse vs high-frequency for DEM 90m vs fine depth.
"""

from typing import Optional

import numpy as np


def gaussian_lowpass(
    array: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Gaussian lowpass, NaN-aware via mean fill."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        # fallback: uniform mean
        return np.full_like(
            array,
            float(np.nanmean(array[np.isfinite(array)]))
            if np.any(np.isfinite(array))
            else 0.0,
            dtype=np.float64,
        )
    if mask is not None:
        valid = mask.astype(bool) & np.isfinite(array)
    else:
        valid = np.isfinite(array)
    if np.sum(valid) == 0:
        return np.full_like(array, np.nan, dtype=np.float64)
    mean_val = float(np.mean(array[valid]))
    filled = np.where(valid, array.astype(np.float64), mean_val)
    low = gaussian_filter(filled, sigma=sigma)
    # restore NaN where originally invalid? Keep low everywhere for decomposition
    return low.astype(np.float64)


def highpass_component(
    array: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    low = gaussian_lowpass(array, sigma, mask)
    return array.astype(np.float64) - low


def decompose_dem_and_depth(
    dem: np.ndarray,
    depth: np.ndarray,
    sigma: float = 10.0,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Decompose both DEM and depth into coarse + highfreq."""
    dem_low = gaussian_lowpass(dem, sigma, mask)
    depth_low = gaussian_lowpass(depth, sigma, mask)
    dem_high = dem.astype(np.float64) - dem_low
    depth_high = depth.astype(np.float64) - depth_low
    # correlations
    try:
        from scipy.stats import pearsonr

        if mask is not None:
            valid = mask.astype(bool) & np.isfinite(dem) & np.isfinite(depth)
        else:
            valid = np.isfinite(dem) & np.isfinite(depth)
        coarse_corr = float("nan")
        high_corr = float("nan")
        if np.sum(valid) >= 2:
            dl, el = depth_low[valid], dem_low[valid]
            dh, eh = depth_high[valid], dem_high[valid]
            if np.std(dl) > 0 and np.std(el) > 0:
                coarse_corr, _ = pearsonr(dl, el)
                coarse_corr = float(coarse_corr)
            if np.std(dh) > 0 and np.std(eh) > 0:
                high_corr, _ = pearsonr(dh, eh)
                high_corr = float(high_corr)
    except Exception:
        coarse_corr, high_corr = float("nan"), float("nan")
    return {
        "dem_coarse": dem_low,
        "dem_highfreq": dem_high,
        "depth_coarse": depth_low,
        "depth_highfreq": depth_high,
        "sigma": sigma,
        "correlation_coarse": coarse_corr,
        "correlation_highfreq": high_corr,
    }


def frequency_fusion_reconstruction(
    depth: np.ndarray,
    dem: np.ndarray,
    scale: float,
    offset: float,
    sigma: float = 10.0,
    alpha: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    H_pred = (scale*depth+offset) + alpha*(dem - lowpass(dem))
    Assumption: coarse DEM low-freq is truth, high-freq DEM adds detail; monocular high-freq is not directly added to avoid sharpening artifacts.
    Documented as estimated, not ground truth.
    """
    base = scale * depth.astype(np.float64) + offset
    dem_low = gaussian_lowpass(dem, sigma, mask)
    dem_high = dem.astype(np.float64) - dem_low
    return (base + alpha * dem_high).astype(np.float32)


def compute_frequency_metrics(
    dem: np.ndarray, depth: np.ndarray, sigma: float = 10.0
) -> dict:
    decomp = decompose_dem_and_depth(dem, depth, sigma)
    dem_low, dem_high = decomp["dem_coarse"], decomp["dem_highfreq"]
    depth_low, depth_high = decomp["depth_coarse"], decomp["depth_highfreq"]
    # variance ratios
    try:
        dem_total_var = float(np.nanvar(dem))
        depth_total_var = float(np.nanvar(depth))
        dem_low_var = float(np.nanvar(dem_low))
        depth_low_var = float(np.nanvar(depth_low))
        dem_high_var = float(np.nanvar(dem_high))
        depth_high_var = float(np.nanvar(depth_high))
    except Exception:
        dem_total_var = depth_total_var = dem_low_var = depth_low_var = dem_high_var = (
            depth_high_var
        ) = float("nan")
    return {
        "sigma": sigma,
        "dem_total_var": dem_total_var,
        "depth_total_var": depth_total_var,
        "dem_low_var": dem_low_var,
        "depth_low_var": depth_low_var,
        "dem_high_var": dem_high_var,
        "depth_high_var": depth_high_var,
        "correlation_coarse": decomp["correlation_coarse"],
        "correlation_highfreq": decomp["correlation_highfreq"],
    }
