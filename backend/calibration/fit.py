import numpy as np
from sklearn.linear_model import LinearRegression

def fit_scale(relative_depth: np.ndarray, srtm_elevation: np.ndarray, method="linear"):
    """
    Fits relative depth to metric SRTM elevation.
    """
    if method == "linear":
        # Flatten arrays and remove any NaNs from the reference data
        valid_mask = ~np.isnan(srtm_elevation) & ~np.isnan(relative_depth)
        
        x = relative_depth[valid_mask].reshape(-1, 1)
        y = srtm_elevation[valid_mask].reshape(-1, 1)
        
        if len(x) < 2:
            print("Not enough valid points for linear regression, falling back to simple scaling.")
            return relative_depth * 100.0 # arbitrary scaling fallback
            
        # Optional: subsample for speed if x is huge
        if len(x) > 10000:
            indices = np.random.choice(len(x), 10000, replace=False)
            x_sample = x[indices]
            y_sample = y[indices]
        else:
            x_sample, y_sample = x, y
            
        model = LinearRegression()
        model.fit(x_sample, y_sample)
        
        # Apply the learned scale and offset to the entire relative depth map
        scale = model.coef_[0][0]
        offset = model.intercept_[0]
        
        calibrated_dsm = (relative_depth * scale) + offset
        print(f"Calibration applied: DSM = Depth * {scale:.2f} + {offset:.2f}")
        return calibrated_dsm
        
    elif method == "patch_stats":
        print("Warning: patch_stats method not fully implemented. Falling back to linear.")
        return fit_scale(relative_depth, srtm_elevation, method="linear")
    else:
        raise ValueError(f"Unknown calibration method: {method}")

def refine_with_gcps(elevation: np.ndarray, gcp_points: list):
    """
    Stub for lightweight bias correction using Ground Control Points.
    """
    print("GCP refinement is not implemented yet. Returning original elevation.")
    return elevation
