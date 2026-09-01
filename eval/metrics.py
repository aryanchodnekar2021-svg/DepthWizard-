import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def compute_metrics(predicted_dsm: np.ndarray, reference_dsm: np.ndarray):
    """
    Computes RMSE, MAE, and Pearson correlation between predicted and reference DSMs.
    Handles NaN/nodata masking.
    Assuming both arrays are already aligned/resampled to the same shape.
    """
    if predicted_dsm.shape != reference_dsm.shape:
        raise ValueError(f"Shape mismatch: Predicted {predicted_dsm.shape} vs Reference {reference_dsm.shape}. Please resample first.")
        
    # Mask out invalid data (NaNs)
    valid_mask = ~np.isnan(predicted_dsm) & ~np.isnan(reference_dsm)
    
    pred_valid = predicted_dsm[valid_mask]
    ref_valid = reference_dsm[valid_mask]
    
    if len(pred_valid) < 2:
        return {"rmse": None, "mae": None, "correlation": None, "notes": "Not enough valid pixels."}
        
    rmse = np.sqrt(mean_squared_error(ref_valid, pred_valid))
    mae = mean_absolute_error(ref_valid, pred_valid)
    
    # Pearson correlation can fail if variance is 0
    try:
        corr, _ = pearsonr(ref_valid, pred_valid)
    except Exception:
        corr = np.nan
        
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(corr),
        "valid_pixels": int(len(pred_valid))
    }
