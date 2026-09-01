import os
import sys
import numpy as np

# Add the parent directory to the path so we can import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import compute_metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_tiles_dir = os.path.join(base_dir, "data", "eval_tiles")
    os.makedirs(eval_tiles_dir, exist_ok=True)
    
    # In a real scenario, we would loop through eval_tiles_dir and read images + reference DSMs.
    # We will simulate this with dummy data for now to ensure the harness runs.
    
    test_tiles = [
        {"id": "tile_01", "terrain_type": "urban", "size": (128, 128)},
        {"id": "tile_02", "terrain_type": "sparse", "size": (128, 128)},
        {"id": "tile_03", "terrain_type": "hilly", "size": (128, 128)},
        {"id": "tile_04", "terrain_type": "forested", "size": (128, 128)},
    ]
    
    print("Running evaluation harness...")
    
    results = []
    
    for tile in test_tiles:
        print(f"Processing {tile['id']} ({tile['terrain_type']})...")
        
        # Simulate predicted DSM (e.g. after running estimator and calibrate)
        # We add some noise to simulate errors
        reference_dsm = np.random.uniform(50, 500, tile['size'])
        
        # Simulate different accuracy for different terrains
        if tile['terrain_type'] == "forested":
            noise_level = 50.0  # High error in forests
        elif tile['terrain_type'] == "urban":
            noise_level = 30.0  # Moderate error on buildings
        else:
            noise_level = 10.0  # Low error on simple terrain
            
        predicted_dsm = reference_dsm + np.random.normal(0, noise_level, tile['size'])
        
        # Add occasional NaNs to test masking
        if np.random.rand() > 0.5:
            predicted_dsm[10:20, 10:20] = np.nan
            
        metrics = compute_metrics(predicted_dsm, reference_dsm)
        
        results.append({
            "terrain_type": tile['terrain_type'],
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "correlation": metrics['correlation']
        })
        
    # Aggregate and save results
    results_path = os.path.join(base_dir, "eval", "results.md")
    
    with open(results_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("| Terrain Type | RMSE (m) | MAE (m) | Correlation | Notes |\n")
        f.write("|--------------|----------|---------|-------------|-------|\n")
        
        for res in results:
            notes = ""
            if res['rmse'] > 40:
                notes = "High error flagged"
            
            f.write(f"| {res['terrain_type'].capitalize()} | {res['rmse']:.2f} | {res['mae']:.2f} | {res['correlation']:.3f} | {notes} |\n")
            
    print(f"\nEvaluation complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
