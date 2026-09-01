import os
import time
import argparse
import numpy as np
from PIL import Image
from estimator import estimate_depth

def create_dummy_image(path):
    print(f"Creating a dummy image at {path} for testing...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create a simple synthetic scene: gradient background with a rectangle
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        img[y, :, 0] = y // 2 # Blueish sky gradient
        img[y, :, 1] = 100
        img[y, :, 2] = 255 - (y // 2)
        
    # Draw a "building" in the middle
    img[256:512, 200:300] = [100, 100, 100]
    
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description="Test Depth Estimation")
    parser.add_argument("--image", type=str, default="dummy.jpg", help="Image filename in data/samples/")
    parser.add_argument("--backbone", type=str, default="depth_anything_v2", help="Model backbone")
    args = parser.parse_args()
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_dir = os.path.join(base_dir, "data", "samples")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    image_path = os.path.join(samples_dir, args.image)
    
    if not os.path.exists(image_path) and args.image == "dummy.jpg":
        create_dummy_image(image_path)
    elif not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    print(f"Loading image {image_path}...")
    img = np.array(Image.open(image_path).convert("RGB"))
    
    print(f"Running depth estimation with {args.backbone}...")
    start_time = time.time()
    
    # Run inference
    depth_map = estimate_depth(img, args.backbone)
    
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds.")
    
    # Normalize depth map for visualization (0-255 grayscale)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    print(f"Depth map raw range: {depth_min:.2f} to {depth_max:.2f}")
    
    if depth_max > depth_min:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_map)
        
    depth_vis = (depth_norm * 255).astype(np.uint8)
    
    # Save output
    output_filename = f"depth_{args.backbone}_{os.path.basename(args.image)}.png"
    output_path = os.path.join(outputs_dir, output_filename)
    Image.fromarray(depth_vis).save(output_path)
    print(f"Saved depth visualization to {output_path}")

if __name__ == "__main__":
    main()
