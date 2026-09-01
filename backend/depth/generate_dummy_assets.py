import os
import numpy as np
from PIL import Image

def generate_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_dir = os.path.join(base_dir, "data", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    size = 256
    
    # 1. Generate Dummy Heightmap (grayscale)
    # A simple gradient with a peak in the middle
    heightmap = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            # Distance from center
            dist = np.sqrt((x - size/2)**2 + (y - size/2)**2)
            max_dist = size / 1.5
            heightmap[y, x] = max(0, 1.0 - (dist / max_dist))
            
    # Add some random noise
    heightmap += np.random.normal(0, 0.05, (size, size))
    heightmap = np.clip(heightmap, 0, 1)
    
    heightmap_8bit = (heightmap * 255).astype(np.uint8)
    Image.fromarray(heightmap_8bit).save(os.path.join(samples_dir, "dummy_heightmap.png"))
    
    # 2. Generate Dummy Texture (RGB)
    # Green/brown grid pattern mapping over the terrain
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x // 32) % 2 == (y // 32) % 2:
                texture[y, x] = [34, 139, 34] # Forest Green
            else:
                texture[y, x] = [139, 69, 19] # Saddle Brown
                
    Image.fromarray(texture).save(os.path.join(samples_dir, "dummy_texture.png"))
    print("Successfully generated dummy_heightmap.png and dummy_texture.png in data/samples/")

if __name__ == "__main__":
    generate_assets()
