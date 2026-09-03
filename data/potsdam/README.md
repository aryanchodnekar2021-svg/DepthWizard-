# ISPRS Potsdam Dataset — Download & Preparation

## Overview

The ISPRS Potsdam 2D Semantic Labeling dataset provides:

- **38 tiles**, each **6000×6000 pixels** at **5 cm/pixel** resolution
- **True Orthophoto (TOP)** images in RGB, IRRG, and RGBIR compositions
- **Digital Surface Model (DSM)** as 32-bit float GeoTIFF
- **CRS:** UTM WGS84 (EPSG:32633)
- **Total size:** ~13.3 GB

## Download

### Source

- **URL:** https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/
- **Password:** Available at https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
- **License:** ISPRS benchmark terms of use

### Required Files

1. `1_TOP_mosaic_iron_04cl.tif` — Top True Orthophoto mosaic (optional, for reference)
2. `2_DSM_mosaic_iron_04cl.tif` — DSM mosaic (optional, for reference)
3. Individual tile files:
   - `4_Ortho_RGBIR.zip` — RGBIR tiles (4 channels: R, G, B, IR)
   - `5_DSM.zip` — DSM tiles (1 channel, float32)
   - `5_Labels_all.zip` — Semantic labels (for reference only)

### Alternative: Individual Tiles

If full dataset is too large, download individual tiles:

- Each tile: ~10-15 MB (RGB), ~14 MB (DSM)
- Minimum: 2-3 tiles for initial testing

## Expected Directory Structure

After downloading and extracting:

```
data/potsdam/
├── raw/                          # Raw downloaded files
│   ├── 4_Ortho_RGBIR/           # RGBIR tiles
│   │   ├── top_potsdam_2_10_RGBIR.tif
│   │   ├── top_potsdam_2_10_RGB.tif
│   │   └── ...
│   ├── 5_DSM/                   # DSM tiles
│   │   ├── top_potsdam_2_10_DSM.tif
│   │   └── ...
│   └── 5_Labels_all/            # Labels (reference)
│       ├── top_potsdam_2_10_label.tif
│       └── ...
├── prepared/                    # Prepared benchmark pairs (generated)
│   ├── tile_2_10.npz           # RGB + DSM pair
│   ├── tile_2_11.npz
│   ├── manifest.json           # Dataset manifest
│   └── ...
└── README.md                   # This file
```

## Preparation

### Option 1: Full Dataset

```bash
# After downloading raw files to data/potsdam/raw/
python -c "
from backend.geo.potsdam_adapter import prepare_potsdam_for_benchmark
manifest = prepare_potsdam_for_benchmark(
    root='data/potsdam/raw',
    output_dir='data/potsdam/prepared',
    split='train',
)
print(f'Prepared {len(manifest[\"tiles_prepared\"])} tiles')
"
```

### Option 2: Subset for Testing

```bash
# Prepare only first 3 tiles
python -c "
from backend.geo.potsdam_adapter import prepare_potsdam_for_benchmark
manifest = prepare_potsdam_for_benchmark(
    root='data/potsdam/raw',
    output_dir='data/potsdam/prepared',
    max_tiles=3,
    split='train',
)
"
```

### Option 3: Resize for Speed

```bash
# Resize to 1000px max dimension (faster inference)
python -c "
from backend.geo.potsdam_adapter import prepare_potsdam_for_benchmark
manifest = prepare_potsdam_for_benchmark(
    root='data/potsdam/raw',
    output_dir='data/potsdam/prepared_small',
    max_tiles=5,
    max_size=1000,
)
"
```

## Tile Naming Convention

- **Row:** 2-7 (top to bottom, north to south)
- **Column:** 7-15 (left to right, west to east)
- **Example:** `top_potsdam_2_10_RGB.tif` = row 2, column 10, RGB composition

### Training Tiles (24 tiles)

2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12,
5_10, 5_11, 5_12, 6_7, 6_8, 6_9, 6_10, 6_11, 6_12,
7_7, 7_8, 7_9, 7_10, 7_11, 7_12

### Test Tiles (14 tiles)

2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15,
5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13

## Metadata

Each prepared tile (`.npz`) contains:

- `rgb`: uint8 RGB array (H, W, 3)
- `dsm`: float64 DSM array (H, W) in meters
- `tile_id`: string (e.g., "2_10")
- `row`, `col`: int
- `split`: "train" or "test"
- `crs`: "EPSG:32633"
- `resolution_m`: 0.05 (5 cm)

## Validation

After preparation, validate with:

```bash
python -c "
from backend.geo.potsdam_adapter import discover_potsdam_tiles
info = discover_potsdam_tiles('data/potsdam/raw')
print(f'Tiles found: {info.total_tiles}')
print(f'With RGB: {info.tiles_with_rgb}')
print(f'With DSM: {info.tiles_with_dsm}')
print(f'With both: {info.tiles_with_both}')
"
```

## Notes

- **DSM provenance:** Generated via dense image matching (Trimble INPHO 5.6)
- **Vertical datum:** WGS84 ellipsoid (not EGM96 geoid like SRTM)
- **Resolution:** 5 cm/pixel (much finer than SRTM 90m)
- **Usage:** RGB→depth→calibration→metric DSM vs independent reference DSM
- **No ground truth labels needed** for calibration benchmark (only RGB+DSM pairs)
