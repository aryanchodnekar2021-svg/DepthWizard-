# Depth Estimation Notes

## Models Used
- **Depth Anything V2** (Default): `depth-anything/Depth-Anything-V2-Small-hf`
- **MiDaS** (Fallback): `Intel/dpt-large`
- **ZoeDepth** (Placeholder): Currently maps to `Intel/dpt-large`.

## Inference Performance
- **Model**: `depth-anything/Depth-Anything-V2-Small-hf`
- **Environment**: CPU (Windows)
- **Time**: ~42.18 seconds on a dummy 512x512 image during initial run (including model load time / compilation overhead). Note that subsequent inferences would be faster.

## Observations
- Tested on a synthetic dummy image featuring a flat colored rectangle on a gradient background.
- Since we don't have real `urban / sparse / hilly / forested` images in `data/samples/` yet, specific domain-gap issues (like flattened forest canopies or noisy flat urban blocks) cannot be conclusively assessed at this stage. 
- *Next steps:* Once real satellite/drone imagery is available, run `python backend/depth/run_sample.py --image [filename]` and update these notes with real-world observations.
