# Mesh Visualization QA Checklist

Before finalizing the frontend integration, manually verify the following criteria using the running Three.js application.

## 1. Navigability Check
- [ ] **First-Person Lock**: Clicking on the canvas should lock the cursor and hide it.
- [ ] **Movement**: WASD keys should smoothly translate the camera across the X/Z plane.
- [ ] **Vertical Movement**: Spacebar and Shift should smoothly translate the camera up and down.
- [ ] **Look Mechanics**: Moving the mouse should fluidly rotate the camera view without jitter.
- [ ] **Orbit Toggle**: Pressing 'C' should unlock the cursor and switch to an orbital view around the center of the mesh.

## 2. Mesh Artifact Check
Load a complex terrain (e.g., `forested` or `urban` tile) and inspect for:
- [ ] **Spikes/Towers**: Unrealistic, sharp vertical extrusions. Often caused by depth estimation struggling with shadows or building edges.
- [ ] **Holes/Pits**: Sudden extreme drops in the mesh. Often caused by NaN values or nodata pixels not being properly masked out during rendering.
- [ ] **Texture Tearing**: Ensure the RGB texture map aligns 1:1 with the displaced vertices without stretching weirdly at the edges.

*How to reproduce issues*: If you spot an artifact, note the rough XYZ coordinates of the camera and verify if the raw depth map (saved in `outputs/`) contains the anomaly.

## 3. Stability Check
- [ ] **Memory Leaks**: Continuously upload 5 different images in a row. The browser should not crash, and old meshes must be properly disposed of from GPU memory.
- [ ] **Extended Session**: Leave the flythrough running for 5 straight minutes. Verify that framerates remain stable (>30fps) and the application doesn't freeze.
