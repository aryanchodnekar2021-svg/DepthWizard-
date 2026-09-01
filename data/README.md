# Data Directory

This directory is intended for sample images and Digital Elevation Model (DEM) tiles used in the DepthWizard pipeline. These files are typically large and thus gitignored.

## Sourcing Data

### Sample Images
Pull sample satellite/drone RGB images from the project source or other sample sets, such as:
`github.com/IMG-PROCESS-SAC/SIH2026`

Place them in `data/samples/`.

### DEM Tiles
SRTM 30m DEM tiles for scale calibration can be downloaded from:
- [OpenTopography](https://opentopography.org/)
- [USGS EarthExplorer](https://earthexplorer.usgs.gov/)

Ensure they match the test region `[my test region]` you intend to process. Place them in `data/srtm/`.
