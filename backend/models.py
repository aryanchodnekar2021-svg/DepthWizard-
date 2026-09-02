"""
Data models for the DepthWizard geospatial pipeline.

Explicitly distinguishes RELATIVE from ABSOLUTE elevation representations.
Never labels relative predictions as metric.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ElevationMode(str, Enum):
    """Whether the output represents relative or absolute elevation."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class CalibrationSource(str, Enum):
    """Source of metric calibration data."""

    NONE = "none"
    SRTM = "srtm"
    GCP = "gcp"
    LOCAL_DEM = "local_dem"


@dataclass
class RasterMetadata:
    """Metadata extracted from any raster file (GeoTIFF or plain image)."""

    is_georeferenced: bool
    file_path: str
    width: int
    height: int
    band_count: int
    dtype: str
    # Geospatial fields (None if not georeferenced)
    crs: Optional[str] = None
    epsg: Optional[int] = None
    transform: Optional[list] = None  # Affine coefficients [a, b, c, d, e, f]
    bounds: Optional[dict] = None  # {left, bottom, right, top}
    resolution: Optional[dict] = None  # {x, y}
    nodata: Optional[float] = None
    geographic_extent: Optional[dict] = None  # {west, south, east, north} in WGS84


@dataclass
class CalibrationMetadata:
    """Metadata about the calibration process."""

    applied: bool
    source: CalibrationSource
    reason: Optional[str] = None  # Why calibration was not applied
    scale: Optional[float] = None  # a in H_metric = a * D_relative + b
    offset: Optional[float] = None  # b in H_metric = a * D_relative + b
    fit_method: Optional[str] = None  # e.g., "linear", "ransac"
    valid_samples: Optional[int] = None  # Number of pixels used for fitting
    residual_rmse: Optional[float] = None  # RMSE of fit residuals
    residual_mae: Optional[float] = None  # MAE of fit residuals
    dem_tile_ids: Optional[List[str]] = field(default_factory=list)  # SRTM tiles used


@dataclass
class ReconstructionResult:
    """
    Complete result of a depth reconstruction job.

    For relative output:
        - mode = RELATIVE
        - elevation values are normalized 0-1 (NOT meters)
        - NEVER label as metric/absolute

    For absolute output:
        - mode = ABSOLUTE
        - elevation values are in meters above sea level
        - CRS, transform, and calibration metadata are populated
    """

    mode: ElevationMode
    is_georeferenced: bool
    # Input metadata
    input_raster: RasterMetadata
    # Output arrays
    elevation_array: object  # np.ndarray - the DSM/rDSM
    heightmap_array: object  # np.ndarray - normalized for visualization
    # Output metadata
    crs: Optional[str] = None
    bounds: Optional[dict] = None
    resolution: Optional[dict] = None
    transform: Optional[list] = None
    # Calibration
    calibration: CalibrationMetadata = field(
        default_factory=lambda: CalibrationMetadata(
            applied=False, source=CalibrationSource.NONE
        )
    )
    # Output paths
    dsm_path: Optional[str] = None
    heightmap_path: Optional[str] = None
    texture_path: Optional[str] = None
    # Warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def units(self) -> str:
        """Return the elevation units string."""
        if self.mode == ElevationMode.ABSOLUTE:
            return "meters above sea level"
        return "relative (normalized 0-1, no metric meaning)"

    def to_api_response(self) -> dict:
        """Convert to API response dict."""
        return {
            "status": "ok",
            "mode": self.mode.value,
            "units": self.units,
            "is_georeferenced": self.is_georeferenced,
            "crs": self.crs,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "width": self.input_raster.width,
            "height": self.input_raster.height,
            "calibration": {
                "applied": self.calibration.applied,
                "source": self.calibration.source.value,
                "reason": self.calibration.reason,
                "scale": self.calibration.scale,
                "offset": self.calibration.offset,
                "fit_method": self.calibration.fit_method,
                "valid_samples": self.calibration.valid_samples,
                "residual_rmse": self.calibration.residual_rmse,
                "residual_mae": self.calibration.residual_mae,
                "dem_tile_ids": self.calibration.dem_tile_ids,
            },
            "dsm_url": self.dsm_path,
            "heightmap_url": self.heightmap_path,
            "texture_url": self.texture_path,
            "warnings": self.warnings,
        }
