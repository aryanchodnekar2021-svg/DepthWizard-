"""
Test suite for the DepthWizard geospatial pipeline.

Tests cover:
- Package imports
- Raster detection and metadata extraction
- SRTM tile selection
- DEM reprojection and alignment
- Calibration fitting
- Nodata handling
- Relative and absolute export
- API response structure

All tests use tiny synthetic fixtures — no model download required.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Fixtures — tiny synthetic rasters
# ============================================================


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="dw_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def plain_png(tmp_dir):
    """Create a plain PNG image (not georeferenced)."""
    from PIL import Image

    path = os.path.join(tmp_dir, "test.png")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def plain_jpg(tmp_dir):
    """Create a plain JPG image (not georeferenced)."""
    from PIL import Image

    path = os.path.join(tmp_dir, "test.jpg")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def plain_tif(tmp_dir):
    """Create a plain TIFF without geospatial reference."""
    from PIL import Image

    path = os.path.join(tmp_dir, "test.tif")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def geotiff_4326(tmp_dir):
    """Create a small GeoTIFF with EPSG:4326 (WGS84)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    path = os.path.join(tmp_dir, "geo_test.tif")
    data = np.random.uniform(100, 500, (64, 64)).astype(np.float32)
    transform = from_origin(77.0, 28.0, 0.001, 0.001)  # ~100m resolution

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def geotiff_32643(tmp_dir):
    """Create a small GeoTIFF with EPSG:32643 (UTM Zone 43N)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    path = os.path.join(tmp_dir, "geo_utm.tif")
    data = np.random.uniform(100, 500, (64, 64)).astype(np.float32)
    transform = from_origin(770000, 3100000, 30, 30)  # 30m UTM

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def geotiff_with_nodata(tmp_dir):
    """Create a GeoTIFF with explicit nodata value."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    path = os.path.join(tmp_dir, "geo_nodata.tif")
    data = np.random.uniform(100, 500, (64, 64)).astype(np.float32)
    data[0:10, 0:10] = -9999.0  # Nodata region
    transform = from_origin(77.0, 28.0, 0.001, 0.001)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def mock_srtm_tile(tmp_dir):
    """Create a mock SRTM tile (1° x 1° at ~30m)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    srtm_dir = os.path.join(tmp_dir, "srtm")
    os.makedirs(srtm_dir, exist_ok=True)

    # N28E077 tile covers lat 28-29, lon 77-78
    path = os.path.join(srtm_dir, "N28E077.tif")
    data = np.random.uniform(200, 400, (120, 120)).astype(np.float32)
    # Simulate elevation gradient
    for i in range(120):
        data[i, :] += i * 0.5

    transform = from_origin(77.0, 29.0, 1.0 / 120, 1.0 / 120)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=120,
        width=120,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)

    return srtm_dir, path


# ============================================================
# Test: Package Imports (Step 1)
# ============================================================


class TestPackageImports:
    def test_import_models(self):
        from backend.models import (
            ElevationMode,
            CalibrationSource,
            CalibrationMetadata,
            ReconstructionResult,
            RasterMetadata,
        )

        assert ElevationMode.RELATIVE.value == "relative"
        assert ElevationMode.ABSOLUTE.value == "absolute"

    def test_import_raster_inspect(self):
        from backend.geo.raster_inspect import inspect_raster, validate_input

        assert callable(inspect_raster)

    def test_import_srtm_provider(self):
        from backend.geo.srtm_provider import SRTMProvider, lat_lon_to_tile_id

        assert callable(SRTMProvider)

    def test_import_calibration(self):
        from backend.calibration.pipeline import calibrate, save_output
        from backend.calibration.fit import (
            calibrate_depth_to_elevation,
            apply_calibration,
        )
        from backend.calibration.georef import read_geotiff_metadata

        assert callable(calibrate)
        assert callable(calibrate_depth_to_elevation)

    def test_import_metrics(self):
        from eval.metrics import compute_metrics

        assert callable(compute_metrics)


# ============================================================
# Test: Raster Detection (Step 2)
# ============================================================


class TestRasterDetection:
    def test_plain_png(self, plain_png):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(plain_png)
        assert meta.is_georeferenced is False
        assert meta.width == 64
        assert meta.height == 64
        assert meta.crs is None

    def test_plain_jpg(self, plain_jpg):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(plain_jpg)
        assert meta.is_georeferenced is False
        assert meta.width == 64

    def test_plain_tif(self, plain_tif):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(plain_tif)
        # Plain TIFF without CRS
        assert meta.is_georeferenced is False

    def test_geotiff_detected(self, geotiff_4326):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(geotiff_4326)
        assert meta.is_georeferenced is True
        assert meta.crs is not None
        assert "4326" in meta.crs
        assert meta.epsg == 4326
        assert meta.bounds is not None
        assert meta.transform is not None

    def test_geotiff_utm(self, geotiff_32643):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(geotiff_32643)
        assert meta.is_georeferenced is True
        assert meta.epsg == 32643

    def test_geotiff_with_nodata(self, geotiff_with_nodata):
        from backend.geo.raster_inspect import inspect_raster

        meta = inspect_raster(geotiff_with_nodata)
        assert meta.is_georeferenced is True
        assert meta.nodata == -9999.0

    def test_file_not_found(self):
        from backend.geo.raster_inspect import inspect_raster

        with pytest.raises(FileNotFoundError):
            inspect_raster("/nonexistent/file.png")

    def test_unsupported_format(self, tmp_dir):
        from backend.geo.raster_inspect import inspect_raster

        path = os.path.join(tmp_dir, "test.bmp")
        with open(path, "wb") as f:
            f.write(b"BM")
        with pytest.raises(ValueError, match="Unsupported file format"):
            inspect_raster(path)


# ============================================================
# Test: GeoTIFF Metadata (Step 2)
# ============================================================


class TestGeoTIFFMetadata:
    def test_read_metadata_4326(self, geotiff_4326):
        from backend.calibration.georef import read_geotiff_metadata

        meta = read_geotiff_metadata(geotiff_4326)
        assert meta is not None
        assert meta["crs"] is not None
        assert meta["epsg"] == 4326
        assert meta["bounds_dict"]["left"] == pytest.approx(77.0, abs=0.01)
        assert meta["shape"] == (64, 64)

    def test_read_metadata_utm(self, geotiff_32643):
        from backend.calibration.georef import read_geotiff_metadata

        meta = read_geotiff_metadata(geotiff_32643)
        assert meta is not None
        assert meta["epsg"] == 32643
        # UTM bounds should be in meters
        assert meta["bounds_dict"]["left"] > 700000

    def test_read_metadata_plain_image(self, plain_png):
        from backend.calibration.georef import read_geotiff_metadata

        meta = read_geotiff_metadata(plain_png)
        assert meta is None  # No geospatial data


# ============================================================
# Test: SRTM Tile Selection (Step 4)
# ============================================================


class TestSRTMTileSelection:
    def test_tile_id_from_lat_lon(self):
        from backend.geo.srtm_provider import lat_lon_to_tile_id

        assert lat_lon_to_tile_id(28.5, 77.5) == "N28E077"
        assert lat_lon_to_tile_id(-33.9, 18.4) == "S34E018"
        assert lat_lon_to_tile_id(28.0, 77.0) == "N28E077"
        assert lat_lon_to_tile_id(-1.0, -1.0) == "S01W001"  # floor(-1) = -1

    def test_tile_bounds(self):
        from backend.geo.srtm_provider import tile_id_to_bounds

        min_lat, max_lat, min_lon, max_lon = tile_id_to_bounds("N28E077")
        assert min_lat == 28.0
        assert max_lat == 29.0
        assert min_lon == 77.0
        assert max_lon == 78.0

    def test_find_tiles(self, mock_srtm_tile):
        srtm_dir, tile_path = mock_srtm_tile
        from backend.geo.srtm_provider import find_tiles_for_aoi

        tiles = find_tiles_for_aoi(
            min_lat=28.0,
            max_lat=29.0,
            min_lon=77.0,
            max_lon=78.0,
            srtm_dir=srtm_dir,
        )
        assert len(tiles) == 1
        assert tiles[0].tile_id == "N28E077"

    def test_find_tiles_multi(self, tmp_dir):
        """Test finding tiles when AOI spans multiple tiles."""
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS

        srtm_dir = os.path.join(tmp_dir, "srtm")
        os.makedirs(srtm_dir, exist_ok=True)

        # Create two tiles
        for tile_id in ["N28E077", "N28E078"]:
            path = os.path.join(srtm_dir, f"{tile_id}.tif")
            data = np.ones((120, 120), dtype=np.float32) * 300
            if tile_id == "N28E077":
                transform = from_origin(77.0, 29.0, 1.0 / 120, 1.0 / 120)
            else:
                transform = from_origin(78.0, 29.0, 1.0 / 120, 1.0 / 120)

            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=120,
                width=120,
                count=1,
                dtype="float32",
                crs=CRS.from_epsg(4326),
                transform=transform,
            ) as dst:
                dst.write(data, 1)

        from backend.geo.srtm_provider import find_tiles_for_aoi

        tiles = find_tiles_for_aoi(
            min_lat=28.5,
            max_lat=28.9,
            min_lon=77.5,
            max_lon=78.5,
            srtm_dir=srtm_dir,
        )
        assert len(tiles) == 2
        tile_ids = {t.tile_id for t in tiles}
        assert "N28E077" in tile_ids
        assert "N28E078" in tile_ids

    def test_find_tiles_empty_dir(self, tmp_dir):
        from backend.geo.srtm_provider import find_tiles_for_aoi

        tiles = find_tiles_for_aoi(28.0, 29.0, 77.0, 78.0, tmp_dir)
        assert len(tiles) == 0

    def test_find_tiles_nonexistent_dir(self):
        from backend.geo.srtm_provider import find_tiles_for_aoi

        tiles = find_tiles_for_aoi(28.0, 29.0, 77.0, 78.0, "/nonexistent")
        assert len(tiles) == 0


# ============================================================
# Test: DEM Alignment (Step 5)
# ============================================================


class TestDEMAlignment:
    def test_align_same_crs(self, geotiff_4326, mock_srtm_tile):
        """Test aligning DEM to image grid when CRS matches."""
        import rasterio
        from backend.calibration.georef import align_dem_to_image_grid

        srtm_dir, tile_path = mock_srtm_tile

        with rasterio.open(geotiff_4326) as img_src:
            img_shape = (img_src.height, img_src.width)
            img_transform = img_src.transform
            img_crs = str(img_src.crs)

        with rasterio.open(tile_path) as dem_src:
            dem_data = dem_src.read(1).astype(np.float32)
            dem_transform = dem_src.transform
            dem_crs = str(dem_src.crs)

        aligned = align_dem_to_image_grid(
            dem_data,
            dem_transform,
            dem_crs,
            img_shape,
            img_transform,
            img_crs,
        )

        assert aligned.shape == img_shape
        assert aligned.dtype == np.float32
        # Should have some valid values
        assert np.sum(~np.isnan(aligned)) > 0


# ============================================================
# Test: Calibration (Step 6)
# ============================================================


class TestCalibration:
    def test_calibration_basic(self):
        from backend.calibration.fit import calibrate_depth_to_elevation

        # Create synthetic depth and DEM with known relationship
        # DEM = 2.0 * depth + 100
        depth = np.random.uniform(0.1, 0.9, (32, 32)).astype(np.float32)
        dem = 2.0 * depth + 100.0
        dem += np.random.normal(0, 0.5, (32, 32))  # Small noise

        result = calibrate_depth_to_elevation(depth, dem)

        # Should recover scale ≈ 2.0 and offset ≈ 100.0
        assert abs(result.scale - 2.0) < 0.5
        assert abs(result.offset - 100.0) < 5.0
        assert result.valid_samples > 0
        assert result.residual_rmse < 5.0

    def test_calibration_with_nodata(self):
        from backend.calibration.fit import calibrate_depth_to_elevation

        depth = np.random.uniform(0.1, 0.9, (32, 32)).astype(np.float32)
        dem = 3.0 * depth + 50.0
        dem[0:5, 0:5] = np.nan  # Nodata region

        result = calibrate_depth_to_elevation(depth, dem)
        assert result.valid_samples < 32 * 32  # Some pixels excluded

    def test_calibration_shape_mismatch(self):
        from backend.calibration.fit import calibrate_depth_to_elevation

        depth = np.ones((32, 32), dtype=np.float32)
        dem = np.ones((64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            calibrate_depth_to_elevation(depth, dem)

    def test_calibration_insufficient_data(self):
        from backend.calibration.fit import calibrate_depth_to_elevation

        depth = np.ones((4, 4), dtype=np.float32)
        dem = np.full((4, 4), np.nan, dtype=np.float32)

        with pytest.raises(ValueError, match="Not enough valid"):
            calibrate_depth_to_elevation(depth, dem)

    def test_apply_calibration(self):
        from backend.calibration.fit import apply_calibration

        depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = apply_calibration(depth, scale=100.0, offset=200.0)
        expected = np.array([[200.0, 250.0, 300.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


# ============================================================
# Test: Nodata Handling (Step 7)
# ============================================================


class TestNodataHandling:
    def test_metrics_with_nodata(self):
        from eval.metrics import compute_metrics

        pred = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        ref = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        metrics = compute_metrics(pred, ref)
        assert metrics["valid_pixels"] == 5  # One NaN excluded
        assert metrics["rmse"] is not None
        assert metrics["coverage_pct"] < 100.0

    def test_metrics_all_valid(self):
        from eval.metrics import compute_metrics

        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])

        metrics = compute_metrics(pred, ref)
        assert metrics["rmse"] == 0.0
        assert metrics["correlation"] == 1.0
        assert metrics["coverage_pct"] == 100.0

    def test_metrics_shape_mismatch(self):
        from eval.metrics import compute_metrics

        pred = np.ones((32, 32))
        ref = np.ones((64, 64))

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_metrics(pred, ref)


# ============================================================
# Test: Export (Step 8)
# ============================================================


class TestExport:
    def test_save_relative_png(self, tmp_dir):
        from backend.calibration.pipeline import save_output

        array = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
        path = os.path.join(tmp_dir, "relative.png")
        save_output(path, array, metadata=None, mode="relative")

        assert os.path.exists(path)
        from PIL import Image

        img = Image.open(path)
        assert img.size == (64, 64)

    def test_save_metric_geotiff(self, geotiff_4326, tmp_dir):
        from backend.calibration.pipeline import save_output
        from backend.calibration.georef import read_geotiff_metadata

        meta = read_geotiff_metadata(geotiff_4326)
        array = np.random.uniform(100, 500, (64, 64)).astype(np.float32)

        path = os.path.join(tmp_dir, "metric.tif")
        metadata_dict = {
            "crs": meta["crs_string"],
            "transform": [
                meta["transform"].a,
                meta["transform"].b,
                meta["transform"].c,
                meta["transform"].d,
                meta["transform"].e,
                meta["transform"].f,
            ],
        }
        save_output(path, array, metadata_dict, mode="metric")

        assert os.path.exists(path)

        # Round-trip test
        import rasterio

        with rasterio.open(path) as src:
            assert src.crs is not None
            assert src.width == 64
            assert src.height == 64
            written = src.read(1)
            np.testing.assert_array_almost_equal(written, array, decimal=1)


# ============================================================
# Test: API Response Structure (Step 9)
# ============================================================


class TestAPIResponse:
    def test_reconstruction_result_relative(self):
        from backend.models import (
            ReconstructionResult,
            RasterMetadata,
            ElevationMode,
            CalibrationMetadata,
            CalibrationSource,
        )
        import numpy as np

        input_meta = RasterMetadata(
            is_georeferenced=False,
            file_path="test.png",
            width=64,
            height=64,
            band_count=3,
            dtype="uint8",
        )

        result = ReconstructionResult(
            mode=ElevationMode.RELATIVE,
            is_georeferenced=False,
            input_raster=input_meta,
            elevation_array=np.zeros((64, 64)),
            heightmap_array=np.zeros((64, 64)),
        )

        response = result.to_api_response()
        assert response["mode"] == "relative"
        assert response["is_georeferenced"] is False
        assert response["calibration"]["applied"] is False
        assert "relative" in response["units"].lower()

    def test_reconstruction_result_absolute(self):
        from backend.models import (
            ReconstructionResult,
            RasterMetadata,
            ElevationMode,
            CalibrationMetadata,
            CalibrationSource,
        )
        import numpy as np

        input_meta = RasterMetadata(
            is_georeferenced=True,
            file_path="test.tif",
            width=64,
            height=64,
            band_count=1,
            dtype="float32",
            crs="EPSG:4326",
            epsg=4326,
        )

        cal = CalibrationMetadata(
            applied=True,
            source=CalibrationSource.SRTM,
            scale=2.5,
            offset=100.0,
            valid_samples=4096,
        )

        result = ReconstructionResult(
            mode=ElevationMode.ABSOLUTE,
            is_georeferenced=True,
            input_raster=input_meta,
            elevation_array=np.ones((64, 64)) * 300,
            heightmap_array=np.ones((64, 64)),
            crs="EPSG:4326",
            calibration=cal,
        )

        response = result.to_api_response()
        assert response["mode"] == "absolute"
        assert response["calibration"]["applied"] is True
        assert response["calibration"]["scale"] == 2.5
        assert "meter" in response["units"].lower()


# ============================================================
# Test: SRTM Provider Integration
# ============================================================


class TestSRTMProvider:
    def test_provider_with_tiles(self, mock_srtm_tile):
        from backend.geo.srtm_provider import SRTMProvider

        srtm_dir, _ = mock_srtm_tile
        provider = SRTMProvider(srtm_dir=srtm_dir)

        aoi = {"left": 77.0, "bottom": 28.0, "right": 78.0, "top": 29.0}
        dem, meta, error = provider.get_dem(aoi, (64, 64))

        assert error is None
        assert dem is not None
        assert dem.shape == (64, 64)
        assert meta is not None
        assert "N28E077" in meta["tile_ids"]

    def test_provider_no_tiles(self, tmp_dir):
        from backend.geo.srtm_provider import SRTMProvider

        provider = SRTMProvider(srtm_dir=tmp_dir)
        aoi = {"left": 77.0, "bottom": 28.0, "right": 78.0, "top": 29.0}
        dem, meta, error = provider.get_dem(aoi, (64, 64))

        assert dem is None
        assert error is not None
        assert "No SRTM tiles" in error

    def test_provider_list_tiles(self, mock_srtm_tile):
        from backend.geo.srtm_provider import SRTMProvider

        srtm_dir, _ = mock_srtm_tile
        provider = SRTMProvider(srtm_dir=srtm_dir)
        tiles = provider.get_available_tiles()
        assert "N28E077" in tiles
