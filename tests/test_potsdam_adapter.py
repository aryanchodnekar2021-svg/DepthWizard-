"""
Tests for Potsdam dataset adapter and real benchmark pipeline.
"""

import os
import tempfile
import numpy as np
import pytest


class TestPotsdamAdapter:
    """Test Potsdam adapter discovery, inspection, and loading."""

    def test_parse_tile_id_standard(self):
        from backend.geo.potsdam_adapter import _parse_tile_id

        assert _parse_tile_id("top_potsdam_2_10_RGB.tif") == (2, 10)
        assert _parse_tile_id("top_potsdam_5_12_DSM.tif") == (5, 12)
        assert _parse_tile_id("top_potsdam_7_15_RGBIR.tif") == (7, 15)

    def test_parse_tile_id_simple(self):
        from backend.geo.potsdam_adapter import _parse_tile_id

        assert _parse_tile_id("2_10") == (2, 10)
        assert _parse_tile_id("6_7.tif") == (6, 7)

    def test_parse_tile_id_invalid(self):
        from backend.geo.potsdam_adapter import _parse_tile_id

        assert _parse_tile_id("random_file.tif") is None
        assert _parse_tile_id("top_potsdam_rgb.tif") is None

    def test_train_test_tiles(self):
        from backend.geo.potsdam_adapter import TRAIN_TILES, TEST_TILES

        assert len(TRAIN_TILES) == 24
        assert len(TEST_TILES) == 14
        assert len(set(TRAIN_TILES) & set(TEST_TILES)) == 0  # no overlap

    def test_discover_empty_dir(self):
        from backend.geo.potsdam_adapter import discover_potsdam_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            info = discover_potsdam_tiles(tmpdir)
            assert info.exists is True
            assert info.total_tiles == 0

    def test_discover_nonexistent_dir(self):
        from backend.geo.potsdam_adapter import discover_potsdam_tiles

        info = discover_potsdam_tiles("/nonexistent/path")
        assert info.exists is False
        assert len(info.errors) > 0

    def test_discover_with_mock_tiles(self):
        from backend.geo.potsdam_adapter import discover_potsdam_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock TIFF files
            import rasterio
            from rasterio.transform import from_bounds

            for tile_id in ["2_10", "2_11"]:
                # Create RGB mock
                rgb_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_RGB.tif")
                transform = from_bounds(360000, 5800000, 360300, 5800300, 100, 100)
                with rasterio.open(
                    rgb_path,
                    "w",
                    driver="GTiff",
                    height=100,
                    width=100,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:32633",
                    transform=transform,
                ) as dst:
                    dst.write(np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8))

                # Create DSM mock
                dsm_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_DSM.tif")
                with rasterio.open(
                    dsm_path,
                    "w",
                    driver="GTiff",
                    height=100,
                    width=100,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32633",
                    transform=transform,
                    nodata=-9999,
                ) as dst:
                    dst.write(
                        np.random.uniform(40, 80, (1, 100, 100)).astype(np.float32)
                    )

            info = discover_potsdam_tiles(tmpdir)
            assert info.total_tiles == 2
            assert info.tiles_with_rgb == 2
            assert info.tiles_with_dsm == 2
            assert info.tiles_with_both == 2

    def test_inspect_tile(self):
        from backend.geo.potsdam_adapter import PotsdamTileInfo, inspect_tile

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio
            from rasterio.transform import from_bounds

            tile_id = "3_10"
            rgb_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_RGB.tif")
            dsm_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_DSM.tif")
            transform = from_bounds(360000, 5800000, 360300, 5800300, 100, 100)

            with rasterio.open(
                rgb_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype="uint8",
                crs="EPSG:32633",
                transform=transform,
            ) as dst:
                dst.write(np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8))

            with rasterio.open(
                dsm_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype="float32",
                crs="EPSG:32633",
                transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(np.random.uniform(40, 80, (1, 100, 100)).astype(np.float32))

            tile = PotsdamTileInfo(
                tile_id=tile_id,
                row=3,
                col=10,
                split="train",
                rgb_path=rgb_path,
                dsm_path=dsm_path,
            )
            tile = inspect_tile(tile)

            assert tile.rgb_shape == (100, 100, 3)
            assert tile.dsm_shape == (100, 100, 1)
            assert tile.rgb_crs == "EPSG:32633"
            assert tile.dsm_crs == "EPSG:32633"
            assert tile.crs_match is True
            assert tile.rgb_dsm_match is True
            assert tile.dsm_elevation_min is not None
            assert tile.dsm_elevation_max is not None

    def test_load_tile_rgb(self):
        from backend.geo.potsdam_adapter import PotsdamTileInfo, load_tile_rgb

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio
            from rasterio.transform import from_bounds

            tile_id = "4_10"
            rgb_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_RGB.tif")
            transform = from_bounds(360000, 5800000, 360300, 5800300, 100, 100)

            with rasterio.open(
                rgb_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype="uint8",
                crs="EPSG:32633",
                transform=transform,
            ) as dst:
                dst.write(np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8))

            tile = PotsdamTileInfo(
                tile_id=tile_id,
                row=4,
                col=10,
                split="train",
                rgb_path=rgb_path,
            )
            rgb = load_tile_rgb(tile)
            assert rgb is not None
            assert rgb.shape == (100, 100, 3)
            assert rgb.dtype == np.uint8

    def test_load_tile_dsm(self):
        from backend.geo.potsdam_adapter import PotsdamTileInfo, load_tile_dsm

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio
            from rasterio.transform import from_bounds

            tile_id = "5_10"
            dsm_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_DSM.tif")
            transform = from_bounds(360000, 5800000, 360300, 5800300, 100, 100)

            with rasterio.open(
                dsm_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype="float32",
                crs="EPSG:32633",
                transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(np.random.uniform(40, 80, (1, 100, 100)).astype(np.float32))

            tile = PotsdamTileInfo(
                tile_id=tile_id,
                row=5,
                col=10,
                split="train",
                dsm_path=dsm_path,
                dsm_nodata=-9999,
            )
            dsm = load_tile_dsm(tile)
            assert dsm is not None
            assert dsm.shape == (100, 100)
            assert dsm.dtype == np.float64

    def test_load_tile_rgb_with_resize(self):
        from backend.geo.potsdam_adapter import PotsdamTileInfo, load_tile_rgb

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio
            from rasterio.transform import from_bounds

            tile_id = "6_7"
            rgb_path = os.path.join(tmpdir, f"top_potsdam_{tile_id}_RGB.tif")
            transform = from_bounds(360000, 5800000, 360300, 5800300, 200, 200)

            with rasterio.open(
                rgb_path,
                "w",
                driver="GTiff",
                height=200,
                width=200,
                count=3,
                dtype="uint8",
                crs="EPSG:32633",
                transform=transform,
            ) as dst:
                dst.write(np.random.randint(0, 255, (3, 200, 200), dtype=np.uint8))

            tile = PotsdamTileInfo(
                tile_id=tile_id,
                row=6,
                col=7,
                split="train",
                rgb_path=rgb_path,
            )
            rgb = load_tile_rgb(tile, max_size=100)
            assert rgb is not None
            assert max(rgb.shape[:2]) <= 100


class TestPotsdamMetadata:
    """Test Potsdam tile metadata and validation."""

    def test_tile_info_defaults(self):
        from backend.geo.potsdam_adapter import PotsdamTileInfo

        tile = PotsdamTileInfo(
            tile_id="2_10",
            row=2,
            col=10,
            split="train",
        )
        assert tile.tile_id == "2_10"
        assert tile.split == "train"
        assert tile.rgb_path is None
        assert tile.dsm_path is None
        assert tile.warnings == []
        assert tile.errors == []

    def test_dataset_info_defaults(self):
        from backend.geo.potsdam_adapter import PotsdamDatasetInfo

        info = PotsdamDatasetInfo(root="/tmp", exists=True)
        assert info.exists is True
        assert info.total_tiles == 0
        assert info.tiles == []


class TestPotsdamIntegration:
    """Integration tests requiring actual data."""

    @pytest.mark.skipif(
        not os.path.isdir("data/potsdam/prepared"), reason="Potsdam data not prepared"
    )
    def test_prepared_tiles_exist(self):
        from pathlib import Path

        tiles = list(Path("data/potsdam/prepared").glob("tile_*.npz"))
        assert len(tiles) > 0

    @pytest.mark.skipif(
        not os.path.isdir("data/potsdam/prepared"), reason="Potsdam data not prepared"
    )
    def test_load_prepared_tile(self):
        from pathlib import Path

        tiles = list(Path("data/potsdam/prepared").glob("tile_*.npz"))
        if tiles:
            data = np.load(str(tiles[0]), allow_pickle=True)
            assert "rgb" in data
            assert "dsm" in data
            assert data["rgb"].ndim == 3
            assert data["dsm"].ndim == 2


class TestRealBenchmarkConfig:
    """Test real benchmark configuration."""

    def test_config_loads(self):
        import yaml

        with open("configs/real_benchmark.yaml", "r") as f:
            config = yaml.safe_load(f)
        assert "methods" in config
        assert "data_dir" in config
        assert config["model"] == "depth-anything/Depth-Anything-V2-Small-hf"

    def test_all_methods_enabled(self):
        import yaml

        with open("configs/real_benchmark.yaml", "r") as f:
            config = yaml.safe_load(f)
        for method_name, method_cfg in config["methods"].items():
            assert method_cfg.get("enabled", True) is True, (
                f"Method {method_name} not enabled"
            )

    def test_spatial_cv_config(self):
        import yaml

        with open("configs/real_benchmark.yaml", "r") as f:
            config = yaml.safe_load(f)
        sc = config["spatial_cv"]
        assert sc["n_folds"] >= 2
        assert sc["block_size_px"] >= 16
        assert sc["min_separation_blocks"] >= 1


class TestRealBenchmarkRunner:
    """Test the real benchmark runner."""

    def test_runner_imports(self):
        import eval.run_real_benchmark

        assert hasattr(eval.run_real_benchmark, "main")
        assert hasattr(eval.run_real_benchmark, "run_single_potsdam_experiment")

    def test_runner_main_exists(self):
        from eval.run_real_benchmark import main

        assert callable(main)
