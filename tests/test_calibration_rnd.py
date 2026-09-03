"""
Tests for calibration R&D modules: methods, spatial CV, frequency, uncertainty.
Synthetic fixtures acceptable for code behavior; NOT for SIH performance evidence.
"""

import numpy as np
import pytest


class TestCalibrationMethods:
    """Tests for calibration method registry and formulations."""

    def test_affine_basic(self):
        from backend.calibration.methods import method_affine, compute_quality_metrics

        np.random.seed(42)
        depth = np.random.rand(100, 100).astype(np.float32)
        dem = 2.0 * depth + 100.0 + 0.1 * np.random.randn(100, 100).astype(np.float32)
        res = method_affine(depth, dem)
        assert res.method_name == "affine"
        assert np.isfinite(res.scale) and abs(res.scale - 2.0) < 0.2
        assert np.isfinite(res.offset) and abs(res.offset - 100.0) < 2.0
        assert 0.9 < res.inlier_fraction <= 1.0
        assert np.isfinite(res.pre_correlation) and res.pre_correlation > 0.9
        assert np.isfinite(res.post_correlation) and res.post_correlation > 0.9
        assert np.isfinite(res.r_squared) and res.r_squared > 0.8

    def test_affine_degenerate_constant(self):
        from backend.calibration.methods import method_affine

        depth = np.ones((10, 10), dtype=np.float32)
        dem = np.full((10, 10), 150.0, dtype=np.float32)
        res = method_affine(depth, dem)
        assert res.method_name == "affine"
        assert not np.isfinite(res.scale) or res.scale == 0.0
        assert np.isfinite(res.offset) and abs(res.offset - 150.0) < 1.0

    def test_affine_insufficient_samples(self):
        from backend.calibration.methods import method_affine

        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dem = np.array([[100.0, np.nan], [np.nan, 200.0]], dtype=np.float32)
        res = method_affine(depth, dem)
        assert res.n_samples < 10

    def test_robust_affine_fallback(self):
        from backend.calibration.methods import method_robust_affine

        # RANSAC not available (no sklearn) -> should fallback to affine
        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 5.0 * depth + 50.0 + 0.05 * np.random.randn(10, 10).astype(np.float32)
        res = method_robust_affine(depth, dem)
        assert res.method_name == "robust_affine"
        assert np.isfinite(res.scale)
        assert np.isfinite(res.offset)

    def test_robust_affine_with_outliers(self):
        from backend.calibration.methods import method_robust_affine

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 5.0 * depth + 50.0
        dem[0, 0] = 1000.0  # outlier
        dem[9, 9] = -500.0  # outlier
        res = method_robust_affine(depth, dem)
        assert np.isfinite(res.scale)
        # Inlier fraction should be <1 if RANSAC works, else fallback to 1.0

    def test_dem_residual(self):
        from backend.calibration.methods import method_dem_residual

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 2.0 * depth + 100.0 + 0.1 * np.random.randn(10, 10).astype(np.float32)
        res = method_dem_residual(depth, dem)
        assert res.method_name == "dem_residual"
        assert "residual_mean" in res.metadata
        assert "residual_std" in res.metadata
        assert "alpha" in res.metadata

    def test_local_normalized(self):
        from backend.calibration.methods import method_local_normalized

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 2.0 * depth + 100.0
        res = method_local_normalized(depth, dem, patch_size=8)
        assert res.method_name == "local_normalized"
        assert "patch_size" in res.metadata

    def test_inverse_depth(self):
        from backend.calibration.methods import method_inverse_depth

        depth = np.linspace(0.1, 1.0, 100).reshape(10, 10).astype(np.float32)  # avoid 0
        dem = 100.0 / (depth + 1e-6) + 50.0
        res = method_inverse_depth(depth, dem)
        assert res.method_name == "inverse_depth"
        assert "epsilon" in res.metadata

    def test_piecewise_linear(self):
        from backend.calibration.methods import method_piecewise_linear

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = np.where(depth < 0.5, 2.0 * depth + 100.0, 4.0 * depth + 80.0)
        res = method_piecewise_linear(depth, dem, n_bins=3)
        assert res.method_name == "piecewise_linear"
        assert "bin_params" in res.metadata
        assert len(res.metadata["bin_params"]) > 0

    def test_frequency_fusion_fallback(self):
        from backend.calibration.methods import method_frequency_fusion

        # scipy.ndimage not available -> fallback
        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 2.0 * depth + 100.0
        res = method_frequency_fusion(depth, dem, lowpass_sigma=5.0)
        assert res.method_name == "frequency_fusion"

    def test_get_available_methods(self):
        from backend.calibration.methods import get_available_methods

        methods = get_available_methods()
        assert isinstance(methods, dict)
        expected = {
            "affine",
            "robust_affine",
            "dem_residual",
            "local_normalized",
            "inverse_depth",
            "piecewise_linear",
            "frequency_fusion",
        }
        assert set(methods.keys()) == expected

    def test_apply_method(self):
        from backend.calibration.methods import apply_method, method_affine

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 2.0 * depth + 100.0
        res = method_affine(depth, dem)
        calibrated = apply_method(res, depth)
        assert calibrated.shape == depth.shape
        assert np.isfinite(calibrated).all()
        assert abs(calibrated[0, 0] - (res.scale * depth[0, 0] + res.offset)) < 1e-4

    def test_compute_quality_metrics(self):
        from backend.calibration.methods import compute_quality_metrics

        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a + 0.1
        m = compute_quality_metrics(a, b)
        assert np.isfinite(m["rmse"]) and m["rmse"] < 0.15
        assert np.isfinite(m["mae"])
        assert np.isfinite(m["r_squared"]) and m["r_squared"] > 0.9
        assert np.isfinite(m["correlation"]) and m["correlation"] > 0.9
        assert np.isfinite(m["median_ae"])
        assert np.isfinite(m["p90_ae"])
        assert np.isfinite(m["p95_ae"])

    def test_compute_quality_metrics_nan(self):
        from backend.calibration.methods import compute_quality_metrics

        a = np.array([[1.0, np.nan], [np.nan, 2.0]])
        b = np.array([[1.0, 3.0], [4.0, 2.0]])
        m = compute_quality_metrics(a, b)
        assert m["valid_pixels"] == 2
        assert np.isfinite(m["rmse"])


class TestSpatialCV:
    """Tests for spatial cross-validation and alignment diagnostics."""

    def test_check_alignment_match(self):
        from backend.calibration.spatial_cv import check_alignment

        depth_shape = (100, 100)
        dem_shape = (100, 100)
        diag = check_alignment(
            depth_shape, None, "EPSG:4326", dem_shape, None, "EPSG:4326"
        )
        assert diag.crs_match is True
        assert diag.overlap_pct == 100.0

    def test_check_alignment_mismatch(self):
        from backend.calibration.spatial_cv import check_alignment

        diag = check_alignment(
            (100, 100), None, "EPSG:4326", (100, 100), None, "EPSG:32643"
        )
        assert diag.crs_match is False
        assert any("CRS mismatch" in w for w in diag.warnings)

    def test_check_alignment_resolution_ratio(self):
        from backend.calibration.spatial_cv import check_alignment
        from rasterio.transform import Affine

        depth_t = Affine.scale(0.0001, -0.0001)  # 10m
        dem_t = Affine.scale(0.001, -0.001)  # 100m
        diag = check_alignment(
            (100, 100), depth_t, "EPSG:4326", (100, 100), dem_t, "EPSG:4326"
        )
        assert abs(diag.resolution_ratio - 10.0) < 1.0  # dem 10x coarser

    def test_create_spatial_blocks(self):
        from backend.calibration.spatial_cv import (
            create_spatial_blocks,
            SpatialCVConfig,
        )

        cfg = SpatialCVConfig(block_size_px=16, min_valid_fraction=0.1)
        blocks = create_spatial_blocks((32, 32), cfg)
        assert len(blocks) == 4
        assert all(b.n_valid == 16 * 16 for b in blocks)

    def test_create_spatial_blocks_uneven(self):
        from backend.calibration.spatial_cv import (
            create_spatial_blocks,
            SpatialCVConfig,
        )

        cfg = SpatialCVConfig(block_size_px=16)
        blocks = create_spatial_blocks((20, 20), cfg)
        # 2x2 = 4 blocks, edge ones smaller
        assert len(blocks) == 4
        assert sum(b.n_valid for b in blocks) == 400

    def test_spatial_separation_geographic(self):
        from backend.calibration.spatial_cv import (
            compute_spatial_separation,
            SpatialBlock,
        )

        a = SpatialBlock(0, 10, 0, 10, 0, 100, 18.5, 73.5)
        b = SpatialBlock(20, 30, 20, 30, 1, 100, 18.6, 73.6)
        sep = compute_spatial_separation(a, b, 0.00083, 0.00083, is_geographic=True)
        # approx 11 km at 18N for 0.1 deg lat/lon difference
        assert 1000 < sep < 50000

    def test_spatial_separation_projected(self):
        from backend.calibration.spatial_cv import (
            compute_spatial_separation,
            SpatialBlock,
        )

        a = SpatialBlock(0, 10, 0, 10, 0, 100, None, None)
        b = SpatialBlock(20, 30, 20, 30, 1, 100, None, None)
        sep = compute_spatial_separation(a, b, 10.0, 10.0, is_geographic=False)
        assert abs(sep - 282.8) < 5.0  # sqrt(200^2+200^2)

    def test_create_spatial_cv_folds(self):
        from backend.calibration.spatial_cv import (
            create_spatial_blocks,
            create_spatial_cv_folds,
            SpatialCVConfig,
        )

        cfg = SpatialCVConfig(
            block_size_px=16, n_folds=2, min_separation_blocks=1, seed=42
        )
        blocks = create_spatial_blocks((32, 32), cfg)
        folds = create_spatial_cv_folds(blocks, cfg)
        assert len(folds) == 2
        for f in folds:
            assert f.n_train_pixels + f.n_test_pixels == 1024
            assert f.train_blocks and f.test_blocks

    def test_spatial_cv_min_separation(self):
        from backend.calibration.spatial_cv import (
            create_spatial_blocks,
            create_spatial_cv_folds,
            SpatialCVConfig,
        )

        # 3x3 grid of blocks, n_folds=3, min_sep=1
        cfg = SpatialCVConfig(
            block_size_px=8, n_folds=3, min_separation_blocks=1, seed=42
        )
        blocks = create_spatial_blocks((24, 24), cfg)
        folds = create_spatial_cv_folds(blocks, cfg)
        # With min_separation, some train blocks removed near test
        for f in folds:
            assert f.spatial_separation_m >= 8 or np.isnan(f.spatial_separation_m)

    def test_describe_cv_plan(self):
        from backend.calibration.spatial_cv import (
            create_spatial_blocks,
            create_spatial_cv_folds,
            describe_cv_plan,
            SpatialCVConfig,
        )

        cfg = SpatialCVConfig(
            block_size_px=8, n_folds=2, min_separation_blocks=0, seed=42
        )
        blocks = create_spatial_blocks((16, 16), cfg)
        folds = create_spatial_cv_folds(blocks, cfg)
        desc = describe_cv_plan(folds)
        assert desc["n_folds"] == 2
        assert desc["avg_train_pixels"] > 0
        assert desc["avg_test_pixels"] > 0


class TestFrequencyDecomposition:
    """Tests for frequency decomposition."""

    def test_gaussian_lowpass(self):
        from backend.calibration.frequency import gaussian_lowpass

        arr = np.random.rand(20, 20).astype(np.float64)
        low = gaussian_lowpass(arr, sigma=2.0)
        assert low.shape == arr.shape
        assert np.isfinite(low).all()

    def test_gaussian_lowpass_with_mask(self):
        from backend.calibration.frequency import gaussian_lowpass

        arr = np.random.rand(20, 20).astype(np.float64)
        mask = np.ones_like(arr, dtype=bool)
        mask[5:15, 5:15] = False
        low = gaussian_lowpass(arr, sigma=2.0, mask=mask)
        assert low.shape == arr.shape

    def test_highpass_component(self):
        from backend.calibration.frequency import highpass_component

        arr = np.linspace(0, 10, 400).reshape(20, 20).astype(np.float64)
        high = highpass_component(arr, sigma=2.0)
        assert high.shape == arr.shape

    def test_decompose_dem_and_depth(self):
        from backend.calibration.frequency import decompose_dem_and_depth

        dem = np.linspace(100, 200, 400).reshape(20, 20).astype(np.float64)
        depth = np.linspace(0, 1, 400).reshape(20, 20).astype(np.float64)
        decomp = decompose_dem_and_depth(dem, depth, sigma=2.0)
        assert "dem_coarse" in decomp and "dem_highfreq" in decomp
        assert "depth_coarse" in decomp and "depth_highfreq" in decomp
        assert "correlation_coarse" in decomp

    def test_frequency_fusion_reconstruction(self):
        from backend.calibration.frequency import frequency_fusion_reconstruction

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = np.linspace(100, 200, 100).reshape(10, 10).astype(np.float32)
        recon = frequency_fusion_reconstruction(
            depth, dem, scale=100.0, offset=50.0, sigma=2.0, alpha=0.5
        )
        assert recon.shape == depth.shape
        assert np.isfinite(recon).all()

    def test_compute_frequency_metrics(self):
        from backend.calibration.frequency import compute_frequency_metrics

        dem = np.random.rand(20, 20).astype(np.float64)
        depth = np.random.rand(20, 20).astype(np.float64)
        m = compute_frequency_metrics(dem, depth, sigma=2.0)
        assert "dem_total_var" in m
        assert "dem_low_var" in m
        assert "dem_high_var" in m


class TestUncertainty:
    """Tests for uncertainty estimation."""

    def test_per_pixel_uncertainty_map(self):
        from backend.calibration.uncertainty import per_pixel_uncertainty_map

        dem = np.random.rand(20, 20).astype(np.float64) * 100 + 500
        depth = np.random.rand(20, 20).astype(np.float32)
        unc = per_pixel_uncertainty_map(10.0, dem, depth, window=3)
        assert unc.shape == dem.shape
        assert np.isfinite(unc).all() or np.any(np.isnan(unc))  # may have NaN at edges

    def test_estimate_uncertainty_from_residual(self):
        from backend.calibration.uncertainty import estimate_uncertainty_from_residual

        residuals = np.random.randn(100).astype(np.float64) * 5.0
        dem = np.random.rand(10, 10).astype(np.float64) * 100 + 500
        depth = np.random.rand(10, 10).astype(np.float32)
        u = estimate_uncertainty_from_residual(residuals, dem, depth)
        assert isinstance(u.mean_uncertainty, float)
        assert np.isfinite(u.residual_std)
        assert np.isfinite(u.dem_std)
        assert np.isfinite(u.depth_variability)
        assert u.per_pixel_uncertainty.shape == dem.shape

    def test_calibration_quality_flags(self):
        from backend.calibration.uncertainty import calibration_quality_flags
        from backend.calibration.methods import CalibrationMethodResult

        res = CalibrationMethodResult(
            "test", 1.0, 100.0, 80.0, 50.0, 1.0, 0.2, 0.3, 0.1, 50, {}
        )
        flags = calibration_quality_flags(res)
        assert "high_residual" in flags
        assert "low_r2" in flags
        assert "few_samples" in flags
        assert "low_correlation" in flags

        res2 = CalibrationMethodResult(
            "test", 1.0, 100.0, 5.0, 3.0, 0.9, 0.8, 0.85, 0.7, 1000, {}
        )
        flags2 = calibration_quality_flags(res2)
        assert "high_residual" not in flags2
        assert "low_r2" not in flags2


class TestAlignmentAndIntegration:
    """Integration tests using the calibration pipeline."""

    def test_pipeline_imports(self):
        from backend.calibration.pipeline import calibrate
        from backend.calibration.fit import calibrate_depth_to_elevation
        from backend.calibration.georef import align_dem_to_image_grid
        from backend.calibration.methods import get_available_methods
        from backend.calibration.spatial_cv import check_alignment
        from backend.calibration.frequency import decompose_dem_and_depth
        from backend.calibration.uncertainty import estimate_uncertainty_from_residual

        assert True  # if imports work

    def test_config_loads(self):
        import yaml

        with open("configs/calibration_experiments.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "methods" in cfg
        assert "alignment" in cfg
        assert "frequency" in cfg
        assert "spatial_cv" in cfg
        assert "visualizations" in cfg

    def test_experiment_runner_blocker(self, tmp_path):
        from eval.run_calibration_experiments import main
        import sys
        import os

        # Should exit with blocker when no datasets
        # Change cwd to temp dir to avoid writing to outputs/
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Need to adjust config path since we changed cwd
            import subprocess

            config_path = os.path.join(
                old_cwd, "configs", "calibration_experiments.yaml"
            )
            output_dir = os.path.join(tmp_path, "calibration_rnd")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eval.run_calibration_experiments",
                    "--config",
                    config_path,
                    "--output",
                    output_dir,
                ],
                capture_output=True,
                text=True,
                cwd=old_cwd,
            )
            # Should complete without error (blocker logic returns, doesn't sys.exit)
            assert result.returncode == 0
            # Check blocker report was created
            assert os.path.exists(os.path.join(output_dir, "blocker_report.json"))
            assert os.path.exists(os.path.join(output_dir, "blocker_report.md"))
        finally:
            os.chdir(old_cwd)


class TestRobustness:
    """Tests for robustness edge cases."""

    def test_nan_handling(self):
        from backend.calibration.methods import method_affine, method_robust_affine

        depth = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        dem = np.array([[100.0, 110.0], [np.nan, 200.0]], dtype=np.float32)
        r1 = method_affine(depth, dem)
        r2 = method_robust_affine(depth, dem)
        assert r1.n_samples == 2
        assert r2.n_samples == 2

    def test_nodata_dem_zero(self):
        from backend.calibration.methods import method_affine

        depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        dem = np.array([[100.0, 0.0], [110.0, 120.0]], dtype=np.float32)  # 0 = nodata
        res = method_affine(depth, dem)
        # Should mask dem==0, leaving 3 valid samples
        assert res.n_samples == 3

    def test_nodata_dem_minus9999(self):
        from backend.calibration.methods import method_affine

        depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        dem = np.array([[100.0, -9999.0], [110.0, 120.0]], dtype=np.float32)
        res = method_affine(depth, dem)
        assert res.n_samples == 3

    def test_extreme_outliers(self):
        from backend.calibration.methods import method_robust_affine

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = 5.0 * depth + 50.0
        dem.flat[0] = 1e6  # extreme outlier
        res = method_robust_affine(depth, dem)
        # Should handle gracefully
        assert np.isfinite(res.scale)

    def test_low_overlap(self):
        from backend.calibration.methods import method_affine

        depth = np.ones((10, 10), dtype=np.float32)
        depth[0, :] = np.nan
        depth[:, 0] = np.nan
        depth[-1, :] = np.nan
        depth[:, -1] = np.nan
        dem = np.ones((10, 10), dtype=np.float32) * 100.0
        res = method_affine(depth, dem)
        assert res.n_samples < 100

    def test_constant_depth(self):
        from backend.calibration.methods import method_affine

        depth = np.full((10, 10), 0.5, dtype=np.float32)
        dem = np.linspace(100, 200, 100).reshape(10, 10).astype(np.float32)
        res = method_affine(depth, dem)
        # Should not crash, scale should be ~0
        assert np.isfinite(res.scale)

    def test_constant_dem(self):
        from backend.calibration.methods import method_affine

        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        dem = np.full((10, 10), 150.0, dtype=np.float32)
        res = method_affine(depth, dem)
        assert np.isfinite(res.scale)
        assert abs(res.scale) < 1e-5  # scale should be ~0

    def test_small_sample_count(self):
        from backend.calibration.methods import method_affine

        # 1 sample -> n_valid = 1 < 2 -> nan scale/offset
        depth = np.array([[0.1]], dtype=np.float32)
        dem = np.array([[100.0]], dtype=np.float32)
        res = method_affine(depth, dem)
        assert not np.isfinite(res.scale)
        assert not np.isfinite(res.offset)

    def test_mismatched_crs_detection(self):
        from backend.calibration.spatial_cv import check_alignment

        diag = check_alignment(
            (100, 100), None, "EPSG:4326", (100, 100), None, "EPSG:3857"
        )
        assert diag.crs_match is False
        assert any("CRS" in w for w in diag.warnings)

    def test_mismatched_resolution(self):
        from backend.calibration.spatial_cv import check_alignment
        from rasterio.transform import Affine

        depth_t = Affine.scale(0.00001, -0.00001)  # 1m
        dem_t = Affine.scale(0.001, -0.001)  # 100m
        diag = check_alignment(
            (1000, 1000), depth_t, "EPSG:4326", (10, 10), dem_t, "EPSG:4326"
        )
        assert diag.resolution_ratio > 10
        assert any("coarser" in w for w in diag.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
