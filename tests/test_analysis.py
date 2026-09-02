"""
Tests for the analysis modules: slope, comparison, and classification.

All tests use synthetic deterministic data — no model download required.
Slope tests verify against analytically known results for planar surfaces.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analysis.slope import compute_slope, compute_slope_percent
from backend.analysis.comparison import compute_error_map
from backend.analysis.classification import (
    classify_terrain,
    get_category_name,
    get_category_legend,
    TERRAIN_CATEGORIES,
)


# ============================================================
# Slope tests
# ============================================================


class TestComputeSlope:
    """Tests for compute_slope()."""

    def test_flat_surface(self):
        """A flat surface should have slope = 0 everywhere."""
        flat = np.ones((10, 10)) * 5.0
        slope = compute_slope(flat, cell_size_x=1.0, cell_size_y=1.0)
        assert slope.shape == flat.shape
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)

    def test_planar_surface_dx(self):
        """
        Planar surface z = 3*x (slope in x-direction only).

        dz/dx = 3, dz/dy = 0
        slope = arctan(sqrt(3^2 + 0^2)) = arctan(3) ≈ 71.565 degrees
        """
        rows, cols = 20, 20
        x = np.arange(cols, dtype=np.float64)[np.newaxis, :]
        z = np.broadcast_to(3.0 * x, (rows, cols)).copy().astype(np.float64)

        expected_deg = np.degrees(np.arctan(3.0))
        slope = compute_slope(z, cell_size_x=1.0, cell_size_y=1.0)

        # Interior pixels (not boundary) should match expected
        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(interior, expected_deg, atol=1.0)

    def test_planar_surface_dy(self):
        """
        Planar surface z = 5*y (slope in y-direction only).

        dz/dy = 5, dz/dx = 0
        slope = arctan(5) ≈ 78.690 degrees
        """
        rows, cols = 20, 20
        y = np.arange(rows, dtype=np.float64)[:, np.newaxis]
        z = np.broadcast_to(5.0 * y, (rows, cols)).copy().astype(np.float64)

        expected_deg = np.degrees(np.arctan(5.0))
        slope = compute_slope(z, cell_size_x=1.0, cell_size_y=1.0)

        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(interior, expected_deg, atol=1.0)

    def test_planar_surface_combined(self):
        """
        z = 2*x + 3*y → slope = arctan(sqrt(4 + 9)) = arctan(sqrt(13))
        """
        rows, cols = 20, 20
        x = np.arange(cols, dtype=np.float64)[np.newaxis, :]
        y = np.arange(rows, dtype=np.float64)[:, np.newaxis]
        z = 2.0 * x + 3.0 * y

        expected_deg = np.degrees(np.arctan(np.sqrt(13.0)))
        slope = compute_slope(z, cell_size_x=1.0, cell_size_y=1.0)

        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(interior, expected_deg, atol=1.0)

    def test_nodata_handling(self):
        """NaN pixels in input should produce NaN in slope output."""
        z = np.ones((10, 10)) * 5.0
        z[3, 4] = np.nan
        z[7, 2] = np.nan

        slope = compute_slope(z)
        assert np.isnan(slope[3, 4])
        assert np.isnan(slope[7, 2])
        # Pixels far from any NaN should be finite (flat surface → slope ≈ 0)
        # Corners and center are safe (distance > 1 from any NaN pixel)
        assert np.isfinite(slope[0, 0])
        assert np.isfinite(slope[5, 5])
        assert np.isfinite(slope[9, 9])

    def test_non_square_cell_size(self):
        """
        z = 1*x, cell_size_x = 2.0 → effective gradient = 1/2 = 0.5
        slope = arctan(0.5) ≈ 26.565 degrees
        """
        rows, cols = 20, 20
        x = np.arange(cols, dtype=np.float64)[np.newaxis, :]
        z = np.broadcast_to(x, (rows, cols)).copy().astype(np.float64)

        expected_deg = np.degrees(np.arctan(0.5))
        slope = compute_slope(z, cell_size_x=2.0, cell_size_y=1.0)

        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(interior, expected_deg, atol=1.0)

    def test_rejects_1d_input(self):
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="Expected 2D"):
            compute_slope(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_input(self):
        """Should raise ValueError for 3D input."""
        with pytest.raises(ValueError, match="Expected 2D"):
            compute_slope(np.ones((4, 4, 4)))

    def test_all_nan(self):
        """All-NaN input should produce all-NaN output."""
        z = np.full((5, 5), np.nan)
        slope = compute_slope(z)
        assert np.all(np.isnan(slope))


class TestComputeSlopePercent:
    """Tests for compute_slope_percent()."""

    def test_flat_surface(self):
        """Flat surface → 0% slope."""
        flat = np.ones((10, 10)) * 5.0
        pct = compute_slope_percent(flat)
        np.testing.assert_allclose(pct, 0.0, atol=1e-10)

    def test_known_gradient(self):
        """
        z = 1*x → dz/dx = 1, dz/dy = 0
        percent slope = 100 * sqrt(1) = 100%
        """
        rows, cols = 20, 20
        x = np.arange(cols, dtype=np.float64)[np.newaxis, :]
        z = np.broadcast_to(x, (rows, cols)).copy().astype(np.float64)

        pct = compute_slope_percent(z, cell_size_x=1.0, cell_size_y=1.0)
        interior = pct[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 100.0, atol=1.0)

    def test_nodata_handling(self):
        """NaN input → NaN output."""
        z = np.ones((8, 8)) * 3.0
        z[2, 5] = np.nan
        pct = compute_slope_percent(z)
        assert np.isnan(pct[2, 5])


# ============================================================
# Comparison tests
# ============================================================


class TestComputeErrorMap:
    """Tests for compute_error_map()."""

    def test_identical_arrays(self):
        """Identical arrays → zero error, perfect correlation."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = compute_error_map(a, a.copy())

        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["bias"] == pytest.approx(0.0, abs=1e-10)
        assert result["correlation"] == pytest.approx(1.0, abs=1e-10)
        assert result["valid_pixels"] == 6
        assert result["coverage_pct"] == pytest.approx(100.0)

    def test_known_error(self):
        """pred = [10, 20], ref = [8, 22] → errors = [2, -2]"""
        pred = np.array([[10.0, 20.0]])
        ref = np.array([[8.0, 22.0]])
        result = compute_error_map(pred, ref)

        assert result["rmse"] == pytest.approx(2.0, abs=1e-10)
        assert result["mae"] == pytest.approx(2.0, abs=1e-10)
        assert result["bias"] == pytest.approx(0.0, abs=1e-10)
        assert result["valid_pixels"] == 2

    def test_overprediction_bias(self):
        """pred consistently > ref → positive bias."""
        pred = np.array([[10.0, 10.0]])
        ref = np.array([[5.0, 5.0]])
        result = compute_error_map(pred, ref)

        assert result["bias"] == pytest.approx(5.0, abs=1e-10)
        assert result["mae"] == pytest.approx(5.0, abs=1e-10)

    def test_nan_masking(self):
        """NaN values in either array should be excluded from metrics."""
        pred = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        ref = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        result = compute_error_map(pred, ref)

        # Only 4 valid pixels (positions 0,0; 1,0; 1,1; 1,2)
        assert result["valid_pixels"] == 4
        assert result["coverage_pct"] == pytest.approx(4.0 / 6.0 * 100.0)

    def test_error_map_values(self):
        """Verify error_map array contents."""
        pred = np.array([[10.0, 20.0], [30.0, 40.0]])
        ref = np.array([[8.0, 20.0], [35.0, 40.0]])
        result = compute_error_map(pred, ref)

        error_map = result["error_map"]
        np.testing.assert_allclose(error_map, [[2.0, 0.0], [-5.0, 0.0]])

    def test_shape_mismatch_raises(self):
        """Different shapes should raise ValueError."""
        a = np.ones((3, 3))
        b = np.ones((4, 4))
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_error_map(a, b)

    def test_insufficient_valid_pixels(self):
        """Fewer than 2 valid pixels → None metrics."""
        a = np.full((3, 3), np.nan)
        a[0, 0] = 1.0
        result = compute_error_map(a, a.copy())

        assert result["rmse"] is None
        assert result["valid_pixels"] == 1

    def test_constant_arrays(self):
        """Constant but different arrays → zero variance, should handle gracefully."""
        pred = np.full((5, 5), 10.0)
        ref = np.full((5, 5), 8.0)
        result = compute_error_map(pred, ref)

        assert result["rmse"] == pytest.approx(2.0, abs=1e-10)
        assert result["bias"] == pytest.approx(2.0, abs=1e-10)
        # Both constant → std=0 → correlation should be NaN
        assert result["correlation"] is None or np.isnan(result["correlation"])

    def test_median_percentile_errors(self):
        """Verify median and percentile error computation."""
        pred = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
        ref = np.zeros((10, 10))
        result = compute_error_map(pred, ref)

        # median of [1..100] = 50.5, p90 ≈ 90.5, p95 ≈ 95.5 (numpy default interpolation)
        assert result["median_abs_error"] == pytest.approx(50.5, abs=1e-10)
        assert result["p90_abs_error"] == pytest.approx(90.5, abs=1.0)
        assert result["p95_abs_error"] == pytest.approx(95.5, abs=1.0)


# ============================================================
# Classification tests
# ============================================================


class TestClassifyTerrain:
    """Tests for classify_terrain()."""

    def test_flat_terrain(self):
        """Slope < 2 degrees → category 0 (flat)."""
        slope = np.full((10, 10), 1.0)
        cls = classify_terrain(slope)
        assert cls.dtype == np.uint8
        assert np.all(cls == 0)

    def test_gentle_terrain(self):
        """Slope 2-8 degrees → category 1 (gentle)."""
        slope = np.full((10, 10), 5.0)
        cls = classify_terrain(slope)
        assert np.all(cls == 1)

    def test_moderate_terrain(self):
        """Slope 8-15 degrees → category 2 (moderate)."""
        slope = np.full((10, 10), 12.0)
        cls = classify_terrain(slope)
        assert np.all(cls == 2)

    def test_steep_terrain(self):
        """Slope 15-30 degrees → category 3 (steep)."""
        slope = np.full((10, 10), 25.0)
        cls = classify_terrain(slope)
        assert np.all(cls == 3)

    def test_very_steep_terrain(self):
        """Slope > 30 degrees → category 4 (very_steep)."""
        slope = np.full((10, 10), 45.0)
        cls = classify_terrain(slope)
        assert np.all(cls == 4)

    def test_nodata_slope(self):
        """NaN slope → category 255 (nodata)."""
        slope = np.full((10, 10), 5.0)
        slope[3, 4] = np.nan
        cls = classify_terrain(slope)

        assert cls[3, 4] == 255
        assert cls[0, 0] == 1

    def test_boundary_values(self):
        """Test exact boundary thresholds."""
        slope = np.array([[1.9, 2.0, 7.9, 8.0, 14.9, 15.0, 29.9, 30.0, 30.1]])
        cls = classify_terrain(slope)
        expected = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4]])
        np.testing.assert_array_equal(cls, expected)

    def test_mixed_terrain(self):
        """Multiple categories in one array."""
        slope = np.array(
            [
                [0.5, 5.0, 12.0],
                [25.0, 45.0, np.nan],
            ]
        )
        cls = classify_terrain(slope)
        expected = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(cls, expected)

    def test_rejects_1d(self):
        """1D input should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 2D"):
            classify_terrain(np.array([1.0, 5.0, 10.0]))


class TestTerrainLegend:
    """Tests for legend and category name helpers."""

    def test_category_names(self):
        """All 5 categories should have names."""
        assert get_category_name(0) == "flat"
        assert get_category_name(1) == "gentle"
        assert get_category_name(2) == "moderate"
        assert get_category_name(3) == "steep"
        assert get_category_name(4) == "very_steep"
        assert get_category_name(99) == "unknown"

    def test_legend_structure(self):
        """Legend should have 5 entries with id and description."""
        legend = get_category_legend()
        assert len(legend) == 5
        for name, info in legend.items():
            assert "id" in info
            assert "description" in info
            assert isinstance(info["id"], int)

    def test_terrain_categories_constant(self):
        """TERRAIN_CATEGORIES dict should have 5 entries."""
        assert len(TERRAIN_CATEGORIES) == 5
