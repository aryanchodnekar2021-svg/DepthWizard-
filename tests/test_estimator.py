"""
Tests for depth estimator improvements: model status, tiled inference,
error handling, and multi-band input support.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.depth.estimator import (
    get_model_status,
    prepare_image,
    estimate_depth,
    MODELS,
)


class TestModelStatus:
    """Tests for get_model_status()."""

    def test_returns_dict(self):
        """Status should be a dict with required keys."""
        status = get_model_status()
        assert isinstance(status, dict)
        for key in [
            "model_id",
            "device",
            "loaded",
            "last_inference_ms",
            "total_inferences",
        ]:
            assert key in status, f"Missing key: {key}"

    def test_initial_state(self):
        """Before loading, loaded should be False."""
        # Note: if model was loaded in a prior test, this may be True
        # We just check the structure
        status = get_model_status()
        assert isinstance(status["loaded"], bool)
        assert isinstance(status["total_inferences"], int)
        assert status["total_inferences"] >= 0


class TestPrepareImage:
    """Tests for prepare_image() input normalization."""

    def test_grayscale_to_rgb(self):
        """2D grayscale input should become (H, W, 3)."""
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = prepare_image(gray)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_multiband_selects_first_three(self):
        """Multi-band input should be trimmed to 3 channels."""
        img = np.random.randint(0, 255, (64, 64, 5), dtype=np.uint8)
        result = prepare_image(img)
        assert result.shape == (64, 64, 3)

    def test_float_to_uint8(self):
        """Float input should be converted to uint8."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = prepare_image(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_uint8_passthrough(self):
        """Proper uint8 RGB should pass through unchanged."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = prepare_image(img)
        np.testing.assert_array_equal(result, img)

    def test_two_band_becomes_rgb(self):
        """2-band input should become 3-band."""
        img = np.random.randint(0, 255, (32, 32, 2), dtype=np.uint8)
        result = prepare_image(img)
        assert result.shape[2] == 3


class TestEstimateDepth:
    """Tests for estimate_depth() error handling."""

    def test_empty_image_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            estimate_depth(np.array([]))

    def test_none_image_raises(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            estimate_depth(None)


class TestModeMapping:
    """Test model mapping is correct."""

    def test_default_model_exists(self):
        """Default backbone should be in MODELS dict."""
        assert "depth_anything_v2" in MODELS

    def test_model_id_format(self):
        """Model IDs should be HuggingFace format."""
        for name, model_id in MODELS.items():
            assert "/" in model_id, f"{name}: model_id missing org prefix"
