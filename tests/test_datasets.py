"""
Tests for dataset adapters: SRTM and ISPRS Potsdam.
"""

import numpy as np
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datasets.srtm_adapter import SRTMAdapter
from backend.datasets.isprs_adapter import ISPRSAdapter


class TestSRTMAdapter:
    """Tests for SRTM tile adapter."""

    def test_list_samples_empty_dir(self):
        """Empty/missing directory should return empty list."""
        adapter = SRTMAdapter(srtm_dir="/nonexistent/path")
        assert adapter.list_samples() == []

    def test_list_samples_empty_real_dir(self):
        """data/srtm/ exists but is empty — should return empty list."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        srtm_dir = os.path.join(base_dir, "data", "srtm")
        adapter = SRTMAdapter(srtm_dir=srtm_dir)
        result = adapter.list_samples()
        assert isinstance(result, list)
        # May be empty since no tiles are present

    def test_load_sample_missing_raises(self):
        """Loading a nonexistent tile should raise FileNotFoundError."""
        adapter = SRTMAdapter(srtm_dir="/tmp/fake_srtm")
        with pytest.raises(FileNotFoundError):
            adapter.load_sample("N00E000")

    def test_get_reference_dsm_missing(self):
        """Reference DSM for nonexistent tile should return None."""
        adapter = SRTMAdapter(srtm_dir="/tmp/fake_srtm")
        result = adapter.get_reference_dsm("N00E000")
        assert result is None


class TestISPRSAdapter:
    """Tests for ISPRS Potsdam adapter."""

    def test_list_samples_empty_dir(self):
        """Missing directory should return empty list."""
        adapter = ISPRSAdapter(data_dir="/nonexistent/path")
        assert adapter.list_samples() == []

    def test_list_samples_empty_real_dir(self):
        """data/isprs_potsdam/ doesn't exist — should return empty list."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data", "isprs_potsdam")
        adapter = ISPRSAdapter(data_dir=data_dir)
        result = adapter.list_samples()
        assert isinstance(result, list)

    def test_load_sample_missing_raises(self):
        """Loading a nonexistent tile should raise FileNotFoundError."""
        adapter = ISPRSAdapter(data_dir="/tmp/fake_isprs")
        with pytest.raises(FileNotFoundError):
            adapter.load_sample("nonexistent_tile")

    def test_get_reference_dsm_missing(self):
        """Reference DSM for nonexistent tile should return None."""
        adapter = ISPRSAdapter(data_dir="/tmp/fake_isprs")
        result = adapter.get_reference_dsm("nonexistent_tile")
        assert result is None
