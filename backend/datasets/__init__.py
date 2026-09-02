"""
Dataset adapters for loading real imagery and elevation data.

Provides a unified interface for accessing:
- SRTM elevation tiles
- ISPRS Potsdam Vaihingen DSM/RGB pairs
"""

from backend.datasets.base import DatasetAdapter
from backend.datasets.srtm_adapter import SRTMAdapter
from backend.datasets.isprs_adapter import ISPRSAdapter

__all__ = ["DatasetAdapter", "SRTMAdapter", "ISPRSAdapter"]
