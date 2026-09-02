"""
Abstract base class for dataset adapters.

All dataset adapters implement a common interface for listing samples,
loading imagery, and retrieving reference DSMs.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

import numpy as np


class DatasetAdapter(ABC):
    """Abstract interface for dataset access."""

    @abstractmethod
    def list_samples(self) -> List[Dict]:
        """
        List available samples in this dataset.

        Returns:
            List of dicts with at least 'id', 'description', and any
            dataset-specific metadata (bounds, resolution, etc.)
        """
        ...

    @abstractmethod
    def load_sample(self, sample_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a sample image for depth estimation.

        Args:
            sample_id: Identifier from list_samples()

        Returns:
            (image_array, metadata_dict)
            image_array is uint8 RGB of shape (H, W, 3)

        Raises:
            FileNotFoundError: If sample_id not found
            ValueError: If file is corrupt or unreadable
        """
        ...

    @abstractmethod
    def get_reference_dsm(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Get the reference DSM for a sample (if available).

        Args:
            sample_id: Identifier from list_samples()

        Returns:
            float32 array of elevation in meters, or None if no reference
        """
        ...
