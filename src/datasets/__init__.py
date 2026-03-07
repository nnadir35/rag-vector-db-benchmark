"""Dataset loader implementations.

This package provides concrete implementations for loading standard datasets.
"""

from .config import SQuADDatasetConfig
from .squad_loader import SQuADLoader

__all__ = [
    "SQuADDatasetConfig",
    "SQuADLoader",
]
