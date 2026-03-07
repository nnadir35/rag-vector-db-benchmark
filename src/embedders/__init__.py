"""Embedder implementations.

This package provides concrete implementations of the Embedder interface.
"""

from .config import SentenceTransformersEmbedderConfig
from .sentence_transformers_embedder import SentenceTransformersEmbedder

__all__ = [
    "SentenceTransformersEmbedder",
    "SentenceTransformersEmbedderConfig",
]
