"""Pytest configuration and shared fixtures.

This module provides common test fixtures used across multiple test files.
"""

from typing import Sequence
from unittest.mock import MagicMock

import pytest

from src.core.embedding import Embedder
from src.core.types import (
    Chunk,
    ChunkMetadata,
    DocumentMetadata,
    Embedding,
    Query,
)


class MockEmbedder(Embedder):
    """Mock embedder for testing.
    
    This embedder generates deterministic embeddings based on input text,
    making tests reproducible. The embedding is simply a hash-based vector.
    """
    
    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock embedder.
        
        Args:
            dimension: Dimension of embeddings to produce
        """
        self._dimension = dimension
    
    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """Generate deterministic embedding for a chunk."""
        # Simple hash-based embedding for determinism
        vector = [float(hash(chunk.id + str(i)) % 100) / 100.0 for i in range(self._dimension)]
        return Embedding(vector=vector, dimension=self._dimension)
    
    def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        """Generate embeddings for multiple chunks."""
        return [self.embed_chunk(chunk) for chunk in chunks]
    
    def embed_query(self, query: Query) -> Embedding:
        """Generate deterministic embedding for a query."""
        # Simple hash-based embedding for determinism
        vector = [float(hash(query.text + str(i)) % 100) / 100.0 for i in range(self._dimension)]
        return Embedding(vector=vector, dimension=self._dimension)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk-1",
            content="This is the first chunk about machine learning.",
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_index=0,
                start_char=0,
                end_char=50,
            ),
        ),
        Chunk(
            id="chunk-2",
            content="This is the second chunk about neural networks.",
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_index=1,
                start_char=51,
                end_char=100,
            ),
        ),
        Chunk(
            id="chunk-3",
            content="This is the third chunk about deep learning.",
            metadata=ChunkMetadata(
                document_id="doc-2",
                chunk_index=0,
                start_char=0,
                end_char=45,
            ),
        ),
    ]


@pytest.fixture
def sample_query() -> Query:
    """Create a sample query for testing."""
    return Query(
        id="query-1",
        text="What is machine learning?",
    )


@pytest.fixture
def sample_embeddings(mock_embedder: MockEmbedder, sample_chunks: list[Chunk]) -> list[Embedding]:
    """Create sample embeddings for testing."""
    return mock_embedder.embed_chunks(sample_chunks)
