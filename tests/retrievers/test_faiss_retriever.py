"""Tests for FAISSRetriever."""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.core.types import (
    Chunk,
    ChunkMetadata,
    Embedding,
    Query,
)
from src.retrievers.faiss_retriever import FAISSRetriever
from src.retrievers.config import FAISSRetrieverConfig


# We create a simple mock embedder that outputs deterministic vectors
class DummyEmbedder:
    def __init__(self, dim=3):
        self.dim = dim

    def embed_chunk(self, chunk):
        return Embedding(vector=[0.1] * self.dim, dimension=self.dim)

    def embed_chunks(self, chunks):
        return [self.embed_chunk(c) for c in chunks]

    def embed_query(self, query):
        return Embedding(vector=[0.1] * self.dim, dimension=self.dim)

    def get_dimension(self):
        return self.dim


def test_add_chunks_and_retrieve_returns_results():
    """Test that adding 3 chunks and running 1 query returns results."""
    config = FAISSRetrieverConfig(distance_metric="ip")
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config, embedder)

    chunks = [
        Chunk(id="c1", content="a", metadata=ChunkMetadata(document_id="d", chunk_index=0, start_char=0, end_char=1)),
        Chunk(id="c2", content="b", metadata=ChunkMetadata(document_id="d", chunk_index=1, start_char=1, end_char=2)),
        Chunk(id="c3", content="c", metadata=ChunkMetadata(document_id="d", chunk_index=2, start_char=2, end_char=3)),
    ]
    embeddings = [
        Embedding(vector=[1.0, 0.0, 0.0], dimension=3),
        Embedding(vector=[0.0, 1.0, 0.0], dimension=3),
        Embedding(vector=[0.0, 0.0, 1.0], dimension=3),
    ]
    retriever.add_chunks(chunks, embeddings)

    query = Query(id="q1", text="some query")
    result = retriever.retrieve(query, top_k=2)

    assert len(result.chunks) == 2
    assert "embedding_latency_seconds" in result.metadata
    assert "retrieval_latency_seconds" in result.metadata
    assert result.metadata["num_results"] == 2
    assert result.metadata["index_total_vectors"] == 3


def test_retrieve_empty_index_raises_runtime_error():
    """Test retrieving from empty index raises RuntimeError."""
    config = FAISSRetrieverConfig()
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(config, embedder)

    query_emb = Embedding(vector=[0.1, 0.1, 0.1], dimension=3)

    with pytest.raises(RuntimeError, match="Cannot retrieve from FAISS: index is empty"):
        retriever.retrieve_with_embedding(query_emb, top_k=5)


def test_add_chunks_length_mismatch_raises_value_error():
    """Test adding chunks with length mismatch raises ValueError."""
    config = FAISSRetrieverConfig()
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config, embedder)

    chunks = [
        Chunk(id="c1", content="a", metadata=ChunkMetadata(document_id="d", chunk_index=0, start_char=0, end_char=1)),
        Chunk(id="c2", content="b", metadata=ChunkMetadata(document_id="d", chunk_index=1, start_char=1, end_char=2)),
        Chunk(id="c3", content="c", metadata=ChunkMetadata(document_id="d", chunk_index=2, start_char=2, end_char=3)),
    ]
    embeddings = [
        Embedding(vector=[0.1]*3, dimension=3),
        Embedding(vector=[0.2]*3, dimension=3),
    ]

    with pytest.raises(ValueError, match="Number of chunks"):
        retriever.add_chunks(chunks, embeddings)


def test_cosine_normalizes_vectors():
    """Test that cosine distance metric normalizes vectors using faiss.normalize_L2."""
    config = FAISSRetrieverConfig(distance_metric="cosine")
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(config, embedder)

    with patch("src.retrievers.faiss_retriever.faiss") as mock_faiss:
        retriever._ensure_faiss_imported = MagicMock()

        vecs = np.array([[1.0, 2.0]], dtype=np.float32)
        retriever._normalize_if_cosine(vecs)

        mock_faiss.normalize_L2.assert_called_once_with(vecs)


def test_l2_distance_inverted_to_negative_score():
    """Test that L2 distance score is inverted to negative value."""
    config = FAISSRetrieverConfig(distance_metric="l2")
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config, embedder)

    chunks = [
        Chunk(id="c1", content="a", metadata=ChunkMetadata(document_id="d", chunk_index=0, start_char=0, end_char=1)),
    ]
    embeddings = [
        Embedding(vector=[1.0, 0.0, 0.0], dimension=3),
    ]
    retriever.add_chunks(chunks, embeddings)

    query_emb = Embedding(vector=[2.0, 0.0, 0.0], dimension=3)
    result = retriever.retrieve_with_embedding(query_emb, top_k=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].score == -1.0
    assert result.chunks[0].score <= 0.0


def test_clear_resets_state():
    """Test that clear resets retriever state."""
    config = FAISSRetrieverConfig()
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config, embedder)

    chunks = [
        Chunk(id="c1", content="a", metadata=ChunkMetadata(document_id="d", chunk_index=0, start_char=0, end_char=1)),
    ]
    embeddings = [
        Embedding(vector=[0.1]*3, dimension=3),
    ]
    retriever.add_chunks(chunks, embeddings)

    assert retriever._index is not None
    assert len(retriever._id_to_chunk) == 1
    assert retriever._next_id == 1

    retriever.clear()

    assert retriever._index is None
    assert len(retriever._id_to_chunk) == 0
    assert retriever._next_id == 0


def test_config_rejects_invalid_distance_metric():
    """Test that config rejects invalid distance metrics."""
    with pytest.raises(ValueError, match="distance_metric must be one of"):
        FAISSRetrieverConfig(distance_metric="invalid_metric")
