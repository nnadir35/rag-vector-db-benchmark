"""Tests for FAISSRetriever."""

import os
import pytest
from unittest.mock import MagicMock

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


def test_add_and_retrieve_flat_ip():
    """Test inserting documents and successfully retrieving them using cosine (flat_ip)."""
    # 1. Arrange
    config = FAISSRetrieverConfig(
        collection_name="test_faiss_ip",
        distance_metric="cosine"
    )
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config=config, embedder=embedder)

    chunk1 = Chunk(
        id="chunk_1",
        content="This is the first test chunk.",
        metadata=ChunkMetadata(document_id="doc_1", chunk_index=0, start_char=0, end_char=10)
    )
    chunk2 = Chunk(
        id="chunk_2",
        content="This is the second test chunk.",
        metadata=ChunkMetadata(document_id="doc_2", chunk_index=0, start_char=0, end_char=10)
    )
    chunks = [chunk1, chunk2]
    embeddings = [
        Embedding(vector=[1.0, 0.0, 0.0], dimension=3),
        Embedding(vector=[0.0, 1.0, 0.0], dimension=3)
    ]

    # 2. Act (Add)
    retriever.add_chunks(chunks, embeddings)

    # 3. Act (Retrieve with custom query embedding matching chunk 1 exactly)
    query_emb = Embedding(vector=[1.0, 0.0, 0.0], dimension=3)
    result = retriever.retrieve_with_embedding(query_emb, top_k=1, query_id="q1")

    # 4. Assert
    assert len(result.chunks) == 1
    top_hit = result.chunks[0]

    assert top_hit.chunk.id == "chunk_1"
    assert top_hit.chunk.content == "This is the first test chunk."
    assert top_hit.chunk.metadata.document_id == "doc_1"
    assert top_hit.score >= 0.99


def test_add_and_retrieve_flat_l2():
    """Test inserting documents and successfully retrieving them using L2 (flat_l2)."""
    # 1. Arrange
    config = FAISSRetrieverConfig(
        collection_name="test_faiss_l2",
        distance_metric="l2"
    )
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config=config, embedder=embedder)

    chunk1 = Chunk(
        id="chunk_1",
        content="This is the first test chunk.",
        metadata=ChunkMetadata(document_id="doc_1", chunk_index=0, start_char=0, end_char=10)
    )
    chunk2 = Chunk(
        id="chunk_2",
        content="This is the second test chunk.",
        metadata=ChunkMetadata(document_id="doc_2", chunk_index=0, start_char=0, end_char=10)
    )
    chunks = [chunk1, chunk2]
    embeddings = [
        Embedding(vector=[1.0, 0.0, 0.0], dimension=3),
        Embedding(vector=[0.0, 1.0, 0.0], dimension=3)
    ]

    # 2. Act (Add)
    retriever.add_chunks(chunks, embeddings)

    # 3. Act (Retrieve with query closer to chunk 2)
    query_emb = Embedding(vector=[0.1, 0.9, 0.0], dimension=3)
    result = retriever.retrieve_with_embedding(query_emb, top_k=2, query_id="q2")

    # 4. Assert
    assert len(result.chunks) == 2
    # chunk 2 is closer to [0.1, 0.9, 0.0] than chunk 1 is.
    # distance to chunk 2: (0.1-0)^2 + (0.9-1)^2 + 0 = 0.01 + 0.01 = 0.02, score = -0.02
    # distance to chunk 1: (0.1-1)^2 + (0.9-0)^2 + 0 = 0.81 + 0.81 = 1.62, score = -1.62
    top_hit = result.chunks[0]
    assert top_hit.chunk.id == "chunk_2"
    assert top_hit.score == pytest.approx(-0.02)

    second_hit = result.chunks[1]
    assert second_hit.chunk.id == "chunk_1"
    assert second_hit.score == pytest.approx(-1.62)


def test_clear_collection():
    """Test that clearing the retriever wipes all chunks."""
    config = FAISSRetrieverConfig(
        collection_name="test_faiss_clear",
        distance_metric="cosine"
    )
    embedder = DummyEmbedder(dim=3)
    retriever = FAISSRetriever(config=config, embedder=embedder)

    chunk = Chunk(
        id="test_chunk",
        content="test content",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=5)
    )
    emb = Embedding(vector=[1.0, 1.0, 1.0], dimension=3)

    retriever.add_chunks([chunk], [emb])

    # Verify it was added
    res1 = retriever.retrieve_with_embedding(emb, top_k=1)
    assert len(res1.chunks) == 1

    # Clear collection
    retriever.clear()

    # Verify empty
    res2 = retriever.retrieve_with_embedding(emb, top_k=1)
    assert len(res2.chunks) == 0


def test_persistence(tmp_path):
    """Test index serialization and deserialization."""
    persist_path = os.path.join(tmp_path, "my_faiss_index")
    config = FAISSRetrieverConfig(
        collection_name="test_faiss_persist",
        distance_metric="cosine",
        persist_path=persist_path
    )
    embedder = DummyEmbedder(dim=3)
    
    # 1. Instantiate retriever, add chunks, it should write files to disk
    retriever1 = FAISSRetriever(config=config, embedder=embedder)
    chunk = Chunk(
        id="persist_chunk",
        content="persist content",
        metadata=ChunkMetadata(document_id="doc_p", chunk_index=0, start_char=0, end_char=15)
    )
    emb = Embedding(vector=[1.0, 0.0, 0.0], dimension=3)
    retriever1.add_chunks([chunk], [emb])

    assert os.path.exists(f"{persist_path}.index")
    assert os.path.exists(f"{persist_path}.pkl")

    # 2. Instantiate a new retriever with same config, search without adding chunks first
    retriever2 = FAISSRetriever(config=config, embedder=embedder)
    result = retriever2.retrieve_with_embedding(emb, top_k=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk.id == "persist_chunk"
    assert result.chunks[0].chunk.content == "persist content"

    # 3. Clear should delete the files
    retriever2.clear()
    assert not os.path.exists(f"{persist_path}.index")
    assert not os.path.exists(f"{persist_path}.pkl")


def test_import_error_faiss_missing(monkeypatch):
    """Test error is thrown when faiss is missing."""
    import sys
    monkeypatch.setitem(sys.modules, "faiss", None)

    import src.retrievers.faiss_retriever as fr
    monkeypatch.setattr(fr, "faiss", None)

    config = FAISSRetrieverConfig(collection_name="test")
    embedder = DummyEmbedder()

    retriever = FAISSRetriever(config, embedder)

    with pytest.raises(ImportError, match="faiss package is required for FAISSRetriever"):
        # Triggers lazy load check
        retriever._ensure_faiss_imported()
