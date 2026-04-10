"""Tests for ChromaRetriever using a real Ephemeral/Persistent client."""

import pytest

from src.core.types import (
    Chunk,
    ChunkMetadata,
    Embedding,
)
from src.retrievers.chroma_retriever import ChromaRetriever
from src.retrievers.config import ChromaRetrieverConfig


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


@pytest.fixture
def chroma_retriever(tmp_path):
    """Fixture supplying a real but isolated ChromaRetriever instance."""
    config = ChromaRetrieverConfig(
        collection_name="test_collection",
        persist_directory=str(tmp_path),
        distance_metric="cosine"
    )
    embedder = DummyEmbedder(dim=3)
    retriever = ChromaRetriever(config=config, embedder=embedder)

    # ensure it's clean (though tmp_path is fresh anyway)
    retriever.clear()

    yield retriever

    # Teardown
    retriever.clear()


def test_add_and_retrieve(chroma_retriever):
    """Test inserting documents and successfully retrieving them."""
    # 1. Arrange
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
    chroma_retriever.add_chunks(chunks, embeddings)

    # 3. Act (Retrieve with custom query embedding matching chunk 1 exactly)
    # Cosine distance to [1.0, 0.0, 0.0] should be 0, score = 1.0
    query_emb = Embedding(vector=[1.0, 0.0, 0.0], dimension=3)
    result = chroma_retriever.retrieve_with_embedding(query_emb, top_k=1, query_id="q1")

    # 4. Assert
    assert len(result.chunks) == 1
    top_hit = result.chunks[0]

    assert top_hit.chunk.id == "chunk_1"
    assert top_hit.chunk.content == "This is the first test chunk."
    assert top_hit.chunk.metadata.document_id == "doc_1"
    # Near perfect cosine score
    assert top_hit.score >= 0.99


def test_clear_collection(chroma_retriever):
    """Test that clearing the retriever wipes all chunks."""
    chunk = Chunk(
        id="test_chunk",
        content="test content",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=5)
    )
    emb = Embedding(vector=[1.0, 1.0, 1.0], dimension=3)

    chroma_retriever.add_chunks([chunk], [emb])

    # Verify it was added
    res1 = chroma_retriever.retrieve_with_embedding(emb, top_k=1)
    assert len(res1.chunks) == 1

    # Clear collection
    chroma_retriever.clear()

    # Verify empty
    res2 = chroma_retriever.retrieve_with_embedding(emb, top_k=1)
    assert len(res2.chunks) == 0


def test_import_error_chromadb_missing(monkeypatch):
    """Test error is thrown when chromadb is missing."""
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", None)

    import src.retrievers.chroma_retriever as cr
    monkeypatch.setattr(cr, "chromadb", None)

    config = ChromaRetrieverConfig(collection_name="test")
    embedder = DummyEmbedder()

    retriever = ChromaRetriever(config, embedder)

    with pytest.raises(ImportError, match="chromadb package is required for ChromaRetriever"):
        # Accessing the property triggers the lazy load check
        _ = retriever._chroma_collection
