"""Tests for PineconeRetriever implementation.

These tests use mocks to avoid requiring actual Pinecone API access,
ensuring tests are fast, deterministic, and can run in CI/CD environments.
"""

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from src.core.types import Query, RetrievedChunk, RetrievalResult
from src.retrievers import PineconeRetriever, PineconeRetrieverConfig
from tests.conftest import MockEmbedder, sample_chunks, sample_embeddings, sample_query


class MockPineconeIndex:
    """Mock Pinecone Index for testing."""
    
    def __init__(self) -> None:
        """Initialize mock index with empty storage."""
        self._vectors: dict[str, dict[str, Any]] = {}
        self._namespace: str | None = None
    
    def upsert(self, vectors: list[dict[str, Any]], namespace: str | None = None) -> None:
        """Mock upsert operation."""
        self._namespace = namespace
        for vector_data in vectors:
            self._vectors[vector_data["id"]] = {
                "values": vector_data["values"],
                "metadata": vector_data.get("metadata", {}),
            }
    
    def query(
        self,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        include_metadata: bool = True,
    ) -> MagicMock:
        """Mock query operation.
        
        Returns mock results sorted by a simple similarity score.
        """
        self._namespace = namespace
        
        # Create mock matches
        matches = []
        for i, (chunk_id, vector_data) in enumerate(list(self._vectors.items())[:top_k]):
            # Simple mock score (decreasing with index)
            score = 1.0 - (i * 0.1)
            
            match = MagicMock()
            match.id = chunk_id
            match.score = score
            match.metadata = vector_data["metadata"]
            matches.append(match)
        
        # Create mock response
        response = MagicMock()
        response.matches = matches
        return response
    
    def delete(self, delete_all: bool = False, namespace: str | None = None) -> None:
        """Mock delete operation."""
        if delete_all:
            if namespace is None or self._namespace == namespace:
                self._vectors.clear()


class MockPineconeClient:
    """Mock Pinecone client for testing."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize mock client."""
        self._api_key = api_key
        self._indices: dict[str, MockPineconeIndex] = {}
    
    def Index(self, name: str) -> MockPineconeIndex:
        """Get or create mock index."""
        if name not in self._indices:
            self._indices[name] = MockPineconeIndex()
        return self._indices[name]


@pytest.fixture
def pinecone_config() -> PineconeRetrieverConfig:
    """Create Pinecone retriever config for testing."""
    return PineconeRetrieverConfig(
        api_key="test-api-key",
        index_name="test-index",
        dimension=384,
        metric="cosine",
    )


@pytest.fixture
def mock_pinecone_client() -> MockPineconeClient:
    """Create mock Pinecone client."""
    return MockPineconeClient(api_key="test-api-key")


class TestPineconeRetrieverInitialization:
    """Test PineconeRetriever initialization."""
    
    def test_init_success(
        self,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
    ) -> None:
        """Test successful initialization."""
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        assert retriever._config == pinecone_config
        assert retriever._embedder == mock_embedder
    
    def test_init_dimension_mismatch(
        self,
        pinecone_config: PineconeRetrieverConfig,
    ) -> None:
        """Test initialization fails when embedder dimension doesn't match config."""
        embedder = MockEmbedder(dimension=512)  # Different dimension
        
        with pytest.raises(ValueError, match="does not match"):
            PineconeRetriever(pinecone_config, embedder)


class TestPineconeRetrieverAddChunks:
    """Test add_chunks method."""
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_add_chunks_success(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
        sample_embeddings: list,
    ) -> None:
        """Test successfully adding chunks."""
        # Setup mock
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        # Create retriever and add chunks
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        retriever.add_chunks(sample_chunks, sample_embeddings)
        
        # Verify chunks were stored
        assert len(mock_index._vectors) == len(sample_chunks)
        for chunk in sample_chunks:
            assert chunk.id in mock_index._vectors
    
    def test_add_chunks_length_mismatch(
        self,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
    ) -> None:
        """Test add_chunks fails when chunks and embeddings have different lengths."""
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        wrong_embeddings = mock_embedder.embed_chunks(sample_chunks[:2])  # One less
        
        with pytest.raises(ValueError, match="must match"):
            retriever.add_chunks(sample_chunks, wrong_embeddings)
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_add_chunks_dimension_mismatch(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
    ) -> None:
        """Test add_chunks fails when embedding dimension doesn't match."""
        from src.core.types import Embedding
        
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        
        # Create embedding with wrong dimension
        wrong_embedding = Embedding(
            vector=[0.1] * 512,  # Wrong dimension
            dimension=512,
        )
        
        with pytest.raises(ValueError, match="dimension"):
            retriever.add_chunks(sample_chunks, [wrong_embedding] * len(sample_chunks))


class TestPineconeRetrieverRetrieve:
    """Test retrieve method."""
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_retrieve_success(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
        sample_embeddings: list,
        sample_query,
    ) -> None:
        """Test successful retrieval."""
        # Setup mock
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        # Add chunks first
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        retriever.add_chunks(sample_chunks, sample_embeddings)
        
        # Retrieve
        result = retriever.retrieve(sample_query, top_k=2)
        
        # Verify result
        assert isinstance(result, RetrievalResult)
        assert result.query.id == sample_query.id
        assert result.query.text == sample_query.text  # Should preserve query text
        assert len(result.chunks) <= 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in result.chunks)
        assert "embedding_latency_seconds" in result.metadata
        assert "retrieval_latency_seconds" in result.metadata
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_retrieve_with_embedding(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
        sample_embeddings: list,
    ) -> None:
        """Test retrieve_with_embedding method."""
        # Setup mock
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        # Add chunks first
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        retriever.add_chunks(sample_chunks, sample_embeddings)
        
        # Create query embedding
        query_embedding = mock_embedder.embed_query(
            Query(id="test-query", text="test query")
        )
        
        # Retrieve with embedding
        result = retriever.retrieve_with_embedding(
            query_embedding=query_embedding,
            top_k=2,
            query_id="test-query",
        )
        
        # Verify result
        assert isinstance(result, RetrievalResult)
        assert result.query.id == "test-query"
        assert len(result.chunks) <= 2
        assert "retrieval_latency_seconds" in result.metadata
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_retrieve_with_embedding_dimension_mismatch(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
    ) -> None:
        """Test retrieve_with_embedding fails with wrong dimension."""
        from src.core.types import Embedding
        
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        
        # Create embedding with wrong dimension
        wrong_embedding = Embedding(
            vector=[0.1] * 512,
            dimension=512,
        )
        
        with pytest.raises(ValueError, match="dimension"):
            retriever.retrieve_with_embedding(wrong_embedding, top_k=10)


class TestPineconeRetrieverClear:
    """Test clear method."""
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_clear_success(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
        sample_embeddings: list,
    ) -> None:
        """Test successfully clearing index."""
        # Setup mock
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        # Add chunks first
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        retriever.add_chunks(sample_chunks, sample_embeddings)
        assert len(mock_index._vectors) > 0
        
        # Clear
        retriever.clear()
        
        # Verify cleared
        assert len(mock_index._vectors) == 0


class TestPineconeRetrieverMetadata:
    """Test metadata handling."""
    
    @patch("src.retrievers.pinecone_retriever.pinecone")
    def test_chunk_metadata_preservation(
        self,
        mock_pinecone: MagicMock,
        pinecone_config: PineconeRetrieverConfig,
        mock_embedder: MockEmbedder,
        sample_chunks: list,
        sample_embeddings: list,
        sample_query,
    ) -> None:
        """Test that chunk metadata is preserved through storage and retrieval."""
        # Setup mock
        mock_index = MockPineconeIndex()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone.Pinecone.return_value = mock_client
        
        # Add chunks
        retriever = PineconeRetriever(pinecone_config, mock_embedder)
        retriever.add_chunks(sample_chunks, sample_embeddings)
        
        # Retrieve
        result = retriever.retrieve(sample_query, top_k=10)
        
        # Verify metadata is preserved
        for retrieved_chunk in result.chunks:
            chunk = retrieved_chunk.chunk
            assert chunk.metadata.document_id is not None
            assert chunk.metadata.chunk_index is not None
            assert chunk.metadata.start_char is not None
            assert chunk.metadata.end_char is not None
