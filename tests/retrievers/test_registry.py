"""Tests for retriever registry."""

import pytest

from src.core.retrieval import Retriever
from src.retrievers import (
    RETRIEVER_REGISTRY,
    get_retriever,
    list_retrievers,
    register_retriever,
    unregister_retriever,
)
from src.retrievers.pinecone_retriever import PineconeRetriever


class TestRetrieverRegistry:
    """Test retriever registry functionality."""

    def test_pinecone_registered(self) -> None:
        """Test that PineconeRetriever is registered."""
        assert "pinecone" in RETRIEVER_REGISTRY
        assert RETRIEVER_REGISTRY["pinecone"] == PineconeRetriever

    def test_get_retriever_success(self) -> None:
        """Test getting a registered retriever."""
        retriever_class = get_retriever("pinecone")
        assert retriever_class == PineconeRetriever

    def test_get_retriever_not_found(self) -> None:
        """Test getting a non-existent retriever raises error."""
        with pytest.raises(KeyError, match="not registered"):
            get_retriever("nonexistent-retriever")

    def test_list_retrievers(self) -> None:
        """Test listing all registered retrievers."""
        retrievers = list_retrievers()
        assert isinstance(retrievers, list)
        assert "pinecone" in retrievers

    def test_register_retriever_success(self) -> None:
        """Test registering a new retriever."""
        # Create a dummy retriever class
        class DummyRetriever(Retriever):
            def add_chunks(self, chunks, embeddings):
                pass

            def retrieve(self, query, top_k=10):
                pass

            def retrieve_with_embedding(self, query_embedding, top_k=10, query_id=None):
                pass

            def clear(self):
                pass

        # Register it
        register_retriever("dummy", DummyRetriever)

        # Verify it's registered
        assert "dummy" in RETRIEVER_REGISTRY
        assert get_retriever("dummy") == DummyRetriever

        # Cleanup
        unregister_retriever("dummy")

    def test_register_retriever_duplicate(self) -> None:
        """Test registering duplicate retriever name raises error."""
        class DummyRetriever(Retriever):
            def add_chunks(self, chunks, embeddings):
                pass

            def retrieve(self, query, top_k=10):
                pass

            def retrieve_with_embedding(self, query_embedding, top_k=10, query_id=None):
                pass

            def clear(self):
                pass

        register_retriever("dummy2", DummyRetriever)

        with pytest.raises(ValueError, match="already registered"):
            register_retriever("dummy2", DummyRetriever)

        # Cleanup
        unregister_retriever("dummy2")

    def test_register_retriever_invalid_class(self) -> None:
        """Test registering non-Retriever class raises error."""
        class NotARetriever:
            pass

        with pytest.raises(ValueError, match="must be a subclass"):
            register_retriever("invalid", NotARetriever)

    def test_unregister_retriever(self) -> None:
        """Test unregistering a retriever."""
        class DummyRetriever(Retriever):
            def add_chunks(self, chunks, embeddings):
                pass

            def retrieve(self, query, top_k=10):
                pass

            def retrieve_with_embedding(self, query_embedding, top_k=10, query_id=None):
                pass

            def clear(self):
                pass

        register_retriever("dummy3", DummyRetriever)
        assert "dummy3" in RETRIEVER_REGISTRY

        unregister_retriever("dummy3")
        assert "dummy3" not in RETRIEVER_REGISTRY

    def test_unregister_retriever_not_found(self) -> None:
        """Test unregistering non-existent retriever raises error."""
        with pytest.raises(KeyError, match="not registered"):
            unregister_retriever("nonexistent")
