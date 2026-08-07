"""Tests for MilvusRetriever."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

# Setup mock for pymilvus before importing anything else
mock_pymilvus = MagicMock()

class MockDataType:
    INT64 = 5
    VARCHAR = 21
    FLOAT_VECTOR = 101

mock_pymilvus.DataType = MockDataType


class MockMilvusClient:
    """Mock implementation of pymilvus.MilvusClient for testing."""

    def __init__(self, uri: str = ":memory:") -> None:
        self.uri = uri
        self.collections: dict[str, Any] = {}
        self.inserted_data: dict[str, list[dict[str, Any]]] = {}

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")
        return self.collections[collection_name]

    def create_schema(self, auto_id: bool = False, enable_dynamic_field: bool = True) -> Any:
        class Schema:
            def __init__(self) -> None:
                self.fields: list[dict[str, Any]] = []

            def add_field(
                self,
                field_name: str,
                datatype: Any,
                is_primary: bool = False,
                max_length: int | None = None,
                dim: int | None = None,
            ) -> None:
                params = {}
                if dim is not None:
                    params["dim"] = dim
                self.fields.append({
                    "name": field_name,
                    "type": datatype,
                    "params": params,
                    "is_primary": is_primary,
                })
        return Schema()

    def prepare_index_params(self) -> Any:
        class IndexParams:
            def __init__(self) -> None:
                self.indexes: list[dict[str, Any]] = []

            def add_index(
                self,
                field_name: str,
                index_type: str,
                metric_type: str,
                params: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                self.indexes.append({
                    "field_name": field_name,
                    "index_type": index_type,
                    "metric_type": metric_type,
                    "params": params,
                    **kwargs,
                })
        return IndexParams()

    def create_collection(
        self,
        collection_name: str,
        schema: Any,
        index_params: Any = None,
    ) -> None:
        self.collections[collection_name] = {
            "schema": {
                "fields": schema.fields,
            }
        }
        self.inserted_data[collection_name] = []

    def load_collection(self, collection_name: str) -> None:
        pass

    def drop_collection(self, collection_name: str) -> None:
        if collection_name in self.collections:
            del self.collections[collection_name]
        if collection_name in self.inserted_data:
            del self.inserted_data[collection_name]

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> int:
        if collection_name not in self.inserted_data:
            self.inserted_data[collection_name] = []
        self.inserted_data[collection_name].extend(data)
        return len(data)

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        anns_field: str,
        search_params: dict[str, Any],
        limit: int,
        output_fields: list[str],
    ) -> list[list[dict[str, Any]]]:
        results = []
        for _ in data:
            hits = []
            for i, entity in enumerate(self.inserted_data.get(collection_name, [])[:limit]):
                distance = 0.95 - (i * 0.1)
                hits.append({
                    "id": entity["id"],
                    "distance": distance,
                    "entity": {
                        "chunk_id": entity["chunk_id"],
                        "content": entity["content"],
                        "document_id": entity["document_id"],
                        "chunk_index": entity["chunk_index"],
                        "start_char": entity["start_char"],
                        "end_char": entity["end_char"],
                    },
                })
            results.append(hits)
        return results


mock_pymilvus.MilvusClient = MockMilvusClient
sys.modules["pymilvus"] = mock_pymilvus

from src.core.types import Chunk, ChunkMetadata, Embedding, Query
from src.retrievers.config import MilvusRetrieverConfig
from src.retrievers.milvus_retriever import MilvusRetriever


class DummyEmbedder:
    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def embed_chunk(self, chunk: Chunk) -> Embedding:
        return Embedding(vector=[0.1] * self.dim, dimension=self.dim)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        return [self.embed_chunk(c) for c in chunks]

    def embed_query(self, query: Query) -> Embedding:
        return Embedding(vector=[0.1] * self.dim, dimension=self.dim)

    def get_dimension(self) -> int:
        return self.dim


def test_add_chunks_and_retrieve_returns_results() -> None:
    """Test that adding chunks and retrieving works, returning results."""
    config = MilvusRetrieverConfig(collection_name="test_col", distance_metric="cosine")
    embedder = DummyEmbedder(dim=3)
    retriever = MilvusRetriever(config, embedder)

    chunks = [
        Chunk(
            id="c1",
            content="first chunk",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=11),
        ),
        Chunk(
            id="c2",
            content="second chunk",
            metadata=ChunkMetadata(document_id="d1", chunk_index=1, start_char=12, end_char=24),
        ),
    ]
    embeddings = [
        Embedding(vector=[1.0, 0.0, 0.0], dimension=3),
        Embedding(vector=[0.0, 1.0, 0.0], dimension=3),
    ]

    retriever.add_chunks(chunks, embeddings)
    assert retriever._collection_ready is True

    query = Query(id="q1", text="find something")
    result = retriever.retrieve(query, top_k=2)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk.id == "c1"
    assert result.chunks[0].chunk.content == "first chunk"
    assert result.chunks[0].chunk.metadata.document_id == "d1"
    assert result.chunks[0].score == 0.95
    assert result.chunks[1].chunk.id == "c2"
    assert result.chunks[1].score == 0.85
    assert "retrieval_latency_seconds" in result.metadata
    assert result.metadata["num_results"] == 2
    assert result.metadata["collection_name"] == "test_col"


def test_retrieve_before_add_raises_runtime_error() -> None:
    """Test retrieving from empty index raises RuntimeError."""
    config = MilvusRetrieverConfig()
    embedder = DummyEmbedder(dim=3)
    retriever = MilvusRetriever(config, embedder)

    query = Query(id="q1", text="query")
    with pytest.raises(RuntimeError, match="Cannot retrieve from Milvus: collection is empty"):
        retriever.retrieve(query, top_k=1)


def test_add_chunks_length_mismatch_raises_value_error() -> None:
    """Test adding chunks with length mismatch raises ValueError."""
    config = MilvusRetrieverConfig()
    embedder = DummyEmbedder(dim=3)
    retriever = MilvusRetriever(config, embedder)

    chunks = [
        Chunk(
            id="c1",
            content="first chunk",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=11),
        ),
    ]
    embeddings: list[Embedding] = []

    with pytest.raises(ValueError, match="Number of chunks"):
        retriever.add_chunks(chunks, embeddings)


def test_metric_type_mapping_cosine_returns_COSINE() -> None:
    """Test distance metric mapping for cosine."""
    config = MilvusRetrieverConfig(distance_metric="cosine")
    retriever = MilvusRetriever(config, DummyEmbedder())
    assert retriever._metric_type() == "COSINE"


def test_metric_type_mapping_l2_returns_L2() -> None:
    """Test distance metric mapping for l2."""
    config = MilvusRetrieverConfig(distance_metric="l2")
    retriever = MilvusRetriever(config, DummyEmbedder())
    assert retriever._metric_type() == "L2"


def test_clear_drops_collection_and_resets_state() -> None:
    """Test that clear drops the collection and resets retriever state."""
    config = MilvusRetrieverConfig(collection_name="clear_col")
    embedder = DummyEmbedder(dim=3)
    retriever = MilvusRetriever(config, embedder)

    chunks = [
        Chunk(
            id="c1",
            content="content",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=7),
        ),
    ]
    embeddings = [Embedding(vector=[1.0, 0.0, 0.0], dimension=3)]

    retriever.add_chunks(chunks, embeddings)
    client = retriever._get_client()
    assert client.has_collection("clear_col") is True
    assert retriever._collection_ready is True
    assert retriever._next_id == 1

    retriever.clear()
    assert client.has_collection("clear_col") is False
    assert retriever._collection_ready is False
    assert retriever._next_id == 0


def test_config_rejects_invalid_distance_metric() -> None:
    """Test that config rejects invalid distance metrics."""
    with pytest.raises(ValueError, match="distance_metric must be one of"):
        MilvusRetrieverConfig(distance_metric="invalid_metric")


def test_in_memory_client_uses_memory_uri() -> None:
    """Test that in_memory config matches local file URI and remote uses env."""
    config_mem = MilvusRetrieverConfig(in_memory=True)
    retriever_mem = MilvusRetriever(config_mem, DummyEmbedder())
    client_mem = retriever_mem._get_client()
    assert client_mem.uri == f"{config_mem.collection_name}.db"

    config_remote = MilvusRetrieverConfig(in_memory=False)
    retriever_remote = MilvusRetriever(config_remote, DummyEmbedder())

    with patch.dict(os.environ, {"MILVUS_HOST": "my-milvus-host", "MILVUS_PORT": "12345"}):
        client_remote = retriever_remote._get_client()
        assert client_remote.uri == "http://my-milvus-host:12345"
