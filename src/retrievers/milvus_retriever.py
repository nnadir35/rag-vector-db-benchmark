"""Milvus retriever implementation.

This module provides a concrete implementation of the Retriever interface
that uses Milvus as the vector database backend via the high-level MilvusClient.
"""

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymilvus import DataType, MilvusClient
else:
    MilvusClient = None
    DataType = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    Embedding,
    Query,
    RetrievalResult,
    RetrievedChunk,
)
from .config import MilvusRetrieverConfig


class MilvusRetriever(Retriever):
    """Milvus-based retriever implementation."""

    def __init__(
        self,
        config: MilvusRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize Milvus retriever.

        Args:
            config: Configuration for Milvus usage
            embedder: Embedder instance to use for queries
        """
        self._config = config
        self._embedder = embedder

        self._client: Any | None = None
        self._collection_ready: bool = False
        self._next_id: int = 0

    def _ensure_milvus_imported(self) -> None:
        """Ensure pymilvus module is lazily imported.

        Raises:
            ImportError: If pymilvus package is not installed
        """
        global MilvusClient, DataType
        if MilvusClient is None or DataType is None:
            try:
                from pymilvus import DataType as _DataType
                from pymilvus import MilvusClient as _MilvusClient

                MilvusClient = _MilvusClient
                DataType = _DataType
            except ImportError:
                raise ImportError(
                    "pymilvus package is required for MilvusRetriever. "
                    "Install it with: pip install pymilvus>=2.3.0"
                ) from None

    def _get_client(self) -> "MilvusClient":
        """Get or create MilvusClient client.

        Connection resolution order:
        1. If in_memory is True, use MilvusClient(":memory:")
        2. Else, resolve MILVUS_HOST (default: localhost) and MILVUS_PORT (default: 19530)
           and connect to http://host:port.
        """
        self._ensure_milvus_imported()

        if self._client is None:
            try:
                if self._config.in_memory:
                    # pymilvus 3.x requires a local file ending with .db for embedded mode
                    self._client = MilvusClient(f"{self._config.collection_name}.db")
                else:
                    host = os.getenv("MILVUS_HOST", "localhost").strip()
                    port = int(os.getenv("MILVUS_PORT", "19530"))
                    self._client = MilvusClient(uri=f"http://{host}:{port}")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Milvus: {e}") from e

        return self._client

    def _metric_type(self) -> str:
        """Map config distance metric string to Milvus metric type."""
        mapping = {
            "cosine": "COSINE",
            "l2": "L2",
            "ip": "IP",
        }
        return mapping[self._config.distance_metric]

    def _ensure_collection(self, vector_size: int) -> None:
        """Create collection if missing or dimension changed.

        Args:
            vector_size: Dimension of the vectors to store
        """
        client = self._get_client()
        name = self._config.collection_name

        exists = client.has_collection(collection_name=name)

        if exists:
            desc = client.describe_collection(collection_name=name)
            current_size = None
            for field_info in desc.get("schema", {}).get("fields", []):
                if field_info.get("name") == "vector":
                    current_size = field_info.get("params", {}).get("dim")
                    break

            if current_size != vector_size:
                client.drop_collection(collection_name=name)
                exists = False
                self._next_id = 0

        if not exists:
            # Re-ensure DataType is loaded
            self._ensure_milvus_imported()

            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(field_name="start_char", datatype=DataType.INT64)
            schema.add_field(field_name="end_char", datatype=DataType.INT64)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type=self._metric_type(),
                params={
                    "M": self._config.hnsw_m,
                    "efConstruction": self._config.hnsw_ef_construction,
                },
            )

            client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
            # Load collection for search
            client.load_collection(collection_name=name)

        self._collection_ready = True

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Add chunks and embeddings to Milvus.

        Args:
            chunks: Sequence of chunks to add
            embeddings: Sequence of embeddings corresponding to chunks
                (must be same length and order as chunks)

        Raises:
            ValueError: If chunks and embeddings have different lengths
            RuntimeError: If storage fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            return

        try:
            first_dim = embeddings[0].dimension
            for emb in embeddings:
                if emb.dimension != first_dim:
                    raise ValueError(
                        "All embeddings must share the same dimension for Milvus indexing"
                    )

            self._ensure_collection(vector_size=first_dim)

            client = self._get_client()
            entities = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=False)):
                entities.append({
                    "id": self._next_id + i,
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "vector": list(emb.vector),
                })

            batch_size = 25
            for start in range(0, len(entities), batch_size):
                client.insert(
                    collection_name=self._config.collection_name,
                    data=entities[start : start + batch_size],
                )

            # Flush and load to ensure inserted data is immediately searchable
            if hasattr(client, "flush"):
                client.flush(collection_name=self._config.collection_name)
            client.load_collection(collection_name=self._config.collection_name)

            self._next_id += len(chunks)
            self._collection_ready = True

        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to Milvus: {e}") from e

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query using the embedder.

        Args:
            query: The query to retrieve chunks for
            top_k: Maximum number of chunks to retrieve

        Returns:
            RetrievalResult containing ranked chunks and metadata
        """
        start_time = time.time()
        query_embedding = self._embedder.embed_query(query)
        embedding_time = time.time() - start_time

        result = self.retrieve_with_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            query_id=query.id,
        )

        metadata = dict(result.metadata)
        metadata["embedding_latency_seconds"] = embedding_time

        return RetrievalResult(
            query=query,
            chunks=result.chunks,
            metadata=metadata,
        )

    def retrieve_with_embedding(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        query_id: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed embedding for the query
            top_k: Maximum number of chunks to retrieve
            query_id: Optional query ID for the result metadata

        Returns:
            RetrievalResult containing ranked chunks and metadata

        Raises:
            RuntimeError: If collection is not ready or search fails
        """
        if not self._collection_ready:
            raise RuntimeError(
                "Cannot retrieve from Milvus: collection is empty. Call add_chunks first."
            )

        try:
            start_time = time.time()
            client = self._get_client()

            results = client.search(
                collection_name=self._config.collection_name,
                data=[list(query_embedding.vector)],
                anns_field="vector",
                search_params={"metric_type": self._metric_type(), "params": {"ef": self._config.hnsw_ef_search}},
                limit=top_k,
                output_fields=[
                    "chunk_id",
                    "content",
                    "document_id",
                    "chunk_index",
                    "start_char",
                    "end_char",
                ],
            )

            retrieval_time = time.time() - start_time

            retrieved_chunks: list[RetrievedChunk] = []
            if results and len(results) > 0:
                from ..core.types import ChunkMetadata

                for rank, hit in enumerate(results[0]):
                    entity = hit.get("entity", hit)
                    chunk_metadata = ChunkMetadata(
                        document_id=entity["document_id"],
                        chunk_index=int(entity["chunk_index"]),
                        start_char=int(entity["start_char"]),
                        end_char=int(entity["end_char"]),
                    )
                    chunk = Chunk(
                        id=str(entity["chunk_id"]),
                        content=str(entity["content"]),
                        metadata=chunk_metadata,
                    )
                    score = float(hit["distance"])
                    if self._config.distance_metric == "l2":
                        score = -score

                    retrieved_chunks.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=score,
                            rank=rank,
                        )
                    )

            query = Query(id=query_id or "unknown", text="")

            metadata = {
                "retrieval_latency_seconds": retrieval_time,
                "num_results": len(retrieved_chunks),
                "collection_name": self._config.collection_name,
            }

            return RetrievalResult(
                query=query,
                chunks=retrieved_chunks,
                metadata=metadata,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve from Milvus: {e}") from e

    def clear(self) -> None:
        """Clear the collection and reset state.

        Always attempts server-side deletion via a live client, regardless of
        in-process ``_client``/``_collection_ready`` state — a fresh
        ``MilvusRetriever`` (e.g. the start of every benchmark_db.py process)
        starts with ``_client is None`` even though a same-named collection may
        already exist on a persistent Milvus server from a prior run. Skipping
        the delete in that case silently accumulates/mismatches stale entities
        across runs. ``drop_collection`` is a no-op when the collection does not
        exist (verified: no exception is raised).
        """
        try:
            client = self._get_client()
            client.drop_collection(self._config.collection_name)
            self._collection_ready = False
            self._next_id = 0
            # Also clean up the local db file if running in memory mode
            if self._config.in_memory:
                db_path = f"{self._config.collection_name}.db"
                if os.path.exists(db_path):
                    try:
                        if os.path.isdir(db_path):
                            import shutil
                            shutil.rmtree(db_path)
                        else:
                            os.remove(db_path)
                    except OSError:
                        pass
        except Exception as e:
            raise RuntimeError(f"Failed to clear Milvus: {e}") from e

    def describe_index(self) -> dict[str, Any]:
        """Inspect and return runtime index configuration for Milvus."""
        info: dict[str, Any] = {
            "in_memory": self._config.in_memory,
            "collection_name": self._config.collection_name,
            "distance_metric": self._config.distance_metric,
        }
        info["hnsw_ef_search"] = self._config.hnsw_ef_search
        if self._client is not None and self._collection_ready:
            try:
                client = self._get_client()
                index_details = client.describe_index(collection_name=self._config.collection_name, index_name="vector")
                info["index_details"] = index_details
                info["index_type"] = index_details.get("index_type")
                # pymilvus returns HNSW params ("M", "efConstruction") as top-level keys
                # in index_details, not nested under a "params" key.
                params = index_details.get("params", index_details)
                info["params"] = params
                if "M" in params:
                    info["hnsw_m"] = int(params["M"])
                if "efConstruction" in params:
                    info["hnsw_ef_construction"] = int(params["efConstruction"])
            except Exception as e:
                info["describe_error"] = str(e)
        return info

