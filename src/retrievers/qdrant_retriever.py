"""Qdrant retriever implementation.

This module provides a concrete implementation of the Retriever interface
that uses Qdrant as the vector database backend.
"""

import json
import os
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
else:
    QdrantClient = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    Embedding,
    Query,
    RetrievalResult,
    RetrievedChunk,
)
from .config import QdrantRetrieverConfig


class QdrantRetriever(Retriever):
    """Qdrant-based retriever implementation."""

    def __init__(
        self,
        config: QdrantRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize Qdrant retriever.

        Args:
            config: Configuration for Qdrant usage
            embedder: Embedder instance to use for queries
        """
        self._config = config
        self._embedder = embedder

        self._client: Any | None = None
        self._collection_ready: bool = False

    def _ensure_qdrant_imported(self) -> None:
        """Ensure qdrant_client module is lazily imported."""
        global QdrantClient
        if QdrantClient is None:
            try:
                from qdrant_client import QdrantClient as _QdrantClient

                QdrantClient = _QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client package is required for QdrantRetriever. "
                    "Install it with: pip install qdrant-client"
                ) from None

    def _get_client(self) -> "QdrantClient":
        """Get or create Qdrant client.

        Connection resolution order (first match wins):

        1. ``QDRANT_URL`` — full URL (e.g. ``http://qdrant-server:6333``).
        2. ``QDRANT_HOST`` (optional ``QDRANT_PORT``, default ``6333``) — HTTP URL built
           for the official Qdrant server container.
        3. ``in_memory`` in config — embedded ``:memory:`` instance.
        4. Local persistence — ``persist_path`` on disk.

        Environment variables allow deployment (Docker Compose, k8s) without hardcoding
        hosts in application code.
        """
        self._ensure_qdrant_imported()

        if self._client is None:
            try:
                url = os.getenv("QDRANT_URL", "").strip()
                host = os.getenv("QDRANT_HOST", "").strip()
                if url:
                    self._client = QdrantClient(url=url)
                elif host:
                    port = int(os.getenv("QDRANT_PORT", "6333"))
                    self._client = QdrantClient(url=f"http://{host}:{port}")
                elif self._config.in_memory:
                    self._client = QdrantClient(":memory:")
                else:
                    if self._config.persist_path is None:
                        raise RuntimeError(
                            "persist_path must be set when in_memory is False "
                            "and QDRANT_URL/QDRANT_HOST are not set"
                        )
                    self._client = QdrantClient(path=self._config.persist_path)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Qdrant: {e}") from e

        return self._client

    def _distance_to_qdrant(self) -> Any:
        """Map config distance string to Qdrant Distance enum."""
        from qdrant_client.models import Distance

        mapping = {
            "cosine": Distance.COSINE,
            "l2": Distance.EUCLID,
            "euclidean": Distance.EUCLID,
            "ip": Distance.DOT,
            "dot": Distance.DOT,
        }
        return mapping[self._config.distance_metric]

    def _ensure_collection(self, vector_size: int) -> None:
        """Create collection if missing or dimension changed."""
        from qdrant_client.models import VectorParams

        client = self._get_client()
        name = self._config.collection_name

        collections = client.get_collections().collections
        exists = any(c.name == name for c in collections)

        if exists:
            info = client.get_collection(collection_name=name)
            params_vectors = info.config.params.vectors
            if isinstance(params_vectors, dict):
                current_size = next(iter(params_vectors.values())).size
            else:
                current_size = params_vectors.size
            if current_size != vector_size:
                client.delete_collection(collection_name=name)
                exists = False

        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self._distance_to_qdrant(),
                ),
            )

        self._collection_ready = True

    def _chunk_metadata_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        """Convert Chunk metadata to Qdrant payload fields."""
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.metadata.document_id,
            "chunk_index": chunk.metadata.chunk_index,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
        }

        for key, value in chunk.metadata.custom.items():
            if isinstance(value, (str, int, float, bool)):
                payload[f"custom_{key}"] = value
            else:
                try:
                    payload[f"custom_{key}"] = json.dumps(value)
                except (TypeError, ValueError):
                    pass
        return payload

    def _payload_to_chunk(self, payload: dict[str, Any]) -> Chunk:
        """Reconstruct Chunk from Qdrant payload."""
        from ..core.types import ChunkMetadata

        custom_metadata: dict[str, Any] = {}
        for key, value in payload.items():
            if key.startswith("custom_"):
                custom_key = key[7:]
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        custom_metadata[custom_key] = parsed
                    except (TypeError, ValueError):
                        custom_metadata[custom_key] = value
                else:
                    custom_metadata[custom_key] = value

        chunk_metadata = ChunkMetadata(
            document_id=payload["document_id"],
            chunk_index=payload["chunk_index"],
            start_char=payload["start_char"],
            end_char=payload["end_char"],
            custom=custom_metadata,
        )

        return Chunk(
            id=str(payload["chunk_id"]),
            content=str(payload["content"]),
            metadata=chunk_metadata,
        )

    @staticmethod
    def _point_id_for_chunk(chunk_id: str) -> str:
        """Build a deterministic UUID point id from chunk id."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Add chunks and embeddings to Qdrant."""
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
                        "All embeddings must share the same dimension for Qdrant indexing"
                    )

            self._ensure_collection(vector_size=first_dim)

            from qdrant_client.models import PointStruct

            client = self._get_client()
            points: list[Any] = []
            for chunk, emb in zip(chunks, embeddings, strict=False):
                payload = self._chunk_metadata_to_payload(chunk)
                points.append(
                    PointStruct(
                        id=self._point_id_for_chunk(chunk.id),
                        vector=list(emb.vector),
                        payload=payload,
                    )
                )

            batch_size = 100
            for i in range(0, len(points), batch_size):
                client.upsert(
                    collection_name=self._config.collection_name,
                    points=points[i : i + batch_size],
                )

        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to Qdrant: {e}") from e

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query using the embedder."""
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
        """Retrieve relevant chunks using a pre-computed query embedding."""
        try:
            start_time = time.time()
            if not self._collection_ready:
                raise RuntimeError(
                    "Cannot retrieve from Qdrant: collection is empty. Call add_chunks first."
                )

            client = self._get_client()

            hits = client.query_points(
                collection_name=self._config.collection_name,
                query=list(query_embedding.vector),
                limit=top_k,
                with_payload=True,
            ).points

            retrieval_time = time.time() - start_time

            retrieved_chunks: list[RetrievedChunk] = []
            for rank, hit in enumerate(hits):
                if hit.payload is None:
                    continue
                chunk = self._payload_to_chunk(hit.payload)
                score = float(hit.score)
                if self._config.distance_metric in ("l2", "euclidean"):
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
            raise RuntimeError(f"Failed to retrieve from Qdrant: {e}") from e

    def clear(self) -> None:
        """Clear the collection."""
        try:
            if self._client is not None and self._collection_ready:
                self._get_client().delete_collection(
                    collection_name=self._config.collection_name
                )
            self._collection_ready = False
        except Exception as e:
            raise RuntimeError(f"Failed to clear Qdrant: {e}") from e
