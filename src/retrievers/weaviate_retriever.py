# src/retrievers/weaviate_retriever.py
"""Weaviate retriever implementation.

Implements the generic :class:`src.core.retrieval.Retriever` interface using
Weaviate as the vector store. The implementation supports two modes:

* **in_memory=True** – a lightweight pure‑Python mock that stores embeddings
  in a dictionary and performs cosine similarity with NumPy.
* **in_memory=False** – a real Weaviate client (`weaviate-client` v4).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    import weaviate
    import weaviate.classes as wvc
else:
    weaviate = None
    wvc = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    ChunkMetadata,
    Embedding,
    Query,
    RetrievalResult,
    RetrievedChunk,
)
from .config import WeaviateRetrieverConfig


class WeaviateRetriever(Retriever):
    """Retriever that stores chunks in Weaviate.

    Parameters
    ----------
    config: WeaviateRetrieverConfig
        Configuration describing the collection name, distance metric, host etc.
    embedder: Embedder
        The embedder used for ``retrieve``
    """

    def __init__(self, config: WeaviateRetrieverConfig, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._client: Any | None = None
        self._collection: Any | None = None
        # In‑memory mock storage: id -> (vector, Chunk)
        self._store: Dict[str, tuple[np.ndarray, Chunk]] = {}

    def _ensure_weaviate_imported(self) -> None:
        """Import ``weaviate`` lazily when required."""
        global weaviate, wvc
        if weaviate is None or wvc is None:
            try:
                import weaviate as wv
                import weaviate.classes as wv_classes
            except ImportError as exc:
                raise ImportError(
                    "weaviate-client>=4.0.0 is required for WeaviateRetriever. "
                    "Install it with: pip install weaviate-client"
                ) from exc
            weaviate = wv
            wvc = wv_classes

    def _get_client(self) -> Any:
        if self._client is None:
            self._ensure_weaviate_imported()
            port = int(self._config.host.split(":")[-1])
            host = self._config.host.split("://")[-1].split(":")[0]
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
            )
        return self._client

    def _ensure_collection(self) -> None:
        client = self._get_client()
        if client.collections.exists(self._config.collection_name):
            self._collection = client.collections.get(self._config.collection_name)
            return

        # Map distance metric
        distance = wvc.config.VectorDistances.COSINE
        if self._config.distance_metric == "dot":
            distance = wvc.config.VectorDistances.DOT
        elif self._config.distance_metric == "l2-squared":
            distance = wvc.config.VectorDistances.L2_SQUARED
        elif self._config.distance_metric == "manhattan":
            distance = wvc.config.VectorDistances.MANHATTAN
        elif self._config.distance_metric == "hamming":
            distance = wvc.config.VectorDistances.HAMMING

        self._collection = client.collections.create(
            name=self._config.collection_name,
            properties=[
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="start_char", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="end_char", data_type=wvc.config.DataType.INT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=distance,
                max_connections=self._config.hnsw_m,
                ef_construction=self._config.hnsw_ef_construction,
                ef=self._config.hnsw_ef_search,
            ),
        )

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )
        if not chunks:
            return
        if self._config.in_memory:
            for chunk, emb in zip(chunks, embeddings):
                vec = np.asarray(emb.vector, dtype=np.float32)
                self._store[chunk.id] = (vec, chunk)
            return
            
        self._ensure_collection()
        with self._collection.batch.dynamic() as batch:
            for chunk, emb in zip(chunks, embeddings):
                meta = chunk.metadata
                properties = {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "document_id": meta.document_id,
                    "chunk_index": meta.chunk_index,
                    "start_char": meta.start_char,
                    "end_char": meta.end_char,
                }
                batch.add_object(
                    properties=properties,
                    vector=list(emb.vector),
                )
        if self._collection.batch.failed_objects:
            raise RuntimeError(f"Weaviate batch indexing errors: {self._collection.batch.failed_objects}")

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        start = time.time()
        query_emb = self._embedder.embed_query(query)
        embed_latency = time.time() - start
        result = self.retrieve_with_embedding(query_emb, top_k=top_k, query_id=query.id)
        metadata = dict(result.metadata)
        metadata["embedding_latency_seconds"] = embed_latency
        return RetrievalResult(query=query, chunks=result.chunks, metadata=metadata)

    def retrieve_with_embedding(
        self, query_embedding: Embedding, top_k: int = 10, query_id: str | None = None
    ) -> RetrievalResult:
        if self._config.in_memory:
            q_vec = np.asarray(query_embedding.vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                raise ValueError("Query embedding has zero magnitude.")
            q_vec = q_vec / q_norm
            scores_chunks: List[tuple[float, Chunk]] = []
            for vec, chunk in self._store.values():
                c_norm = np.linalg.norm(vec)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, vec / c_norm))
                scores_chunks.append((sim, chunk))
            scores_chunks.sort(key=lambda x: x[0], reverse=True)
            top = scores_chunks[:top_k]
            retrieved: List[RetrievedChunk] = []
            for rank, (score, chunk) in enumerate(top):
                retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))
            metadata = {
                "retrieval_latency_seconds": 0.0,
                "num_results": len(retrieved),
                "collection_name": self._config.collection_name,
                "mode": "in_memory",
            }
            query_obj = Query(id=query_id or "unknown", text="")
            return RetrievalResult(query=query_obj, chunks=retrieved, metadata=metadata)

        self._ensure_collection()
        start = time.time()
        
        response = self._collection.query.near_vector(
            near_vector=list(query_embedding.vector),
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
            return_properties=["chunk_id", "content", "document_id", "chunk_index", "start_char", "end_char"]
        )
        latency = time.time() - start
        
        retrieved: List[RetrievedChunk] = []
        for rank, obj in enumerate(response.objects):
            props = obj.properties
            meta = ChunkMetadata(
                document_id=props.get("document_id"),
                chunk_index=props.get("chunk_index"),
                start_char=props.get("start_char"),
                end_char=props.get("end_char"),
            )
            chunk = Chunk(id=props.get("chunk_id"), content=props.get("content"), metadata=meta)
            # Distance from weaviate
            dist = obj.metadata.distance if obj.metadata.distance is not None else 0.0
            # Convert distance to similarity score
            score = 1.0 - dist
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))
            
        metadata = {
            "retrieval_latency_seconds": latency,
            "num_results": len(retrieved),
            "collection_name": self._config.collection_name,
            "mode": "weaviate",
        }
        query_obj = Query(id=query_id or "unknown", text="")
        return RetrievalResult(query=query_obj, chunks=retrieved, metadata=metadata)

    def clear(self) -> None:
        if self._config.in_memory:
            self._store.clear()
            return
        client = self._get_client()
        if client.collections.exists(self._config.collection_name):
            client.collections.delete(self._config.collection_name)
        self._collection = None

    def describe_index(self) -> dict[str, Any]:
        """Inspect and return runtime index configuration for Weaviate."""
        info: dict[str, Any] = {
            "in_memory": self._config.in_memory,
            "collection_name": self._config.collection_name,
            "distance_metric": self._config.distance_metric,
        }
        if not self._config.in_memory:
            try:
                client = self._get_client()
                if client.collections.exists(self._config.collection_name):
                    col = client.collections.get(self._config.collection_name)
                    col_cfg = col.config.get()
                    info["vector_index_config"] = str(getattr(col_cfg, "vector_index_config", None))
                    vec_cfg = getattr(col_cfg, "vector_index_config", None)
                    if vec_cfg is not None:
                        info["hnsw_m"] = getattr(vec_cfg, "max_connections", None)
                        info["hnsw_ef_construction"] = getattr(vec_cfg, "ef_construction", None)
                        info["hnsw_ef_search"] = getattr(vec_cfg, "ef", None)
            except Exception:
                pass
        return info

