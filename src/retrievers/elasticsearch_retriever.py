# src/retrievers/elasticsearch_retriever.py
"""ElasticSearch retriever implementation.

Implements the generic :class:`src.core.retrieval.Retriever` interface using
Elasticsearch as the vector store.  The implementation supports two modes:

* **in_memory=True** – a lightweight pure‑Python mock that stores embeddings
  in a dictionary and performs cosine similarity with NumPy.  This mode is
  useful for CI, unit tests and environments where an Elasticsearch cluster is
  unavailable.
* **in_memory=False** – a real Elasticsearch client (`elasticsearch‑py`).  The
  index is created with a dense_vector mapping that respects the configured
  ``distance_metric`` (cosine, dot_product or l2_norm).  Bulk indexing and KNN
  search are used for efficiency.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    # Only imported for type checking – actual import is lazy.
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
else:
    Elasticsearch = None  # type: ignore
    bulk = None  # type: ignore

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
from .config import ElasticSearchRetrieverConfig


class ElasticSearchRetriever(Retriever):
    """Retriever that stores chunks in Elasticsearch.

    Parameters
    ----------
    config: ElasticSearchRetrieverConfig
        Configuration describing the index name, distance metric, host etc.
    embedder: Embedder
        The embedder used for ``retrieve`` – it is *not* used for
        ``add_chunks`` because embeddings are supplied explicitly.
    """

    def __init__(self, config: ElasticSearchRetrieverConfig, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._client: Any | None = None  # Elasticsearch client when real mode
        # In‑memory mock storage: id -> (vector, Chunk)
        self._store: Dict[str, tuple[np.ndarray, Chunk]] = {}

    # ---------------------------------------------------------------------
    # Lazy client handling
    # ---------------------------------------------------------------------
    def _ensure_es_imported(self) -> None:
        """Import ``elasticsearch`` lazily when required.

        Raises
        ------
        ImportError
            If the ``elasticsearch`` package is not installed.
        """
        global Elasticsearch, bulk  # pylint: disable=global-statement
        if Elasticsearch is None or bulk is None:
            try:
                from elasticsearch import Elasticsearch as ES
                from elasticsearch import __version__ as es_version
                from elasticsearch.helpers import bulk as es_bulk
            except ImportError as exc:
                raise ImportError(
                    "elasticsearch>=8.12.0 is required for ElasticSearchRetriever. "
                    "Install it with: pip install 'elasticsearch>=8.12.0'"
                ) from exc
            if es_version < (8, 12, 0):
                raise ImportError(
                    "elasticsearch>=8.12.0 is required for ElasticSearchRetriever. Install it with: pip install 'elasticsearch>=8.12.0'"
                )
            Elasticsearch = ES
            bulk = es_bulk

    def _get_client(self) -> Any:
        """Create (or reuse) an Elasticsearch client.

        The client is created on first use so that unit tests which keep
        ``in_memory=True`` never touch the network.
        """
        if self._client is None:
            self._ensure_es_imported()
            self._client = Elasticsearch([self._config.host], request_timeout=300.0)
        return self._client

    # ---------------------------------------------------------------------
    # Helper conversion methods
    # ---------------------------------------------------------------------
    def _chunk_to_doc(self, chunk: Chunk) -> Dict[str, Any]:
        """Serialize a :class:`Chunk` to a JSON document suitable for ES.

        The mapping used in ``_ensure_index`` stores the embedding in a
        ``dense_vector`` field called ``embedding`` and keeps the original chunk
        fields for debugging / inspection.
        """
        meta = chunk.metadata
        return {
            "id": chunk.id,
            "content": chunk.content,
            "document_id": meta.document_id,
            "chunk_index": meta.chunk_index,
            "start_char": meta.start_char,
            "end_char": meta.end_char,
            "custom": meta.custom,
            # ``embedding`` will be filled by the caller – it expects a list of
            # floats.
        }

    def _doc_to_chunk(self, doc: Dict[str, Any]) -> Chunk:
        """Re‑create a :class:`Chunk` from an ES document.
        """
        meta = ChunkMetadata(
            document_id=doc.get("document_id"),
            chunk_index=doc.get("chunk_index"),
            start_char=doc.get("start_char"),
            end_char=doc.get("end_char"),
            custom=doc.get("custom"),
        )
        return Chunk(id=doc.get("id"), content=doc.get("content"), metadata=meta)

    # ---------------------------------------------------------------------
    # Index management
    # ---------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Create the index with proper mapping if it does not yet exist.
        """
        client = self._get_client()
        if client.indices.exists(index=self._config.index_name):
            return
        # Mapping follows the ElasticSearch KNN dense_vector spec.
        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self._config.dims,
                        "index": True,
                        "similarity": self._config.distance_metric,
                        "index_options": {
                            "type": "hnsw",
                            "m": self._config.hnsw_m,
                            "ef_construction": self._config.hnsw_ef_construction,
                        },
                    },
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "document_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                    "custom": {"type": "object"},
                }
            }
        }
        client.indices.create(index=self._config.index_name, body=mapping)

    # ---------------------------------------------------------------------
    # Retriever interface implementation
    # ---------------------------------------------------------------------
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """Add ``chunks`` together with their pre‑computed ``embeddings``.

        In ``in_memory`` mode the data is stored in a local dictionary.  In the
        real mode bulk‑index operations are used for speed.
        """
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
        # Real Elasticsearch path ------------------------------------------------
        self._ensure_index()
        client = self._get_client()
        actions: List[Dict[str, Any]] = []
        for chunk, emb in zip(chunks, embeddings):
            doc = self._chunk_to_doc(chunk)
            doc["embedding"] = list(emb.vector)
            actions.append({"_index": self._config.index_name, "_source": doc})
        # ``bulk`` returns a tuple (successes, errors).  Errors are raised as an
        # exception for visibility.
        success, errors = bulk(client, actions, request_timeout=300.0)
        if errors:
            raise RuntimeError(f"Elasticsearch bulk indexing errors: {errors}")
        client.indices.refresh(index=self._config.index_name)

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Embed ``query`` using the configured ``embedder`` and delegate.
        """
        start = time.time()
        query_emb = self._embedder.embed_query(query)
        embed_latency = time.time() - start
        result = self.retrieve_with_embedding(query_emb, top_k=top_k, query_id=query.id)
        # Attach embedding latency to the metadata for benchmarking purposes.
        metadata = dict(result.metadata)
        metadata["embedding_latency_seconds"] = embed_latency
        return RetrievalResult(query=query, chunks=result.chunks, metadata=metadata)

    def retrieve_with_embedding(
        self, query_embedding: Embedding, top_k: int = 10, query_id: str | None = None
    ) -> RetrievalResult:
        """Perform a KNN search using a pre‑computed ``query_embedding``.
        """
        if self._config.in_memory:
            # Simple cosine similarity implementation for the mock.
            q_vec = np.asarray(query_embedding.vector, dtype=np.float32)
            # Normalise for cosine similarity.
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                raise ValueError("Query embedding has zero magnitude.")
            q_vec = q_vec / q_norm
            scores_chunks: List[tuple[float, Chunk]] = []
            for vec, chunk in self._store.values():
                # Cosine similarity = dot product of normalized vectors.
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
                "index_name": self._config.index_name,
                "mode": "in_memory",
            }
            query_obj = Query(id=query_id or "unknown", text="")
            return RetrievalResult(query=query_obj, chunks=retrieved, metadata=metadata)

        # Real Elasticsearch path ------------------------------------------------
        client = self._get_client()
        self._ensure_index()
        knn_query = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": list(query_embedding.vector),
                "k": top_k,
                "num_candidates": self._config.num_candidates
            },
            "_source": ["id", "content", "document_id", "chunk_index", "start_char", "end_char", "custom"],
        }
        start = time.time()
        resp = client.search(index=self._config.index_name, body=knn_query)
        latency = time.time() - start
        hits = resp.get("hits", {}).get("hits", [])
        retrieved: List[RetrievedChunk] = []
        for rank, hit in enumerate(hits):
            source = hit["_source"]
            chunk = self._doc_to_chunk(source)
            # ``_score`` is the similarity (higher = better).  For consistency
            # with other retrievers we keep the raw score.
            score = float(hit.get("_score", 0.0))
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))
        metadata = {
            "retrieval_latency_seconds": latency,
            "num_results": len(retrieved),
            "index_name": self._config.index_name,
            "mode": "elasticsearch",
        }
        query_obj = Query(id=query_id or "unknown", text="")
        return RetrievalResult(query=query_obj, chunks=retrieved, metadata=metadata)

    def clear(self) -> None:
        """Remove all data from the retriever.
        """
        if self._config.in_memory:
            self._store.clear()
            return
        client = self._get_client()
        if client.indices.exists(index=self._config.index_name):
            client.indices.delete(index=self._config.index_name)
        # Reset internal client – a fresh index will be created on the next call.
        self._client = None

    def describe_index(self) -> dict[str, Any]:
        """Inspect and return runtime index configuration for ElasticSearch."""
        info: dict[str, Any] = {
            "in_memory": self._config.in_memory,
            "index_name": self._config.index_name,
            "distance_metric": self._config.distance_metric,
            "num_candidates": self._config.num_candidates,
        }
        if not self._config.in_memory:
            try:
                client = self._get_client()
                if client.indices.exists(index=self._config.index_name):
                    mapping = client.indices.get_mapping(index=self._config.index_name)
                    info["mapping"] = mapping.body if hasattr(mapping, "body") else dict(mapping)
                    props = info["mapping"].get(self._config.index_name, {}).get("mappings", {}).get("properties", {}).get("embedding", {})
                    idx_opts = props.get("index_options", {})
                    info["index_options"] = idx_opts
                    if "m" in idx_opts:
                        info["hnsw_m"] = idx_opts["m"]
                    if "ef_construction" in idx_opts:
                        info["hnsw_ef_construction"] = idx_opts["ef_construction"]
            except Exception:
                pass
        return info


