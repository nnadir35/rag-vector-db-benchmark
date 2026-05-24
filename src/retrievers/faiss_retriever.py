"""FAISS retriever implementation.

This module provides a concrete implementation of the Retriever interface
that uses FAISS as the vector database backend.
"""

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import faiss
else:
    # Lazy import - only import when actually used
    faiss = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    Embedding,
    Query,
    RetrievalResult,
    RetrievedChunk,
)
from .config import FAISSRetrieverConfig


class FAISSRetriever(Retriever):
    """FAISS-based retriever implementation.

    This retriever stores chunks and their embeddings in a FAISS index
    and performs similarity search to retrieve relevant chunks for queries.
    """

    def __init__(
        self,
        config: FAISSRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize FAISS retriever.

        Args:
            config: Configuration for FAISS usage
            embedder: Embedder instance to use for query embedding
        """
        self._config = config
        self._embedder = embedder

        self._index: Any | None = None
        self._metadata: dict[int, Chunk] = {}
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_id: int = 0

    def _ensure_faiss_imported(self) -> None:
        """Ensure faiss module is imported.

        Raises:
            ImportError: If faiss package is not installed
        """
        global faiss
        if faiss is None:
            try:
                import faiss as _faiss
                faiss = _faiss
            except ImportError:
                raise ImportError(
                    "faiss package is required for FAISSRetriever. "
                    "Install it with: pip install faiss-cpu"
                ) from None

    def _ensure_initialized(self, dimension: int) -> None:
        """Ensure the FAISS index is initialized and loaded from disk if persisted.

        Args:
            dimension: Dimension of the vector space

        Raises:
            RuntimeError: If loading from disk fails
            ValueError: If the distance metric is unsupported
        """
        self._ensure_faiss_imported()
        if self._index is not None:
            return

        # Check if persist files exist
        if self._config.persist_path:
            index_path = f"{self._config.persist_path}.index"
            pkl_path = f"{self._config.persist_path}.pkl"
            if os.path.exists(index_path) and os.path.exists(pkl_path):
                try:
                    self._index = faiss.read_index(index_path)
                    import pickle
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        self._metadata = data.get("metadata", {})
                        self._id_to_int = data.get("id_to_int", {})
                        self._int_to_id = data.get("int_to_id", {})
                        self._next_id = data.get("next_id", 0)
                    return
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load FAISS index from persist path: {e}"
                    ) from e

        # Create new index
        metric = self._config.distance_metric
        if metric in ("cosine", "ip"):
            sub_index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            sub_index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
        self._index = faiss.IndexIDMap(sub_index)

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Add chunks and their embeddings to FAISS index.

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
            self._ensure_initialized(dimension=first_dim)

            vecs_list = []
            for i, emb in enumerate(embeddings):
                if emb.dimension != first_dim:
                    raise ValueError(
                        "All embeddings must share the same dimension for FAISS indexing"
                    )
                vecs_list.append(emb.vector)

            vecs = np.array(vecs_list, dtype=np.float32)

            if self._config.distance_metric == "cosine":
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms

            ids = []
            for chunk in chunks:
                if chunk.id not in self._id_to_int:
                    self._id_to_int[chunk.id] = self._next_id
                    self._int_to_id[self._next_id] = chunk.id
                    self._next_id += 1
                int_id = self._id_to_int[chunk.id]
                self._metadata[int_id] = chunk
                ids.append(int_id)

            ids_arr = np.array(ids, dtype=np.int64)

            self._index.add_with_ids(vecs, ids_arr)

            # Persist to disk if persist_path is set
            if self._config.persist_path:
                dir_name = os.path.dirname(self._config.persist_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                faiss.write_index(self._index, f"{self._config.persist_path}.index")
                import pickle
                data = {
                    "metadata": self._metadata,
                    "id_to_int": self._id_to_int,
                    "int_to_id": self._int_to_id,
                    "next_id": self._next_id,
                }
                with open(f"{self._config.persist_path}.pkl", "wb") as f:
                    pickle.dump(data, f)

        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to FAISS: {e}") from e

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

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
            ValueError: If embedding dimension doesn't match index
            RuntimeError: If retrieval fails
        """
        if query_embedding.dimension != self._embedder.get_dimension():
            raise ValueError(
                f"Query embedding dimension ({query_embedding.dimension}) does not match "
                f"embedder dimension ({self._embedder.get_dimension()})"
            )

        try:
            self._ensure_initialized(dimension=query_embedding.dimension)

            if self._index.ntotal == 0:
                query = Query(id=query_id or "unknown", text="")
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    metadata={
                        "retrieval_latency_seconds": 0.0,
                        "num_results": 0,
                        "collection_name": self._config.collection_name,
                    },
                )

            query_vec = np.array([query_embedding.vector], dtype=np.float32)

            if self._config.distance_metric == "cosine":
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm

            start_time = time.time()
            distances, ids = self._index.search(query_vec, top_k)
            retrieval_time = time.time() - start_time

            retrieved_chunks = []
            for rank, (dist, idx) in enumerate(zip(distances[0], ids[0], strict=False)):
                if idx == -1:
                    continue

                chunk = self._metadata.get(idx)
                if chunk is None:
                    continue

                score = float(dist)
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
            raise RuntimeError(f"Failed to retrieve from FAISS: {e}") from e

    def clear(self) -> None:
        """Clear all chunks from the FAISS index."""
        try:
            self._ensure_faiss_imported()
            self._index = None
            self._metadata.clear()
            self._id_to_int.clear()
            self._int_to_id.clear()
            self._next_id = 0

            if self._config.persist_path:
                index_path = f"{self._config.persist_path}.index"
                pkl_path = f"{self._config.persist_path}.pkl"
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
        except Exception as e:
            raise RuntimeError(f"Failed to clear FAISS index: {e}") from e
