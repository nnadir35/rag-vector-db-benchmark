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


FAISS_ADD_BATCH_SIZE = 512  # modül seviyesinde sabit


class FAISSRetriever(Retriever):
    """FAISS-based retriever implementation."""

    def __init__(
        self,
        config: FAISSRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize FAISS retriever.

        Args:
            config: Configuration for FAISS connection and settings
            embedder: Embedder instance to use for query embedding
        """
        self._config = config
        self._embedder = embedder

        self._index: Any | None = None
        self._id_to_chunk: dict[int, Chunk] = {}
        self._next_id: int = 0

    def _ensure_faiss_imported(self) -> None:
        """Ensure faiss module is imported.

        Raises:
            ImportError: If faiss package is not installed
        """
        global faiss
        if faiss is None:
            import os
            # Set OMP_NUM_THREADS to 1 to avoid thread/OpenMP conflicts on macOS
            os.environ["OMP_NUM_THREADS"] = "1"
            try:
                import faiss as _faiss
                faiss = _faiss
            except ImportError:
                raise ImportError(
                    "faiss-cpu package is required for FAISSRetriever. "
                    "Install it with: pip install faiss-cpu"
                ) from None

    def _get_or_create_index(self, vector_size: int) -> Any:
        """Get the existing FAISS index or create a new one.

        Args:
            vector_size: Dimension of the vector space

        Returns:
            The FAISS index instance
        """
        self._ensure_faiss_imported()
        if self._index is None:
            if self._config.persist_path:
                self._load_from_disk()

        if self._index is None:
            metric = self._config.distance_metric
            if metric == "l2":
                self._index = faiss.IndexFlatL2(vector_size)
            else:  # cosine veya ip
                self._index = faiss.IndexFlatIP(vector_size)

        return self._index

    def _normalize_if_cosine(self, vectors: np.ndarray) -> np.ndarray:
        """Cosine modunda vektörleri normalize eder, yeni array döner."""
        if self._config.distance_metric == "cosine":
            # C-contiguous, writable ve float32 mi kontrol et. Değilse kopyasını al, segfault'u önle.
            if not (
                isinstance(vectors, np.ndarray)
                and vectors.flags.c_contiguous
                and vectors.flags.writeable
                and vectors.dtype == np.float32
            ):
                vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
            self._ensure_faiss_imported()
            faiss.normalize_L2(vectors)
        return vectors

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
            for emb in embeddings:
                if emb.dimension != first_dim:
                    raise ValueError(
                        "All embeddings must share the same dimension for FAISS indexing"
                    )

            self._get_or_create_index(first_dim)

            # Batch'li insert — büyük vektör matrisinde segfault önler
            for batch_start in range(0, len(chunks), FAISS_ADD_BATCH_SIZE):
                batch_end = batch_start + FAISS_ADD_BATCH_SIZE
                chunk_batch = chunks[batch_start:batch_end]
                emb_batch = embeddings[batch_start:batch_end]

                vectors = np.array(
                    [list(e.vector) for e in emb_batch],
                    dtype=np.float32,
                    order="C",
                    copy=True
                )
                vectors = self._normalize_if_cosine(vectors)
                self._index.add(vectors)

                for chunk in chunk_batch:
                    self._id_to_chunk[self._next_id] = chunk
                    self._next_id += 1

            if self._config.persist_path:
                self._save_to_disk()

        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to FAISS: {e}") from e

    def _save_to_disk(self) -> None:
        """Save FAISS index and metadata to disk."""
        self._ensure_faiss_imported()
        if not self._config.persist_path:
            return

        try:
            dir_name = os.path.dirname(self._config.persist_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            faiss.write_index(self._index, self._config.persist_path + ".index")

            import pickle
            with open(self._config.persist_path + ".meta", "wb") as f:
                pickle.dump(self._id_to_chunk, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index to disk: {e}") from e

    def _load_from_disk(self) -> None:
        """Load FAISS index and metadata from disk."""
        self._ensure_faiss_imported()
        if not self._config.persist_path:
            return

        index_path = self._config.persist_path + ".index"
        meta_path = self._config.persist_path + ".meta"

        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self._index = faiss.read_index(index_path)
                import pickle
                with open(meta_path, "rb") as f:
                    self._id_to_chunk = pickle.load(f)
                self._next_id = len(self._id_to_chunk)
            except Exception as e:
                raise RuntimeError(f"Failed to load FAISS index from disk: {e}") from e

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

        if self._index is None and self._config.persist_path:
            self._load_from_disk()

        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError(
                "Cannot retrieve from FAISS: index is empty. Call add_chunks first."
            )

        try:
            # Sorgu vektörü: C-contiguous, writable, float32
            query_vector = np.array(
                [list(query_embedding.vector)], dtype=np.float32, order="C", copy=True
            )
            query_vector = self._normalize_if_cosine(query_vector)

            start_time = time.time()
            distances, indices = self._index.search(query_vector, top_k)
            retrieval_time = time.time() - start_time

            retrieved_chunks = []
            for rank in range(len(indices[0])):
                idx = indices[0][rank]
                if idx == -1:
                    continue

                chunk = self._id_to_chunk.get(idx)
                if chunk is None:
                    continue

                score = float(distances[0][rank])
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
                "index_total_vectors": self._index.ntotal,
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
            self._id_to_chunk = {}
            self._next_id = 0

            if self._config.persist_path:
                index_path = self._config.persist_path + ".index"
                meta_path = self._config.persist_path + ".meta"
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
        except Exception as e:
            raise RuntimeError(f"Failed to clear FAISS index: {e}") from e
