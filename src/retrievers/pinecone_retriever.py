"""Pinecone retriever implementation.

This module provides a concrete implementation of the Retriever interface
that uses Pinecone as the vector database backend.
"""

import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pinecone
else:
    # Lazy import - only import when actually used
    pinecone = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    Embedding,
    Query,
    RetrievalResult,
    RetrievedChunk,
)
from .config import PineconeRetrieverConfig


class PineconeRetriever(Retriever):
    """Pinecone-based retriever implementation.

    This retriever stores chunks and their embeddings in a Pinecone index
    and performs similarity search to retrieve relevant chunks for queries.

    The retriever uses an Embedder instance to convert queries to embeddings,
    maintaining separation of concerns between embedding and retrieval logic.

    Example:
        ```python
        from src.core.embedding import Embedder
        from src.retrievers import PineconeRetriever, PineconeRetrieverConfig

        config = PineconeRetrieverConfig(
            index_name="my-index",
            dimension=384
        )
        embedder = SomeEmbedder()  # Your embedder implementation
        retriever = PineconeRetriever(config, embedder)

        # Add chunks
        retriever.add_chunks(chunks, embeddings)

        # Retrieve
        result = retriever.retrieve(query)
        ```
    """

    def __init__(
        self,
        config: PineconeRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize Pinecone retriever.

        Args:
            config: Configuration for Pinecone connection and index settings
            embedder: Embedder instance to use for query embedding

        Raises:
            ValueError: If embedder dimension doesn't match config dimension
            RuntimeError: If Pinecone connection fails
        """
        self._config = config
        self._embedder = embedder

        # Validate embedder dimension matches config
        if embedder.get_dimension() != config.dimension:
            raise ValueError(
                f"Embedder dimension ({embedder.get_dimension()}) does not match "
                f"config dimension ({config.dimension})"
            )

        # Lazy initialization - client and index will be created on first use
        self._client: Any | None = None  # pinecone.Pinecone
        self._index: Any | None = None  # pinecone.Index

    def _ensure_pinecone_imported(self) -> None:
        """Ensure pinecone module is imported.

        Raises:
            ImportError: If pinecone package is not installed
        """
        global pinecone
        if pinecone is None:
            try:
                import pinecone
            except ImportError:
                raise ImportError(
                    "pinecone package is required for PineconeRetriever. "
                    "Install it with: pip install pinecone-client or pip install pinecone"
                )

    @property
    def _pinecone_index(self) -> "pinecone.Index":
        """Get or create Pinecone index connection.

        This property implements lazy initialization - the Pinecone client
        and index connection are only created when first needed.

        Returns:
            Pinecone Index instance

        Raises:
            RuntimeError: If connection fails
            ImportError: If pinecone package is not installed
        """
        self._ensure_pinecone_imported()

        if self._index is None:
            try:
                # Initialize Pinecone client
                self._client = pinecone.Pinecone(api_key=self._config.api_key)

                # Get index
                self._index = self._client.Index(
                    name=self._config.index_name,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to Pinecone index '{self._config.index_name}': {e}"
                ) from e

        return self._index

    def _chunk_metadata_to_dict(self, chunk: Chunk) -> dict:
        """Convert Chunk metadata to Pinecone-compatible dictionary.

        Pinecone metadata must be JSON-serializable and contain only
        scalar values (str, int, float, bool, None) or lists/dicts of scalars.

        Args:
            chunk: Chunk to extract metadata from

        Returns:
            Dictionary suitable for Pinecone metadata field
        """
        metadata = {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.metadata.document_id,
            "chunk_index": chunk.metadata.chunk_index,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
        }

        # Add custom metadata, ensuring JSON serializability
        for key, value in chunk.metadata.custom.items():
            try:
                # Test JSON serializability
                json.dumps(value)
                metadata[f"custom_{key}"] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue

        return metadata

    def _dict_to_chunk(self, chunk_id: str, metadata: dict) -> Chunk:
        """Reconstruct Chunk from Pinecone metadata.

        This reverses the _chunk_metadata_to_dict operation, reconstructing
        a Chunk object from the stored metadata.

        Args:
            chunk_id: The chunk ID stored in Pinecone
            metadata: Metadata dictionary from Pinecone

        Returns:
            Reconstructed Chunk object
        """
        from ..core.types import ChunkMetadata

        # Extract custom metadata (keys starting with "custom_")
        custom_metadata = {}
        for key, value in metadata.items():
            if key.startswith("custom_"):
                custom_key = key[7:]  # Remove "custom_" prefix
                custom_metadata[custom_key] = value

        chunk_metadata = ChunkMetadata(
            document_id=metadata["document_id"],
            chunk_index=metadata["chunk_index"],
            start_char=metadata["start_char"],
            end_char=metadata["end_char"],
            custom=custom_metadata,
        )

        return Chunk(
            id=chunk_id,
            content=metadata["content"],
            metadata=chunk_metadata,
        )

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Add chunks and their embeddings to Pinecone index.

        This method stores chunks and embeddings in Pinecone. The chunks
        are stored as metadata, and embeddings are stored as vectors.

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

        # Validate embedding dimensions
        for i, embedding in enumerate(embeddings):
            if embedding.dimension != self._config.dimension:
                raise ValueError(
                    f"Embedding at index {i} has dimension {embedding.dimension}, "
                    f"expected {self._config.dimension}"
                )

        try:
            index = self._pinecone_index

            # Prepare vectors for upsert
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                metadata = self._chunk_metadata_to_dict(chunk)
                vectors_to_upsert.append({
                    "id": chunk.id,
                    "values": list(embedding.vector),
                    "metadata": metadata,
                })

            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                index.upsert(
                    vectors=batch,
                    namespace=self._config.namespace,
                )

        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to Pinecone: {e}") from e

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        This method embeds the query using the embedder and searches
        for similar chunks in Pinecone.

        Args:
            query: The query to retrieve chunks for
            top_k: Maximum number of chunks to retrieve

        Returns:
            RetrievalResult containing ranked chunks and metadata

        Raises:
            ValueError: If query is invalid
            RuntimeError: If retrieval fails
        """
        # Embed query
        start_time = time.time()
        query_embedding = self._embedder.embed_query(query)
        embedding_time = time.time() - start_time

        # Retrieve using embedding
        result = self.retrieve_with_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            query_id=query.id,
        )

        # Add embedding latency to metadata and preserve original query
        metadata = dict(result.metadata)
        metadata["embedding_latency_seconds"] = embedding_time

        # Create result with original query (preserving query text)
        return RetrievalResult(
            query=query,  # Use original query to preserve text
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

        This method performs similarity search in Pinecone using the
        provided embedding vector.

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
        if query_embedding.dimension != self._config.dimension:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.dimension}) does not match "
                f"index dimension ({self._config.dimension})"
            )

        try:
            start_time = time.time()
            index = self._pinecone_index

            # Query Pinecone
            query_response = index.query(
                vector=list(query_embedding.vector),
                top_k=top_k,
                namespace=self._config.namespace,
                include_metadata=True,
            )

            retrieval_time = time.time() - start_time

            # Convert results to RetrievedChunk objects
            retrieved_chunks = []
            for rank, match in enumerate(query_response.matches):
                chunk = self._dict_to_chunk(
                    chunk_id=match.id,
                    metadata=match.metadata,
                )

                # Pinecone returns scores, but the interpretation depends on metric
                # For cosine: higher is better (already normalized)
                # For euclidean: lower is better (distance)
                # For dotproduct: higher is better
                score = match.score if match.score is not None else 0.0

                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=score,
                        rank=rank,
                    )
                )

            # Create a dummy query if query_id not provided
            # (needed for RetrievalResult, but actual query text not available)
            query = Query(
                id=query_id or "unknown",
                text="",  # Not available when using pre-computed embedding
            )

            metadata = {
                "retrieval_latency_seconds": retrieval_time,
                "num_results": len(retrieved_chunks),
                "index_name": self._config.index_name,
                "namespace": self._config.namespace,
            }

            return RetrievalResult(
                query=query,
                chunks=retrieved_chunks,
                metadata=metadata,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve from Pinecone: {e}") from e

    def clear(self) -> None:
        """Clear all chunks from the Pinecone index.

        This method deletes all vectors from the index namespace,
        effectively resetting the retriever to an empty state.

        Raises:
            RuntimeError: If clearing fails
        """
        try:
            index = self._pinecone_index
            index.delete(delete_all=True, namespace=self._config.namespace)
        except Exception as e:
            raise RuntimeError(f"Failed to clear Pinecone index: {e}") from e
