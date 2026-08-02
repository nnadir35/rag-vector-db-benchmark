"""Interface for retrieval operations.

Retrieval is the process of finding relevant document chunks for a given query
using vector similarity search.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .types import Chunk, Embedding, Query, RetrievalResult


class Retriever(ABC):
    """Abstract interface for retrieval operations."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """Add chunks and their embeddings to the retriever's index."""
        pass

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        pass

    @abstractmethod
    def retrieve_with_embedding(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        query_id: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using a pre-computed query embedding."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks from the retriever's index."""
        pass

    def describe_index(self) -> dict[str, Any]:
        """Inspect and return effective runtime index configuration from vector database.

        Returns:
            Dictionary containing index metadata (e.g. index_type, hnsw params, in_memory status).
        """
        return {}
