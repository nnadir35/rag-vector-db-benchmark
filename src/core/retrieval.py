"""Interface for retrieval operations.

Retrieval is the process of finding relevant document chunks for a given query
using vector similarity search.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .types import Chunk, Embedding, Query, RetrievalResult


class Retriever(ABC):
    """Abstract interface for retrieval operations.

    Retrievers are responsible for finding relevant document chunks given
    a query. They encapsulate the vector database, embedding model, and
    retrieval logic (similarity search, filtering, ranking, etc.).

    Implementations must handle:
    - Storing embedded chunks in a vector database
    - Performing similarity search given a query embedding
    - Ranking results by relevance
    - Returning top-k results

    Implementations must NOT:
    - Know about generation or LLMs
    - Know about evaluation or benchmarking
    - Perform generation tasks
    """

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """Add chunks and their embeddings to the retriever's index.

        This method stores chunks and their corresponding embeddings in the
        retriever's vector database. The chunks can later be retrieved via
        retrieve() or retrieve_with_embedding().

        Args:
            chunks: Sequence of chunks to add
            embeddings: Sequence of embeddings corresponding to chunks
                (must be same length and order as chunks)

        Raises:
            ValueError: If chunks and embeddings have different lengths
            RuntimeError: If storage fails
        """
        pass

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        This method performs end-to-end retrieval: it embeds the query
        (using the retriever's internal embedder) and searches for similar
        chunks in the vector database.

        Args:
            query: The query to retrieve chunks for
            top_k: Maximum number of chunks to retrieve

        Returns:
            RetrievalResult containing ranked chunks and metadata

        Raises:
            ValueError: If query is invalid
            RuntimeError: If retrieval fails
        """
        pass

    @abstractmethod
    def retrieve_with_embedding(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        query_id: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using a pre-computed query embedding.

        This method allows callers to provide a pre-computed embedding,
        which can be useful for optimization or when using a different
        embedding model for queries.

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
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks from the retriever's index.

        This method removes all stored chunks and embeddings, effectively
        resetting the retriever to an empty state. Useful for testing or
        re-indexing scenarios.

        Raises:
            RuntimeError: If clearing fails
        """
        pass
