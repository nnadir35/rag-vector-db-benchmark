"""Interface for text embedding.

Embedding is the process of converting text (chunks or queries) into
dense vector representations suitable for semantic search.
"""

from abc import ABC, abstractmethod
from typing import Sequence

from .types import Chunk, Embedding, Query


class Embedder(ABC):
    """Abstract interface for text embedding.
    
    Embedders are responsible for converting text into dense vector
    representations. The embedding model, dimension, and normalization
    strategy are implementation-specific.
    
    Implementations must handle:
    - Converting text to embeddings
    - Maintaining consistent embedding dimensions
    - Handling batch processing efficiently
    - Managing embedding model lifecycle
    
    Implementations must NOT:
    - Know about retrieval or generation
    - Know about vector database storage
    - Perform chunking or document processing
    """
    
    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """Embed a single chunk.
        
        Args:
            chunk: The chunk to embed
            
        Returns:
            Embedding vector for the chunk
            
        Raises:
            ValueError: If chunk is invalid
            RuntimeError: If embedding fails
        """
        pass
    
    @abstractmethod
    def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        """Embed multiple chunks in batch.
        
        This method allows implementations to optimize batch processing.
        The order of returned embeddings must match the order of input chunks.
        
        Args:
            chunks: Sequence of chunks to embed
            
        Returns:
            Sequence of embeddings, one per chunk, in the same order
            
        Raises:
            ValueError: If any chunk is invalid
            RuntimeError: If embedding fails
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: Query) -> Embedding:
        """Embed a query.
        
        Queries may be embedded differently than chunks (e.g., using
        a different model or normalization). This is implementation-specific.
        
        Args:
            query: The query to embed
            
        Returns:
            Embedding vector for the query
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If embedding fails
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder.
        
        Returns:
            The dimension of embedding vectors (must be consistent)
        """
        pass
