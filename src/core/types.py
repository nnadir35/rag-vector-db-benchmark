"""Core data types for the RAG benchmark framework.

This module defines the fundamental data structures used throughout the framework.
All types are immutable and designed to represent clear I/O contracts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable metadata associated with a document.
    
    This type represents arbitrary key-value metadata that can be attached
    to documents. All values must be JSON-serializable to ensure
    compatibility with various storage backends.
    
    Attributes:
        source: Optional source identifier (e.g., file path, URL, database ID)
        title: Optional document title
        author: Optional document author
        created_at: Optional creation timestamp
        custom: Additional custom metadata fields
    """
    
    source: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    """Represents a complete document in the system.
    
    Documents are the atomic unit of ingestion. They contain raw text content
    and metadata, but are not yet processed (chunked, embedded, etc.).
    
    Attributes:
        id: Unique identifier for the document
        content: Raw text content of the document
        metadata: Associated metadata
    """
    
    id: str
    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


@dataclass(frozen=True)
class ChunkMetadata:
    """Immutable metadata associated with a chunk.
    
    Chunk metadata extends document metadata with chunk-specific information
    such as position within the source document.
    
    Attributes:
        document_id: ID of the source document
        chunk_index: Zero-based index of this chunk within the document
        start_char: Character offset where chunk starts in source document
        end_char: Character offset where chunk ends in source document
        custom: Additional custom metadata fields
    """
    
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Represents a text chunk derived from a document.
    
    Chunks are the unit of retrieval. They are created by chunking documents
    and are the entities that get embedded and stored in vector databases.
    
    Attributes:
        id: Unique identifier for the chunk
        content: Text content of the chunk
        metadata: Associated metadata including source document info
    """
    
    id: str
    content: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class Embedding:
    """Represents a vector embedding.
    
    Embeddings are dense vector representations of text (chunks or queries).
    The dimension is determined by the embedding model used.
    
    Attributes:
        vector: The embedding vector as a sequence of floats
        dimension: The dimension of the embedding vector
    """
    
    vector: Sequence[float]
    dimension: int
    
    def __post_init__(self) -> None:
        """Validate that vector length matches dimension."""
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector length {len(self.vector)} does not match dimension {self.dimension}"
            )


@dataclass(frozen=True)
class Query:
    """Represents a user query.
    
    Queries are the input to the retrieval and generation components.
    They contain the query text and optional metadata.
    
    Attributes:
        id: Unique identifier for the query
        text: The query text
        metadata: Optional query metadata
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a chunk retrieved by a retriever.
    
    This extends Chunk with retrieval-specific information such as
    relevance score and rank.
    
    Attributes:
        chunk: The retrieved chunk
        score: Relevance score (higher is more relevant)
        rank: Zero-based rank in the retrieval results
    """
    
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a retrieval operation.
    
    Contains the ranked list of retrieved chunks along with metadata
    about the retrieval operation itself.
    
    Attributes:
        query: The query that was used for retrieval
        chunks: Ranked list of retrieved chunks (ordered by relevance)
        metadata: Additional metadata about the retrieval (latency, model used, etc.)
    """
    
    query: Query
    chunks: Sequence[RetrievedChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Result of a generation operation.
    
    Contains the generated response along with metadata about the
    generation process.
    
    Attributes:
        query: The query that was used for generation
        response: The generated text response
        retrieved_chunks: The chunks that were provided as context
        metadata: Additional metadata (latency, model used, tokens, etc.)
    """
    
    query: Query
    response: str
    retrieved_chunks: Sequence[RetrievedChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)
