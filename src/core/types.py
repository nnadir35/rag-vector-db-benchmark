"""Core data types for the RAG benchmark framework.

This module defines the fundamental data structures used throughout the framework.
All types are immutable and designed to represent clear I/O contracts.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


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

    source: str | None = None
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    custom: dict[str, Any] = field(default_factory=dict)


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
    custom: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGResponse:
    """Represents the complete end-to-end RAG pipeline response.

    This type encapsulates the full result of a RAG pipeline execution,
    combining the query, retrieved context, and generated response.
    It serves as the primary output type for RAG pipelines and the
    input type for end-to-end evaluation.

    This separation from GenerationResult allows for clear distinction
    between component-level results (GenerationResult) and pipeline-level
    results (RAGResponse), enabling independent evaluation of components
    versus the full system.

    Attributes:
        query: The original query that initiated the RAG pipeline
        retrieved_chunks: The chunks retrieved by the retriever component
        response: The final generated response from the generator component
        metadata: Pipeline-level metadata (total latency, component versions, etc.)
    """

    query: Query
    retrieved_chunks: Sequence[RetrievedChunk]
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Generic container for evaluation results without prescribed metrics.

    This type represents the output of an evaluation operation without
    embedding specific metric definitions. Metrics are computed separately
    by evaluators and stored externally, allowing for flexible metric
    composition and comparison across different evaluation strategies.

    The generic design enables future-proofing: new metrics can be added
    without modifying this core type, and different evaluators can produce
    different metric sets while using the same result container.

    This separation ensures that the core data model remains stable while
    evaluation logic evolves independently, supporting the framework's
    principle of separating evaluation concerns from pipeline logic.

    Attributes:
        subject_id: Identifier for what was evaluated (e.g., query ID, experiment ID)
        subject_type: Type of subject (e.g., 'retrieval', 'generation', 'rag_response')
        data: Generic evaluation data (ground truth, predictions, intermediate results)
        metadata: Additional metadata about the evaluation (evaluator name, timestamp, etc.)
    """

    subject_id: str
    subject_type: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
