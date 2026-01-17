"""Core interfaces and data types for the RAG benchmark framework."""

from .types import (
    Document,
    Chunk,
    Query,
    Embedding,
    RetrievalResult,
    GenerationResult,
    DocumentMetadata,
    ChunkMetadata,
)

from .ingestion import DocumentIngester
from .chunking import Chunker
from .embedding import Embedder
from .retrieval import Retriever
from .generation import Generator

__all__ = [
    # Data types
    "Document",
    "Chunk",
    "Query",
    "Embedding",
    "RetrievalResult",
    "GenerationResult",
    "DocumentMetadata",
    "ChunkMetadata",
    # Interfaces
    "DocumentIngester",
    "Chunker",
    "Embedder",
    "Retriever",
    "Generator",
]
