"""Core interfaces and data types for the RAG benchmark framework."""

from .chunking import Chunker
from .embedding import Embedder
from .generation import Generator
from .ingestion import DocumentIngester
from .retrieval import Retriever
from .types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    Embedding,
    GenerationResult,
    Query,
    RetrievalResult,
)

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
