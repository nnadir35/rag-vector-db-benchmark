"""Retriever implementations for the RAG benchmark framework.

This module contains concrete retriever implementations that integrate
with various vector databases and retrieval systems.
"""

from .chroma_retriever import ChromaRetriever
from .factory import build_retriever_from_yaml
from .config import (
    ChromaRetrieverConfig,
    PineconeRetrieverConfig,
    QdrantRetrieverConfig,
)
from .pinecone_retriever import PineconeRetriever
from .qdrant_retriever import QdrantRetriever
from .registry import (
    RETRIEVER_REGISTRY,
    get_retriever,
    list_retrievers,
    register_retriever,
    unregister_retriever,
)

# Register Retrievers
register_retriever("pinecone", PineconeRetriever)
register_retriever("chroma", ChromaRetriever)
register_retriever("qdrant", QdrantRetriever)

__all__ = [
    # Retriever implementations
    "PineconeRetriever",
    "ChromaRetriever",
    "QdrantRetriever",
    "build_retriever_from_yaml",
    # Configuration
    "PineconeRetrieverConfig",
    "ChromaRetrieverConfig",
    "QdrantRetrieverConfig",
    # Registry
    "RETRIEVER_REGISTRY",
    "register_retriever",
    "get_retriever",
    "list_retrievers",
    "unregister_retriever",
]
