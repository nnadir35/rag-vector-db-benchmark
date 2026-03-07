"""Retriever implementations for the RAG benchmark framework.

This module contains concrete retriever implementations that integrate
with various vector databases and retrieval systems.
"""

from .chroma_retriever import ChromaRetriever
from .config import ChromaRetrieverConfig, PineconeRetrieverConfig
from .pinecone_retriever import PineconeRetriever
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

__all__ = [
    # Retriever implementations
    "PineconeRetriever",
    "ChromaRetriever",
    # Configuration
    "PineconeRetrieverConfig",
    "ChromaRetrieverConfig",
    # Registry
    "RETRIEVER_REGISTRY",
    "register_retriever",
    "get_retriever",
    "list_retrievers",
    "unregister_retriever",
]
