"""Retriever implementations for the RAG benchmark framework.

This module contains concrete retriever implementations that integrate
with various vector databases and retrieval systems.
"""

from .config import PineconeRetrieverConfig
from .pinecone_retriever import PineconeRetriever
from .registry import (
    RETRIEVER_REGISTRY,
    get_retriever,
    list_retrievers,
    register_retriever,
    unregister_retriever,
)

# Register PineconeRetriever
register_retriever("pinecone", PineconeRetriever)

__all__ = [
    # Retriever implementations
    "PineconeRetriever",
    # Configuration
    "PineconeRetrieverConfig",
    # Registry
    "RETRIEVER_REGISTRY",
    "register_retriever",
    "get_retriever",
    "list_retrievers",
    "unregister_retriever",
]
