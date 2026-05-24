"""Retriever implementations for the RAG benchmark framework.

This module contains concrete retriever implementations that integrate
with various vector databases and retrieval systems.
"""

from .chroma_retriever import ChromaRetriever
from .config import (
    ChromaRetrieverConfig,
    FAISSRetrieverConfig,
    PineconeRetrieverConfig,
    QdrantRetrieverConfig,
)
from .faiss_retriever import FAISSRetriever
from .factory import build_retriever_from_yaml
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
register_retriever("faiss", FAISSRetriever)

__all__ = [
    # Retriever implementations
    "PineconeRetriever",
    "ChromaRetriever",
    "QdrantRetriever",
    "FAISSRetriever",
    "build_retriever_from_yaml",
    # Configuration
    "PineconeRetrieverConfig",
    "ChromaRetrieverConfig",
    "QdrantRetrieverConfig",
    "FAISSRetrieverConfig",
    # Registry
    "RETRIEVER_REGISTRY",
    "register_retriever",
    "get_retriever",
    "list_retrievers",
    "unregister_retriever",
]

