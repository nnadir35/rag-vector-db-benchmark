"""Configuration classes for retriever implementations.

This module defines configuration dataclasses for various retriever
implementations. All configurations are designed to be serializable
and loadable from configuration files.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PineconeRetrieverConfig:
    """Configuration for PineconeRetriever.

    This configuration encapsulates all settings needed to initialize
    and operate a Pinecone retriever. API keys can be provided directly
    or read from environment variables for security.

    Attributes:
        api_key: Pinecone API key. If not provided, will attempt to read
            from PINECONE_API_KEY environment variable.
        environment: Pinecone environment (legacy, only needed for older accounts).
            For new accounts using serverless, this can be None.
        index_name: Name of the Pinecone index to use
        dimension: Dimension of the embedding vectors. Should match the
            dimension of the embedder being used.
        metric: Similarity metric to use. Options: 'cosine', 'euclidean', 'dotproduct'.
            Defaults to 'cosine'.
        namespace: Optional namespace within the index. Useful for isolating
            different datasets or experiments.
        timeout: Optional timeout in seconds for API requests. Defaults to None
            (uses Pinecone client default).
    """

    index_name: str
    dimension: int
    api_key: str | None = field(default=None)
    environment: str | None = field(default=None)
    metric: str = field(default="cosine")
    namespace: str | None = field(default=None)
    timeout: float | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate configuration and resolve API key from environment if needed."""
        # Validate metric
        valid_metrics = {"cosine", "euclidean", "dotproduct"}
        if self.metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{self.metric}'"
            )

        # Validate dimension
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")

        # Resolve API key from environment if not provided
        if self.api_key is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "api_key must be provided either directly or via PINECONE_API_KEY "
                    "environment variable"
                )
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(self, "api_key", api_key)


@dataclass(frozen=True)
class ChromaRetrieverConfig:
    """Configuration for ChromaRetriever.

    Attributes:
        collection_name: Name of the ChromaDB collection to use.
        persist_directory: Optional directory path to persist database. If None,
            runs completely in-memory.
        distance_metric: Default is 'cosine' for most modern embeddings.
    """

    collection_name: str = field(default="squad_benchmark")
    persist_directory: str | None = field(default=None)
    distance_metric: str = field(default="cosine")

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_metrics = {"cosine", "l2", "ip"}
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}, got '{self.distance_metric}'")

        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")


@dataclass(frozen=True)
class QdrantRetrieverConfig:
    """Configuration for QdrantRetriever.

    Attributes:
        collection_name: Name of the Qdrant collection to use.
        distance_metric: Distance function for vector search. Supported values
            align with Chroma-style names: 'cosine', 'l2' (or 'euclidean'), 'ip' (or 'dot').
        in_memory: If True, use an ephemeral in-memory Qdrant instance (``:memory:``).
        persist_path: When ``in_memory`` is False, local persistence directory for Qdrant.
    """

    collection_name: str = field(default="rag_benchmark")
    distance_metric: str = field(default="cosine")
    in_memory: bool = field(default=True)
    persist_path: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_metrics = {"cosine", "l2", "euclidean", "ip", "dot"}
        if self.distance_metric not in valid_metrics:
            raise ValueError(
                f"distance_metric must be one of {valid_metrics}, "
                f"got '{self.distance_metric}'"
            )

        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.in_memory and self.persist_path is None:
            # Remote Qdrant (Docker / k8s) uses QDRANT_URL or QDRANT_HOST from the environment
            # instead of a local persist_path.
            if not (os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST")):
                raise ValueError(
                    "persist_path must be provided when in_memory is False "
                    "unless QDRANT_URL or QDRANT_HOST is set for remote mode"
                )
