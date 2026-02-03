"""Configuration classes for retriever implementations.

This module defines configuration dataclasses for various retriever
implementations. All configurations are designed to be serializable
and loadable from configuration files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


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
    api_key: Optional[str] = field(default=None)
    environment: Optional[str] = field(default=None)
    metric: str = field(default="cosine")
    namespace: Optional[str] = field(default=None)
    timeout: Optional[float] = field(default=None)
    
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
