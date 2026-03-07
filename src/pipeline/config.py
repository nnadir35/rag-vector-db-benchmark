"""Configuration classes for RAG pipeline.

This module defines configuration data structures controlling the overarching
behavior of the pipeline, including evaluating stages and cutoff metrics.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RAGPipelineConfig:
    """Configuration for RAG Pipeline execution.
    
    Attributes:
        top_k: Number of documents to retrieve.
        evaluate_retrieval: Whether to execute the retrieval evaluator logic.
        evaluate_generation: Whether to execute generation evaluator logic (future use).
    """
    
    top_k: int = field(default=5)
    evaluate_retrieval: bool = field(default=True)
    evaluate_generation: bool = field(default=False)
    
    def __post_init__(self) -> None:
        """Validate pipeline configuration limits."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be strictly positive, got {self.top_k}")
