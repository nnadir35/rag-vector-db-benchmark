"""Data types for pipeline results.

This module encapsulates end-to-end pipeline responses.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.types import Query, RAGResponse


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end result of a single pipeline execution.
    
    This captures the input query, the generated output from the RAG system,
    any evaluated metrics, and the entire operational latency.
    
    Attributes:
        query: The string representation of the user's question.
        rag_response: The core generation response.
        retrieval_metrics: Evaluated retrieval performance, if evaluated.
        total_latency_seconds: Total time taken by the full execution.
    """
    
    query: Query
    rag_response: RAGResponse
    total_latency_seconds: float
    retrieval_metrics: Optional[Dict[str, float]] = None
