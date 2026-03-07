"""Interface for evaluation.

Evaluation is the process of computing metrics to measure the automated
performance of retrievers or generators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import RAGResponse, RetrievalResult


class Evaluator(ABC):
    """Abstract interface for evaluation components."""
    
    @abstractmethod
    def evaluate(
        self,
        result: Any,
        ground_truth: Any,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            result: The component output (e.g., RetrievalResult)
            ground_truth: The target reference to compare against
            
        Returns:
            A dictionary of calculated metrics
        """
        pass
