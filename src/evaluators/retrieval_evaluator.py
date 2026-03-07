"""Retrieval evaluator implementation.

This module provides the evaluator class designed to score RetrievalResult
outputs against known ground truth data.
"""

from typing import Dict, List, Set

from ..core.evaluation import Evaluator
from ..core.types import RetrievalResult
from .config import RetrievalEvaluatorConfig
from .metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class RetrievalEvaluator(Evaluator):
    """Retrieval component evaluator.
    
    Computes standard ranking metrics (Precision, Recall, MRR, nDCG) for
    a given RetrievalResult against ground-truth document/chunk IDs.
    """
    
    def __init__(self, config: RetrievalEvaluatorConfig) -> None:
        """Initialize the evaluator.
        
        Args:
            config: Configuration detailing k-cutoffs.
        """
        self._config = config
        
    def evaluate(
        self,
        result: RetrievalResult,
        ground_truth_ids: Set[str],
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a single retrieval result.
        
        Args:
            result: The retrieval response containing ranked RetrievedChunk instances.
            ground_truth_ids: A set or list of document/chunk IDs considered relevant
                to the query.
                
        Returns:
            A dictionary of metric names and their calculated float values.
        """
        # Safely extract IDs in rank order
        retrieved_ids = [rc.chunk.id for rc in result.chunks]
        relevant_list = list(ground_truth_ids)
        
        metrics: Dict[str, float] = {}
        
        # Calculate MRR (independent of K)
        metrics["mrr"] = mrr(retrieved_ids, relevant_list)
        
        # Calculate K-dependent metrics
        for k in self._config.k_values:
            metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_list, k)
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_list, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_list, k)
            
        return metrics
