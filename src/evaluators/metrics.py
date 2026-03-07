"""Retrieval metrics computations.

This module provides pure mathematical functions for evaluating retrieval
quality using standard metrics like Precision@K, Recall@K, MRR, and nDCG.
"""

import math
from typing import List


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Precision at K.
    
    Args:
        retrieved_ids: List of retrieved item IDs, ordered by rank.
        relevant_ids: List of ground truth relevant item IDs.
        k: The rank cutoff.
        
    Returns:
        Precision score (0.0 to 1.0). Returns 0.0 if retrieved_ids is empty or k=0.
    """
    if not retrieved_ids or k <= 0:
        return 0.0
        
    k_items = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    hits = sum(1 for item_id in k_items if item_id in relevant_set)
    return hits / len(k_items)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Recall at K.
    
    Args:
        retrieved_ids: List of retrieved item IDs, ordered by rank.
        relevant_ids: List of ground truth relevant item IDs.
        k: The rank cutoff.
        
    Returns:
        Recall score (0.0 to 1.0). Returns 0.0 if either list is empty or k=0.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
        
    k_items = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    hits = sum(1 for item_id in k_items if item_id in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank.
    
    Args:
        retrieved_ids: List of retrieved item IDs, ordered by rank.
        relevant_ids: List of ground truth relevant item IDs.
        
    Returns:
        Reciprocal Rank score (0.0 to 1.0). Returns 0.0 if not found.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
        
    relevant_set = set(relevant_ids)
    
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in relevant_set:
            return 1.0 / rank
            
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.
    
    Binary relevance is assumed (1 if item in relevant_set else 0).
    
    Args:
        retrieved_ids: List of retrieved item IDs, ordered by rank.
        relevant_ids: List of ground truth relevant item IDs.
        k: The rank cutoff.
        
    Returns:
        nDCG score (0.0 to 1.0). Returns 0.0 if empty or k=0.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
        
    k_items = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item_id in enumerate(k_items):
        if item_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(rank+1) and rank is 1-indexed
            
    # Calculate Ideal DCG@K
    # IDCG assumes all relevant items appear first, up to min(k, len(relevant_ids))
    idcg = 0.0
    ideal_count = min(k, len(relevant_set))
    for i in range(ideal_count):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
