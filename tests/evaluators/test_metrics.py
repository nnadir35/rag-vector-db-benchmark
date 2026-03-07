"""Unit tests for pure mathematical retrieval metric functions."""

import math

from src.evaluators.metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_with_partial_match():
    """Test precision computations with known overlaps."""
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D", "X"]
    
    # At 1: ["A"] intersect ["B","D","X"] = 0
    assert precision_at_k(retrieved, relevant, 1) == 0.0
    
    # At 2: ["A", "B"] intersect = ["B"] -> 1/2
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    
    # At 4: ["A", "B", "C", "D"] intersect = ["B", "D"] -> 2/4
    assert precision_at_k(retrieved, relevant, 4) == 0.5


def test_recall_at_k_with_missing_relevant():
    """Test recall calculations."""
    retrieved = ["A", "B"]
    relevant = ["B", "C", "D"]
    
    # At 1: ["A"] intersect ["B","C","D"] = 0 -> 0/3
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    
    # At 2: ["A", "B"] intersect = ["B"] -> 1/3
    assert recall_at_k(retrieved, relevant, 2) == 1.0 / 3.0


def test_mrr_when_first_result_is_correct():
    """Test MRR when the very first item is relevant."""
    retrieved = ["A", "B", "C"]
    relevant = ["A"]
    
    assert mrr(retrieved, relevant) == 1.0


def test_mrr_when_correct_result_is_third():
    """Test MRR when item is further down."""
    retrieved = ["X", "Y", "A"]
    relevant = ["A", "Z"]
    
    assert mrr(retrieved, relevant) == 1.0 / 3.0


def test_ndcg_at_k_perfect_ranking():
    """Test nDCG matching exact ideal."""
    retrieved = ["A", "B", "C"]
    relevant = ["A", "B", "C"]
    
    # DCG should equal IDCG
    assert ndcg_at_k(retrieved, relevant, 1) == 1.0
    assert ndcg_at_k(retrieved, relevant, 2) == 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0
    
    # Let's test non-perfect
    # DCG_2: 0 + 1 / log2(3) -> 1 / 1.5849 = ~0.63
    # IDCG_2: 1 / log2(2) + 1 / log2(3) = 1 + 0.63 = 1.63
    # nDCG_2 = 0.63 / 1.63
    non_perfect_retrieved = ["X", "B", "Y"]
    assert ndcg_at_k(non_perfect_retrieved, relevant, 2) < 1.0


def test_all_metrics_return_zero_for_empty_retrieved():
    """Test that zeros are returned strictly when there is empty data or k=0."""
    ret = []
    rel = ["A"]
    
    assert precision_at_k(ret, rel, 1) == 0.0
    assert recall_at_k(ret, rel, 1) == 0.0
    assert mrr(ret, rel) == 0.0
    assert ndcg_at_k(ret, rel, 1) == 0.0
    
    assert precision_at_k(["A"], rel, 0) == 0.0
