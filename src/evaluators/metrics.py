"""Retrieval metrics computations.

This module provides pure mathematical functions for evaluating retrieval
quality using standard metrics like Precision@K, Recall@K, MRR, and nDCG.
"""

import math
import re
import string
from collections import Counter


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
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


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
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


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
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


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
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


def normalize_answer(s: str) -> str:
    """SQuAD resmi normalizasyonu: küçük harfe çevir, noktalama işaretlerini
    kaldır, 'a/an/the' artikellerini kaldır, fazla boşlukları temizle."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    """Normalize edilmiş prediction, gold_answers'daki HERHANGİ birine birebir
    eşleşiyorsa 1.0, değilse 0.0 döner. gold_answers boşsa (cevaplanamaz soru)
    ve prediction da normalize sonrası boşsa 1.0 döner (SQuAD 2.0 kuralı)."""
    if not gold_answers:
        return 1.0 if not normalize_answer(prediction) else 0.0
    
    for gold in gold_answers:
        if normalize_answer(prediction) == normalize_answer(gold):
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Normalize edilmiş prediction ile her bir gold answer arasında token-bazlı
    F1 hesapla (ortak token sayısına dayalı precision/recall), gold_answers
    içindeki EN YÜKSEK F1'i döndür. Token overlap boşsa 0.0 döner (istisna:
    her ikisi de boşsa 1.0)."""
    if not gold_answers:
        return 1.0 if not normalize_answer(prediction) else 0.0

    def _get_f1(pred: str, gold: str) -> float:
        pred_toks = normalize_answer(pred).split()
        gold_toks = normalize_answer(gold).split()
        
        if not pred_toks or not gold_toks:
            # Both empty gives 1.0, otherwise 0.0
            return 1.0 if pred_toks == gold_toks else 0.0
        
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max(_get_f1(prediction, gold) for gold in gold_answers)
