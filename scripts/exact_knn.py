"""Memory-safe, genuinely exact brute-force k-NN — the ground truth for exact-vs-ANN
recall experiments (see exact_vs_ann_benchmark.py, ann_param_sweep.py).

Never materializes a full (num_queries x num_corpus) similarity matrix at once, so it
stays safe at 200K+ corpus scale. Prefers `faiss.IndexFlatIP` (or `IndexFlatL2`) built
directly via the raw `faiss` API — deliberately not `FAISSRetriever`/`FAISSRetrieverConfig`
— so the exact baseline can never accidentally end up using an approximate (HNSW) index.
`IndexFlat*` is exact brute-force by construction: no graph, no quantization, no
approximation. Falls back to batched NumPy (bounded per-block memory) if `faiss` is not
importable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.core.types import Embedding


def _to_matrix(embeddings: Sequence[Embedding]) -> np.ndarray:
    return np.asarray([e.vector for e in embeddings], dtype=np.float32)


def _exact_topk_faiss(
    query_matrix: np.ndarray,
    corpus_matrix: np.ndarray,
    chunk_ids: list[str],
    k: int,
    distance_metric: str,
) -> dict[str, list[str]]:
    import faiss  # local import: optional dependency, guarded by caller's try/except

    dim = corpus_matrix.shape[1]
    index: Any = faiss.IndexFlatIP(dim) if distance_metric == "cosine" else faiss.IndexFlatL2(dim)
    assert index.__class__.__name__ in ("IndexFlatIP", "IndexFlatL2"), (
        "Exact baseline must use an exact Flat index, never an approximate one "
        f"(got {index.__class__.__name__})."
    )
    index.add(corpus_matrix)
    _, indices = index.search(query_matrix, min(k, len(chunk_ids)))
    return {
        str(q_idx): [chunk_ids[i] for i in row if i >= 0]
        for q_idx, row in enumerate(indices)
    }


def _exact_topk_numpy(
    query_matrix: np.ndarray,
    corpus_matrix: np.ndarray,
    chunk_ids: list[str],
    k: int,
    batch_size: int,
) -> dict[str, list[str]]:
    """Batched NumPy fallback — bounded memory: only one (num_queries x batch_size)
    similarity block is ever materialized at once, never the full corpus."""
    num_queries = query_matrix.shape[0]
    k_eff = min(k, len(chunk_ids))
    best_scores = np.full((num_queries, k_eff), -np.inf, dtype=np.float32)
    best_ids: list[list[str]] = [[""] * k_eff for _ in range(num_queries)]

    for start in range(0, corpus_matrix.shape[0], batch_size):
        block = corpus_matrix[start:start + batch_size]
        block_ids = chunk_ids[start:start + batch_size]
        sims = query_matrix @ block.T  # (num_queries, block_size) only — bounded

        combined_scores = np.concatenate([best_scores, sims], axis=1)
        for qi in range(num_queries):
            combined_ids = best_ids[qi] + list(block_ids)
            top_idx = np.argpartition(-combined_scores[qi], min(k_eff, len(combined_ids) - 1))[:k_eff]
            top_idx = top_idx[np.argsort(-combined_scores[qi][top_idx])]
            best_scores[qi] = combined_scores[qi][top_idx]
            best_ids[qi] = [combined_ids[i] for i in top_idx]

    return {str(qi): ids for qi, ids in enumerate(best_ids)}


def exact_topk_per_query(
    query_embeddings: dict[str, Embedding],
    corpus_embeddings: Sequence[Embedding],
    chunk_ids: list[str],
    k: int,
    distance_metric: str = "cosine",
    batch_size: int = 2048,
) -> tuple[dict[str, list[str]], str]:
    """Compute exact top-k chunk IDs per query.

    Returns ``(results, exact_method)`` where ``exact_method`` is
    ``"faiss.IndexFlatIP"``/``"faiss.IndexFlatL2"`` or ``"numpy_batched"`` — recorded in
    result metadata so the "exact, not approximate" property is verifiable after the fact.
    """
    query_ids = list(query_embeddings.keys())
    query_matrix = _to_matrix([query_embeddings[q] for q in query_ids])
    corpus_matrix = _to_matrix(corpus_embeddings)

    try:
        results_by_idx = _exact_topk_faiss(query_matrix, corpus_matrix, chunk_ids, k, distance_metric)
        method = f"faiss.IndexFlat{'IP' if distance_metric == 'cosine' else 'L2'}"
    except ImportError:
        logging.info("faiss not importable; falling back to batched NumPy exact k-NN.")
        results_by_idx = _exact_topk_numpy(query_matrix, corpus_matrix, chunk_ids, k, batch_size)
        method = "numpy_batched"

    return {query_ids[int(idx)]: ids for idx, ids in results_by_idx.items()}, method
