"""Unit tests for the memory-safe exact k-NN helper (scripts/exact_knn.py)."""

from typing import Any

import numpy as np
import pytest

from scripts.exact_knn import exact_topk_per_query
from src.core.types import Embedding


def _naive_topk(
    query_vecs: dict[str, Any], corpus_vecs: Any, chunk_ids: list[str], k: int
) -> dict[str, list[str]]:
    """Reference O(n*m) brute force, used only to check correctness in this test."""
    results = {}
    for qid, qv in query_vecs.items():
        sims = [(float(np.dot(qv, cv)), cid) for cv, cid in zip(corpus_vecs, chunk_ids)]
        sims.sort(key=lambda x: -x[0])
        results[qid] = [cid for _, cid in sims[:k]]
    return results


@pytest.fixture
def tiny_corpus() -> tuple[Any, list[Embedding], list[str], Any, dict[str, Embedding]]:
    rng = np.random.default_rng(42)
    corpus_vecs = rng.normal(size=(30, 8)).astype(np.float32)
    corpus_vecs /= np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    chunk_ids = [f"chunk_{i}" for i in range(30)]
    corpus_embeddings = [Embedding(vector=v.tolist(), dimension=8) for v in corpus_vecs]

    query_vecs = rng.normal(size=(5, 8)).astype(np.float32)
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_ids = [f"q{i}" for i in range(5)]
    query_embeddings = {
        qid: Embedding(vector=v.tolist(), dimension=8) for qid, v in zip(query_ids, query_vecs)
    }
    return corpus_vecs, corpus_embeddings, chunk_ids, query_vecs, query_embeddings


def test_exact_topk_matches_naive_reference(
    tiny_corpus: tuple[Any, list[Embedding], list[str], Any, dict[str, Embedding]]
) -> None:
    corpus_vecs, corpus_embeddings, chunk_ids, query_vecs, query_embeddings = tiny_corpus

    results, method = exact_topk_per_query(query_embeddings, corpus_embeddings, chunk_ids, k=5)

    naive = _naive_topk(
        {qid: query_vecs[i] for i, qid in enumerate(query_embeddings)},
        corpus_vecs,
        chunk_ids,
        k=5,
    )

    assert set(results.keys()) == set(naive.keys())
    for qid in results:
        # Compare as sets: ties at the boundary can permute order across backends.
        assert set(results[qid]) == set(naive[qid])
    assert method in ("faiss.IndexFlatIP", "faiss.IndexFlatL2", "numpy_batched")


def test_exact_topk_batched_numpy_matches_naive(
    tiny_corpus: tuple[Any, list[Embedding], list[str], Any, dict[str, Embedding]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the batched-NumPy fallback path (small batch_size) and check correctness."""
    corpus_vecs, corpus_embeddings, chunk_ids, query_vecs, query_embeddings = tiny_corpus
    from scripts import exact_knn as ek

    def _no_faiss(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("forced for test")

    monkeypatch.setattr(ek, "_exact_topk_faiss", _no_faiss)

    results, method = exact_topk_per_query(
        query_embeddings, corpus_embeddings, chunk_ids, k=5, batch_size=7
    )
    assert method == "numpy_batched"

    naive = _naive_topk(
        {qid: query_vecs[i] for i, qid in enumerate(query_embeddings)},
        corpus_vecs,
        chunk_ids,
        k=5,
    )
    for qid in results:
        assert set(results[qid]) == set(naive[qid])
