#!/usr/bin/env python3
"""Benchmark ChromaDB vs Qdrant: indexing (embed + persist) and retrieval-only latency.

Loads N documents from SQuAD, indexes each vector DB with the same embeddings,
measures mean retrieval time (query embed + vector search, no LLM) in milliseconds,
and compares Recall@K. Does not call any generator / LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunkers.config import FixedSizeChunkerConfig
from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.types import Chunk, Embedding, Query
from src.datasets.config import SQuADDatasetConfig
from src.datasets.squad_loader import SQuADLoader
from src.embedders.config import SentenceTransformersEmbedderConfig
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.evaluators.config import RetrievalEvaluatorConfig
from src.evaluators.retrieval_evaluator import RetrievalEvaluator
from src.retrievers.chroma_retriever import ChromaRetriever
from src.retrievers.config import ChromaRetrieverConfig, QdrantRetrieverConfig
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.utils.config_loader import build_component_configs, load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _select_queries_for_documents(
    queries: List[Query],
    ground_truth: Dict[str, Set[str]],
    doc_ids: Set[str],
    num_queries: int,
) -> List[Query]:
    """Pick queries whose ground-truth context document is in ``doc_ids``."""
    selected: List[Query] = []
    for q in queries:
        if len(selected) >= num_queries:
            break
        gt = ground_truth.get(q.id, set())
        if not gt:
            continue
        if gt.issubset(doc_ids):
            selected.append(q)
    return selected


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def _winner_speed(chroma_ms: float, qdrant_ms: float) -> str:
    if chroma_ms < qdrant_ms:
        return "ChromaDB"
    if qdrant_ms < chroma_ms:
        return "Qdrant"
    return "tie"


def _winner_recall(chroma_r: float, qdrant_r: float) -> str:
    if chroma_r > qdrant_r:
        return "ChromaDB"
    if qdrant_r > chroma_r:
        return "Qdrant"
    return "tie"


def run_benchmark(
    *,
    num_documents: int,
    num_queries: int,
    top_k: int,
    chunker_cfg: FixedSizeChunkerConfig,
    embedder_cfg: SentenceTransformersEmbedderConfig,
    chroma_cfg: ChromaRetrieverConfig,
    qdrant_cfg: QdrantRetrieverConfig,
    k_values: List[int],
) -> Dict[str, Any]:
    """Execute indexing and retrieval benchmarks; return a JSON-serializable report."""
    if top_k not in k_values:
        k_values = sorted({*k_values, top_k})

    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=k_values))
    recall_key = f"recall@{top_k}"

    logging.info("Loading SQuAD (validation, full split)...")
    loader = SQuADLoader(
        SQuADDatasetConfig(split="validation", max_samples=None, version="squad_v2")
    )
    all_queries, ground_truth = loader.load()
    all_documents = loader.load_documents()

    if len(all_documents) < num_documents:
        raise ValueError(
            f"Need at least {num_documents} unique SQuAD contexts; got {len(all_documents)}"
        )

    documents = all_documents[:num_documents]
    doc_ids = {d.id for d in documents}

    bench_queries = _select_queries_for_documents(
        all_queries, ground_truth, doc_ids, num_queries
    )
    if len(bench_queries) < num_queries:
        logging.warning(
            "Only %s queries have ground truth inside the selected %s documents "
            "(requested %s). Using available queries.",
            len(bench_queries),
            num_documents,
            num_queries,
        )

    logging.info("Chunking %s documents...", len(documents))
    chunker = FixedSizeChunker(chunker_cfg)
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    embedder = SentenceTransformersEmbedder(embedder_cfg)

    logging.info("Embedding %s chunks (single pass, shared by both DBs)...", len(chunks))
    t0 = time.perf_counter()
    embeddings: List[Embedding] = embedder.embed_chunks(chunks)
    embedding_seconds = time.perf_counter() - t0

    # --- Chroma: persist timing ---
    chroma = ChromaRetriever(config=chroma_cfg, embedder=embedder)
    chroma.clear()
    t_c0 = time.perf_counter()
    chroma.add_chunks(chunks, embeddings)
    chroma_add_seconds = time.perf_counter() - t_c0

    # --- Retrieval + accuracy (Chroma) ---
    chroma_latencies_ms: List[float] = []
    chroma_rows: List[Dict[str, float]] = []
    for q in bench_queries:
        t_r0 = time.perf_counter()
        result = chroma.retrieve(q, top_k=top_k)
        chroma_latencies_ms.append((time.perf_counter() - t_r0) * 1000.0)
        gt = set(ground_truth.get(q.id, set()))
        chroma_rows.append(evaluator.evaluate(result, gt))

    chroma_mean = _mean_metrics(chroma_rows)
    chroma_recall = chroma_mean.get(recall_key, float("nan"))

    # --- Qdrant: persist timing ---
    qdrant = QdrantRetriever(config=qdrant_cfg, embedder=embedder)
    qdrant.clear()
    t_q0 = time.perf_counter()
    qdrant.add_chunks(chunks, embeddings)
    qdrant_add_seconds = time.perf_counter() - t_q0

    qdrant_latencies_ms: List[float] = []
    qdrant_rows: List[Dict[str, float]] = []
    for q in bench_queries:
        t_r0 = time.perf_counter()
        result = qdrant.retrieve(q, top_k=top_k)
        qdrant_latencies_ms.append((time.perf_counter() - t_r0) * 1000.0)
        gt = set(ground_truth.get(q.id, set()))
        qdrant_rows.append(evaluator.evaluate(result, gt))

    qdrant_mean = _mean_metrics(qdrant_rows)
    qdrant_recall = qdrant_mean.get(recall_key, float("nan"))

    avg_chroma_ms = _mean(chroma_latencies_ms)
    avg_qdrant_ms = _mean(qdrant_latencies_ms)

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),
        "top_k": top_k,
        "embedding_seconds": embedding_seconds,
        "chroma_add_seconds": chroma_add_seconds,
        "qdrant_add_seconds": qdrant_add_seconds,
        "chroma_indexing_total_seconds": embedding_seconds + chroma_add_seconds,
        "qdrant_indexing_total_seconds": embedding_seconds + qdrant_add_seconds,
        "retrieval_avg_ms_chroma": avg_chroma_ms,
        "retrieval_avg_ms_qdrant": avg_qdrant_ms,
        "recall_at_k_metric": recall_key,
        "mean_recall_chroma": chroma_recall,
        "mean_recall_qdrant": qdrant_recall,
        "mean_metrics_chroma": chroma_mean,
        "mean_metrics_qdrant": qdrant_mean,
        "winner_faster_retrieval": _winner_speed(avg_chroma_ms, avg_qdrant_ms),
        "winner_higher_recall": _winner_recall(chroma_recall, qdrant_recall),
    }


def _print_table(report: Dict[str, Any]) -> None:
    ck = report["recall_at_k_metric"]
    print("\n" + "=" * 76)
    print("VECTOR DB BENCHMARK (SQuAD — LLM devre dışı, sadece embed + retrieval)")
    print("=" * 76)
    print(
        f"Döküman: {report['num_documents']}  |  "
        f"Chunk: {report['num_chunks']}  |  "
        f"Soru: {report['num_queries_evaluated']}  |  top_k={report['top_k']}"
    )
    print("-" * 76)
    print(f"{'Metrik':<42} | {'ChromaDB':>14} | {'Qdrant':>14}")
    print("-" * 76)
    print(
        f"{'İndeksleme: embedding (ortak, tek geçiş) [s]':<42} | "
        f"{report['embedding_seconds']:14.3f} | {'—':>14}"
    )
    print(
        f"{'İndeksleme: DB yazma (add_chunks) [s]':<42} | "
        f"{report['chroma_add_seconds']:14.3f} | {report['qdrant_add_seconds']:14.3f}"
    )
    print(
        f"{'İndeksleme: toplam (embedding + bu DB) [s]':<42} | "
        f"{report['chroma_indexing_total_seconds']:14.3f} | "
        f"{report['qdrant_indexing_total_seconds']:14.3f}"
    )
    print(
        f"{'Ortalama retrieval süresi [ms] (sorgu embed + arama)':<42} | "
        f"{report['retrieval_avg_ms_chroma']:14.2f} | {report['retrieval_avg_ms_qdrant']:14.2f}"
    )
    print(
        f"{('Ortalama ' + ck):<42} | "
        f"{report['mean_recall_chroma']:14.4f} | {report['mean_recall_qdrant']:14.4f}"
    )
    print("-" * 76)
    print(
        f"Daha hızlı retrieval: {report['winner_faster_retrieval']}  |  "
        f"Daha yüksek {ck}: {report['winner_higher_recall']}"
    )
    print("=" * 76 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChromaDB vs Qdrant: indexing ve retrieval benchmark (LLM yok)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="İsteğe bağlı YAML (chunker, embedder, evaluator, pipeline, "
        "chroma_retriever, qdrant_retriever).",
    )
    parser.add_argument("--num-documents", type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.config:
        raw = load_yaml(os.path.abspath(args.config))
        exp = build_component_configs(raw)
        chunker_cfg = exp.chunker
        embedder_cfg = exp.embedder
        k_values = sorted({*exp.evaluator.k_values, args.top_k})
        chroma_dict = {k: v for k, v in raw.get("chroma_retriever", {}).items() if k != "type"}
        qdrant_dict = {k: v for k, v in raw.get("qdrant_retriever", {}).items() if k != "type"}
        chroma_cfg = ChromaRetrieverConfig(**chroma_dict) if chroma_dict else ChromaRetrieverConfig()
        qdrant_cfg = QdrantRetrieverConfig(**qdrant_dict) if qdrant_dict else QdrantRetrieverConfig()
    else:
        chunker_cfg = FixedSizeChunkerConfig()
        embedder_cfg = SentenceTransformersEmbedderConfig()
        k_values = sorted({1, 3, 5, 10, args.top_k})
        chroma_cfg = ChromaRetrieverConfig(collection_name="bench_chroma_squad")
        qdrant_cfg = QdrantRetrieverConfig(
            collection_name="bench_qdrant_squad",
            in_memory=True,
        )

    top_k = args.top_k
    report = run_benchmark(
        num_documents=args.num_documents,
        num_queries=args.num_queries,
        top_k=top_k,
        chunker_cfg=chunker_cfg,
        embedder_cfg=embedder_cfg,
        chroma_cfg=chroma_cfg,
        qdrant_cfg=qdrant_cfg,
        k_values=k_values,
    )

    _print_table(report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_db_{ts}.json")
    payload = {
        "timestamp": ts,
        "config_path": os.path.abspath(args.config) if args.config else None,
        **report,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.info("Sonuçlar yazıldı: %s", out_path)


if __name__ == "__main__":
    main()
