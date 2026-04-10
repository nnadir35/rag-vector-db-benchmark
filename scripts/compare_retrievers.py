#!/usr/bin/env python3
"""Compare ChromaDB and Qdrant retrievers on the same retrieval benchmark."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.retrieval import Retriever
from src.core.types import Query
from src.datasets.squad_loader import SQuADLoader
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.evaluators.config import RetrievalEvaluatorConfig
from src.evaluators.retrieval_evaluator import RetrievalEvaluator
from src.retrievers.chroma_retriever import ChromaRetriever
from src.retrievers.config import ChromaRetrieverConfig, QdrantRetrieverConfig
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.utils.config_loader import build_component_configs, load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPORT_METRIC_KEYS = ("mrr", "precision@1", "recall@3")


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric key across per-query rows."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(r[key] for r in rows) / len(rows) for key in keys}


def _run_retriever_eval(
    retriever: Retriever,
    queries: list[Query],
    ground_truth: dict[str, set[str]],
    evaluator: RetrievalEvaluator,
    top_k: int,
) -> tuple[dict[str, float], float, list[dict[str, float]]]:
    """Evaluate one retriever; return mean metrics, mean latency, per-query metrics."""
    per_query_metrics: list[dict[str, float]] = []
    latencies: list[float] = []

    for q in queries:
        t0 = time.perf_counter()
        result = retriever.retrieve(q, top_k=top_k)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        gt_ids = set(ground_truth.get(q.id, set()))
        per_query_metrics.append(evaluator.evaluate(result, gt_ids))

    mean_m = _mean_metrics(per_query_metrics)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return mean_m, avg_latency, per_query_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ChromaDB vs Qdrant on the same queries (retrieval only)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "experiments",
            "configs",
            "baseline_ollama.yaml",
        ),
        help="YAML experiment config (dataset, chunker, embedder, pipeline, optional "
        "chroma_retriever / qdrant_retriever sections).",
    )
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    raw_config = load_yaml(config_path)
    exp_config = build_component_configs(raw_config)

    k_values = sorted({*exp_config.evaluator.k_values, 1, 3})
    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=k_values))

    top_k = max(3, exp_config.pipeline.top_k)

    logging.info("Loading dataset...")
    loader = SQuADLoader(exp_config.dataset)
    queries, ground_truth = loader.load()
    raw_documents = loader.load_documents()

    logging.info("Chunking and embedding...")
    chunker = FixedSizeChunker(exp_config.chunker)
    chunks = []
    for doc in raw_documents:
        chunks.extend(chunker.chunk(doc))

    embedder = SentenceTransformersEmbedder(exp_config.embedder)
    embeddings = embedder.embed_chunks(chunks)

    chroma_dict = raw_config.get("chroma_retriever", {})
    chroma_dict = {k: v for k, v in chroma_dict.items() if k != "type"}
    qdrant_dict = raw_config.get("qdrant_retriever", {})
    qdrant_dict = {k: v for k, v in qdrant_dict.items() if k != "type"}

    chroma_config = ChromaRetrieverConfig(**chroma_dict)
    qdrant_config = QdrantRetrieverConfig(**qdrant_dict)

    results_summary: dict[str, Any] = {
        "experiment_name": exp_config.name,
        "config_path": config_path,
        "num_queries": len(queries),
        "top_k": top_k,
        "chroma": {},
        "qdrant": {},
    }

    # Chroma
    logging.info("Running Chroma retriever...")
    chroma = ChromaRetriever(config=chroma_config, embedder=embedder)
    chroma.clear()
    chroma.add_chunks(chunks, embeddings)
    chroma_mean, chroma_lat, chroma_per_q = _run_retriever_eval(
        chroma, queries, ground_truth, evaluator, top_k
    )
    results_summary["chroma"] = {
        "metrics": chroma_mean,
        "avg_latency_seconds": chroma_lat,
        "per_query_metrics": chroma_per_q,
    }

    # Qdrant
    logging.info("Running Qdrant retriever...")
    qdrant = QdrantRetriever(config=qdrant_config, embedder=embedder)
    qdrant.clear()
    qdrant.add_chunks(chunks, embeddings)
    qdrant_mean, qdrant_lat, qdrant_per_q = _run_retriever_eval(
        qdrant, queries, ground_truth, evaluator, top_k
    )
    results_summary["qdrant"] = {
        "metrics": qdrant_mean,
        "avg_latency_seconds": qdrant_lat,
        "per_query_metrics": qdrant_per_q,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiments/results", exist_ok=True)
    out_path = f"experiments/results/comparison_{ts}.json"
    results_summary["timestamp"] = ts

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Terminal table
    print("\n" + "=" * 72)
    print("RETRIEVER COMPARISON (same queries, retrieval only)")
    print("=" * 72)
    print(f"{'Metric':<22} | {'Chroma':>14} | {'Qdrant':>14}")
    print("-" * 72)
    for key in REPORT_METRIC_KEYS:
        c_val = chroma_mean.get(key, float("nan"))
        q_val = qdrant_mean.get(key, float("nan"))
        print(f"{key:<22} | {c_val:14.4f} | {q_val:14.4f}")
    print(
        f"{'avg_latency_seconds':<22} | {chroma_lat:14.4f} | {qdrant_lat:14.4f}"
    )
    print("=" * 72)
    print(f"Results saved to {out_path}\n")

    logging.info("Comparison complete.")


if __name__ == "__main__":
    main()
