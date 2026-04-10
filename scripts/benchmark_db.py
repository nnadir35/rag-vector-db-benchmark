#!/usr/bin/env python3
"""Benchmark ChromaDB vs Qdrant: indexing (embed + persist) and retrieval-only latency.

Loads N documents from SQuAD (validation, then train if needed), indexes each vector DB
with the same embeddings, measures mean retrieval time (query embed + vector search, no LLM),
compares Recall@K, and records peak RSS and average CPU during index/write and retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import psutil
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "psutil is required for benchmark_db.py. Install with: pip install psutil"
    ) from exc

from tqdm import tqdm

from src.chunkers.config import FixedSizeChunkerConfig
from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.types import Chunk, Document, Embedding, Query
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

T = TypeVar("T")


@dataclass(frozen=True)
class ResourceStats:
    """Peak resident set size (MB) and mean sampled CPU usage (%) for an interval."""

    peak_memory_mb: float
    avg_cpu_percent: float


class ResourceSampler:
    """Background RSS / CPU sampling for the current process (thread-safe)."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval = max(0.02, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_bytes: list[int] = []
        self._cpu_samples: list[float] = []

    def start(self) -> None:
        self._rss_bytes.clear()
        self._cpu_samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            proc = psutil.Process(os.getpid())
        except psutil.Error:
            return
        try:
            proc.cpu_percent(interval=None)
        except psutil.Error:
            return
        # Sample first, then sleep — avoids zero samples when the workload finishes
        # faster than one polling interval.
        while not self._stop.is_set():
            try:
                self._rss_bytes.append(proc.memory_info().rss)
                self._cpu_samples.append(proc.cpu_percent(interval=None))
            except psutil.Error:
                break
            if self._stop.wait(self._interval):
                break

    def stop(self) -> ResourceStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self._rss_bytes:
            return ResourceStats(peak_memory_mb=0.0, avg_cpu_percent=0.0)
        peak_mb = max(self._rss_bytes) / (1024.0 * 1024.0)
        raw_cpu = sum(self._cpu_samples) / max(1, len(self._cpu_samples))
        # psutil process CPU can exceed 100% on multi-core; normalize to [0, 100] machine share.
        logical = psutil.cpu_count(logical=True) or 1
        avg_cpu = min(100.0, raw_cpu / logical)
        return ResourceStats(peak_memory_mb=peak_mb, avg_cpu_percent=avg_cpu)


def run_with_resource_stats(work: Callable[[], T], sampler_interval: float = 0.05) -> tuple[T, ResourceStats]:
    """Run ``work`` while sampling memory/CPU in a background thread."""
    sampler = ResourceSampler(interval_seconds=sampler_interval)
    sampler.start()
    try:
        result = work()
    finally:
        stats = sampler.stop()
    return result, stats


def load_squad_corpus(
    num_documents: int,
    version: str,
) -> tuple[list[Document], list[Query], dict[str, set[str]], list[str]]:
    """Load up to ``num_documents`` unique SQuAD contexts (validation first, then train)."""
    if num_documents <= 0:
        raise ValueError(f"num_documents must be positive, got {num_documents}")

    val_loader = SQuADLoader(
        SQuADDatasetConfig(split="validation", max_samples=None, version=version)
    )
    logging.info("Loading SQuAD validation split...")
    val_queries, val_gt = val_loader.load()
    val_docs = val_loader.load_documents()

    splits_used: list[str] = ["validation"]

    if len(val_docs) >= num_documents:
        documents = val_docs[:num_documents]
        return documents, val_queries, dict(val_gt), splits_used

    logging.info(
        "Validation has %s unique contexts (< %s). Loading train split...",
        len(val_docs),
        num_documents,
    )
    train_loader = SQuADLoader(
        SQuADDatasetConfig(split="train", max_samples=None, version=version)
    )
    train_queries, train_gt = train_loader.load()
    train_docs = train_loader.load_documents()

    merged: list[Document] = list(val_docs)
    seen: set[str] = {d.id for d in merged}
    for doc in train_docs:
        if len(merged) >= num_documents:
            break
        if doc.id not in seen:
            merged.append(doc)
            seen.add(doc.id)

    if len(merged) < num_documents:
        raise ValueError(
            f"Need {num_documents} unique SQuAD contexts but only "
            f"{len(merged)} available across validation+train."
        )

    documents = merged[:num_documents]
    splits_used.append("train")

    combined_gt: dict[str, set[str]] = {**val_gt, **train_gt}
    combined_queries = val_queries + train_queries

    return documents, combined_queries, combined_gt, splits_used


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _select_queries_for_documents(
    queries: list[Query],
    ground_truth: dict[str, set[str]],
    doc_ids: set[str],
    num_queries: int,
) -> list[Query]:
    """Pick queries whose ground-truth context document is in ``doc_ids``."""
    selected: list[Query] = []
    for q in queries:
        if len(selected) >= num_queries:
            break
        gt = ground_truth.get(q.id, set())
        if not gt:
            continue
        if gt.issubset(doc_ids):
            selected.append(q)
    return selected


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
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
    k_values: list[int],
    squad_version: str,
    show_progress: bool,
) -> dict[str, Any]:
    """Execute indexing and retrieval benchmarks; return a JSON-serializable report."""
    if top_k not in k_values:
        k_values = sorted({*k_values, top_k})

    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=k_values))
    recall_key = f"recall@{top_k}"

    documents, all_queries, ground_truth, splits_used = load_squad_corpus(
        num_documents, squad_version
    )
    doc_ids = {d.id for d in documents}

    bench_queries = _select_queries_for_documents(
        all_queries, ground_truth, doc_ids, num_queries
    )
    if len(bench_queries) < num_queries:
        logging.warning(
            "Only %s queries have ground truth inside the selected %s documents "
            "(requested %s). Using available queries.",
            len(bench_queries),
            len(documents),
            num_queries,
        )

    logging.info("Chunking %s documents...", len(documents))
    chunker = FixedSizeChunker(chunker_cfg)
    chunks: list[Chunk] = []
    doc_iter = tqdm(
        documents,
        desc="Chunking documents",
        unit="doc",
        disable=not show_progress,
    )
    for doc in doc_iter:
        chunks.extend(chunker.chunk(doc))

    embedder = SentenceTransformersEmbedder(embedder_cfg)

    logging.info("Embedding %s chunks (single pass, shared by both DBs)...", len(chunks))
    t0 = time.perf_counter()
    embeddings: list[Embedding] = embedder.embed_chunks(chunks)
    embedding_seconds = time.perf_counter() - t0

    # --- Chroma: persist + resource stats ---
    chroma = ChromaRetriever(config=chroma_cfg, embedder=embedder)
    chroma.clear()
    t_idx0 = time.perf_counter()

    def _chroma_index() -> None:
        chroma.add_chunks(chunks, embeddings)

    _, chroma_index_stats = run_with_resource_stats(_chroma_index)
    chroma_add_seconds = time.perf_counter() - t_idx0

    # --- Retrieval + accuracy (Chroma) ---
    chroma_latencies_ms: list[float] = []
    chroma_rows: list[dict[str, float]] = []

    def _chroma_retrieval_pass() -> None:
        for q in tqdm(
            bench_queries,
            desc="Chroma retrieval",
            unit="q",
            leave=False,
            disable=not show_progress,
        ):
            t_r0 = time.perf_counter()
            result = chroma.retrieve(q, top_k=top_k)
            chroma_latencies_ms.append((time.perf_counter() - t_r0) * 1000.0)
            gt = set(ground_truth.get(q.id, set()))
            chroma_rows.append(evaluator.evaluate(result, gt))

    _, chroma_retrieval_stats = run_with_resource_stats(_chroma_retrieval_pass)

    chroma_mean = _mean_metrics(chroma_rows)
    chroma_recall = chroma_mean.get(recall_key, float("nan"))

    # --- Qdrant: persist + resource stats ---
    qdrant = QdrantRetriever(config=qdrant_cfg, embedder=embedder)
    qdrant.clear()
    t_qidx0 = time.perf_counter()

    def _qdrant_index() -> None:
        qdrant.add_chunks(chunks, embeddings)

    _, qdrant_index_stats = run_with_resource_stats(_qdrant_index)
    qdrant_add_seconds = time.perf_counter() - t_qidx0

    qdrant_latencies_ms: list[float] = []
    qdrant_rows: list[dict[str, float]] = []

    def _qdrant_retrieval_pass() -> None:
        for q in tqdm(
            bench_queries,
            desc="Qdrant retrieval",
            unit="q",
            leave=False,
            disable=not show_progress,
        ):
            t_r0 = time.perf_counter()
            result = qdrant.retrieve(q, top_k=top_k)
            qdrant_latencies_ms.append((time.perf_counter() - t_r0) * 1000.0)
            gt = set(ground_truth.get(q.id, set()))
            qdrant_rows.append(evaluator.evaluate(result, gt))

    _, qdrant_retrieval_stats = run_with_resource_stats(_qdrant_retrieval_pass)

    qdrant_mean = _mean_metrics(qdrant_rows)
    qdrant_recall = qdrant_mean.get(recall_key, float("nan"))

    avg_chroma_ms = _mean(chroma_latencies_ms)
    avg_qdrant_ms = _mean(qdrant_latencies_ms)

    chroma_block = {
        "indexing": {
            "memory_usage_mb": round(chroma_index_stats.peak_memory_mb, 3),
            "cpu_percent": round(chroma_index_stats.avg_cpu_percent, 2),
        },
        "retrieval": {
            "memory_usage_mb": round(chroma_retrieval_stats.peak_memory_mb, 3),
            "cpu_percent": round(chroma_retrieval_stats.avg_cpu_percent, 2),
        },
    }
    qdrant_block = {
        "indexing": {
            "memory_usage_mb": round(qdrant_index_stats.peak_memory_mb, 3),
            "cpu_percent": round(qdrant_index_stats.avg_cpu_percent, 2),
        },
        "retrieval": {
            "memory_usage_mb": round(qdrant_retrieval_stats.peak_memory_mb, 3),
            "cpu_percent": round(qdrant_retrieval_stats.avg_cpu_percent, 2),
        },
    }

    return {
        "squad_version": squad_version,
        "splits_used": splits_used,
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
        "chroma": chroma_block,
        "qdrant": qdrant_block,
        "winner_faster_retrieval": _winner_speed(avg_chroma_ms, avg_qdrant_ms),
        "winner_higher_recall": _winner_recall(chroma_recall, qdrant_recall),
    }


def _print_table(report: dict[str, Any]) -> None:
    ck = report["recall_at_k_metric"]
    ch = report["chroma"]
    qd = report["qdrant"]
    print("\n" + "=" * 92)
    print("VECTOR DB BENCHMARK (SQuAD — LLM devre dışı, sadece embed + retrieval)")
    print("=" * 92)
    print(
        f"Splits: {report.get('splits_used')}  |  "
        f"Döküman: {report['num_documents']}  |  "
        f"Chunk: {report['num_chunks']}  |  "
        f"Soru: {report['num_queries_evaluated']}  |  top_k={report['top_k']}"
    )
    print("-" * 92)
    print(f"{'Metrik':<50} | {'ChromaDB':>16} | {'Qdrant':>16}")
    print("-" * 92)
    print(
        f"{'İndeksleme: embedding (ortak, tek geçiş) [s]':<50} | "
        f"{report['embedding_seconds']:16.3f} | {'—':>16}"
    )
    print(
        f"{'İndeksleme: DB yazma (add_chunks) [s]':<50} | "
        f"{report['chroma_add_seconds']:16.3f} | {report['qdrant_add_seconds']:16.3f}"
    )
    print(
        f"{'İndeksleme: peak RAM (RSS) [MB]':<50} | "
        f"{ch['indexing']['memory_usage_mb']:16.3f} | {qd['indexing']['memory_usage_mb']:16.3f}"
    )
    print(
        f"{'İndeksleme: ort. CPU [%]':<50} | "
        f"{ch['indexing']['cpu_percent']:16.2f} | {qd['indexing']['cpu_percent']:16.2f}"
    )
    print(
        f"{'İndeksleme: toplam (embedding + bu DB) [s]':<50} | "
        f"{report['chroma_indexing_total_seconds']:16.3f} | "
        f"{report['qdrant_indexing_total_seconds']:16.3f}"
    )
    print(
        f"{'Ortalama retrieval [ms] (sorgu embed + arama)':<50} | "
        f"{report['retrieval_avg_ms_chroma']:16.2f} | {report['retrieval_avg_ms_qdrant']:16.2f}"
    )
    print(
        f"{'Retrieval: peak RAM (RSS) [MB]':<50} | "
        f"{ch['retrieval']['memory_usage_mb']:16.3f} | {qd['retrieval']['memory_usage_mb']:16.3f}"
    )
    print(
        f"{'Retrieval: ort. CPU [%]':<50} | "
        f"{ch['retrieval']['cpu_percent']:16.2f} | {qd['retrieval']['cpu_percent']:16.2f}"
    )
    print(
        f"{('Ortalama ' + ck):<50} | "
        f"{report['mean_recall_chroma']:16.4f} | {report['mean_recall_qdrant']:16.4f}"
    )
    print("-" * 92)
    print(
        f"Daha hızlı retrieval: {report['winner_faster_retrieval']}  |  "
        f"Daha yüksek {ck}: {report['winner_higher_recall']}"
    )
    print("=" * 92 + "\n")


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
    parser.add_argument(
        "--squad-version",
        type=str,
        default="squad_v2",
        help="HuggingFace datasets adı (varsayılan: squad_v2).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm ilerleme çubuklarını kapat (log/CI için).",
    )
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
        squad_version=args.squad_version,
        show_progress=not args.no_progress,
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
