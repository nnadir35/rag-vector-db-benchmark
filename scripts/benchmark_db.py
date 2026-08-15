#!/usr/bin/env python3
"""Benchmark ChromaDB vs Qdrant vs FAISS vs Milvus vs ElasticSearch: indexing (embed + persist) and retrieval-only latency.

Loads N documents from SQuAD (validation, then train if needed), indexes each vector DB
with the same embeddings, measures mean retrieval time (query embed + vector search, no LLM),
compares Recall@K, and records peak RSS and average CPU during index/write and retrieval.
Requires docker-compose up elasticsearch if in_memory=false for ES.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, TypeVar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import psutil
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "psutil is required for benchmark_db.py. Install with: pip install psutil"
    ) from exc

import numpy as np
from tqdm import tqdm

from src.chunkers.config import FixedSizeChunkerConfig
from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.types import Chunk, Document, Embedding, Query
from src.datasets import (
    DatasetLoader,
    MSMARCODatasetConfig,
    MSMARCOLoader,
    SQuADDatasetConfig,
    SQuADLoader,
)
from src.embedders.config import SentenceTransformersEmbedderConfig
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.evaluators.config import RetrievalEvaluatorConfig
from src.evaluators.retrieval_evaluator import RetrievalEvaluator
from src.retrievers.chroma_retriever import ChromaRetriever
from src.retrievers.config import (
    ChromaRetrieverConfig,
    ElasticSearchRetrieverConfig,
    FAISSRetrieverConfig,
    MilvusRetrieverConfig,
    QdrantRetrieverConfig,
    WeaviateRetrieverConfig,
)
from src.retrievers.elasticsearch_retriever import ElasticSearchRetriever
from src.retrievers.faiss_retriever import FAISSRetriever
from src.retrievers.milvus_retriever import MilvusRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.weaviate_retriever import WeaviateRetriever
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

    val_loader: DatasetLoader = SQuADLoader(
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
    train_loader: DatasetLoader = SQuADLoader(
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


def load_dataset_corpus(
    dataset_config: SQuADDatasetConfig | MSMARCODatasetConfig,
    num_documents: int,
) -> tuple[list[Document], list[Query], dict[str, set[str]], list[str]]:
    """Load a bounded corpus from the configured benchmark dataset."""
    if isinstance(dataset_config, MSMARCODatasetConfig):
        loader: DatasetLoader = MSMARCOLoader(replace(dataset_config, max_documents=num_documents))
        queries, ground_truth = loader.load()
        documents = loader.load_documents()
        return documents, queries, ground_truth, ["local-msmarco"]
    return load_squad_corpus(num_documents, dataset_config.version)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


WARMUP_COUNT = 10


def _evaluate_with_repeats(
    *,
    name: str,
    retrieve_fn: Callable[[Embedding, int, str], Any],
    bench_queries: list[Query],
    query_embeddings: dict[str, Embedding],
    top_k: int,
    ground_truth: dict[str, set[str]],
    evaluator: RetrievalEvaluator,
    num_repeats: int,
    show_progress: bool,
    dump_per_query: bool = False,
) -> dict[str, Any]:
    """Run the retrieval pass for one DB ``num_repeats`` times.

    ``retrieve_fn`` takes a pre-computed query embedding (``retrieve_with_embedding``
    underneath) so the measured latency here is search-only — query embedding cost
    (constant, DB-independent) is measured once by the caller and reported separately.
    """
    per_repeat_avg_ms: list[float] = []
    per_repeat_p50_ms: list[float] = []
    per_repeat_p95_ms: list[float] = []
    per_repeat_metrics: list[dict[str, float]] = []
    per_query_details: list[dict[str, Any]] = []

    def _do_all_repeats() -> None:
        warmup_queries = bench_queries[:WARMUP_COUNT] if len(bench_queries) >= WARMUP_COUNT else []
        for repeat_idx in range(num_repeats):
            for q in warmup_queries:
                retrieve_fn(query_embeddings[q.id], top_k, q.id)

            latencies_ms: list[float] = []
            rows: list[dict[str, float]] = []
            for q in tqdm(
                bench_queries,
                desc=f"{name} retrieval (repeat {repeat_idx + 1}/{num_repeats})",
                unit="q",
                leave=False,
                disable=not show_progress,
            ):
                t_r0 = time.perf_counter()
                result = retrieve_fn(query_embeddings[q.id], top_k, q.id)
                lat_ms = (time.perf_counter() - t_r0) * 1000.0
                latencies_ms.append(lat_ms)
                gt = set(ground_truth.get(q.id, set()))
                rows.append(evaluator.evaluate(result, gt))

                if dump_per_query and repeat_idx == 0:
                    retrieved_chunks_info = [
                        {
                            "chunk_id": rc.chunk.id,
                            "doc_id": rc.chunk.id.split('_chunk_')[0] if '_chunk_' in rc.chunk.id else rc.chunk.id,
                            "score": float(rc.score),
                            "rank": rc.rank,
                        }
                        for rc in result.chunks
                    ]
                    per_query_details.append({
                        "query_id": q.id,
                        "latency_ms": lat_ms,
                        "retrieved": retrieved_chunks_info,
                    })

            per_repeat_avg_ms.append(_mean(latencies_ms))
            per_repeat_p50_ms.append(_percentile(latencies_ms, 50))
            per_repeat_p95_ms.append(_percentile(latencies_ms, 95))
            per_repeat_metrics.append(_mean_metrics(rows))

    _, resource_stats = run_with_resource_stats(_do_all_repeats)

    metric_keys = per_repeat_metrics[0].keys() if per_repeat_metrics else []
    metrics_mean = {k: float(np.mean([m[k] for m in per_repeat_metrics])) for k in metric_keys}
    metrics_std = {k: float(np.std([m[k] for m in per_repeat_metrics])) for k in metric_keys}

    res = {
        "resource_stats": resource_stats,
        "avg_ms_mean": float(np.mean(per_repeat_avg_ms)) if per_repeat_avg_ms else float("nan"),
        "avg_ms_std": float(np.std(per_repeat_avg_ms)) if per_repeat_avg_ms else float("nan"),
        "p50_ms_mean": float(np.mean(per_repeat_p50_ms)) if per_repeat_p50_ms else float("nan"),
        "p50_ms_std": float(np.std(per_repeat_p50_ms)) if per_repeat_p50_ms else float("nan"),
        "p95_ms_mean": float(np.mean(per_repeat_p95_ms)) if per_repeat_p95_ms else float("nan"),
        "p95_ms_std": float(np.std(per_repeat_p95_ms)) if per_repeat_p95_ms else float("nan"),
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "num_repeats": num_repeats,
    }
    if dump_per_query:
        res["per_query_details"] = per_query_details
    return res


def _select_queries_for_documents(
    queries: list[Query],
    ground_truth: dict[str, set[str]],
    doc_ids: set[str],
    num_queries: int,
    query_seed: int = 42,
) -> list[Query]:
    """Pick queries whose ground-truth context document is in ``doc_ids`` using a seeded random selection."""
    candidate_queries: list[Query] = []
    for q in queries:
        gt = ground_truth.get(q.id, set())
        if not gt:
            continue
        if gt.issubset(doc_ids):
            candidate_queries.append(q)

    if len(candidate_queries) <= num_queries:
        return candidate_queries

    import random
    rng = random.Random(query_seed)
    selected_indices = rng.sample(range(len(candidate_queries)), num_queries)
    return [candidate_queries[i] for i in sorted(selected_indices)]












def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def _winner_speed_multi(db_ms: dict[str, float]) -> str:
    """En düşük latency'e sahip DB adını (veya eşitlik durumunda 'TIE') döner."""
    if not db_ms:
        return "N/A"
    min_val = min(db_ms.values())
    winners = [k for k, v in db_ms.items() if abs(v - min_val) < 1e-9]
    if len(winners) > 1:
        if len(winners) == len(db_ms):
            return "TIE (All Equal)"
        return f"TIE ({', '.join(winners)})"
    return winners[0]


def _winner_recall_multi(db_recall: dict[str, float]) -> str:
    """En yüksek recall'a sahip DB adını (veya eşitlik durumunda 'TIE') döner."""
    if not db_recall:
        return "N/A"
    max_val = max(db_recall.values())
    winners = [k for k, v in db_recall.items() if abs(v - max_val) < 1e-9]
    if len(winners) > 1:
        if len(winners) == len(db_recall):
            return "TIE (All Equal)"
        return f"TIE ({', '.join(winners)})"
    return winners[0]


# Docker-based DB'ler için: (container adı adayları, konteyner içindeki veri yolu).
# Weaviate docker-compose.yml'de container_name/volume tanımlamıyor; adı proje
# önekiyle otomatik üretiliyor ve verisi container'ın yazılabilir katmanında
# tutuluyor, bu yüzden isim "weaviate" içeren çalışan konteyner otomatik bulunur.
_DOCKER_DB_PATHS: dict[str, tuple[list[str], str]] = {
    "qdrant": (["qdrant_db"], "/qdrant/storage"),
    "milvus": (["milvus_db"], "/var/lib/milvus"),
    "elasticsearch": (["elasticsearch_db"], "/usr/share/elasticsearch/data"),
    "weaviate": (["weaviate"], "/var/lib/weaviate"),
    "chroma": (["chroma_db"], "/chroma/chroma"),
}


def _find_container_name(candidates: list[str]) -> str | None:
    """Find a running container whose name matches or contains one of the candidates."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    names = proc.stdout.strip().splitlines()
    for cand in candidates:
        for n in names:
            if cand == n or cand in n:
                return n
    return None


def _parse_size_token(token: str) -> float | None:
    """Parse a docker-formatted size token (e.g. '12.3MB', '1.2GB') into megabytes."""
    m = re.match(r"^([\d.]+)\s*(GB|MB|kB|B)$", token.strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    multiplier = {"B": 1e-6, "kB": 1e-3, "MB": 1.0, "GB": 1000.0}[unit]
    return value * multiplier


def _docker_exec_du_mb(container: str, path: str) -> float | None:
    """Primary strategy: du -sm run inside the container (avoids host/VM volume path issues)."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "du", "-sm", path],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.split()[0])
    except (ValueError, IndexError):
        return None


def _docker_system_df_mb(container: str) -> float | None:
    """Fallback strategy: parse `docker system df -v` for the container's size."""
    try:
        result = subprocess.run(
            ["docker", "system", "df", "-v"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if container not in line:
            continue
        for token in line.split():
            size = _parse_size_token(token)
            if size is not None:
                return size
    return None


def _docker_stats_blkio_write_mb(container: str) -> float | None:
    """Last-resort fallback: cumulative block I/O write reported by `docker stats`.

    Not a true before/after delta (container may have pre-existing I/O), but the
    best signal available without instrumenting indexing calls per DB.
    """
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.BlockIO}}", container],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split("/")
    if len(parts) != 2:
        return None
    return _parse_size_token(parts[1].strip())


def measure_disk_size_mb(db_name: str, persist_path: str | None) -> dict[str, Any]:
    """Measure on-disk index size for a retriever.

    For in-process DBs (chroma, faiss) without a persist path, there is genuinely
    no disk footprint to measure (RAM-only) — this is reported as 0.0 with a flag,
    not an error. For Docker-backed DBs, tries docker exec + du, then
    `docker system df -v`, then a docker stats block-I/O fallback.
    """
    chroma_in_docker = db_name == "chroma" and bool(os.getenv("CHROMA_HOST", "").strip())
    if db_name in ("chroma", "faiss") and not chroma_in_docker:
        if not persist_path or not os.path.exists(persist_path):
            return {"disk_size_mb": 0.0, "disk_size_flag": "in_memory_no_disk"}
        try:
            result = subprocess.run(
                ["du", "-sm", persist_path], capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode == 0:
                return {"disk_size_mb": float(result.stdout.split()[0])}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return {"disk_size_mb": None, "disk_size_error": "du_failed_on_persist_path"}

    candidates, container_path = _DOCKER_DB_PATHS[db_name]
    container = _find_container_name(candidates)
    if container is None:
        return {
            "disk_size_mb": None,
            "disk_size_error": f"container_not_found (tried {candidates})",
        }

    size = _docker_exec_du_mb(container, container_path)
    if size is not None:
        return {"disk_size_mb": size}

    size = _docker_system_df_mb(container)
    if size is not None:
        return {"disk_size_mb": size, "disk_size_flag": "measured_via_docker_system_df_fallback"}

    blkio = _docker_stats_blkio_write_mb(container)
    if blkio is not None:
        return {
            "disk_size_mb": None,
            "disk_size_error": "docker_exec_du_and_system_df_failed",
            "disk_io_write_mb": blkio,
        }

    return {"disk_size_mb": None, "disk_size_error": "all_measurement_strategies_failed"}


def validate_effective_config(effective_cfg: dict[str, dict[str, Any]]) -> None:
    """Check effective DB configurations and output warnings to stderr if unexpected modes are used."""
    expected_defaults = {
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
        "hnsw_ef_search": 64,
    }
    # ChromaDB ve FAISS'in in_memory kavramı yok; her zaman in-process çalışırlar.
    # in_memory=True bu iki DB için beklenen durumdur, mock anlamına gelmez.
    inherently_in_process = {"chroma", "faiss"}
    warnings: list[str] = []
    for db_name, cfg in effective_cfg.items():
        if cfg.get("in_memory") is True and db_name not in inherently_in_process:
            warnings.append(f"[{db_name.upper()}] in_memory=True (MOCK / embedded in-process mode running!)")
        if db_name == "faiss" and cfg.get("index_type") != "hnsw":
            warnings.append(f"[FAISS] index_type='{cfg.get('index_type')}' (expected 'hnsw')")
        for key in ["hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"]:
            if key in expected_defaults:
                actual = cfg.get(key)
                exp_val = expected_defaults[key]
                if actual is None:
                    warnings.append(
                        f"[{db_name.upper()}] {key} doğrulanamadı "
                        "(describe_index başarısız veya alan eksik)"
                    )
                elif actual != exp_val:
                    warnings.append(f"[{db_name.upper()}] {key}={actual} (expected {exp_val})")

    if warnings:
        sys.stderr.write("\n" + "!" * 80 + "\n")
        sys.stderr.write("⚠️  KONFİGÜRASYON UYUMSUZLUĞU / MOCK UYARISI:\n")
        for w in warnings:
            sys.stderr.write(f"  - {w}\n")
        sys.stderr.write("!" * 80 + "\n\n")


def run_benchmark(
    *,
    num_documents: int,
    num_queries: int,
    top_k: int,
    chunker_cfg: FixedSizeChunkerConfig,
    embedder_cfg: SentenceTransformersEmbedderConfig,
    chroma_cfg: ChromaRetrieverConfig,
    qdrant_cfg: QdrantRetrieverConfig,
    faiss_cfg: FAISSRetrieverConfig,
    milvus_cfg: MilvusRetrieverConfig,
    elasticsearch_cfg: ElasticSearchRetrieverConfig,
    weaviate_cfg: WeaviateRetrieverConfig,
    k_values: list[int],
    dataset_config: SQuADDatasetConfig | MSMARCODatasetConfig,
    show_progress: bool,
    fixed_query_ids: set[str] | None = None,
    num_repeats: int = 3,
    query_seed: int = 42,
    dump_per_query: bool = False,
    include_faiss: bool = False,
) -> dict[str, Any]:
    """Execute indexing and retrieval benchmarks; return a JSON-serializable report."""








    if top_k not in k_values:
        k_values = sorted({*k_values, top_k})

    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=k_values))
    recall_key = f"recall@{top_k}"

    documents, all_queries, ground_truth, splits_used = load_dataset_corpus(
        dataset_config, num_documents
    )
    doc_ids = {d.id for d in documents}

    if fixed_query_ids is not None:
        # Sabit set: dışarıdan verilmiş ID listesine filtrele
        bench_queries = [q for q in all_queries if q.id in fixed_query_ids]
        # Güvenlik: bu sorguların gold'u gerçekten bu doc_ids içinde mi kontrol et
        bench_queries = [
            q for q in bench_queries
            if ground_truth.get(q.id, set()).issubset(doc_ids)
        ]
        if len(bench_queries) < len(fixed_query_ids):
            logging.warning(
                "fixed_query_ids has %d IDs but only %d are valid for this scale "
                "(%d docs). Continuing with %d queries.",
                len(fixed_query_ids), len(bench_queries), num_documents, len(bench_queries)
            )
    else:
        bench_queries = _select_queries_for_documents(
            all_queries, ground_truth, doc_ids, num_queries, query_seed=query_seed
        )
        if len(bench_queries) < num_queries:
            logging.warning(
                "Only %s queries have ground truth inside the selected %s documents "
                "(requested %s). Using available queries.",
                len(bench_queries),
                len(documents),
                num_queries,
            )

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

    # --- Query embeddings: computed once, shared by every DB's retrieve_with_embedding
    # call, so per-DB retrieval latency below is search-only (index size dependent),
    # not inflated by the ~constant embed_query() cost. ---
    logging.info("Embedding %s queries (single pass, shared by every DB)...", len(bench_queries))
    t_qe0 = time.perf_counter()
    query_embeddings: dict[str, Embedding] = {
        q.id: embedder.embed_query(q)
        for q in tqdm(bench_queries, desc="Embedding queries", unit="q", disable=not show_progress)
    }
    query_embedding_total_ms = (time.perf_counter() - t_qe0) * 1000.0
    query_embedding_avg_ms = (
        query_embedding_total_ms / len(bench_queries) if bench_queries else float("nan")
    )

    # --- Chroma: persist + resource stats ---
    chroma = ChromaRetriever(config=chroma_cfg, embedder=embedder)
    chroma.clear()
    t_idx0 = time.perf_counter()

    def _chroma_index() -> None:
        chroma.add_chunks(chunks, embeddings)

    _, chroma_index_stats = run_with_resource_stats(_chroma_index)
    chroma_add_seconds = time.perf_counter() - t_idx0

    # --- Retrieval + accuracy (Chroma) ---
    chroma_eval = _evaluate_with_repeats(
        name="Chroma",
        retrieve_fn=lambda emb, k, qid: chroma.retrieve_with_embedding(emb, top_k=k, query_id=qid),
        bench_queries=bench_queries,
        query_embeddings=query_embeddings,
        top_k=top_k,
        ground_truth=ground_truth,
        evaluator=evaluator,
        num_repeats=num_repeats,
        show_progress=show_progress,
        dump_per_query=dump_per_query,
    )
    chroma_retrieval_stats = chroma_eval["resource_stats"]
    chroma_mean = chroma_eval["metrics_mean"]
    chroma_std = chroma_eval["metrics_std"]
    chroma_recall = chroma_mean.get(recall_key, float("nan"))

    # --- Qdrant: persist + resource stats ---
    qdrant = QdrantRetriever(config=qdrant_cfg, embedder=embedder)
    qdrant.clear()
    t_qidx0 = time.perf_counter()

    def _qdrant_index() -> None:
        qdrant.add_chunks(chunks, embeddings)

    _, qdrant_index_stats = run_with_resource_stats(_qdrant_index)
    qdrant_add_seconds = time.perf_counter() - t_qidx0

    qdrant_eval = _evaluate_with_repeats(
        name="Qdrant",
        retrieve_fn=lambda emb, k, qid: qdrant.retrieve_with_embedding(emb, top_k=k, query_id=qid),
        bench_queries=bench_queries,
        query_embeddings=query_embeddings,
        top_k=top_k,
        ground_truth=ground_truth,
        evaluator=evaluator,
        num_repeats=num_repeats,
        show_progress=show_progress,
        dump_per_query=dump_per_query,
    )
    qdrant_retrieval_stats = qdrant_eval["resource_stats"]
    qdrant_mean = qdrant_eval["metrics_mean"]
    qdrant_std = qdrant_eval["metrics_std"]
    qdrant_recall = qdrant_mean.get(recall_key, float("nan"))

    # --- FAISS: persist + resource stats (opt-in; excluded from the systematic
    # benchmark by default — in-process library, no network layer, not comparable
    # to the Docker-backed DBs. See CLAUDE.md Pinecone precedent.) ---
    faiss: FAISSRetriever | None = None
    faiss_eval: dict[str, Any] | None = None
    faiss_index_stats: ResourceStats | None = None
    faiss_add_seconds = 0.0
    faiss_mean: dict[str, float] = {}
    faiss_std: dict[str, float] = {}
    faiss_recall = float("nan")

    if include_faiss:
        faiss = FAISSRetriever(config=faiss_cfg, embedder=embedder)
        faiss.clear()
        t_fidx0 = time.perf_counter()

        def _faiss_index() -> None:
            faiss.add_chunks(chunks, embeddings)

        _, faiss_index_stats = run_with_resource_stats(_faiss_index)
        faiss_add_seconds = time.perf_counter() - t_fidx0

        # --- Retrieval + accuracy (FAISS) ---
        faiss_eval = _evaluate_with_repeats(
            name="FAISS",
            retrieve_fn=lambda emb, k, qid: faiss.retrieve_with_embedding(emb, top_k=k, query_id=qid),
            bench_queries=bench_queries,
            query_embeddings=query_embeddings,
            top_k=top_k,
            ground_truth=ground_truth,
            evaluator=evaluator,
            num_repeats=num_repeats,
            show_progress=show_progress,
            dump_per_query=dump_per_query,
        )
        faiss_retrieval_stats = faiss_eval["resource_stats"]
        faiss_mean = faiss_eval["metrics_mean"]
        faiss_std = faiss_eval["metrics_std"]
        faiss_recall = faiss_mean.get(recall_key, float("nan"))

    # --- Milvus: persist + resource stats ---
    milvus = MilvusRetriever(config=milvus_cfg, embedder=embedder)
    milvus.clear()
    t_midx0 = time.perf_counter()

    def _milvus_index() -> None:
        milvus.add_chunks(chunks, embeddings)

    _, milvus_index_stats = run_with_resource_stats(_milvus_index)
    milvus_add_seconds = time.perf_counter() - t_midx0

    # --- Retrieval + accuracy (Milvus) ---
    milvus_eval = _evaluate_with_repeats(
        name="Milvus",
        retrieve_fn=lambda emb, k, qid: milvus.retrieve_with_embedding(emb, top_k=k, query_id=qid),
        bench_queries=bench_queries,
        query_embeddings=query_embeddings,
        top_k=top_k,
        ground_truth=ground_truth,
        evaluator=evaluator,
        num_repeats=num_repeats,
        show_progress=show_progress,
        dump_per_query=dump_per_query,
    )
    milvus_retrieval_stats = milvus_eval["resource_stats"]
    milvus_mean = milvus_eval["metrics_mean"]
    milvus_std = milvus_eval["metrics_std"]
    milvus_recall = milvus_mean.get(recall_key, float("nan"))

    # --- ElasticSearch: persist + resource stats ---
    elasticsearch = ElasticSearchRetriever(config=elasticsearch_cfg, embedder=embedder)
    elasticsearch.clear()
    t_esidx0 = time.perf_counter()

    def _elasticsearch_index() -> None:
        elasticsearch.add_chunks(chunks, embeddings)

    _, elasticsearch_index_stats = run_with_resource_stats(_elasticsearch_index)
    elasticsearch_add_seconds = time.perf_counter() - t_esidx0

    # --- Retrieval + accuracy (ElasticSearch) ---
    elasticsearch_eval = _evaluate_with_repeats(
        name="ElasticSearch",
        retrieve_fn=lambda emb, k, qid: elasticsearch.retrieve_with_embedding(emb, top_k=k, query_id=qid),
        bench_queries=bench_queries,
        query_embeddings=query_embeddings,
        top_k=top_k,
        ground_truth=ground_truth,
        evaluator=evaluator,
        num_repeats=num_repeats,
        show_progress=show_progress,
        dump_per_query=dump_per_query,
    )
    elasticsearch_retrieval_stats = elasticsearch_eval["resource_stats"]
    elasticsearch_mean = elasticsearch_eval["metrics_mean"]
    elasticsearch_std = elasticsearch_eval["metrics_std"]
    elasticsearch_recall = elasticsearch_mean.get(recall_key, float("nan"))

    # --- Weaviate: persist + resource stats ---
    weaviate = WeaviateRetriever(config=weaviate_cfg, embedder=embedder)
    weaviate.clear()
    t_widx0 = time.perf_counter()

    def _weaviate_index() -> None:
        weaviate.add_chunks(chunks, embeddings)

    _, weaviate_index_stats = run_with_resource_stats(_weaviate_index)
    weaviate_add_seconds = time.perf_counter() - t_widx0

    # --- Retrieval + accuracy (Weaviate) ---
    weaviate_eval = _evaluate_with_repeats(
        name="Weaviate",
        retrieve_fn=lambda emb, k, qid: weaviate.retrieve_with_embedding(emb, top_k=k, query_id=qid),
        bench_queries=bench_queries,
        query_embeddings=query_embeddings,
        top_k=top_k,
        ground_truth=ground_truth,
        evaluator=evaluator,
        num_repeats=num_repeats,
        show_progress=show_progress,
        dump_per_query=dump_per_query,
    )
    weaviate_retrieval_stats = weaviate_eval["resource_stats"]
    weaviate_mean = weaviate_eval["metrics_mean"]
    weaviate_std = weaviate_eval["metrics_std"]
    weaviate_recall = weaviate_mean.get(recall_key, float("nan"))

    # eval dicts hold search-only latency (retrieve_with_embedding only). The
    # legacy combined "retrieval_*_ms" fields (embed + search, kept for backward
    # compatibility) are derived by adding the shared query_embedding_avg_ms offset.
    search_only_avg_chroma_ms = chroma_eval["avg_ms_mean"]
    search_only_avg_chroma_ms_std = chroma_eval["avg_ms_std"]
    search_only_p50_chroma_ms = chroma_eval["p50_ms_mean"]
    search_only_p50_chroma_ms_std = chroma_eval["p50_ms_std"]
    search_only_p95_chroma_ms = chroma_eval["p95_ms_mean"]
    search_only_p95_chroma_ms_std = chroma_eval["p95_ms_std"]
    avg_chroma_ms = search_only_avg_chroma_ms + query_embedding_avg_ms
    avg_chroma_ms_std = search_only_avg_chroma_ms_std
    p50_chroma_ms = search_only_p50_chroma_ms + query_embedding_avg_ms
    p50_chroma_ms_std = search_only_p50_chroma_ms_std
    p95_chroma_ms = search_only_p95_chroma_ms + query_embedding_avg_ms
    p95_chroma_ms_std = search_only_p95_chroma_ms_std

    search_only_avg_qdrant_ms = qdrant_eval["avg_ms_mean"]
    search_only_avg_qdrant_ms_std = qdrant_eval["avg_ms_std"]
    search_only_p50_qdrant_ms = qdrant_eval["p50_ms_mean"]
    search_only_p50_qdrant_ms_std = qdrant_eval["p50_ms_std"]
    search_only_p95_qdrant_ms = qdrant_eval["p95_ms_mean"]
    search_only_p95_qdrant_ms_std = qdrant_eval["p95_ms_std"]
    avg_qdrant_ms = search_only_avg_qdrant_ms + query_embedding_avg_ms
    avg_qdrant_ms_std = search_only_avg_qdrant_ms_std
    p50_qdrant_ms = search_only_p50_qdrant_ms + query_embedding_avg_ms
    p50_qdrant_ms_std = search_only_p50_qdrant_ms_std
    p95_qdrant_ms = search_only_p95_qdrant_ms + query_embedding_avg_ms
    p95_qdrant_ms_std = search_only_p95_qdrant_ms_std

    if include_faiss:
        assert faiss_eval is not None
        search_only_avg_faiss_ms = faiss_eval["avg_ms_mean"]
        search_only_avg_faiss_ms_std = faiss_eval["avg_ms_std"]
        search_only_p50_faiss_ms = faiss_eval["p50_ms_mean"]
        search_only_p50_faiss_ms_std = faiss_eval["p50_ms_std"]
        search_only_p95_faiss_ms = faiss_eval["p95_ms_mean"]
        search_only_p95_faiss_ms_std = faiss_eval["p95_ms_std"]
        avg_faiss_ms = search_only_avg_faiss_ms + query_embedding_avg_ms
        avg_faiss_ms_std = search_only_avg_faiss_ms_std
        p50_faiss_ms = search_only_p50_faiss_ms + query_embedding_avg_ms
        p50_faiss_ms_std = search_only_p50_faiss_ms_std
        p95_faiss_ms = search_only_p95_faiss_ms + query_embedding_avg_ms
        p95_faiss_ms_std = search_only_p95_faiss_ms_std

    search_only_avg_milvus_ms = milvus_eval["avg_ms_mean"]
    search_only_avg_milvus_ms_std = milvus_eval["avg_ms_std"]
    search_only_p50_milvus_ms = milvus_eval["p50_ms_mean"]
    search_only_p50_milvus_ms_std = milvus_eval["p50_ms_std"]
    search_only_p95_milvus_ms = milvus_eval["p95_ms_mean"]
    search_only_p95_milvus_ms_std = milvus_eval["p95_ms_std"]
    avg_milvus_ms = search_only_avg_milvus_ms + query_embedding_avg_ms
    avg_milvus_ms_std = search_only_avg_milvus_ms_std
    p50_milvus_ms = search_only_p50_milvus_ms + query_embedding_avg_ms
    p50_milvus_ms_std = search_only_p50_milvus_ms_std
    p95_milvus_ms = search_only_p95_milvus_ms + query_embedding_avg_ms
    p95_milvus_ms_std = search_only_p95_milvus_ms_std

    search_only_avg_elasticsearch_ms = elasticsearch_eval["avg_ms_mean"]
    search_only_avg_elasticsearch_ms_std = elasticsearch_eval["avg_ms_std"]
    search_only_p50_elasticsearch_ms = elasticsearch_eval["p50_ms_mean"]
    search_only_p50_elasticsearch_ms_std = elasticsearch_eval["p50_ms_std"]
    search_only_p95_elasticsearch_ms = elasticsearch_eval["p95_ms_mean"]
    search_only_p95_elasticsearch_ms_std = elasticsearch_eval["p95_ms_std"]
    avg_elasticsearch_ms = search_only_avg_elasticsearch_ms + query_embedding_avg_ms
    avg_elasticsearch_ms_std = search_only_avg_elasticsearch_ms_std
    p50_elasticsearch_ms = search_only_p50_elasticsearch_ms + query_embedding_avg_ms
    p50_elasticsearch_ms_std = search_only_p50_elasticsearch_ms_std
    p95_elasticsearch_ms = search_only_p95_elasticsearch_ms + query_embedding_avg_ms
    p95_elasticsearch_ms_std = search_only_p95_elasticsearch_ms_std

    search_only_avg_weaviate_ms = weaviate_eval["avg_ms_mean"]
    search_only_avg_weaviate_ms_std = weaviate_eval["avg_ms_std"]
    search_only_p50_weaviate_ms = weaviate_eval["p50_ms_mean"]
    search_only_p50_weaviate_ms_std = weaviate_eval["p50_ms_std"]
    search_only_p95_weaviate_ms = weaviate_eval["p95_ms_mean"]
    search_only_p95_weaviate_ms_std = weaviate_eval["p95_ms_std"]
    avg_weaviate_ms = search_only_avg_weaviate_ms + query_embedding_avg_ms
    avg_weaviate_ms_std = search_only_avg_weaviate_ms_std
    p50_weaviate_ms = search_only_p50_weaviate_ms + query_embedding_avg_ms
    p50_weaviate_ms_std = search_only_p50_weaviate_ms_std
    p95_weaviate_ms = search_only_p95_weaviate_ms + query_embedding_avg_ms
    p95_weaviate_ms_std = search_only_p95_weaviate_ms_std

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
    faiss_block: dict[str, Any] | None = None
    if include_faiss:
        assert faiss_index_stats is not None
        faiss_block = {
            "indexing": {
                "memory_usage_mb": round(faiss_index_stats.peak_memory_mb, 3),
                "cpu_percent": round(faiss_index_stats.avg_cpu_percent, 2),
            },
            "retrieval": {
                "memory_usage_mb": round(faiss_retrieval_stats.peak_memory_mb, 3),
                "cpu_percent": round(faiss_retrieval_stats.avg_cpu_percent, 2),
            },
        }
    milvus_block = {
        "indexing": {
            "memory_usage_mb": round(milvus_index_stats.peak_memory_mb, 3),
            "cpu_percent": round(milvus_index_stats.avg_cpu_percent, 2),
        },
        "retrieval": {
            "memory_usage_mb": round(milvus_retrieval_stats.peak_memory_mb, 3),
            "cpu_percent": round(milvus_retrieval_stats.avg_cpu_percent, 2),
        },
    }
    elasticsearch_block = {
        "indexing": {
            "memory_usage_mb": round(elasticsearch_index_stats.peak_memory_mb, 3),
            "cpu_percent": round(elasticsearch_index_stats.avg_cpu_percent, 2),
        },
        "retrieval": {
            "memory_usage_mb": round(elasticsearch_retrieval_stats.peak_memory_mb, 3),
            "cpu_percent": round(elasticsearch_retrieval_stats.avg_cpu_percent, 2),
        },
    }

    weaviate_block = {
        "indexing": {
            "memory_usage_mb": round(weaviate_index_stats.peak_memory_mb, 3),
            "cpu_percent": round(weaviate_index_stats.avg_cpu_percent, 2),
        },
        "retrieval": {
            "memory_usage_mb": round(weaviate_retrieval_stats.peak_memory_mb, 3),
            "cpu_percent": round(weaviate_retrieval_stats.avg_cpu_percent, 2),
        },
    }

    effective_config = {
        "chroma": chroma.describe_index(),
        "qdrant": qdrant.describe_index(),
        "milvus": milvus.describe_index(),
        "elasticsearch": elasticsearch.describe_index(),
        "weaviate": weaviate.describe_index(),
    }
    if include_faiss:
        assert faiss is not None
        effective_config["faiss"] = faiss.describe_index()
    validate_effective_config(effective_config)

    disk_size_info = {
        "chroma": measure_disk_size_mb("chroma", chroma_cfg.persist_directory),
        "qdrant": measure_disk_size_mb("qdrant", None),
        "milvus": measure_disk_size_mb("milvus", None),
        "elasticsearch": measure_disk_size_mb("elasticsearch", None),
        "weaviate": measure_disk_size_mb("weaviate", None),
    }
    if include_faiss:
        disk_size_info["faiss"] = measure_disk_size_mb("faiss", faiss_cfg.persist_path)

    ret_dict = {
        "dataset": "msmarco" if isinstance(dataset_config, MSMARCODatasetConfig) else "squad",
        "dataset_version": (
            None if isinstance(dataset_config, MSMARCODatasetConfig) else dataset_config.version
        ),
        "splits_used": splits_used,
        "fixed_query_set": fixed_query_ids is not None,
        "query_seed": query_seed,
        "query_ids_used": [q.id for q in bench_queries],
        "effective_config": effective_config,

        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),

        "top_k": top_k,
        "num_repeats": num_repeats,
        "embedding_seconds": embedding_seconds,
        "query_embedding_total_ms": query_embedding_total_ms,
        "query_embedding_avg_ms": query_embedding_avg_ms,
        "search_only_avg_ms_chroma": search_only_avg_chroma_ms,
        "search_only_avg_ms_std_chroma": search_only_avg_chroma_ms_std,
        "search_only_p50_ms_chroma": search_only_p50_chroma_ms,
        "search_only_p50_ms_std_chroma": search_only_p50_chroma_ms_std,
        "search_only_p95_ms_chroma": search_only_p95_chroma_ms,
        "search_only_p95_ms_std_chroma": search_only_p95_chroma_ms_std,
        "search_only_avg_ms_qdrant": search_only_avg_qdrant_ms,
        "search_only_avg_ms_std_qdrant": search_only_avg_qdrant_ms_std,
        "search_only_p50_ms_qdrant": search_only_p50_qdrant_ms,
        "search_only_p50_ms_std_qdrant": search_only_p50_qdrant_ms_std,
        "search_only_p95_ms_qdrant": search_only_p95_qdrant_ms,
        "search_only_p95_ms_std_qdrant": search_only_p95_qdrant_ms_std,
        "search_only_avg_ms_milvus": search_only_avg_milvus_ms,
        "search_only_avg_ms_std_milvus": search_only_avg_milvus_ms_std,
        "search_only_p50_ms_milvus": search_only_p50_milvus_ms,
        "search_only_p50_ms_std_milvus": search_only_p50_milvus_ms_std,
        "search_only_p95_ms_milvus": search_only_p95_milvus_ms,
        "search_only_p95_ms_std_milvus": search_only_p95_milvus_ms_std,
        "search_only_avg_ms_elasticsearch": search_only_avg_elasticsearch_ms,
        "search_only_avg_ms_std_elasticsearch": search_only_avg_elasticsearch_ms_std,
        "search_only_p50_ms_elasticsearch": search_only_p50_elasticsearch_ms,
        "search_only_p50_ms_std_elasticsearch": search_only_p50_elasticsearch_ms_std,
        "search_only_p95_ms_elasticsearch": search_only_p95_elasticsearch_ms,
        "search_only_p95_ms_std_elasticsearch": search_only_p95_elasticsearch_ms_std,
        "search_only_avg_ms_weaviate": search_only_avg_weaviate_ms,
        "search_only_avg_ms_std_weaviate": search_only_avg_weaviate_ms_std,
        "search_only_p50_ms_weaviate": search_only_p50_weaviate_ms,
        "search_only_p50_ms_std_weaviate": search_only_p50_weaviate_ms_std,
        "search_only_p95_ms_weaviate": search_only_p95_weaviate_ms,
        "search_only_p95_ms_std_weaviate": search_only_p95_weaviate_ms_std,
        "chroma_add_seconds": chroma_add_seconds,
        "qdrant_add_seconds": qdrant_add_seconds,
        "milvus_add_seconds": milvus_add_seconds,
        "elasticsearch_add_seconds": elasticsearch_add_seconds,
        "weaviate_add_seconds": weaviate_add_seconds,
        "chroma_indexing_total_seconds": embedding_seconds + chroma_add_seconds,
        "qdrant_indexing_total_seconds": embedding_seconds + qdrant_add_seconds,
        "milvus_indexing_total_seconds": embedding_seconds + milvus_add_seconds,
        "elasticsearch_indexing_total_seconds": embedding_seconds + elasticsearch_add_seconds,
        "weaviate_indexing_total_seconds": embedding_seconds + weaviate_add_seconds,
        "retrieval_avg_ms_chroma": avg_chroma_ms,
        "retrieval_avg_ms_std_chroma": avg_chroma_ms_std,
        "retrieval_p50_ms_chroma": p50_chroma_ms,
        "retrieval_p50_ms_std_chroma": p50_chroma_ms_std,
        "retrieval_p95_ms_chroma": p95_chroma_ms,
        "retrieval_p95_ms_std_chroma": p95_chroma_ms_std,
        "retrieval_avg_ms_qdrant": avg_qdrant_ms,
        "retrieval_avg_ms_std_qdrant": avg_qdrant_ms_std,
        "retrieval_p50_ms_qdrant": p50_qdrant_ms,
        "retrieval_p50_ms_std_qdrant": p50_qdrant_ms_std,
        "retrieval_p95_ms_qdrant": p95_qdrant_ms,
        "retrieval_p95_ms_std_qdrant": p95_qdrant_ms_std,
        "retrieval_avg_ms_milvus": avg_milvus_ms,
        "retrieval_avg_ms_std_milvus": avg_milvus_ms_std,
        "retrieval_p50_ms_milvus": p50_milvus_ms,
        "retrieval_p50_ms_std_milvus": p50_milvus_ms_std,
        "retrieval_p95_ms_milvus": p95_milvus_ms,
        "retrieval_p95_ms_std_milvus": p95_milvus_ms_std,
        "retrieval_avg_ms_elasticsearch": avg_elasticsearch_ms,
        "retrieval_avg_ms_std_elasticsearch": avg_elasticsearch_ms_std,
        "retrieval_p50_ms_elasticsearch": p50_elasticsearch_ms,
        "retrieval_p50_ms_std_elasticsearch": p50_elasticsearch_ms_std,
        "retrieval_p95_ms_elasticsearch": p95_elasticsearch_ms,
        "retrieval_p95_ms_std_elasticsearch": p95_elasticsearch_ms_std,
        "retrieval_avg_ms_weaviate": avg_weaviate_ms,
        "retrieval_avg_ms_std_weaviate": avg_weaviate_ms_std,
        "retrieval_p50_ms_weaviate": p50_weaviate_ms,
        "retrieval_p50_ms_std_weaviate": p50_weaviate_ms_std,
        "retrieval_p95_ms_weaviate": p95_weaviate_ms,
        "retrieval_p95_ms_std_weaviate": p95_weaviate_ms_std,
        "recall_at_k_metric": recall_key,
        "mean_recall_chroma": chroma_recall,
        "mean_recall_qdrant": qdrant_recall,
        "mean_recall_milvus": milvus_recall,
        "mean_recall_elasticsearch": elasticsearch_recall,
        "mean_recall_weaviate": weaviate_recall,
        "mean_metrics_chroma": chroma_mean,
        "mean_metrics_qdrant": qdrant_mean,
        "mean_metrics_milvus": milvus_mean,
        "mean_metrics_elasticsearch": elasticsearch_mean,
        "mean_metrics_weaviate": weaviate_mean,
        "std_metrics_chroma": chroma_std,
        "std_metrics_qdrant": qdrant_std,
        "std_metrics_milvus": milvus_std,
        "std_metrics_elasticsearch": elasticsearch_std,
        "std_metrics_weaviate": weaviate_std,
        "chroma": chroma_block,
        "qdrant": qdrant_block,
        "milvus": milvus_block,
        "elasticsearch": elasticsearch_block,
        "weaviate": weaviate_block,
        "winner_faster_retrieval": _winner_speed_multi({
            "ChromaDB": avg_chroma_ms,
            "Qdrant": avg_qdrant_ms,
            "Milvus": avg_milvus_ms,
            "ElasticSearch": avg_elasticsearch_ms,
            "Weaviate": avg_weaviate_ms,
            **({"FAISS": avg_faiss_ms} if include_faiss else {}),
        }),
        "winner_higher_recall": _winner_recall_multi({
            "ChromaDB": chroma_recall,
            "Qdrant": qdrant_recall,
            "Milvus": milvus_recall,
            "ElasticSearch": elasticsearch_recall,
            "Weaviate": weaviate_recall,
            **({"FAISS": faiss_recall} if include_faiss else {}),
        }),
    }
    if include_faiss:
        ret_dict["faiss_add_seconds"] = faiss_add_seconds
        ret_dict["faiss_indexing_total_seconds"] = embedding_seconds + faiss_add_seconds
        ret_dict["retrieval_avg_ms_faiss"] = avg_faiss_ms
        ret_dict["retrieval_avg_ms_std_faiss"] = avg_faiss_ms_std
        ret_dict["retrieval_p50_ms_faiss"] = p50_faiss_ms
        ret_dict["retrieval_p50_ms_std_faiss"] = p50_faiss_ms_std
        ret_dict["retrieval_p95_ms_faiss"] = p95_faiss_ms
        ret_dict["retrieval_p95_ms_std_faiss"] = p95_faiss_ms_std
        ret_dict["mean_recall_faiss"] = faiss_recall
        ret_dict["mean_metrics_faiss"] = faiss_mean
        ret_dict["std_metrics_faiss"] = faiss_std
        ret_dict["faiss"] = faiss_block
        ret_dict["search_only_avg_ms_faiss"] = search_only_avg_faiss_ms
        ret_dict["search_only_avg_ms_std_faiss"] = search_only_avg_faiss_ms_std
        ret_dict["search_only_p50_ms_faiss"] = search_only_p50_faiss_ms
        ret_dict["search_only_p50_ms_std_faiss"] = search_only_p50_faiss_ms_std
        ret_dict["search_only_p95_ms_faiss"] = search_only_p95_faiss_ms
        ret_dict["search_only_p95_ms_std_faiss"] = search_only_p95_faiss_ms_std

    for db_name, info in disk_size_info.items():
        ret_dict[f"disk_size_mb_{db_name}"] = info.get("disk_size_mb")
        if "disk_size_error" in info:
            ret_dict[f"disk_size_error_{db_name}"] = info["disk_size_error"]
        if "disk_size_flag" in info:
            ret_dict[f"disk_size_flag_{db_name}"] = info["disk_size_flag"]
        if "disk_io_write_mb" in info:
            ret_dict[f"disk_io_write_mb_{db_name}"] = info["disk_io_write_mb"]

    if dump_per_query:
        ret_dict["per_query_details_chroma"] = chroma_eval["per_query_details"]
        ret_dict["per_query_details_qdrant"] = qdrant_eval["per_query_details"]
        ret_dict["per_query_details_milvus"] = milvus_eval["per_query_details"]
        ret_dict["per_query_details_elasticsearch"] = elasticsearch_eval["per_query_details"]
        ret_dict["per_query_details_weaviate"] = weaviate_eval["per_query_details"]
        if include_faiss:
            assert faiss_eval is not None
            ret_dict["per_query_details_faiss"] = faiss_eval["per_query_details"]
    return ret_dict








def _format_val(val: float, fmt: str = ".2f") -> str:
    if val is None or (isinstance(val, float) and (val != val or val < 0)):  # NaN check
        return "—"
    return f"{val:{fmt}}"


def _print_table(report: dict[str, Any]) -> None:
    use_color = sys.stdout.isatty()

    COLOR_GREEN = "\033[92m" if use_color else ""
    COLOR_RED = "\033[91m" if use_color else ""
    COLOR_RESET = "\033[0m" if use_color else ""

    dbs = [
        ("ChromaDB", "chroma"),
        ("Qdrant", "qdrant"),
        *([("FAISS", "faiss")] if "mean_recall_faiss" in report else []),
        ("Milvus", "milvus"),
        ("ElasticSearch", "elasticsearch"),
        ("Weaviate", "weaviate"),
    ]

    ck = report.get("recall_at_k_metric", "recall@10")

    def get_val(metric_type: str, key_suffix: str) -> dict[str, float]:
        res = {}
        for db_name, key in dbs:
            if metric_type == "add_seconds":
                res[db_name] = report.get(f"{key}_add_seconds", 0.0)
            elif metric_type == "idx_mem":
                block = report.get(key, {})
                res[db_name] = block.get("indexing", {}).get("memory_usage_mb", 0.0) if block else 0.0
            elif metric_type == "idx_cpu":
                block = report.get(key, {})
                res[db_name] = block.get("indexing", {}).get("cpu_percent", 0.0) if block else 0.0
            elif metric_type == "idx_total":
                res[db_name] = report.get(f"{key}_indexing_total_seconds", 0.0)
            elif metric_type == "ret_avg":
                res[db_name] = report.get(f"retrieval_avg_ms_{key}", 0.0)
            elif metric_type == "ret_p50":
                res[db_name] = report.get(f"retrieval_p50_ms_{key}", 0.0)
            elif metric_type == "ret_p95":
                res[db_name] = report.get(f"retrieval_p95_ms_{key}", 0.0)
            elif metric_type == "search_only_p50":
                res[db_name] = report.get(f"search_only_p50_ms_{key}", 0.0)
            elif metric_type == "ret_mem":
                block = report.get(key, {})
                res[db_name] = block.get("retrieval", {}).get("memory_usage_mb", 0.0) if block else 0.0
            elif metric_type == "ret_cpu":
                block = report.get(key, {})
                res[db_name] = block.get("retrieval", {}).get("cpu_percent", 0.0) if block else 0.0
            elif metric_type == "recall":
                res[db_name] = report.get(f"mean_recall_{key}", float("nan"))
            elif metric_type == "mrr":
                mm = report.get(f"mean_metrics_{key}", {})
                res[db_name] = mm.get("mrr", float("nan"))
            elif metric_type == "ndcg10":
                mm = report.get(f"mean_metrics_{key}", {})
                res[db_name] = mm.get("ndcg@10", float("nan"))
        return res

    def format_row(title: str, val_dict: dict[str, float], fmt: str = ".2f", higher_is_better: bool = False, apply_color: bool = True) -> str:
        valid_vals = [v for v in val_dict.values() if v is not None and isinstance(v, (int, float)) and v == v]
        best_val = max(valid_vals) if (higher_is_better and valid_vals) else (min(valid_vals) if valid_vals else None)
        worst_val = min(valid_vals) if (higher_is_better and valid_vals) else (max(valid_vals) if valid_vals else None)

        cell_strings = []
        for db_name, _ in dbs:
            val = val_dict.get(db_name, float("nan"))
            formatted = _format_val(val, fmt)
            if apply_color and best_val is not None and worst_val is not None and best_val != worst_val:
                if abs(val - best_val) < 1e-9:
                    cell_str = f"{COLOR_GREEN}{formatted:>16}{COLOR_RESET}"
                elif abs(val - worst_val) < 1e-9:
                    cell_str = f"{COLOR_RED}{formatted:>16}{COLOR_RESET}"
                else:
                    cell_str = f"{formatted:>16}"
            else:
                cell_str = f"{formatted:>16}"
            cell_strings.append(cell_str)

        return f"{title:<50} | " + " | ".join(cell_strings)

    header = f"{'Metrik':<50} | " + " | ".join(f"{db_name:>16}" for db_name, _ in dbs)
    sep_width = len(header)

    print("\n" + "=" * sep_width)
    print("VECTOR DB BENCHMARK (SQuAD — LLM devre dışı, sadece embed + retrieval)")
    print("=" * sep_width)
    print(
        f"Splits: {report.get('splits_used')}  |  "
        f"Döküman: {report['num_documents']}  |  "
        f"Chunk: {report['num_chunks']}  |  "
        f"Soru: {report['num_queries_evaluated']}  |  top_k={report['top_k']}  |  "
        f"query embed (ort.): {report.get('query_embedding_avg_ms', float('nan')):.2f} ms"
    )
    print("-" * sep_width)
    print(header)
    print("-" * sep_width)
    embedding_row_cells = " | ".join(
        f"{report['embedding_seconds']:>16.3f}" if db_name == "ChromaDB" else f"{'—':>16}"
        for db_name, _ in dbs
    )
    print(f"{'İndeksleme: embedding (ortak, tek geçiş) [s]':<50} | " + embedding_row_cells)
    print(format_row("İndeksleme: DB yazma (add_chunks) [s]", get_val("add_seconds", ""), fmt=".3f", higher_is_better=False))
    print(format_row("İndeksleme: peak RAM (RSS) [MB]", get_val("idx_mem", ""), fmt=".3f", higher_is_better=False))
    print(format_row("İndeksleme: ort. CPU [%]", get_val("idx_cpu", ""), fmt=".2f", higher_is_better=False))
    print(format_row("İndeksleme: toplam (embedding + bu DB) [s]", get_val("idx_total", ""), fmt=".3f", higher_is_better=False))

    print(format_row("Ortalama retrieval [ms] (sorgu embed + arama)", get_val("ret_avg", ""), fmt=".2f", higher_is_better=False))
    print(format_row("p50 retrieval latency [ms]", get_val("ret_p50", ""), fmt=".2f", higher_is_better=False))
    print(format_row("p50 search-only latency [ms] (embed hariç)", get_val("search_only_p50", ""), fmt=".2f", higher_is_better=False))
    print(format_row("p95 retrieval latency [ms]", get_val("ret_p95", ""), fmt=".2f", higher_is_better=False))

    print(format_row("Retrieval: peak RAM (RSS) [MB]", get_val("ret_mem", ""), fmt=".3f", higher_is_better=False))
    print(format_row("Retrieval: ort. CPU [%]", get_val("ret_cpu", ""), fmt=".2f", higher_is_better=False))

    print(format_row(f"Ortalama {ck}", get_val("recall", ""), fmt=".4f", higher_is_better=True))
    print(format_row("Ortalama MRR", get_val("mrr", ""), fmt=".4f", higher_is_better=True))
    print(format_row("Ortalama nDCG@10", get_val("ndcg10", ""), fmt=".4f", higher_is_better=True))
    print("-" * sep_width)

    p50_dict = get_val("ret_p50", "")
    valid_p50 = {k: v for k, v in p50_dict.items() if v is not None and isinstance(v, (int, float)) and v == v}
    fastest_db = min(valid_p50, key=lambda k: valid_p50[k]) if valid_p50 else "N/A"
    fastest_p50 = valid_p50[fastest_db] if valid_p50 else 0.0

    mrr_dict = get_val("mrr", "")
    valid_mrr = {k: v for k, v in mrr_dict.items() if v is not None and isinstance(v, (int, float)) and v == v}
    best_mrr_db = max(valid_mrr, key=lambda k: valid_mrr[k]) if valid_mrr else "N/A"
    best_mrr_val = valid_mrr[best_mrr_db] if valid_mrr else 0.0

    summary_line = f"En hızlı (p50): {fastest_db} {fastest_p50:.1f}ms | En yüksek MRR: {best_mrr_db} {best_mrr_val:.3f}"
    print(summary_line)
    print("=" * sep_width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChromaDB vs Qdrant vs FAISS vs Milvus vs ElasticSearch vs Weaviate: indexing ve retrieval benchmark (LLM yok)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="İsteğe bağlı YAML (chunker, embedder, evaluator, pipeline, "
        "chroma_retriever, qdrant_retriever, faiss_retriever, milvus_retriever, elasticsearch_retriever, weaviate_retriever).",
    )
    parser.add_argument("--num-documents", type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=3,
        help="Her DB için retrieval-pass'in kaç kez tekrarlanacağı (ortalama±std için). "
        "1 verilirse eski tek-geçiş davranışıyla aynı sonucu üretir.",
    )
    parser.add_argument(
        "--squad-version",
        type=str,
        default="squad_v2",
        help="HuggingFace datasets adı (varsayılan: squad_v2).",
    )
    parser.add_argument("--collection-path", type=str, default=None, help="Local MS MARCO collection TSV/TSV.GZ path.")
    parser.add_argument("--queries-path", type=str, default=None, help="Local MS MARCO queries TSV/TSV.GZ path.")
    parser.add_argument("--qrels-path", type=str, default=None, help="Local MS MARCO qrels TSV/TSV.GZ path.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm ilerleme çubuklarını kapat (log/CI için).",
    )
    parser.add_argument(
        "--query-seed",
        type=int,
        default=42,
        help="Sorgu seçiminde kullanılacak seed değeri.",
    )
    parser.add_argument(
        "--dump-per-query",
        action="store_true",
        help="Her sorgu için metrikleri ayrı bir dosyaya dök.",
    )
    parser.add_argument(
        "--include-faiss",
        action="store_true",
        help="FAISS'i sistematik benchmark'a dahil et. Varsayılan: kapalı — FAISS "
        "in-process bir kütüphane (ağ katmanı yok) ve Docker'daki diğer DB'lerle "
        "adil karşılaştırılamıyor (bkz. CLAUDE.md Pinecone deseni).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Sonuç dosya adına eklenecek opsiyonel ekstra etiket, ör. 'V2' veya 'chunk256'.",
    )
    parser.add_argument(
        "--fixed-query-file",
        type=str,
        default=None,
        help="Sabit sorgu seti dosyası (JSON, {'query_ids': [...]})."
        " Verilirse: bu ID listesi TÜM ölçeklerde kullanılır, ölçeğe özgü sorgu seçimi "
        "tamamen bypass edilir (_select_queries_for_documents hiç çağrılmaz)."
        " Verilmezse: bu koşumda otomatik seçilen sorgu ID'leri "
        "experiments/results/fixed_queries_<timestamp>.json'a yazılır — en küçük ölçekte "
        "(S1) koşulup üretilen bu dosya sonraki ölçeklerde --fixed-query-file olarak verilmelidir.",
    )
    args = parser.parse_args()

    if args.num_repeats < 1:
        raise ValueError(f"--num-repeats must be >= 1, got {args.num_repeats}")

    fixed_query_ids: set[str] | None = None
    if args.fixed_query_file:
        fixed_query_file_path = os.path.abspath(args.fixed_query_file)
        with open(fixed_query_file_path, encoding="utf-8") as f:
            fixed_query_payload = json.load(f)
        fixed_query_ids = set(fixed_query_payload["query_ids"])
        logging.info(
            "Sabit sorgu seti yüklendi: %s (%d ID) — ölçeğe özgü seçim bypass edildi.",
            fixed_query_file_path,
            len(fixed_query_ids),
        )

    if args.config:
        raw = load_yaml(os.path.abspath(args.config))
        exp = build_component_configs(raw)
        chunker_cfg = exp.chunker
        embedder_cfg = exp.embedder
        dataset_config = exp.dataset
        k_values = sorted({*exp.evaluator.k_values, args.top_k})
        chroma_dict = {k: v for k, v in raw.get("chroma_retriever", {}).items() if k != "type"}
        qdrant_dict = {k: v for k, v in raw.get("qdrant_retriever", {}).items() if k != "type"}
        faiss_dict = {k: v for k, v in raw.get("faiss_retriever", {}).items() if k != "type"}
        milvus_dict = {k: v for k, v in raw.get("milvus_retriever", {}).items() if k != "type"}
        elasticsearch_dict = {k: v for k, v in raw.get("elasticsearch_retriever", {}).items() if k != "type"}
        weaviate_dict = {k: v for k, v in raw.get("weaviate_retriever", {}).items() if k != "type"}
        chroma_cfg = ChromaRetrieverConfig(**chroma_dict) if chroma_dict else ChromaRetrieverConfig()
        qdrant_cfg = QdrantRetrieverConfig(**qdrant_dict) if qdrant_dict else QdrantRetrieverConfig(in_memory=True)
        faiss_cfg = FAISSRetrieverConfig(**faiss_dict) if faiss_dict else FAISSRetrieverConfig()
        milvus_cfg = MilvusRetrieverConfig(**milvus_dict) if milvus_dict else MilvusRetrieverConfig(in_memory=True)
        elasticsearch_cfg = ElasticSearchRetrieverConfig(**elasticsearch_dict) if elasticsearch_dict else ElasticSearchRetrieverConfig(in_memory=True)
        weaviate_cfg = WeaviateRetrieverConfig(**weaviate_dict) if weaviate_dict else WeaviateRetrieverConfig(in_memory=True)
    else:
        chunker_cfg = FixedSizeChunkerConfig()
        embedder_cfg = SentenceTransformersEmbedderConfig()
        k_values = sorted({1, 3, 5, 10, args.top_k})
        chroma_cfg = ChromaRetrieverConfig(collection_name="bench_chroma_squad")
        qdrant_cfg = QdrantRetrieverConfig(
            collection_name="bench_qdrant_squad",
            in_memory=True,
        )
        faiss_cfg = FAISSRetrieverConfig(collection_name="bench_faiss_squad")
        milvus_cfg = MilvusRetrieverConfig(
            collection_name="bench_milvus_squad",
            in_memory=True,
        )
        elasticsearch_cfg = ElasticSearchRetrieverConfig(index_name="bench_elasticsearch_squad")
        weaviate_cfg = WeaviateRetrieverConfig(collection_name="bench_weaviate_squad")
        if args.collection_path or args.queries_path or args.qrels_path:
            if not (args.collection_path and args.queries_path and args.qrels_path):
                raise ValueError("--collection-path, --queries-path, and --qrels-path must be supplied together.")
            dataset_config: SQuADDatasetConfig | MSMARCODatasetConfig = MSMARCODatasetConfig(
                collection_path=args.collection_path,
                queries_path=args.queries_path,
                qrels_path=args.qrels_path,
                max_documents=args.num_documents,
                num_queries=args.num_queries,
            )
        else:
            dataset_config = SQuADDatasetConfig(version=args.squad_version)

    top_k = args.top_k
    report = run_benchmark(
        num_documents=args.num_documents,
        num_queries=args.num_queries,
        top_k=top_k,
        chunker_cfg=chunker_cfg,
        embedder_cfg=embedder_cfg,
        chroma_cfg=chroma_cfg,
        qdrant_cfg=qdrant_cfg,
        faiss_cfg=faiss_cfg,
        milvus_cfg=milvus_cfg,
        elasticsearch_cfg=elasticsearch_cfg,
        weaviate_cfg=weaviate_cfg,
        k_values=k_values,
        dataset_config=dataset_config,
        show_progress=not args.no_progress,
        num_repeats=args.num_repeats,
        fixed_query_ids=fixed_query_ids,
        query_seed=args.query_seed,
        dump_per_query=args.dump_per_query,
        include_faiss=args.include_faiss,
    )

    _print_table(report)


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    if not args.fixed_query_file:
        # Bu koşumda ölçeğe özgü seçilen sorgu ID'lerini kalıcı hale getir; sonraki
        # ölçekler (S2-S5) bu dosyayı --fixed-query-file olarak vererek aynı
        # sorgu setini kullanabilir (bkz. DENEY-PLANI-V2.md 0.2).
        fixed_query_out_path = os.path.join(out_dir, f"fixed_queries_{ts}.json")
        with open(fixed_query_out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query_ids": report["query_ids_used"],
                    "num_queries": report["num_queries_evaluated"],
                    "source_num_documents": report["num_documents"],
                    "timestamp": ts,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logging.info(
            "Sabit sorgu seti adayı yazıldı: %s (%d ID, kaynak ölçek: %d doküman)",
            fixed_query_out_path,
            report["num_queries_evaluated"],
            report["num_documents"],
        )

    num_docs = report["num_documents"]
    db_count = 6 if args.include_faiss else 5
    tag_part = f"_{args.output_tag}" if args.output_tag else ""
    out_path = os.path.join(
        out_dir, f"official_scale{tag_part}_{num_docs}docs_{db_count}db_topk{top_k}_{ts}.json"
    )
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
