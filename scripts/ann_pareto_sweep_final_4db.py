#!/usr/bin/env python3
"""ANN query-time parameter sweep / Recall-Latency Pareto analysis for the final 4-DB
comparison (Qdrant, Milvus, Elasticsearch, Weaviate) at 100,000 documents / 500 fixed
queries / top_k=10. This is the final experiment for the thesis's experimental section.

Critical experimental rule: for each DB the index/collection is built EXACTLY ONCE.
The sweep only varies the query-time ANN search-effort parameter on the SAME already-built
index -- it never rebuilds. This isolates search-effort -> recall/latency trade-off from
build-to-build nondeterminism (see analysis_FINAL_4DB_exact_vs_ann_*.json's methodology
note: build nondeterminism was only empirically confirmed for Milvus at 100K and must not
be generalized to Qdrant/Weaviate).

Query-time parameter mapping (verified against the client/server code paths actually
reached by src/retrievers/*.py -- see per-DB comments below):
    Qdrant        -> SearchParams(hnsw_ef=...)   passed on every query_points() call
    Milvus        -> search_params={"params": {"ef": ...}} passed on every search() call
    Elasticsearch -> knn.num_candidates           passed on every search() call
    Weaviate      -> NOT query-time adjustable in this codebase/client version -- `ef` is
                     baked into the collection's vector_index_config at collection-create
                     time (wvc.config.Configure.VectorIndex.hnsw(ef=...)); the v4 client's
                     near_vector() query has no per-request ef override. Rebuilding the
                     collection per sweep value would violate "build once", so Weaviate is
                     EXCLUDED from the actual sweep and reported as a single reference point
                     at the FINAL_4DB default (ef=64, baked in at build time).

Usage:
    python scripts/ann_pareto_sweep_final_4db.py --num-documents 100000
"""

from __future__ import annotations  # noqa: I001

# IMPORTANT: faiss must be imported before torch/sentence-transformers (see
# exact_vs_ann_final_4db.py for the SIGSEGV rationale on macOS ARM).
import faiss  # noqa: F401,E402

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.benchmark_db import (  # noqa: E402
    build_reproducibility_metadata,
    load_dataset_corpus,
)
from scripts.exact_knn import exact_topk_per_query  # noqa: E402
from scripts.query_level_bootstrap_ci import bootstrap_ci  # noqa: E402
from src.chunkers.fixed_size_chunker import FixedSizeChunker  # noqa: E402
from src.core.types import Chunk, Embedding, Query  # noqa: E402
from src.datasets import MSMARCODatasetConfig  # noqa: E402
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: E402
from src.evaluators.config import RetrievalEvaluatorConfig  # noqa: E402
from src.evaluators.retrieval_evaluator import RetrievalEvaluator  # noqa: E402
from src.retrievers.config import (  # noqa: E402
    ChromaRetrieverConfig,
    ElasticSearchRetrieverConfig,
    MilvusRetrieverConfig,
    QdrantRetrieverConfig,
    WeaviateRetrieverConfig,
)
from src.retrievers.elasticsearch_retriever import ElasticSearchRetriever  # noqa: E402
from src.retrievers.milvus_retriever import MilvusRetriever  # noqa: E402
from src.retrievers.qdrant_retriever import QdrantRetriever  # noqa: E402
from src.retrievers.weaviate_retriever import WeaviateRetriever  # noqa: E402
from src.utils.config_loader import build_component_configs, load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FINAL_DBS = ["qdrant", "milvus", "elasticsearch", "weaviate"]
DEFAULT_SWEEP_VALUES = [16, 32, 64, 128, 256]
N_BOOTSTRAP = 2000
SEED = 42
WARMUP_COUNT = 10

# db -> (search-time param field on the retriever's frozen config, whether it is truly
# adjustable per-query without a rebuild)
PARAM_FIELD = {
    "qdrant": "hnsw_ef_search",
    "milvus": "hnsw_ef_search",
    "elasticsearch": "num_candidates",
    "weaviate": "hnsw_ef_search",  # baked in at build time -- NOT swept, see module docstring
}
SWEEP_APPLICABLE = {
    "qdrant": True,
    "milvus": True,
    "elasticsearch": True,
    "weaviate": False,
}


def _ann_recall_vs_exact(ann_ids: list[str], exact_ids: list[str], k: int) -> float:
    if not exact_ids:
        return float("nan")
    ann_top = set(ann_ids[:k])
    exact_top = set(exact_ids[:k])
    return len(ann_top & exact_top) / k


def _get_index_count(db_name: str, retriever: Any) -> int | None:
    """Best-effort live point/entity/document count -- used only as a same-index sanity
    check across sweep points (build-once validation), never as a benchmark metric."""
    try:
        if db_name == "qdrant":
            client = retriever._get_client()
            return int(client.count(collection_name=retriever._config.collection_name, exact=True).count)
        if db_name == "milvus":
            client = retriever._get_client()
            stats = client.get_collection_stats(collection_name=retriever._config.collection_name)
            return int(stats.get("row_count", -1))
        if db_name == "elasticsearch":
            client = retriever._get_client()
            resp = client.count(index=retriever._config.index_name)
            return int(resp.get("count", resp.body.get("count") if hasattr(resp, "body") else -1))
        if db_name == "weaviate":
            collection = retriever._collection
            if collection is None:
                return None
            agg = collection.aggregate.over_all(total_count=True)
            return int(agg.total_count)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not fetch index count for %s: %s", db_name, exc)
        return None
    return None


def _run_repeats_for_value(
    retriever: Any,
    bench_queries: list[Query],
    query_embeddings: dict[str, Embedding],
    top_k: int,
    num_repeats: int,
    show_progress: bool,
) -> tuple[list[list[float]], list[tuple[Query, Any]]]:
    """Runs `num_repeats` independent timing passes over the full query set.

    Returns (per_repeat_latencies_ms, first_repeat_query_results). The quality metrics
    (qrels recall/mrr/ndcg, ann_recall_vs_exact) are computed ONLY from the first repeat's
    results -- deterministic timing repeats are never treated as independent quality
    samples, per the experiment's methodology rule.
    """
    from tqdm import tqdm

    per_repeat_latencies: list[list[float]] = []
    first_repeat_results: list[tuple[Query, Any]] = []

    warmup_queries = bench_queries[:WARMUP_COUNT] if len(bench_queries) >= WARMUP_COUNT else []
    for repeat_idx in range(num_repeats):
        for q in warmup_queries:
            retriever.retrieve_with_embedding(query_embeddings[q.id], top_k=top_k, query_id=q.id)

        latencies_ms: list[float] = []
        for q in tqdm(
            bench_queries,
            desc=f"repeat {repeat_idx + 1}/{num_repeats}",
            unit="q",
            leave=False,
            disable=not show_progress,
        ):
            t0 = time.perf_counter()
            result = retriever.retrieve_with_embedding(query_embeddings[q.id], top_k=top_k, query_id=q.id)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if repeat_idx == 0:
                first_repeat_results.append((q, result))
        per_repeat_latencies.append(latencies_ms)

    return per_repeat_latencies, first_repeat_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "experiments",
                                               "configs", "benchmark_all_dbs.yaml"))
    parser.add_argument("--num-documents", type=int, default=100000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--values", type=str, default="16,32,64,128,256")
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--fixed-query-file", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "experiments",
                                               "results", "fixed_queries_20260803_123156.json"))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    top_k = args.top_k
    sweep_values = [int(v.strip()) for v in args.values.split(",") if v.strip()]

    raw = load_yaml(os.path.abspath(args.config))
    exp = build_component_configs(raw)
    chunker_cfg = exp.chunker
    embedder_cfg = exp.embedder
    dataset_config = exp.dataset

    chroma_dict = {k: v for k, v in raw.get("chroma_retriever", {}).items() if k != "type"}
    qdrant_dict = {k: v for k, v in raw.get("qdrant_retriever", {}).items() if k != "type"}
    milvus_dict = {k: v for k, v in raw.get("milvus_retriever", {}).items() if k != "type"}
    elasticsearch_dict = {k: v for k, v in raw.get("elasticsearch_retriever", {}).items() if k != "type"}
    weaviate_dict = {k: v for k, v in raw.get("weaviate_retriever", {}).items() if k != "type"}
    chroma_cfg = ChromaRetrieverConfig(**chroma_dict) if chroma_dict else ChromaRetrieverConfig()
    qdrant_cfg = QdrantRetrieverConfig(**qdrant_dict) if qdrant_dict else QdrantRetrieverConfig(in_memory=True)
    milvus_cfg = MilvusRetrieverConfig(**milvus_dict) if milvus_dict else MilvusRetrieverConfig(in_memory=True)
    elasticsearch_cfg = (
        ElasticSearchRetrieverConfig(**elasticsearch_dict) if elasticsearch_dict
        else ElasticSearchRetrieverConfig(in_memory=True)
    )
    weaviate_cfg = WeaviateRetrieverConfig(**weaviate_dict) if weaviate_dict else WeaviateRetrieverConfig(in_memory=True)

    default_ef = {
        "qdrant": qdrant_cfg.hnsw_ef_search,
        "milvus": milvus_cfg.hnsw_ef_search,
        "elasticsearch": elasticsearch_cfg.num_candidates,
        "weaviate": weaviate_cfg.hnsw_ef_search,
    }

    with open(args.fixed_query_file, encoding="utf-8") as f:
        fixed_query_ids = set(json.load(f)["query_ids"])
    logging.info("Loaded fixed query set: %d IDs", len(fixed_query_ids))

    documents, all_queries, ground_truth, splits_used = load_dataset_corpus(
        dataset_config, args.num_documents
    )
    doc_ids = {d.id for d in documents}
    bench_queries = [q for q in all_queries if q.id in fixed_query_ids]
    bench_queries = [q for q in bench_queries if ground_truth.get(q.id, set()).issubset(doc_ids)]
    if len(bench_queries) != len(fixed_query_ids):
        logging.warning(
            "fixed_query_ids has %d IDs but only %d are valid for this scale (%d docs).",
            len(fixed_query_ids), len(bench_queries), args.num_documents,
        )
    query_hash = hashlib.sha256(",".join(sorted(q.id for q in bench_queries)).encode("utf-8")).hexdigest()
    logging.info("Bench query set: %d queries, hash=%s", len(bench_queries), query_hash[:16])

    chunker = FixedSizeChunker(chunker_cfg)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    chunk_ids = [c.id for c in chunks]

    embedder = SentenceTransformersEmbedder(embedder_cfg)
    logging.info("Embedding %d chunks...", len(chunks))
    embeddings = embedder.embed_chunks(chunks)
    logging.info("Embedding %d queries...", len(bench_queries))
    query_embeddings = {q.id: embedder.embed_query(q) for q in bench_queries}

    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=[top_k]))
    recall_key, ndcg_key, mrr_key = f"recall@{top_k}", f"ndcg@{top_k}", f"mrr@{top_k}"

    logging.info("Computing exact k-NN baseline (top-%d) via faiss.IndexFlatIP ONCE, reused for all sweep points...", top_k)
    t0 = time.perf_counter()
    exact_topk, exact_method = exact_topk_per_query(
        query_embeddings, embeddings, chunk_ids, k=top_k, distance_metric="cosine"
    )
    exact_seconds = time.perf_counter() - t0
    assert exact_method.startswith("faiss.IndexFlat"), f"Exact baseline must be exact; got {exact_method!r}."
    logging.info("Exact k-NN done in %.2fs via %s", exact_seconds, exact_method)

    retriever_specs: dict[str, tuple[Any, Any]] = {
        "qdrant": (QdrantRetriever, qdrant_cfg),
        "milvus": (MilvusRetriever, milvus_cfg),
        "elasticsearch": (ElasticSearchRetriever, elasticsearch_cfg),
        "weaviate": (WeaviateRetriever, weaviate_cfg),
    }

    all_rows: list[dict[str, Any]] = []
    per_db_report: dict[str, Any] = {}
    crash_log: dict[str, str | None] = {}
    effective_config: dict[str, Any] = {}

    for db_name, (cls, cfg) in retriever_specs.items():
        logging.info("=== %s: building index ONCE (%d docs) ===", db_name, args.num_documents)
        crash_log[db_name] = None
        param_field = PARAM_FIELD[db_name]
        sweep_applicable = SWEEP_APPLICABLE[db_name]
        values_for_db = sweep_values if sweep_applicable else [default_ef[db_name]]

        try:
            retriever = cls(config=cfg, embedder=embedder)
            retriever.clear()
            t_build0 = time.perf_counter()
            retriever.add_chunks(chunks, embeddings)
            build_seconds = time.perf_counter() - t_build0
            effective_config[db_name] = retriever.describe_index()
            build_count = _get_index_count(db_name, retriever)
            logging.info("%s: index built in %.1fs, count=%s", db_name, build_seconds, build_count)

            db_rows: list[dict[str, Any]] = []
            counts_across_sweep: list[int | None] = []
            identical_results_flag = False
            prev_ann_id_sets: list[frozenset[str]] | None = None

            for value in values_for_db:
                before = getattr(retriever._config, param_field)
                object.__setattr__(retriever._config, param_field, value)
                after = getattr(retriever._config, param_field)
                assert after == value, (
                    f"{db_name}: failed to set {param_field}={value} on live config (still {after})."
                )
                logging.info(
                    "%s: %s changed %s -> %s (same index/collection, no rebuild)",
                    db_name, param_field, before, after,
                )

                per_repeat_latencies, first_repeat_results = _run_repeats_for_value(
                    retriever, bench_queries, query_embeddings, top_k, args.num_repeats,
                    show_progress=not args.no_progress,
                )

                query_level: list[dict[str, Any]] = []
                cur_ann_id_sets: list[frozenset[str]] = []
                for q, result in first_repeat_results:
                    ann_ids = [rc.chunk.id for rc in result.chunks]
                    cur_ann_id_sets.append(frozenset(ann_ids[:top_k]))
                    gt = set(ground_truth.get(q.id, set()))
                    qrels_metrics = evaluator.evaluate(result, gt)
                    query_level.append({
                        "query_id": q.id,
                        "ann_recall_vs_exact@10": _ann_recall_vs_exact(ann_ids, exact_topk.get(q.id, []), top_k),
                        "recall@10": qrels_metrics.get(recall_key, float("nan")),
                        "mrr@10": qrels_metrics.get(mrr_key, float("nan")),
                        "ndcg@10": qrels_metrics.get(ndcg_key, float("nan")),
                    })

                if sweep_applicable and prev_ann_id_sets is not None and cur_ann_id_sets == prev_ann_id_sets:
                    identical_results_flag = True
                    logging.warning(
                        "%s: results at %s=%s are BYTE-IDENTICAL to the previous sweep value -- "
                        "parameter may not actually be reaching the search request.",
                        db_name, param_field, value,
                    )
                prev_ann_id_sets = cur_ann_id_sets

                count_now = _get_index_count(db_name, retriever)
                counts_across_sweep.append(count_now)

                p50_per_repeat = [float(np.percentile(lats, 50)) for lats in per_repeat_latencies]
                p95_per_repeat = [float(np.percentile(lats, 95)) for lats in per_repeat_latencies]
                p99_per_repeat = [float(np.percentile(lats, 99)) for lats in per_repeat_latencies]
                mean_per_repeat = [float(np.mean(lats)) for lats in per_repeat_latencies]
                qps_per_repeat = [
                    len(bench_queries) / (sum(lats) / 1000.0) if sum(lats) > 0 else float("nan")
                    for lats in per_repeat_latencies
                ]

                ann_values = [r["ann_recall_vs_exact@10"] for r in query_level]
                ann_mean, ann_lo, ann_hi = bootstrap_ci(ann_values, n_samples=N_BOOTSTRAP, seed=SEED)
                recall_vals = [r["recall@10"] for r in query_level]
                mrr_vals = [r["mrr@10"] for r in query_level]
                ndcg_vals = [r["ndcg@10"] for r in query_level]
                recall_mean, recall_lo, recall_hi = bootstrap_ci(recall_vals, n_samples=N_BOOTSTRAP, seed=SEED)

                row = {
                    "db": db_name,
                    "search_param_name": param_field,
                    "search_param_value": value,
                    "sweep_applicable": sweep_applicable,
                    "num_documents": len(documents),
                    "num_queries": len(query_level),
                    "qrels_recall@10": recall_mean,
                    "qrels_recall@10_ci_low": recall_lo,
                    "qrels_recall@10_ci_high": recall_hi,
                    "qrels_mrr@10": sum(mrr_vals) / len(mrr_vals) if mrr_vals else float("nan"),
                    "qrels_ndcg@10": sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else float("nan"),
                    "ann_recall_vs_exact@10": ann_mean,
                    "ann_recall_vs_exact@10_ci_low": ann_lo,
                    "ann_recall_vs_exact@10_ci_high": ann_hi,
                    "search_only_mean_latency_ms": float(np.mean(mean_per_repeat)),
                    "p50_ms": float(np.mean(p50_per_repeat)),
                    "p95_ms": float(np.mean(p95_per_repeat)),
                    "p99_ms": float(np.mean(p99_per_repeat)),
                    "wall_clock_search_only_qps": float(np.mean(qps_per_repeat)),
                    "num_repeats": args.num_repeats,
                    "index_count_at_this_point": count_now,
                    "results_identical_to_previous_value": (
                        identical_results_flag if value == values_for_db[-1] and sweep_applicable else None
                    ),
                }

                assert 0.0 <= row["p50_ms"], f"{db_name}: non-positive latency at {param_field}={value}"
                assert row["wall_clock_search_only_qps"] > 0, f"{db_name}: non-positive QPS at {param_field}={value}"
                if row["ann_recall_vs_exact@10"] == row["ann_recall_vs_exact@10"]:  # not NaN
                    assert 0.0 <= row["ann_recall_vs_exact@10"] <= 1.0, (
                        f"{db_name}: ann_recall_vs_exact out of [0,1] at {param_field}={value}"
                    )
                assert row["num_queries"] == len(bench_queries), (
                    f"{db_name}: only {row['num_queries']}/{len(bench_queries)} queries completed"
                )

                db_rows.append(row)
                all_rows.append(row)
                logging.info(
                    "%s %s=%s: qrels_recall@10=%.4f ann_recall_vs_exact=%.4f p50=%.2fms p95=%.2fms "
                    "p99=%.2fms qps=%.1f",
                    db_name, param_field, value, row["qrels_recall@10"], row["ann_recall_vs_exact@10"],
                    row["p50_ms"], row["p95_ms"], row["p99_ms"], row["wall_clock_search_only_qps"],
                )

            same_index_across_sweep = (
                len({c for c in counts_across_sweep if c is not None}) <= 1
                and build_count in (counts_across_sweep[0] if counts_across_sweep else None, None)
            )

            def _is_dominated(a: dict[str, Any], b: dict[str, Any]) -> bool:
                return (
                    b["qrels_recall@10"] >= a["qrels_recall@10"]
                    and b["p50_ms"] <= a["p50_ms"]
                    and (b["qrels_recall@10"] > a["qrels_recall@10"] or b["p50_ms"] < a["p50_ms"])
                )

            for row in db_rows:
                row["pareto_optimal"] = not any(
                    _is_dominated(row, other) for other in db_rows if other is not row
                )

            per_db_report[db_name] = {
                "build_seconds": build_seconds,
                "build_index_count": build_count,
                "sweep_applicable": sweep_applicable,
                "sweep_not_applicable_reason": (
                    None if sweep_applicable else
                    "ef is baked into the collection's vector_index_config at collection-create "
                    "time in this codebase/weaviate-client v4 version; near_vector() query has no "
                    "per-request ef override, so varying it would require a rebuild -- excluded "
                    "from the query-time sweep per the experiment's build-once rule. Single "
                    "reference point reported at the FINAL_4DB build-time default."
                ),
                "same_index_reused_across_all_sweep_points": same_index_across_sweep,
                "index_counts_across_sweep": counts_across_sweep,
                "identical_results_anomaly_detected": identical_results_flag,
                "rows": db_rows,
            }
        except Exception as exc:  # noqa: BLE001 - record and continue to next DB
            crash_log[db_name] = f"{type(exc).__name__}: {exc}"
            logging.error("DB %s failed: %s", db_name, crash_log[db_name])
            per_db_report[db_name] = {"error": crash_log[db_name]}
            continue

    any_crash = any(v is not None for v in crash_log.values())

    reproducibility = build_reproducibility_metadata(
        dataset_config=dataset_config,
        splits_used=splits_used,
        num_documents=len(documents),
        num_chunks=len(chunks),
        query_ids_used=[q.id for q in bench_queries],
        query_seed=42,
        chunker_cfg=chunker_cfg,
        embedder_cfg=embedder_cfg,
        embedding_dimension=embeddings[0].dimension if embeddings else None,
        top_k=top_k,
        effective_config=effective_config,
        retriever_cfgs={
            "chroma": chroma_cfg,
            "qdrant": qdrant_cfg,
            "milvus": milvus_cfg,
            "elasticsearch": elasticsearch_cfg,
            "weaviate": weaviate_cfg,
        },
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "experiment": "ann_pareto_sweep_final_4db",
        "timestamp": ts,
        "scope": {"final_dbs": FINAL_DBS, "excluded": ["chroma", "faiss"],
                  "num_documents": args.num_documents, "num_queries": len(bench_queries), "top_k": top_k},
        "dataset": "msmarco" if isinstance(dataset_config, MSMARCODatasetConfig) else "squad",
        "splits_used": splits_used,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),
        "query_id_hash": query_hash,
        "top_k": top_k,
        "sweep_values_requested": sweep_values,
        "num_repeats": args.num_repeats,
        "exact_method": exact_method,
        "exact_knn_seconds": exact_seconds,
        "any_db_crash": any_crash,
        "crash_log": crash_log,
        "param_field_by_db": PARAM_FIELD,
        "sweep_applicable_by_db": SWEEP_APPLICABLE,
        "note": (
            "Build-once rule: each DB's index/collection is created exactly once via "
            "add_chunks(); every sweep point mutates ONLY the retriever's in-process query-time "
            "search-param config field (verified via object.__setattr__ + getattr readback) and "
            "reissues retrieve_with_embedding() against the SAME collection. index_counts_across_"
            "sweep in results_by_db[db] is a live point/entity/doc-count readback proving no "
            "rebuild occurred. Quality metrics (qrels recall/mrr/ndcg, ann_recall_vs_exact) come "
            "ONLY from each sweep point's first timing repeat; the remaining num_repeats-1 "
            "repeats contribute to latency/QPS aggregation only, never treated as independent "
            "quality samples. ann_recall_vs_exact@10 measures agreement with the single reused "
            "exact faiss.IndexFlatIP baseline; qrels_* fields are the separate gold-relevance "
            "RetrievalEvaluator metrics -- the two must not be conflated."
        ),
        "results_by_db": per_db_report,
        "reproducibility": reproducibility,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "analysis_FINAL_4DB_ann_pareto_100K.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", json_path)

    csv_path = os.path.join(out_dir, "analysis_FINAL_4DB_ann_pareto_100K.csv")
    fieldnames = [
        "db", "search_param_name", "search_param_value", "sweep_applicable", "pareto_optimal",
        "qrels_recall@10", "qrels_mrr@10", "qrels_ndcg@10",
        "ann_recall_vs_exact@10", "ann_recall_vs_exact@10_ci_low", "ann_recall_vs_exact@10_ci_high",
        "search_only_mean_latency_ms", "p50_ms", "p95_ms", "p99_ms", "wall_clock_search_only_qps",
        "num_queries", "num_repeats", "index_count_at_this_point",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    logging.info("Wrote %s", csv_path)

    print(f"\n=== ANN Pareto Sweep summary ({args.num_documents} docs, {len(bench_queries)} queries) ===")
    print(f"{'DB':<15}{'Param':<10}{'Value':<8}{'Recall@10':<12}{'ANNvsExact':<12}"
          f"{'p50(ms)':<10}{'p95(ms)':<10}{'p99(ms)':<10}{'QPS':<10}{'Pareto':<8}")
    for row in all_rows:
        print(
            f"{row['db']:<15}{row['search_param_name']:<10}{row['search_param_value']:<8}"
            f"{row['qrels_recall@10']:<12.4f}{row['ann_recall_vs_exact@10']:<12.4f}"
            f"{row['p50_ms']:<10.2f}{row['p95_ms']:<10.2f}{row['p99_ms']:<10.2f}"
            f"{row['wall_clock_search_only_qps']:<10.1f}{'YES' if row['pareto_optimal'] else '':<8}"
        )

    if any_crash:
        print("\nWARNING: at least one DB crashed -- see crash_log in the output JSON.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
