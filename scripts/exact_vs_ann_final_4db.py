#!/usr/bin/env python3
"""Exact-vs-ANN experiment for the final 4-DB comparison (Qdrant, Milvus, Elasticsearch,
Weaviate). ChromaDB and FAISS are excluded from the DB comparison — FAISS is used only as
the exact-kNN baseline helper (raw `faiss.IndexFlatIP`, never an approximate index).

Kept fully separate from the qrels-based official_scale_*.json Recall@K pipeline:
this script additionally reports qrels-based Recall@10/MRR@10/nDCG@10 for context, but
`ann_recall_vs_exact@10` (agreement with THIS RUN'S OWN exact brute-force top-k) answers a
different question and must never be conflated with gold-relevance Recall@K.

Uses the same MS MARCO pipeline, same fixed 500-query file, same chunking/embedding/
distance-metric/HNSW-search config as the FINAL_4DB_S1-S6 runs (loaded from the same YAML
config, same code path as scripts/benchmark_db.py's main()), so reproducibility metadata
(config_hash, chunker, embedding) can be compared directly against those runs.

Usage:
    python scripts/exact_vs_ann_final_4db.py --num-documents 100000 --output-tag 100K
    python scripts/exact_vs_ann_final_4db.py --num-documents 200000 --output-tag 200K
"""

from __future__ import annotations  # noqa: I001

# IMPORTANT: faiss must be imported before torch/sentence-transformers. On this
# environment (macOS ARM, faiss 1.14.2 + torch's bundled OpenMP), importing faiss AFTER
# torch has already initialized its own OpenMP runtime causes a silent SIGSEGV inside
# faiss's IndexFlat*.search() — reproduced directly, confirmed fixed by import order,
# not by KMP_DUPLICATE_LIB_OK. Importing faiss first makes it own the OpenMP init.
import faiss  # noqa: F401,E402

import argparse
import csv
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
from src.core.types import Chunk  # noqa: E402
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

FINAL_DBS = ["qdrant", "milvus", "elasticsearch", "weaviate"]  # Chroma/FAISS excluded
N_BOOTSTRAP = 10000
SEED = 42


def _ann_recall_vs_exact(ann_ids: list[str], exact_ids: list[str], k: int) -> float:
    """|ANN top-k ∩ Exact top-k| / k — literal spec formula (denominator is the
    requested k, not len(exact_ids)); NaN only if the exact side is empty (degenerate
    corpus smaller than k, which should not happen at 100K/200K scale)."""
    if not exact_ids:
        return float("nan")
    ann_top = set(ann_ids[:k])
    exact_top = set(exact_ids[:k])
    return len(ann_top & exact_top) / k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "experiments",
                                               "configs", "benchmark_all_dbs.yaml"))
    parser.add_argument("--num-documents", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fixed-query-file", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "experiments",
                                               "results", "fixed_queries_20260803_123156.json"))
    parser.add_argument("--output-tag", type=str, required=True, help='e.g. "100K" or "200K"')
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    top_k = args.top_k

    # --- Same config-loading path as scripts/benchmark_db.py's main(), so config_hash
    # is directly comparable to the FINAL_4DB_S1-S6 runs. chroma_cfg/faiss_cfg are built
    # (matching FINAL_4DB's include_faiss=False composition) purely to keep the
    # reproducibility config_hash apples-to-apples; neither is ever instantiated/queried. ---
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

    logging.info("Computing exact k-NN ground truth (top-%d) via faiss.IndexFlatIP...", top_k)
    t0 = time.perf_counter()
    exact_topk, exact_method = exact_topk_per_query(
        query_embeddings, embeddings, chunk_ids, k=top_k, distance_metric="cosine"
    )
    exact_seconds = time.perf_counter() - t0
    assert exact_method.startswith("faiss.IndexFlat"), (
        f"Exact baseline must use an exact Flat index; got {exact_method!r}."
    )
    logging.info("Exact k-NN done in %.2fs via %s", exact_seconds, exact_method)

    retriever_specs: dict[str, tuple[Any, Any]] = {
        "qdrant": (QdrantRetriever, qdrant_cfg),
        "milvus": (MilvusRetriever, milvus_cfg),
        "elasticsearch": (ElasticSearchRetriever, elasticsearch_cfg),
        "weaviate": (WeaviateRetriever, weaviate_cfg),
    }

    per_db_results: dict[str, Any] = {}
    effective_config: dict[str, Any] = {}
    crash_log: dict[str, str | None] = {}

    for db_name, (cls, cfg) in retriever_specs.items():
        logging.info("Indexing + querying %s (%d docs)...", db_name, args.num_documents)
        crash_log[db_name] = None
        try:
            retriever = cls(config=cfg, embedder=embedder)
            retriever.clear()
            retriever.add_chunks(chunks, embeddings)
            effective_config[db_name] = retriever.describe_index()

            query_level: list[dict[str, Any]] = []
            for q in bench_queries:
                result = retriever.retrieve_with_embedding(query_embeddings[q.id], top_k=top_k, query_id=q.id)
                ann_ids = [rc.chunk.id for rc in result.chunks]
                gt = set(ground_truth.get(q.id, set()))
                qrels_metrics = evaluator.evaluate(result, gt)
                query_level.append({
                    "query_id": q.id,
                    "ann_recall_vs_exact@10": _ann_recall_vs_exact(ann_ids, exact_topk.get(q.id, []), top_k),
                    "recall@10": qrels_metrics.get(recall_key, float("nan")),
                    "mrr@10": qrels_metrics.get(mrr_key, float("nan")),
                    "ndcg@10": qrels_metrics.get(ndcg_key, float("nan")),
                })
        except Exception as exc:  # noqa: BLE001 - deliberately broad: record and continue
            crash_log[db_name] = f"{type(exc).__name__}: {exc}"
            logging.error("DB %s failed: %s", db_name, crash_log[db_name])
            per_db_results[db_name] = {"error": crash_log[db_name]}
            continue

        ann_values = [r["ann_recall_vs_exact@10"] for r in query_level]
        ann_mean, ann_lo, ann_hi = bootstrap_ci(ann_values, n_samples=N_BOOTSTRAP, seed=SEED)
        ann_std = float(np.std([v for v in ann_values if v == v]))

        recall_vals = [r["recall@10"] for r in query_level]
        mrr_vals = [r["mrr@10"] for r in query_level]
        ndcg_vals = [r["ndcg@10"] for r in query_level]

        per_db_results[db_name] = {
            "num_queries": len(query_level),
            "ann_recall_vs_exact@10": {
                "mean": ann_mean, "std": ann_std, "ci_low": ann_lo, "ci_high": ann_hi,
                "n_bootstrap_samples": N_BOOTSTRAP, "seed": SEED,
            },
            "qrels_recall@10_mean": sum(recall_vals) / len(recall_vals) if recall_vals else float("nan"),
            "qrels_mrr@10_mean": sum(mrr_vals) / len(mrr_vals) if mrr_vals else float("nan"),
            "qrels_ndcg@10_mean": sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else float("nan"),
            "query_level_results": query_level,
        }

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
        "experiment": "exact_vs_ann_final_4db",
        "timestamp": ts,
        "scope": {"final_dbs": FINAL_DBS, "excluded": ["chroma", "faiss"],
                  "faiss_role": "exact_baseline_helper"},
        "dataset": "msmarco" if isinstance(dataset_config, MSMARCODatasetConfig) else "squad",
        "splits_used": splits_used,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),
        "top_k": top_k,
        "exact_method": exact_method,
        "exact_knn_seconds": exact_seconds,
        "any_db_crash": any_crash,
        "crash_log": crash_log,
        "note": (
            "ann_recall_vs_exact@10 measures agreement with THIS RUN'S OWN exact "
            "brute-force top-k (faiss.IndexFlatIP) — it is NOT qrels-based gold "
            "Recall@10. The qrels_* fields alongside it come from the same qrels-based "
            "RetrievalEvaluator used by official_scale_*.json and must not be conflated "
            "with ann_recall_vs_exact."
        ),
        "results_by_db": per_db_results,
        "reproducibility": reproducibility,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"analysis_FINAL_4DB_exact_vs_ann_{args.output_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", json_path)

    csv_path = os.path.join(out_dir, f"analysis_FINAL_4DB_exact_vs_ann_{args.output_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "db", "num_documents", "num_queries", "qrels_recall@10", "qrels_mrr@10",
            "qrels_ndcg@10", "ann_recall_vs_exact_mean", "ann_recall_vs_exact_std",
            "ann_recall_vs_exact_ci_low", "ann_recall_vs_exact_ci_high",
        ])
        for db_name in FINAL_DBS:
            r = per_db_results.get(db_name, {})
            if "error" in r:
                writer.writerow([db_name, len(documents), 0, "ERROR", "ERROR", "ERROR", "", "", "", ""])
                continue
            ann = r["ann_recall_vs_exact@10"]
            writer.writerow([
                db_name, len(documents), r["num_queries"],
                r["qrels_recall@10_mean"], r["qrels_mrr@10_mean"], r["qrels_ndcg@10_mean"],
                ann["mean"], ann["std"], ann["ci_low"], ann["ci_high"],
            ])
    logging.info("Wrote %s", csv_path)

    print(f"\n=== Exact-vs-ANN summary ({args.output_tag}, {len(documents)} docs) ===")
    for db_name in FINAL_DBS:
        r = per_db_results.get(db_name, {})
        if "error" in r:
            print(f"{db_name}: FAILED — {r['error']}")
            continue
        ann = r["ann_recall_vs_exact@10"]
        print(
            f"{db_name}: qrels_recall@10={r['qrels_recall@10_mean']:.4f} "
            f"mrr@10={r['qrels_mrr@10_mean']:.4f} ndcg@10={r['qrels_ndcg@10_mean']:.4f} | "
            f"ann_recall_vs_exact@10={ann['mean']:.4f} (std={ann['std']:.4f}, "
            f"CI95=[{ann['ci_low']:.4f}, {ann['ci_high']:.4f}])"
        )

    if any_crash:
        print("\nWARNING: at least one DB crashed — see crash_log in the output JSON.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
