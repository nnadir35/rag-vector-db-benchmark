#!/usr/bin/env python3
"""ANN parameter sweep infrastructure (item 7) — produces a plain recall-vs-latency table
for a Pareto plot. Deliberately computes NO combined/weighted single score.

For one DB and one ANN parameter (hnsw_ef_search for Chroma/Qdrant/Milvus/Weaviate,
num_candidates for ElasticSearch), sweeps a list of values; for each, (re-)indexes once and
records Recall@K (qrels-based), optionally ANN Recall vs Exact (via scripts/exact_knn.py,
memory-safe at 200K+ scale), p50/p95/p99 search-only latency, and wall-clock search-only
QPS. Writes both JSON and CSV.

Usage:
    python scripts/ann_param_sweep.py --db qdrant --param hnsw_ef_search \
        --values 32,64,128,256 --num-documents 5000 --num-queries 200
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.benchmark_db import (  # noqa: E402
    _evaluate_with_repeats,
    _select_queries_for_documents,
    load_dataset_corpus,
)
from scripts.exact_knn import exact_topk_per_query  # noqa: E402
from src.chunkers.config import FixedSizeChunkerConfig  # noqa: E402
from src.chunkers.fixed_size_chunker import FixedSizeChunker  # noqa: E402
from src.core.types import Chunk, Embedding  # noqa: E402
from src.datasets import MSMARCODatasetConfig, SQuADDatasetConfig  # noqa: E402
from src.embedders.config import SentenceTransformersEmbedderConfig  # noqa: E402
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: E402
from src.evaluators.config import RetrievalEvaluatorConfig  # noqa: E402
from src.evaluators.retrieval_evaluator import RetrievalEvaluator  # noqa: E402
from src.retrievers.chroma_retriever import ChromaRetriever  # noqa: E402
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_RETRIEVER_SPECS = {
    "chroma": (ChromaRetriever, ChromaRetrieverConfig, {"collection_name": "ann_sweep_chroma"}),
    "qdrant": (QdrantRetriever, QdrantRetrieverConfig, {"collection_name": "ann_sweep_qdrant", "in_memory": True}),
    "milvus": (MilvusRetriever, MilvusRetrieverConfig, {"collection_name": "ann_sweep_milvus", "in_memory": True}),
    "elasticsearch": (ElasticSearchRetriever, ElasticSearchRetrieverConfig, {"index_name": "ann_sweep_es"}),
    "weaviate": (WeaviateRetriever, WeaviateRetrieverConfig, {"collection_name": "ann_sweep_weaviate", "in_memory": True}),
}

_VALID_PARAMS = {"hnsw_ef_search", "num_candidates"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, choices=sorted(_RETRIEVER_SPECS.keys()))
    parser.add_argument("--param", required=True, choices=sorted(_VALID_PARAMS))
    parser.add_argument("--values", required=True, help="Comma-separated values, e.g. 32,64,128,256")
    parser.add_argument("--num-documents", type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--squad-version", type=str, default="squad_v2")
    parser.add_argument("--collection-path", type=str, default=None)
    parser.add_argument("--queries-path", type=str, default=None)
    parser.add_argument("--qrels-path", type=str, default=None)
    parser.add_argument("--with-exact", action="store_true",
                         help="Also compute ann_recall_vs_exact@k for each sweep point.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-tag", type=str, default=None)
    args = parser.parse_args()

    cls, cfg_cls, base_kwargs = _RETRIEVER_SPECS[args.db]
    field_names = {f.name for f in dataclasses.fields(cfg_cls)}
    if args.param not in field_names:
        raise ValueError(f"--param {args.param!r} is not a field of {cfg_cls.__name__}.")

    values = [int(v.strip()) for v in args.values.split(",") if v.strip()]
    top_k = args.top_k

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

    documents, all_queries, ground_truth, splits_used = load_dataset_corpus(
        dataset_config, args.num_documents
    )
    doc_ids = {d.id for d in documents}
    bench_queries = _select_queries_for_documents(
        all_queries, ground_truth, doc_ids, args.num_queries, query_seed=args.query_seed
    )

    chunker = FixedSizeChunker(FixedSizeChunkerConfig())
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    chunk_ids = [c.id for c in chunks]

    embedder = SentenceTransformersEmbedder(SentenceTransformersEmbedderConfig())
    embeddings = embedder.embed_chunks(chunks)
    query_embeddings = {q.id: embedder.embed_query(q) for q in bench_queries}

    exact_topk: dict[str, list[str]] = {}
    exact_method = None
    if args.with_exact:
        exact_topk, exact_method = exact_topk_per_query(query_embeddings, embeddings, chunk_ids, k=top_k)

    evaluator = RetrievalEvaluator(RetrievalEvaluatorConfig(k_values=[top_k]))
    recall_key = f"recall@{top_k}"

    rows: list[dict[str, Any]] = []
    for value in values:
        logging.info("=== %s: %s=%s ===", args.db, args.param, value)
        sweep_kwargs: dict[str, Any] = {**base_kwargs, args.param: value}  # type: ignore[dict-item]
        cfg = cfg_cls(**sweep_kwargs)  # type: ignore[arg-type]  # generic over DB config classes by design
        retriever = cls(config=cfg, embedder=embedder)  # type: ignore[arg-type]
        retriever.clear()

        t0 = time.perf_counter()
        retriever.add_chunks(chunks, embeddings)
        index_seconds = time.perf_counter() - t0

        def _retrieve(emb: Embedding, k: int, qid: str, r: Any = retriever) -> Any:
            return r.retrieve_with_embedding(emb, top_k=k, query_id=qid)

        eval_res = _evaluate_with_repeats(
            name=f"{args.db}[{args.param}={value}]",
            retrieve_fn=_retrieve,
            bench_queries=bench_queries,
            query_embeddings=query_embeddings,
            top_k=top_k,
            ground_truth=ground_truth,
            evaluator=evaluator,
            num_repeats=args.num_repeats,
            show_progress=not args.no_progress,
        )

        ann_recall_vs_exact = float("nan")
        if args.with_exact:
            scores = []
            for q in bench_queries:
                result = retriever.retrieve_with_embedding(query_embeddings[q.id], top_k=top_k, query_id=q.id)
                ann_ids = [rc.chunk.id for rc in result.chunks][:top_k]
                exact_ids = set(exact_topk.get(q.id, [])[:top_k])
                if exact_ids:
                    scores.append(len(set(ann_ids) & exact_ids) / len(exact_ids))
            if scores:
                ann_recall_vs_exact = sum(scores) / len(scores)

        row = {
            "param_value": value,
            "index_seconds": index_seconds,
            f"recall@{top_k}": eval_res["metrics_mean"].get(recall_key, float("nan")),
            f"ann_recall_vs_exact@{top_k}": ann_recall_vs_exact,
            "p50_ms": eval_res["p50_ms_mean"],
            "p95_ms": eval_res["p95_ms_mean"],
            "p99_ms": eval_res["p99_ms_mean"],
            "wall_clock_search_only_qps": eval_res["wall_clock_search_only_qps_mean"],
        }
        rows.append(row)
        print(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.output_tag}" if args.output_tag else ""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"ann_sweep_{args.db}_{args.param}{tag_part}_{ts}"

    payload = {
        "experiment": "ann_param_sweep",
        "db": args.db,
        "param": args.param,
        "values": values,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),
        "top_k": top_k,
        "exact_method": exact_method,
        "rows": rows,
        "note": "No combined/weighted score is computed; use `rows` for a recall-latency Pareto plot.",
    }
    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", json_path)

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logging.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
