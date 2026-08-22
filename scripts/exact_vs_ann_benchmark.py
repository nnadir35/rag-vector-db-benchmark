#!/usr/bin/env python3
"""Exact-vs-ANN recall experiment (item 6) — kept fully separate from the official
qrels-based Recall@K / official_scale_* experiments.

For the same fixed query set and the same chunk embeddings used by the systematic
benchmark, computes an exact brute-force k-NN ground truth (scripts/exact_knn.py — never
approximate, memory-safe at 200K+ scale), then compares each DB's ANN top-k against it:

    ann_recall_vs_exact@k = |ann_topk ∩ exact_topk| / k

This measures how close each DB's approximate index gets to *its own* exact nearest
neighbors — a different question from qrels-based Recall@K (which measures relevance to
human-labeled gold answers). Never touches official_scale_*.json or the gold recall@k
pipeline.

FAISS is used here only as the exact-kNN baseline helper (via scripts/exact_knn.py's raw
`faiss.IndexFlatIP`) — it is never counted as a 6th "official benchmark DB" unless the run
was invoked with --include-faiss, mirroring benchmark_db.py's own flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.benchmark_db import (  # noqa: E402
    _select_queries_for_documents,
    load_dataset_corpus,
)
from scripts.exact_knn import exact_topk_per_query  # noqa: E402
from src.chunkers.config import FixedSizeChunkerConfig  # noqa: E402
from src.chunkers.fixed_size_chunker import FixedSizeChunker  # noqa: E402
from src.core.types import Chunk, RetrievedChunk  # noqa: E402
from src.datasets import MSMARCODatasetConfig, SQuADDatasetConfig  # noqa: E402
from src.embedders.config import SentenceTransformersEmbedderConfig  # noqa: E402
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: E402
from src.retrievers.chroma_retriever import ChromaRetriever  # noqa: E402
from src.retrievers.config import (  # noqa: E402
    ChromaRetrieverConfig,
    ElasticSearchRetrieverConfig,
    FAISSRetrieverConfig,
    MilvusRetrieverConfig,
    QdrantRetrieverConfig,
    WeaviateRetrieverConfig,
)
from src.retrievers.elasticsearch_retriever import ElasticSearchRetriever  # noqa: E402
from src.retrievers.faiss_retriever import FAISSRetriever  # noqa: E402
from src.retrievers.milvus_retriever import MilvusRetriever  # noqa: E402
from src.retrievers.qdrant_retriever import QdrantRetriever  # noqa: E402
from src.retrievers.weaviate_retriever import WeaviateRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OFFICIAL_BENCHMARK_DBS = ["chroma", "qdrant", "milvus", "elasticsearch", "weaviate"]


def _chunk_ids_of(result_chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [rc.chunk.id for rc in result_chunks]


def _ann_recall_vs_exact(ann_ids: list[str], exact_ids: list[str], k: int) -> float:
    if not exact_ids:
        return float("nan")
    ann_top = set(ann_ids[:k])
    exact_top = set(exact_ids[:k])
    return len(ann_top & exact_top) / len(exact_top)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-documents", type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--squad-version", type=str, default="squad_v2")
    parser.add_argument("--collection-path", type=str, default=None)
    parser.add_argument("--queries-path", type=str, default=None)
    parser.add_argument("--qrels-path", type=str, default=None)
    parser.add_argument("--include-faiss", action="store_true",
                         help="Also benchmark FAISS as a genuine 6th official-style DB "
                              "(in addition to using it internally as the exact baseline).")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

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
    logging.info("Embedding %d chunks...", len(chunks))
    embeddings = embedder.embed_chunks(chunks)
    logging.info("Embedding %d queries...", len(bench_queries))
    query_embeddings = {q.id: embedder.embed_query(q) for q in bench_queries}

    logging.info("Computing exact k-NN ground truth (top-%d)...", top_k)
    t0 = time.perf_counter()
    exact_topk, exact_method = exact_topk_per_query(query_embeddings, embeddings, chunk_ids, k=top_k)
    exact_seconds = time.perf_counter() - t0
    logging.info("Exact k-NN done in %.2fs via %s", exact_seconds, exact_method)

    retriever_specs = {
        "chroma": (ChromaRetriever, ChromaRetrieverConfig(collection_name="exact_vs_ann_chroma")),
        "qdrant": (QdrantRetriever, QdrantRetrieverConfig(collection_name="exact_vs_ann_qdrant", in_memory=True)),
        "milvus": (MilvusRetriever, MilvusRetrieverConfig(collection_name="exact_vs_ann_milvus", in_memory=True)),
        "elasticsearch": (ElasticSearchRetriever, ElasticSearchRetrieverConfig(index_name="exact_vs_ann_es")),
        "weaviate": (WeaviateRetriever, WeaviateRetrieverConfig(collection_name="exact_vs_ann_weaviate", in_memory=True)),
    }
    official_benchmark_dbs = list(OFFICIAL_BENCHMARK_DBS)
    if args.include_faiss:
        retriever_specs["faiss"] = (FAISSRetriever, FAISSRetrieverConfig(collection_name="exact_vs_ann_faiss"))
        official_benchmark_dbs.append("faiss")

    ann_recall_by_db: dict[str, float] = {}
    for db_name, (cls, cfg) in retriever_specs.items():
        logging.info("Indexing + querying %s...", db_name)
        retriever = cls(config=cfg, embedder=embedder)
        retriever.clear()
        retriever.add_chunks(chunks, embeddings)

        per_query_scores = []
        for q in bench_queries:
            result = retriever.retrieve_with_embedding(query_embeddings[q.id], top_k=top_k, query_id=q.id)
            ann_ids = _chunk_ids_of(result.chunks)
            per_query_scores.append(
                _ann_recall_vs_exact(ann_ids, exact_topk.get(q.id, []), top_k)
            )
        valid_scores = [s for s in per_query_scores if s == s]  # drop NaN
        ann_recall_by_db[db_name] = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.output_tag}" if args.output_tag else ""
    report = {
        "experiment": "exact_vs_ann",
        "timestamp": ts,
        "dataset": "msmarco" if isinstance(dataset_config, MSMARCODatasetConfig) else "squad",
        "splits_used": splits_used,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "num_queries_evaluated": len(bench_queries),
        "top_k": top_k,
        "exact_method": exact_method,
        "exact_knn_seconds": exact_seconds,
        "faiss_role": "official_benchmark_db" if args.include_faiss else "exact_baseline_helper",
        "official_benchmark_dbs": official_benchmark_dbs,
        f"ann_recall_vs_exact@{top_k}": ann_recall_by_db,
        "note": (
            "ann_recall_vs_exact measures agreement with this run's own exact brute-force "
            "top-k, NOT qrels-based gold Recall@K. Kept separate from official_scale_*.json."
        ),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exact_vs_ann_{len(documents)}docs{tag_part}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", out_path)
    for db_name, score in ann_recall_by_db.items():
        print(f"{db_name}: ann_recall_vs_exact@{top_k} = {score:.4f}")


if __name__ == "__main__":
    main()
