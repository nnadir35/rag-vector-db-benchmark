#!/usr/bin/env python3
"""Run a RAG benchmark experiment from a configuration file.

This script acts as the main CLI entry point. It orchestrates reading YAML,
initializing pipeline components, running inference against standard
evaluation datasets, saving results logically, and displaying them.
"""

import argparse
import asyncio
import itertools
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence
import sys

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import build_component_configs, load_yaml
from src.datasets.squad_loader import SQuADLoader
from src.generators.universal_generator import UniversalGenerator
from src.evaluators.retrieval_evaluator import RetrievalEvaluator
from src.evaluators.generation_evaluator import GenerationEvaluator
from src.pipeline.rag_pipeline import RAGPipeline
from src.core.retrieval import Retriever
from src.core.types import Query, RetrievalResult, RetrievedChunk, Chunk, ChunkMetadata, Embedding
from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.retrievers.chroma_retriever import ChromaRetriever
from src.retrievers.config import ChromaRetrieverConfig

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")





def tabulate_results(all_metrics: List[Optional[Dict[str, float]]], total_time: float) -> None:
    """Print terminal-friendly metrics summary table."""
    valid_metrics = [m for m in all_metrics if m is not None]
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(f"Total Queries Evaluated : {len(all_metrics)}")
    print(f"Successful Evaluations  : {len(valid_metrics)}")
    print(f"Total Pipeline Latency  : {total_time:.4f} seconds")
    
    if not valid_metrics:
        print("No evaluation metrics were generated.")
        return
        
    # Aggregate metrics
    keys = valid_metrics[0].keys()
    agg = {key: sum(m[key] for m in valid_metrics) / len(valid_metrics) for key in keys}
    
    print("-" * 50)
    print(f"{'Metric':<20} | {'Average Score':<15}")
    print("-" * 50)
    
    for key in sorted(keys):
        print(f"{key:<20} | {agg[key]:.4f}")
    print("=" * 50 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    # 1. Load configuration
    try:
        raw_config = load_yaml(args.config)
        exp_config = build_component_configs(raw_config)
        logging.info(f"Loaded configuration for experiment: {exp_config.name}")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return

    # 2. Setup Data Loader
    logging.info("Loading dataset...")
    loader = SQuADLoader(exp_config.dataset)
    try:
        queries, ground_truth = loader.load()
        raw_documents = loader.load_documents()
        logging.info(f"Loaded {len(queries)} queries and {len(raw_documents)} documents.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
        
    # Convert documents to Chunks using FixedSizeChunker
    logging.info("Chunking documents...")
    chunker = FixedSizeChunker(exp_config.chunker)
    chunks = []
    for doc in raw_documents:
        chunks.extend(chunker.chunk(doc))
    logging.info(f"Generated {len(chunks)} chunks from {len(raw_documents)} documents.")

    # 3. Component Instantiation
    logging.info("Initializing Generator and Evaluator...")
    generator = UniversalGenerator(exp_config.generator)
    evaluator = RetrievalEvaluator(exp_config.evaluator)
    generation_evaluator = GenerationEvaluator(judge_generator=generator)
    
    logging.info("Initializing Embedder and computing embeddings...")
    embedder = SentenceTransformersEmbedder(exp_config.embedder)
    embeddings = embedder.embed_chunks(chunks)
    
    logging.info("Initializing ChromaRetriever and indexing chunks...")
    retriever_config = ChromaRetrieverConfig(**raw_config.get("retriever", {}))
    retriever = ChromaRetriever(config=retriever_config, embedder=embedder)
    
    # Clear the collection to ensure a fresh index for each run
    retriever.clear()
    retriever.add_chunks(chunks, embeddings)
    
    # Pipeline Setup
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        config=exp_config.pipeline,
        retrieval_evaluator=evaluator,
        generation_evaluator=generation_evaluator
    )
    
    # 4. Run Experiment
    logging.info("Executing pipeline on queries (this may take a while)...")
    
    start_time = time.time()
    query_texts = [q.text for q in queries]
    # Keep ground truth matching based on original query ids
    gt_lists = [list(ground_truth.get(q.id, set())) for q in queries]
    
    # Handle batch processing safely by chunking locally so we don't bombard APIs simultaneously
    batch_size = 5
    all_results = []
    
    for i in range(0, len(query_texts), batch_size):
        batch_q = query_texts[i:i+batch_size]
        batch_gt = gt_lists[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}...")
        results = await pipeline.run_batch(batch_q, batch_gt)
        all_results.extend(results)
        
    execution_time = time.time() - start_time
    
    # 5. Summarize and Print Table
    extracted_metrics = [r.retrieval_metrics for r in all_results]
    tabulate_results(extracted_metrics, execution_time)
    
    # 6. Save JSON Data
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiments/results", exist_ok=True)
    out_file = f"experiments/results/{ts}_{exp_config.name}.json"
    
    output_data = {
        "experiment_name": exp_config.name,
        "timestamp": ts,
        "raw_config": raw_config,
        "metrics_summary": {},
        "results": [
            {
                "query": r.query.text,
                "response": r.rag_response.response,
                "latency": r.total_latency_seconds,
                "metrics": r.retrieval_metrics
            }
            for r in all_results
        ]
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Full details saved to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG Pipeline Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
