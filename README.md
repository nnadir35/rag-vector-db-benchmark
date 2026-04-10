# RAG Vector DB Benchmark

[![Python CI](https://github.com/YOUR_USERNAME/rag-vector-db-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rag-vector-db-benchmark/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Strict Type Checking](https://img.shields.io/badge/mypy-strict-green)](http://mypy-lang.org/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> 🚀 **[Quick Start: Explore the Interactive Demo Notebook here (`notebooks/example_evaluation.ipynb`)](notebooks/example_evaluation.ipynb)**
> **Let your business ask questions to its own data — and measure exactly how well it works.**

This framework benchmarks end-to-end RAG (Retrieval-Augmented Generation) pipelines: how accurately an AI system retrieves relevant documents and generates correct answers from a company's internal knowledge base.

## Real Benchmark Results (SQuAD v2 · 100 queries)

| Metric | Score | What it means |
|---|---|---|
| MRR | **0.707** | On average, the correct answer is ranked 1st or 2nd |
| Precision@1 | **0.610** | 61% of top results are directly relevant |
| Recall@3 | **0.820** | 82% of relevant documents found in top 3 results |

## What This Enables

- **Drop in any LLM**: Ollama (local/private), OpenAI, Anthropic — swap with one config change
- **Drop in any vector DB**: ChromaDB today, Pinecone or Weaviate tomorrow — no code changes
- **Measure before you ship**: Know your retrieval quality with real numbers before going to production
- **Full privacy option**: Runs entirely locally with Ollama — no data leaves your servers

## Stack

Python · ChromaDB · Ollama/llama3 · sentence-transformers · FastAPI · Gradio · LiteLLM

---

> **Repository Structure**: See [STRUCTURE.md](./STRUCTURE.md) for a detailed breakdown of the directory organization, component responsibilities, and design principles.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run benchmarks and tests from the same directory (scripts add the project root to `sys.path`):

```bash
python scripts/run_experiment.py --config experiments/configs/baseline_ollama.yaml
python -m pytest tests/ -q
python api.py   # FastAPI on http://0.0.0.0:8000  (requires Ollama if using default generator)
```

## Problem Definition

RAG systems combine information retrieval with language generation, creating complex interactions between retrieval components (vector databases, embedding models) and generation components (LLMs, prompt engineering). Current evaluation practices often conflate these concerns, making it difficult to:

- **Isolate performance bottlenecks**: Understand whether poor results stem from retrieval failures or generation limitations
- **Compare vector databases fairly**: Evaluate retrieval systems independently of downstream generation choices
- **Reproduce experiments**: Ensure consistent, deterministic evaluation across different environments
- **Scale evaluations**: Systematically test combinations of retrievers, generators, and configurations

This framework addresses these challenges by enforcing strict separation of concerns and providing a reproducible, configurable evaluation infrastructure.

## What is Benchmarked

### In Scope

**Retrieval Components**
- Vector database query performance (latency, throughput)
- Embedding model effectiveness (retrieval accuracy, semantic similarity)
- Retrieval quality metrics (precision, recall, nDCG, MRR)
- Retrieval latency and cost per query

**Generation Components**
- LLM response quality (faithfulness, relevance, answer quality)
- Generation latency and cost per response
- Prompt engineering effectiveness

**End-to-End Pipeline**
- Overall system accuracy and quality
- Total latency (retrieval + generation)
- Total cost per query
- Failure modes and error analysis

### Out of Scope

- **Model training or fine-tuning**: This framework evaluates existing models, not training new ones
- **Data preprocessing pipelines**: Assumes pre-processed, chunked documents are available
- **Production deployment concerns**: Focuses on evaluation, not serving infrastructure
- **Real-time monitoring**: Designed for batch evaluation, not continuous monitoring
- **User experience metrics**: Focuses on technical metrics, not subjective user satisfaction

## RAG Pipeline Overview

The framework models RAG systems as a composition of three distinct, interchangeable components:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retriever  │────▶│   Generator  │────▶│  Response   │
│  (Vector DB │     │     (LLM)    │     │             │
│  + Embed)   │     │              │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│  Retrieval  │     │  Generation  │
│  Evaluator  │     │  Evaluator   │
└─────────────┘     └──────────────┘
```

### Component Responsibilities

**Retriever**
- Accepts a query and returns ranked document chunks
- Encapsulates vector database, embedding model, and retrieval logic
- Evaluated independently on retrieval-specific metrics

**Generator**
- Accepts a query and retrieved context, produces a response
- Encapsulates LLM, prompt template, and generation parameters
- Evaluated independently on generation-specific metrics

**Evaluator**
- Measures component performance using standardized metrics
- Operates independently of pipeline execution
- Produces reproducible, comparable results

### Design Principles

- **Interface-first**: Components communicate through well-defined interfaces, enabling plug-and-play experimentation
- **Separation of concerns**: Retrieval logic never knows about generation, and vice versa
- **No benchmarking leakage**: Evaluation logic is isolated from pipeline execution
- **Full configurability**: No hardcoded models, prompts, or database choices

## Evaluation Philosophy

### Reproducibility First

All experiments must be reproducible. This means:
- Deterministic evaluation where possible (fixed seeds, consistent data splits)
- Versioned configurations for all components
- Immutable experiment artifacts (inputs, outputs, metrics)
- Clear documentation of non-deterministic sources

### Isolated Component Evaluation

**Retrieval evaluation** measures:
- How well the retriever finds relevant documents
- Query performance characteristics (latency, cost)
- Independent of downstream generation quality

**Generation evaluation** measures:
- How well the generator produces accurate, relevant responses
- Response quality given fixed retrieval results
- Independent of retrieval system choices

**Combined evaluation** measures:
- End-to-end system performance
- Interaction effects between components
- Total system cost and latency

### Metrics Hierarchy

1. **Primary metrics**: Core quality measures (accuracy, relevance, faithfulness)
2. **Performance metrics**: Latency and throughput
3. **Cost metrics**: Per-query and per-experiment costs
4. **Failure analysis**: Error rates, timeout frequencies, edge case handling

### Evaluation Workflow

1. **Baseline establishment**: Evaluate each component independently
2. **Component optimization**: Iterate on individual components
3. **Integration testing**: Evaluate component combinations
4. **Comparative analysis**: Compare configurations systematically

## Experiment-Driven Workflow

The framework supports a systematic, experiment-driven approach to RAG system development:

### Experiment Definition

An experiment consists of:
- **Configuration**: Component choices (retriever, generator, evaluators)
- **Dataset**: Query set and ground truth
- **Metrics**: Evaluation criteria and success thresholds
- **Constraints**: Resource limits, timeout values

### Experiment Execution

1. **Configuration loading**: Load experiment configuration from files
2. **Component instantiation**: Create retriever, generator, and evaluator instances
3. **Pipeline execution**: Run queries through the RAG pipeline
4. **Metric collection**: Gather performance, quality, and cost metrics
5. **Result aggregation**: Combine metrics across queries and components

### Experiment Analysis

- **Component-level analysis**: Identify which components drive performance
- **Ablation studies**: Understand contribution of each component
- **Cost-performance tradeoffs**: Analyze efficiency vs. quality curves
- **Failure mode analysis**: Identify systematic weaknesses

### Experiment Reproducibility

- All experiments are defined declaratively (no code changes needed)
- Experiment results are versioned and immutable
- Comparison across experiments is standardized
- Historical experiment data supports longitudinal analysis

## How to Extend the Framework

### Adding a New Retriever

1. **Implement the retriever interface**: Define how your retriever accepts queries and returns results
2. **Encapsulate configuration**: Make all retriever-specific settings configurable
3. **Register the retriever**: Add to the retriever registry with a unique identifier
4. **No generator knowledge**: Retriever implementation must not depend on generator details

### Adding a New Generator

1. **Implement the generator interface**: Define how your generator accepts queries and context
2. **Encapsulate configuration**: Make all generator-specific settings (model, prompt, parameters) configurable
3. **Register the generator**: Add to the generator registry with a unique identifier
4. **No retriever knowledge**: Generator implementation must not depend on retriever details

### Adding a New Evaluator

1. **Implement the evaluator interface**: Define metric computation logic
2. **Specify metric scope**: Clearly indicate whether this evaluates retrieval, generation, or both
3. **Ensure determinism**: Make evaluation deterministic where possible, document randomness sources
4. **Register the evaluator**: Add to the evaluator registry with metric metadata

### Adding a New Metric

1. **Define the metric interface**: Specify inputs, outputs, and computation method
2. **Implement metric computation**: Ensure reproducibility and efficiency
3. **Document metric interpretation**: Explain what the metric measures and how to interpret values
4. **Add to metric registry**: Enable metric discovery and combination

### Adding a New Dataset

1. **Define dataset format**: Specify query, document, and ground truth structure
2. **Implement dataset loader**: Create loader that produces standardized format
3. **Register the dataset**: Add to dataset registry with metadata
4. **Document ground truth**: Explain how ground truth is defined and validated

### Architectural Constraints

When extending the framework, maintain:

- **Interface boundaries**: Components communicate only through defined interfaces
- **Configuration-driven design**: All choices are externalized to configuration
- **No global state**: Components are stateless or manage their own state
- **Composition over inheritance**: Prefer composing components over deep inheritance hierarchies
- **Testability**: All components must be mockable and testable in isolation

## Contributing

This framework is designed for long-term research and engineering. Contributions should:

- Maintain strict separation of concerns
- Add comprehensive type hints and docstrings
- Include tests with mocks for external dependencies
- Follow the interface-first design philosophy
- Document configuration options and their effects

## License

[Specify license]
