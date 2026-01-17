# Repository Structure

This document defines the repository structure for the RAG benchmark framework. The structure is designed to support many experiments without refactoring, maintain clear separation of concerns, and enable extensibility.

## Directory Layout

```
rag-vector-db-benchmark/
├── src/
│   ├── core/
│   ├── retrievers/
│   ├── generators/
│   ├── evaluators/
│   ├── datasets/
│   ├── pipeline/
│   └── utils/
├── experiments/
│   ├── configs/
│   └── results/
├── tests/
├── docs/
└── scripts/
```

## Directory Specifications

### `src/core/`

**Responsibility**: Define abstract interfaces, base types, and shared data structures that establish contracts between components.

**What belongs here**:
- Abstract base classes/interfaces for `Retriever`, `Generator`, `Evaluator`
- Core data types (Query, Document, RetrievalResult, GenerationResult, etc.)
- Metric definitions and interfaces
- Registry interfaces for component discovery
- Shared type definitions and enums
- Configuration schema definitions

**What must NOT belong here**:
- Concrete implementations of retrievers, generators, or evaluators
- Experiment-specific logic
- Benchmarking orchestration code
- Business logic or domain-specific code
- External API client code

---

### `src/retrievers/`

**Responsibility**: Implement retrieval components that query vector databases and return ranked document chunks.

**What belongs here**:
- Concrete retriever implementations (e.g., `PineconeRetriever`, `WeaviateRetriever`)
- Retriever-specific configuration classes
- Embedding model wrappers used by retrievers
- Vector database client abstractions
- Retriever registry and factory functions
- Retriever-specific utilities (e.g., query preprocessing)

**What must NOT belong here**:
- Generator or evaluator code
- Experiment orchestration
- Dataset loading logic
- Metric computation (belongs in evaluators)
- Benchmarking infrastructure
- End-to-end pipeline code

---

### `src/generators/`

**Responsibility**: Implement generation components that produce responses from queries and retrieved context.

**What belongs here**:
- Concrete generator implementations (e.g., `OpenAIGenerator`, `AnthropicGenerator`)
- Generator-specific configuration classes
- Prompt template management
- LLM client abstractions
- Generator registry and factory functions
- Generation-specific utilities (e.g., response post-processing)

**What must NOT belong here**:
- Retriever or evaluator code
- Experiment orchestration
- Dataset loading logic
- Metric computation (belongs in evaluators)
- Benchmarking infrastructure
- Retrieval logic or vector database code

---

### `src/evaluators/`

**Responsibility**: Implement evaluation logic that computes metrics for retrievers, generators, or the full pipeline.

**What belongs here**:
- Concrete evaluator implementations (e.g., `RetrievalEvaluator`, `GenerationEvaluator`)
- Metric computation functions (precision, recall, faithfulness, etc.)
- Evaluator-specific configuration classes
- Ground truth comparison logic
- Evaluator registry and factory functions
- Metric aggregation and statistical utilities

**What must NOT belong here**:
- Retriever or generator implementations
- Experiment orchestration (loading configs, running pipelines)
- Dataset definitions (belongs in datasets/)
- Component-specific business logic
- Benchmarking infrastructure that coordinates experiments

---

### `src/datasets/`

**Responsibility**: Define dataset loaders and data structures for queries, documents, and ground truth.

**What belongs here**:
- Dataset loader implementations
- Dataset-specific data structures
- Dataset registry and factory functions
- Data validation and schema checking
- Dataset metadata and documentation
- Data format converters

**What must NOT belong here**:
- Actual dataset files (use external storage or data/ directory)
- Component implementations (retrievers, generators, evaluators)
- Experiment configurations
- Metric computation logic
- Benchmarking orchestration

---

### `src/pipeline/`

**Responsibility**: Orchestrate RAG pipeline execution by composing retrievers, generators, and evaluators.

**What belongs here**:
- Pipeline execution engine
- Component composition logic
- Pipeline configuration loading
- Execution orchestration (retrieve → generate → evaluate)
- Pipeline result aggregation
- Error handling and retry logic

**What must NOT belong here**:
- Component implementations (belongs in respective component directories)
- Experiment-specific logic (belongs in experiments/)
- Metric computation (belongs in evaluators/)
- Dataset loading (belongs in datasets/)
- Benchmarking infrastructure (experiment management, comparison)

---

### `src/utils/`

**Responsibility**: Provide shared utilities used across multiple components.

**What belongs here**:
- Logging configuration and utilities
- Configuration parsing and validation
- Common data transformations
- File I/O utilities
- Time and cost tracking utilities
- Reproducibility helpers (seed management, etc.)

**What must NOT belong here**:
- Component-specific logic (belongs in component directories)
- Experiment orchestration
- Business logic or domain knowledge
- External API clients (belongs with component implementations)

---

### `experiments/configs/`

**Responsibility**: Store experiment configuration files that define component choices, datasets, and evaluation criteria.

**What belongs here**:
- Experiment configuration files (YAML, JSON, or TOML)
- Configuration templates
- Configuration validation schemas
- Experiment metadata files

**What must NOT belong here**:
- Code (belongs in src/)
- Experiment results (belongs in experiments/results/)
- Dataset files
- Component implementations
- Scripts (belongs in scripts/)

---

### `experiments/results/`

**Responsibility**: Store immutable experiment results, metrics, and artifacts.

**What belongs here**:
- Experiment result files (metrics, logs, outputs)
- Result metadata and provenance information
- Comparison reports
- Visualization data

**What must NOT belong here**:
- Experiment configurations (belongs in configs/)
- Source code
- Temporary or intermediate files (use .gitignore)
- Dataset files
- Component implementations

---

### `tests/`

**Responsibility**: Contain all test code organized to mirror the source structure.

**What belongs here**:
- Unit tests for each component
- Integration tests for pipelines
- Test fixtures and mocks
- Test utilities and helpers
- Test configuration files

**What must NOT belong here**:
- Production code
- Experiment configurations
- Actual dataset files (use fixtures)
- Benchmarking scripts
- Documentation (belongs in docs/)

---

### `docs/`

**Responsibility**: Store documentation, design decisions, and research notes.

**What belongs here**:
- Architecture documentation
- Component usage guides
- Experiment design notes
- Research findings and analysis
- API documentation

**What must NOT belong here**:
- Code
- Configuration files
- Experiment results
- Test files

---

### `scripts/`

**Responsibility**: Provide executable scripts for common tasks like running experiments, analyzing results, and setup.

**What belongs here**:
- Experiment execution scripts
- Result analysis scripts
- Setup and installation scripts
- Data preparation scripts
- Utility scripts (formatting, linting, etc.)

**What must NOT belong here**:
- Core framework code (belongs in src/)
- Test code (belongs in tests/)
- Configuration files (belongs in experiments/configs/)
- Documentation (belongs in docs/)

---

## Design Principles

### Separation of Concerns
- Each directory has a single, well-defined responsibility
- Components cannot import from other component directories (only from `core/` and `utils/`)
- Pipeline code orchestrates but doesn't implement components

### Extensibility
- New retrievers, generators, or evaluators are added to their respective directories
- No changes to existing code required when adding new components
- Registry pattern enables discovery without hardcoding

### Experiment Isolation
- Experiments are defined declaratively in `experiments/configs/`
- Results are stored immutably in `experiments/results/`
- No experiment-specific code in the framework

### Reproducibility
- All configurations are versioned
- Results include full provenance
- No global state or environment-dependent behavior

## File Naming Conventions

- **Interfaces**: `interface_name.py` (e.g., `retriever.py`, `generator.py`)
- **Implementations**: `component_name.py` (e.g., `pinecone_retriever.py`, `openai_generator.py`)
- **Configs**: `experiment_name.yaml` (e.g., `baseline_pinecone_gpt4.yaml`)
- **Tests**: `test_component_name.py` (e.g., `test_pinecone_retriever.py`)

## Import Rules

1. **Core** can import only standard library and external dependencies
2. **Components** can import from `core/` and `utils/` only
3. **Pipeline** can import from all component directories and `core/`, `utils/`
4. **Experiments** should not import framework code directly (use CLI/API)
5. **Tests** can import from anywhere in `src/`
