# RAG Vector DB Benchmark

> **Not:** Bu dosya (CLAUDE.md) hızlı komut referansı, stil kuralları ve sistem talimatlarını içerir. Detaylı ilerleme günlüğü ve modül durumu için `AGENTS.md` dosyasına bakın.

## Commands
- **Lint:** `ruff check src/ scripts/`
- **Type check:** `mypy .`
- **Tests:** `pytest tests/ -q`
- **Single retriever test:** `pytest tests/retrievers/test_faiss_retriever.py -q`
- **Benchmark (quick):** `python scripts/benchmark_db.py --num-documents 50 --num-queries 10`
- **Benchmark (full):** `python scripts/benchmark_db.py --config experiments/configs/benchmark_all_dbs.yaml`
- **API:** `uvicorn api:app --reload`
- **GUI:** `python app_gui.py`

## Architecture
- **Abstract base:** `src/core/retrieval.py` → `Retriever(ABC)` — all retrievers must implement `add_chunks`, `retrieve`, `retrieve_with_embedding`, `clear`
- **Core types:** `src/core/types.py` → `Chunk`, `Embedding`, `Query`, `RetrievalResult`, `RetrievedChunk`
- **Retriever configs:** `src/retrievers/config.py` → frozen dataclasses, one per DB
- **Retrievers:** `src/retrievers/<name>_retriever.py` — lazy import pattern via `TYPE_CHECKING`
- **Config loader:** `src/utils/config_loader.py` → `build_component_configs()` parses YAML → dataclasses
- **Benchmark script:** `scripts/benchmark_db.py` → `run_benchmark()` runs all DBs, `main()` handles CLI + YAML
- **Results:** `experiments/results/` — JSON files, do NOT read these into context

## Adding a New Retriever (ElasticSearch pattern)
1. Add `XRetrieverConfig` frozen dataclass to `src/retrievers/config.py`
2. Create `src/retrievers/x_retriever.py` — follow `faiss_retriever.py` pattern exactly
3. Add `elasticsearch_cfg: ElasticSearchRetrieverConfig` param to `run_benchmark()` in `scripts/benchmark_db.py`
4. Add ES block (index, retrieve, stats) mirroring the Milvus block
5. Add to `winner_faster_retrieval` and `winner_higher_recall` dicts
6. Add section to `experiments/configs/benchmark_all_dbs.yaml`
7. Update `build_component_configs()` in `src/utils/config_loader.py`
8. Add `tests/retrievers/test_elasticsearch_retriever.py` mirroring `test_faiss_retriever.py`

## Code Style — IMPORTANT
- `from __future__ import annotations` on every file
- Lazy DB client imports: `if TYPE_CHECKING: import lib` pattern — never import at module level
- All configs: `@dataclass(frozen=True)` with `__post_init__` validation
- `object.__setattr__` for env var overrides in frozen dataclasses
- mypy strict — no bare `Any`, use `Any | None` with explicit guards
- Line length: 100 (ruff enforced)
- Batch size 100 for bulk index operations

## Environment Variables
- `CHROMA_HOST` / `CHROMA_PORT` — remote Chroma
- `QDRANT_URL` / `QDRANT_HOST` — remote Qdrant
- `ELASTICSEARCH_HOST` — remote ES (falls back to `http://localhost:9200`)
- `PINECONE_API_KEY` — required for Pinecone

## Test Patterns
- All retriever tests use `in_memory=True` — no Docker required
- Fixtures in `tests/conftest.py` — reuse `sample_chunks`, `sample_embeddings`, `sample_query`
- New retriever test file must cover: `add_chunks`, `retrieve`, `retrieve_with_embedding`, `clear`, config validation

## Do NOT
- Edit files under `experiments/results/` — benchmark output only
- Import DB clients at module level — always lazy
- Add task state or notes to this file
- Read entire `scripts/benchmark_db.py` into context — it's 25K bytes; use subagents to investigate specific sections

## When compacting, preserve
- Which retriever files were modified
- Last benchmark command run and its output summary
- Any failing mypy/ruff errors in progress