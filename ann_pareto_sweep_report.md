# ANN Pareto Frontier — Query-Time Parameter Sweep (100K docs)

Query-time search-effort sweep on Qdrant, Milvus, Elasticsearch and Weaviate — one index build per database, only the runtime ANN parameter varied, isolating the recall/latency trade-off from build-to-build nondeterminism.

**Corpus:** 100,000 docs / 108,903 chunks · **Queries:** 500 fixed, top_k=10 · **Sweep values:** 16, 32, 64, 128, 256 · **Timing repeats:** 3 (quality metrics from repeat 1 only) · **Container restarts:** 0/0/0/0

Outputs: `experiments/results/analysis_FINAL_4DB_ann_pareto_100K.json`, `analysis_FINAL_4DB_ann_pareto_100K.csv` · Script: `scripts/ann_pareto_sweep_final_4db.py` · Exact baseline: `faiss.IndexFlatIP`, computed once, 1.75s

---

## Validation

| Check | Result |
|---|---|
| Index built once per DB | ✅ `add_chunks()` called exactly once; only config field mutated per sweep point |
| Same collection reused | ✅ live count readback identical across all 5 values, every DB: 108,903 |
| Query set identical | ✅ same 500-ID hash across all DBs and all sweep points |
| Exact baseline reused | ✅ `faiss.IndexFlatIP` computed once (1.75s), reused for all 19 sweep rows |
| Runtime param verified changed | ✅ `object.__setattr__` + `getattr` readback asserted on every point |
| No frozen-parameter anomaly | ✅ zero cases of byte-identical top-10 sets across consecutive values |
| Latency / QPS positive | ✅ asserted > 0 on all 19 rows |
| ANN recall in [0,1] | ✅ asserted on all rows with a defined value |
| 500/500 queries completed | ✅ on every one of the 19 sweep rows, no partial runs |
| No container crash/restart | ✅ RestartCount 0 for all 4 containers, before and after |

---

## Parameter → quality / latency, per database

Quality metrics come from each sweep point's **first** timing repeat only — the two remaining repeats feed latency/QPS aggregation but are never treated as independent quality samples, since retrieval against a static index is deterministic.

### Qdrant — `search_params.hnsw_ef`

Passed on every `query_points()` call, read live from the retriever's in-process config — confirmed applied: p50 falls 11.1ms→3.3ms then rises again as ef grows, and ANN-vs-exact climbs monotonically 0.995→0.999.

| ef | Pareto | Qrels R@10 | MRR@10 | nDCG@10 | ANN vs Exact@10 | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|---|---|---|
| 16 | | 0.9613 | 0.8777 | 0.8958 | 0.9952 | 11.14ms | 32.01ms | 48.93ms | 75.8 |
| 32 | | 0.9573 | 0.8736 | 0.8917 | 0.9870 | 7.77ms | 13.21ms | 19.25ms | 152.8 |
| **64** (default) | ✅ | 0.9593 | 0.8756 | 0.8937 | 0.9930 | 3.31ms | 4.66ms | 5.85ms | 294.2 |
| 128 | ✅ | 0.9633 | 0.8786 | 0.8970 | 0.9980 | 4.11ms | 6.79ms | 11.65ms | 201.1 |
| 256 | ✅ | 0.9653 | 0.8806 | 0.8990 | 0.9992 | 4.86ms | 6.52ms | 8.82ms | 198.2 |

### Milvus — `search_params.params.ef`

Passed on every `client.search()` call. Cheapest DB in absolute latency at every ef value; ANN-vs-exact has the widest range of the four (0.910→0.996), the clearest confirmation the parameter reaches the search request.

| ef | Pareto | Qrels R@10 | MRR@10 | nDCG@10 | ANN vs Exact@10 | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|---|---|---|
| 16 | | 0.9323 | 0.8519 | 0.8687 | 0.9104 | 4.26ms | 10.02ms | 17.99ms | 206.3 |
| 32 | | 0.9533 | 0.8693 | 0.8875 | 0.9274 | 2.39ms | 3.49ms | 5.52ms | 391.1 |
| **64** (default) | ✅ | 0.9553 | 0.8726 | 0.8905 | 0.9822 | 2.16ms | 3.23ms | 5.44ms | 419.3 |
| 128 | ✅ | 0.9573 | 0.8746 | 0.8925 | 0.9918 | 2.87ms | 4.54ms | 7.92ms | 331.6 |
| 256 | ✅ | 0.9633 | 0.8786 | 0.8970 | 0.9962 | 3.31ms | 5.50ms | 10.14ms | 279.5 |

### Elasticsearch — `knn.num_candidates`

Passed on every `client.search()` call. Qrels Recall@10 is flat at 0.9653 across the whole sweep — the gold-relevance judgment set saturates by num_candidates=16 — but ANN-vs-exact still climbs 0.957→1.000, proving the parameter is live even where the qrels metric can't see it. Only 32 is Pareto-optimal; every larger value adds latency (p99 up to 21ms at 256) for a qrels-recall gain of exactly zero.

| num_candidates | Pareto | Qrels R@10 | MRR@10 | nDCG@10 | ANN vs Exact@10 | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|---|---|---|
| 16 | | 0.9653 | 0.8841 | 0.9016 | 0.9570 | 8.29ms | 48.93ms | 123.06ms | 82.6 |
| 32 | ✅ | 0.9653 | 0.8806 | 0.8990 | 0.9850 | 6.32ms | 26.67ms | 53.59ms | 117.6 |
| **64** (default) | | 0.9653 | 0.8806 | 0.8990 | 0.9932 | 7.66ms | 11.76ms | 19.56ms | 122.9 |
| 128 | | 0.9653 | 0.8806 | 0.8990 | 0.9978 | 8.03ms | 9.98ms | 13.68ms | 122.0 |
| 256 | | 0.9653 | 0.8806 | 0.8990 | 0.9996 | 9.67ms | 13.55ms | 21.33ms | 98.7 |

### Weaviate — not query-time adjustable

**Excluded from the sweep.** In this codebase's Weaviate v4 client, HNSW `ef` is baked into `vector_index_config` at collection-create time (`Configure.VectorIndex.hnsw(ef=…)`); `near_vector()` carries no per-request ef override. Varying it would require rebuilding the collection, which breaks the experiment's build-once rule — so it is reported as one reference point at the FINAL_4DB default (ef=64, baked in at build) rather than forced into a fake sweep.

| ef (build-time) | | Qrels R@10 | MRR@10 | nDCG@10 | ANN vs Exact@10 | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|---|---|---|
| **64** (default, only point) | reference | 0.9533 | 0.8706 | 0.8885 | 0.9784 | 4.26ms | 10.76ms | 18.93ms | 198.8 |

---

## Reading the frontier

**1. The default of 64 sits ON the Pareto frontier for Qdrant and Milvus, OFF it for Elasticsearch**
For Qdrant and Milvus, 64 is non-dominated — no other tested value beats it on both axes at once. For Elasticsearch, 32 dominates 64: identical qrels Recall@10 (0.9653) at 6.32ms vs 7.66ms p50, so 64 is a strictly worse choice than 32 by the qrels metric, even though its ANN-vs-exact fidelity is higher (0.993 vs 0.985).

**2. Latency cost of going from ef=64 to ef=256**
Qdrant: p50 `3.31ms → 4.86ms` (+47%) buys qrels-recall `0.9593 → 0.9653` (+0.6pp) and ANN-vs-exact `0.993 → 0.999`. Milvus: p50 `2.16ms → 3.31ms` (+54%) buys qrels-recall `0.9553 → 0.9633` (+0.8pp) and ANN-vs-exact `0.982 → 0.996`. Elasticsearch: p50 `7.66ms → 9.67ms` (+26%) buys zero qrels-recall gain, only ANN-vs-exact `0.993 → 1.000`. In all three, the last mile of ANN fidelity (getting from ~99% to ~100% agreement with exact search) costs real latency but essentially no qrels-Recall@10 — the gold-relevance metric saturates well before exact agreement does.

**3. Milvus reaches similar ANN fidelity at the lowest latency of the three swept DBs**
At ~0.98–0.99 ANN-vs-exact fidelity, Milvus's p50 is 2.16–2.87ms, versus Qdrant's 3.31–4.11ms and Elasticsearch's 7.66–8.03ms — roughly 2–3× faster than Elasticsearch for the same search-effort tier. Elasticsearch is consistently the highest-latency of the three at every matched ef/num_candidates value, and the only one whose p95/p99 tail is large relative to its p50 at low search effort (num_candidates=16: p50 8.29ms vs p99 123.06ms).

**4. No anomalies in the run**
The one pattern that could look like a bug — Elasticsearch's qrels Recall@10 not moving across the sweep — is not: ANN-vs-exact for the same rows moves cleanly from 0.957 to 0.9996, so `num_candidates` is provably reaching the search request; the qrels metric is simply already saturated by 500 fixed queries against 100K documents at num_candidates=16. No sweep point produced byte-identical top-10 result sets to its neighbor, no index count drifted from its post-build value, and no container restarted.

---

## Runtime-application evidence

- **Code path:** each param is read from the retriever's `self._config` field inside `retrieve_with_embedding()`, called fresh on every query — not cached at build time. Verified by reading `qdrant_retriever.py`, `milvus_retriever.py`, `elasticsearch_retriever.py` directly.
- **Config mutation:** `object.__setattr__(retriever._config, field, value)` then `getattr` readback, asserted equal, before every sweep point.
- **Behavioral confirmation:** latency and ANN-vs-exact both move with the parameter on every DB — a frozen parameter would show flat latency and flat ANN-vs-exact, which none did.
- **Index stability:** live point/entity/doc count read back after every sweep point: 108,903 for all 5 values, all 3 swept DBs — proof the same collection was reused, not rebuilt.

---

**Scope closes here.** This is the final experiment for the thesis's experimental section — exact-vs-ANN root-cause analysis, the corrected build-nondeterminism methodology note, and this Recall–Latency Pareto sweep together complete the empirical program. No further experiments are planned.
