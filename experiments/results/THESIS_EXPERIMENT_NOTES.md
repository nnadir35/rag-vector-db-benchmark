# Thesis Experiment Notes — Methodology Caveats

This file documents methodology decisions and known limitations for the final thesis result set
(`THESIS_RESULTS_MANIFEST.json`, `THESIS_MASTER_RESULTS.csv`). It does not introduce new results —
it explains how to correctly interpret the existing ones.

**Final thesis DB scope:** Qdrant, Milvus, Elasticsearch, Weaviate.

---

## 1. Why ChromaDB was excluded from the final comparison

ChromaDB was part of the original experimental scope and is still present internally in every
`official_scale_FINAL_4DB_S*.json` file (the benchmark script runs all 5 non-FAISS DBs in one pass;
its code was intentionally left unmodified under time pressure). However, the Docker/persistence
configuration used for Chroma in this setup was found to produce unreliable disk-usage measurements
(see `validation_report_s3_chroma_disk_20260822_073001.json`, produced during the S3 disk-usage
investigation). Because disk footprint is one of the compared dimensions, and because comparability
across DBs requires every measured dimension to be trustworthy for every DB, Chroma's results are
excluded from the final quality/latency/resource comparison and from all statistical analyses
(bootstrap CI, paired bootstrap). Its numbers remain in the raw JSON files for transparency but must
not be cited as thesis findings.

FAISS is excluded for a different, structural reason: it has no server/persistence layer comparable
to the other five DBs and is used only as the brute-force exact-search baseline for the
ANN-vs-exact experiments (`exact_method: faiss.IndexFlatIP`), not as a benchmarked retrieval system
in its own right.

## 2. Milvus build-to-build nondeterminism observed at 100K

The exact-vs-ANN experiment at 100K (`analysis_FINAL_4DB_exact_vs_ann_100K.json`) showed Milvus's
agreement with exact brute-force search (`ann_recall_vs_exact@10`) at 0.942 — visibly lower than
Qdrant (0.9998), Elasticsearch (0.9944), and Weaviate (0.9782) at the same scale, despite all four
using comparable default HNSW parameters. This was empirically confirmed to be a **build-to-build**
effect specific to Milvus's HNSW graph construction at 100K, not a general property of ANN search in
this codebase, and **must not be generalized to Qdrant or Weaviate** (both of which show ANN-vs-exact
agreement of ≥0.978 at the same scale). The 200K run does not reproduce this gap as sharply
(Milvus 0.979 vs Qdrant 0.992), which is consistent with build variance rather than a systematic bias
that grows with scale. Because of this, cross-DB comparisons of Milvus's *exact-vs-ANN fidelity*
specifically at 100K should be read as a single observed build's outcome, not as evidence of a stable
per-DB ranking on that metric. This is also why the ANN Pareto sweep (`ann_pareto_sweep_final_4db.py`)
was designed to build each DB's index exactly once and vary only the query-time search-effort
parameter on top of it — isolating the recall/latency trade-off from this build variance rather than
conflating the two.

## 3. Exact-vs-ANN results are single-build point estimates

`analysis_FINAL_4DB_exact_vs_ann_100K.json` and `..._200K.json` each reflect **one index build per
DB**, compared once against a single FAISS `IndexFlatIP` brute-force pass over the same corpus. They
are point estimates of ANN fidelity at that specific build, not repeated-build distributions — no
confidence interval over multiple independent builds exists for `ann_recall_vs_exact@10` itself (the
reported `ci_low`/`ci_high` in that file bootstrap over the *500 queries* of that one build, not over
repeated builds). Given the build-to-build variance documented in Note 2, a single build's
ANN-vs-exact number for any one DB — especially Milvus — should be presented in the thesis as an
illustrative snapshot of ANN behavior at that scale, not as a precise, reproducible property of the
DB's default configuration.

## 4. Weaviate could not be included in the query-time `ef` sweep

The ANN Pareto sweep (`analysis_FINAL_4DB_ann_pareto_100K.json`) varies each DB's search-effort
parameter live, on a single already-built index: Qdrant's `hnsw_ef` (`SearchParams`), Milvus's
`search_params.params.ef`, and Elasticsearch's `knn.num_candidates` are all read fresh on every
query call. Weaviate's HNSW `ef` is not exposed this way in this codebase's v4 client: it is baked
into `vector_index_config` at **collection-create time**
(`wvc.config.Configure.VectorIndex.hnsw(ef=...)`), and `near_vector()` carries no per-request ef
override. Sweeping it would require rebuilding the collection at every sweep point, which breaks the
sweep's core methodological rule (build once, vary only the query-time parameter) and would
reintroduce the build-to-build variance the sweep exists to avoid (see Note 2). Weaviate is therefore
reported as a single reference point at the FINAL_4DB default (`ef=64`, baked in at build time) rather
than forced into a sweep the client does not support.

## 5. Paired-comparison statistics are exploratory, not confirmatory

`analysis_FINAL_4DB_paired_bootstrap.json` reports, for every scale × metric, all `C(4,2) = 6`
pairwise DB comparisons (Qdrant/Milvus/Elasticsearch/Weaviate), across 6 scales × 3 metrics = 18
groups, i.e. 108 paired comparisons in total. **No multiple-comparison correction** (e.g.
Bonferroni, Holm) has been applied to the reported 95% CIs. With this many comparisons run at a
nominal 95% level, some fraction of "statistically distinguishable" results are expected by chance
alone. These results should be presented in the thesis as **exploratory** paired comparisons that
describe the observed data, not as confirmatory hypothesis tests supporting a specific pre-registered
claim. `THESIS_paired_bootstrap_significant.csv` lists only the pairs whose CI excludes 0 for
convenience, but the same caveat applies to every row in it: read effect size (`mean_delta`) alongside
significance, since a statistically distinguishable but practically tiny difference (e.g. 0.001
recall) should not be overstated as a meaningful DB ranking.

---

*No result JSON files were modified to produce this note. No new benchmark, sweep, or repeat run was
executed as part of preparing the thesis inventory.*
