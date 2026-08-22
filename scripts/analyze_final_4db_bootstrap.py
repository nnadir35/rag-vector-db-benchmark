#!/usr/bin/env python3
"""Statistical analysis of the Scale Experiment A FINAL_4DB_S1-S6 results.

Scope: **Qdrant, Milvus, Elasticsearch, Weaviate only.** ChromaDB and FAISS are
deliberately excluded from every statistic this script produces — see the methodology
note embedded in the output JSON's ``"methodology_note"`` field.

Two analyses, both driven off ``query_level_results_{db}`` (per-query rows already
captured by scripts/benchmark_db.py), never off the 3 deterministic repeats:

1. Per-DB, per-scale bootstrap 95% CI for Recall@10, MRR@10 (via the per-query "rr"
   field), and nDCG@10. Bootstrap sample unit = query; 10,000 resamples; seed=42.
2. Paired bootstrap DB-vs-DB comparison (all 6 pairs among the 4 DBs) per scale per
   metric: mean delta, 95% CI, whether the CI contains 0, and an explicit
   "statistically_distinguishable_at_95pct" boolean — never described as "DB A is
   better", and multiple-comparison correction is NOT applied (see note in output).

Reuses scripts/query_level_bootstrap_ci.bootstrap_ci and
scripts/paired_bootstrap_delta.paired_delta_ci — no metric math is reimplemented here.

Usage:
    python scripts/analyze_final_4db_bootstrap.py
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.paired_bootstrap_delta import paired_delta_ci  # noqa: E402
from scripts.query_level_bootstrap_ci import bootstrap_ci  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
FINAL_DBS = ["qdrant", "milvus", "elasticsearch", "weaviate"]  # Chroma/FAISS excluded
METRICS = ["recall@10", "rr", "ndcg@10"]  # "rr" == per-query MRR@10 contribution
METRIC_LABELS = {"recall@10": "recall@10", "rr": "mrr@10", "ndcg@10": "ndcg@10"}
N_SAMPLES = 10000
SEED = 42

METHODOLOGY_NOTE = (
    "ChromaDB başlangıçtaki deney kapsamına dahil edilmiştir. Ancak kullanılan "
    "Docker/persistence konfigürasyonunda kalıcı depolama ölçümünün güvenilir olmadığı "
    "tespit edildiğinden, deneysel koşulların karşılaştırılabilirliğini korumak amacıyla "
    "ChromaDB final performans karşılaştırmasına dahil edilmemiştir. ChromaDB'ye ait ön "
    "deney sonuçları final istatistiksel analizlerde kullanılmamıştır."
)
MULTIPLE_COMPARISON_NOTE = (
    "Each scale x metric produces 6 pairwise comparisons (C(4,2) among 4 DBs), and there "
    "are 6 scales x 3 metrics = 18 such groups (108 paired comparisons total). No "
    "multiple-comparison correction (e.g. Bonferroni/Holm) has been applied to these CIs. "
    "Results are exploratory paired comparisons, not confirmatory hypothesis tests; treat "
    "any single 95% CI that excludes 0 accordingly, especially amid this many comparisons."
)
INTERPRETATION_NOTE = (
    "statistically_distinguishable_at_95pct=true means the paired-bootstrap 95% CI for "
    "the mean per-query delta excludes 0 — it does NOT mean 'DB A is definitively better "
    "than DB B' in any practical sense. Effect size (mean_delta) is reported alongside "
    "every comparison specifically so a statistically distinguishable but practically "
    "tiny difference (e.g. 0.001 recall) is not overstated."
)


def find_final_4db_files() -> dict[str, str]:
    pattern = os.path.join(RESULTS_DIR, "official_scale_FINAL_4DB_S*.json")
    files = sorted(glob.glob(pattern))
    scales: dict[str, str] = {}
    for f in files:
        basename = os.path.basename(f)
        # official_scale_FINAL_4DB_S<n>_...
        marker = "FINAL_4DB_S"
        idx = basename.index(marker) + len(marker)
        scale_num = ""
        for ch in basename[idx:]:
            if ch.isdigit():
                scale_num += ch
            else:
                break
        scales[f"S{scale_num}"] = f
    return dict(sorted(scales.items(), key=lambda kv: int(kv[0][1:])))


def check_naming_artifact(files: dict[str, str]) -> dict[str, Any]:
    """Housekeeping check (item 1): confirm '_5db_' in filenames is a filename-generation
    artifact (db_count is computed from which DBs the script executed, which still
    includes Chroma since the benchmark code itself was not modified — see
    scripts/benchmark_db.py's `db_count = 6 if args.include_faiss else 5`), not a
    reflection of the final analysis scope (4 DBs). Never renames/modifies the files.
    """
    findings = []
    for scale, path in files.items():
        basename = os.path.basename(path)
        has_5db_in_name = "_5db_" in basename
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        chroma_present_in_json = "mean_metrics_chroma" in d
        findings.append({
            "scale": scale,
            "filename": basename,
            "filename_contains_5db": has_5db_in_name,
            "chroma_data_present_in_json": chroma_present_in_json,
            "explanation": (
                "Filename '_5db_' is NOT a naming bug in isolation — it reflects that "
                "the benchmark script (unmodified, per explicit decision to not touch "
                "Chroma's persistence code under time pressure) still executed and "
                "recorded Chroma internally alongside the 4 final DBs. The file's "
                "db_count comes from `db_count = 6 if args.include_faiss else 5` in "
                "scripts/benchmark_db.py's main(), which counts DBs *executed*, not the "
                "'final 4 DB' analysis scope. This script and its outputs use the "
                "correct '4db' label and filter Chroma out of every statistic."
            ) if has_5db_in_name and chroma_present_in_json else (
                "Unexpected: filename/content combination does not match the expected "
                "'ran 5, analyze 4' pattern — flag for manual review."
            ),
        })
    return {"housekeeping_check": findings}


def verify_query_hash_consistency(files: dict[str, str]) -> dict[str, Any]:
    hashes = {}
    config_hashes = {}
    for scale, path in files.items():
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        hashes[scale] = d["reproducibility"]["fixed_query_ids_hash"]
        config_hashes[scale] = d["reproducibility"]["config_hash"]
    unique_query_hashes = set(hashes.values())
    unique_config_hashes = set(config_hashes.values())
    return {
        "per_scale_query_hash": hashes,
        "per_scale_config_hash": config_hashes,
        "query_hash_consistent_across_scales": len(unique_query_hashes) == 1,
        "config_hash_consistent_across_scales": len(unique_config_hashes) == 1,
    }


def load_query_level(path: str, db: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows: list[dict[str, Any]] = d.get(f"query_level_results_{db}", [])
    return rows


def run_bootstrap_ci_analysis(files: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for scale, path in files.items():
        result[scale] = {}
        for db in FINAL_DBS:
            rows = load_query_level(path, db)
            if len(rows) != 500:
                raise ValueError(f"{scale}/{db}: expected 500 query-level rows, got {len(rows)}")
            result[scale][db] = {}
            metric_summary = {}
            for metric in METRICS:
                values = [r.get(metric, float("nan")) for r in rows]
                mean, lo, hi = bootstrap_ci(values, n_samples=N_SAMPLES, seed=SEED, alpha=0.05)
                label = METRIC_LABELS[metric]
                result[scale][db][label] = {
                    "mean": mean, "ci_low": lo, "ci_high": hi,
                    "n_queries": len(values), "n_bootstrap_samples": N_SAMPLES, "seed": SEED,
                }
                metric_summary[label] = (mean, lo, hi)
            summary_rows.append({
                "scale": scale, "db": db,
                "recall@10": metric_summary["recall@10"][0],
                "recall_ci95": f"[{metric_summary['recall@10'][1]:.4f}, {metric_summary['recall@10'][2]:.4f}]",
                "mrr@10": metric_summary["mrr@10"][0],
                "mrr_ci95": f"[{metric_summary['mrr@10'][1]:.4f}, {metric_summary['mrr@10'][2]:.4f}]",
                "ndcg@10": metric_summary["ndcg@10"][0],
                "ndcg_ci95": f"[{metric_summary['ndcg@10'][1]:.4f}, {metric_summary['ndcg@10'][2]:.4f}]",
            })
    return result, summary_rows


def run_paired_bootstrap_analysis(files: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    pairs = list(itertools.combinations(FINAL_DBS, 2))
    for scale, path in files.items():
        result[scale] = {}
        rows_by_db = {db: load_query_level(path, db) for db in FINAL_DBS}
        for db_a, db_b in pairs:
            pair_key = f"{db_a}_vs_{db_b}"
            result[scale][pair_key] = {}
            for metric in METRICS:
                values_a = {r["query_id"]: r.get(metric, float("nan")) for r in rows_by_db[db_a]}
                values_b = {r["query_id"]: r.get(metric, float("nan")) for r in rows_by_db[db_b]}
                comp = paired_delta_ci(values_a, values_b, n_samples=N_SAMPLES, seed=SEED, alpha=0.05)
                label = METRIC_LABELS[metric]
                ci_contains_zero = comp["ci_low"] <= 0.0 <= comp["ci_high"]
                entry = {
                    "db_a": db_a, "db_b": db_b,
                    "mean_delta": comp["mean_delta"],
                    "ci_low": comp["ci_low"], "ci_high": comp["ci_high"],
                    "ci_contains_zero": ci_contains_zero,
                    "statistically_distinguishable_at_95pct": comp["significant_at_alpha"],
                    "num_paired_queries": comp["num_paired_queries"],
                    "dropped_only_in_a": comp["dropped_only_in_a"],
                    "dropped_only_in_b": comp["dropped_only_in_b"],
                }
                result[scale][pair_key][label] = entry
                summary_rows.append({
                    "scale": scale, "metric": label, "db_a": db_a, "db_b": db_b,
                    "mean_delta": comp["mean_delta"],
                    "ci_low": comp["ci_low"], "ci_high": comp["ci_high"],
                    "ci_contains_zero": ci_contains_zero,
                    "distinguishable_at_95pct": comp["significant_at_alpha"],
                })
    return result, summary_rows


def main() -> None:
    files = find_final_4db_files()
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 FINAL_4DB_S1..S6 files, found {len(files)}: {files}")

    housekeeping = check_naming_artifact(files)
    hash_check = verify_query_hash_consistency(files)
    if not hash_check["query_hash_consistent_across_scales"]:
        raise RuntimeError(f"Query hash mismatch across scales: {hash_check['per_scale_query_hash']}")
    if not hash_check["config_hash_consistent_across_scales"]:
        raise RuntimeError(f"Config hash mismatch across scales: {hash_check['per_scale_config_hash']}")

    ci_result, ci_summary_rows = run_bootstrap_ci_analysis(files)
    paired_result, paired_summary_rows = run_paired_bootstrap_analysis(files)

    ci_payload = {
        "analysis": "bootstrap_confidence_intervals",
        "scope": {"final_dbs": FINAL_DBS, "excluded": ["chroma", "faiss"]},
        "methodology_note": METHODOLOGY_NOTE,
        "bootstrap_config": {
            "sample_unit": "query", "n_bootstrap_samples": N_SAMPLES, "seed": SEED,
            "confidence_level": 0.95,
            "note": "3 deterministic repeats are NOT used as independent samples; "
                    "bootstrap resamples the 500 fixed queries.",
        },
        "input_files": files,
        "housekeeping_check": housekeeping["housekeeping_check"],
        "query_hash_check": hash_check,
        "results_by_scale_and_db": ci_result,
    }

    paired_payload = {
        "analysis": "paired_bootstrap_db_vs_db",
        "scope": {"final_dbs": FINAL_DBS, "excluded": ["chroma", "faiss"]},
        "methodology_note": METHODOLOGY_NOTE,
        "multiple_comparison_note": MULTIPLE_COMPARISON_NOTE,
        "interpretation_note": INTERPRETATION_NOTE,
        "bootstrap_config": {
            "sample_unit": "query", "n_bootstrap_samples": N_SAMPLES, "seed": SEED,
            "confidence_level": 0.95,
        },
        "input_files": files,
        "query_hash_check": hash_check,
        "results_by_scale_and_pair": paired_result,
    }

    ci_json_path = os.path.join(RESULTS_DIR, "analysis_FINAL_4DB_bootstrap_ci.json")
    paired_json_path = os.path.join(RESULTS_DIR, "analysis_FINAL_4DB_paired_bootstrap.json")
    with open(ci_json_path, "w", encoding="utf-8") as f:
        json.dump(ci_payload, f, indent=2, ensure_ascii=False)
    with open(paired_json_path, "w", encoding="utf-8") as f:
        json.dump(paired_payload, f, indent=2, ensure_ascii=False)

    ci_csv_path = os.path.join(RESULTS_DIR, "analysis_FINAL_4DB_bootstrap_ci.csv")
    with open(ci_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ci_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ci_summary_rows)

    paired_csv_path = os.path.join(RESULTS_DIR, "analysis_FINAL_4DB_paired_bootstrap.csv")
    with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_summary_rows)

    print(f"Wrote {ci_json_path}")
    print(f"Wrote {paired_json_path}")
    print(f"Wrote {ci_csv_path}")
    print(f"Wrote {paired_csv_path}")
    print()
    print("Query hash consistent across S1-S6:", hash_check["query_hash_consistent_across_scales"])
    print("Config hash consistent across S1-S6:", hash_check["config_hash_consistent_across_scales"])
    print()
    distinguishable = [r for r in paired_summary_rows if r["distinguishable_at_95pct"]]
    print(f"{len(distinguishable)} / {len(paired_summary_rows)} paired comparisons have a 95% CI excluding 0.")


if __name__ == "__main__":
    main()
