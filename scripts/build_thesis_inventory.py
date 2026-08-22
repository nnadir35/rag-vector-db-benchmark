#!/usr/bin/env python3
"""Build the final thesis result inventory from existing FINAL_4DB benchmark outputs.

Reads only. Does not run any benchmark, does not modify any existing result JSON.
Produces:
    experiments/results/THESIS_RESULTS_MANIFEST.json
    experiments/results/THESIS_MASTER_RESULTS.csv
    experiments/results/THESIS_bootstrap_ci_summary.csv
    experiments/results/THESIS_paired_bootstrap_significant.csv
    experiments/results/THESIS_EXPERIMENT_NOTES.md
    experiments/results/thesis_figures/*.svg
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "thesis_figures")
os.makedirs(FIG_DIR, exist_ok=True)

FINAL_DBS = ["qdrant", "milvus", "elasticsearch", "weaviate"]
DB_LABEL = {"qdrant": "Qdrant", "milvus": "Milvus", "elasticsearch": "Elasticsearch", "weaviate": "Weaviate"}
DB_COLOR = {"qdrant": "#5B8FF9", "milvus": "#5AD8A6", "elasticsearch": "#F6BD16", "weaviate": "#E86452"}

SCALES = [
    ("S1", "1K", "official_scale_FINAL_4DB_S1_1000docs_5db_topk10_20260822_103413.json"),
    ("S2", "10K", "official_scale_FINAL_4DB_S2_10000docs_5db_topk10_20260822_103632.json"),
    ("S3", "20K", "official_scale_FINAL_4DB_S3_20000docs_5db_topk10_20260822_103954.json"),
    ("S4", "50K", "official_scale_FINAL_4DB_S4_50000docs_5db_topk10_20260822_104734.json"),
    ("S5", "100K", "official_scale_FINAL_4DB_S5_100000docs_5db_topk10_20260822_112215.json"),
    ("S6", "200K", "official_scale_FINAL_4DB_S6_200000docs_5db_topk10_20260822_115137.json"),
]

ANALYSIS_FILES = {
    "bootstrap_ci": "analysis_FINAL_4DB_bootstrap_ci.json",
    "paired_bootstrap": "analysis_FINAL_4DB_paired_bootstrap.json",
    "exact_vs_ann_100K": "analysis_FINAL_4DB_exact_vs_ann_100K.json",
    "exact_vs_ann_200K": "analysis_FINAL_4DB_exact_vs_ann_200K.json",
    "ann_pareto_100K": "analysis_FINAL_4DB_ann_pareto_100K.json",
}

EXCLUDED_PATTERNS = [
    "official_scale_experiment_A_S1_1000docs_5db_topk10_20260803_123156.json",
    "official_scale_experiment_A_S2_10000docs_5db_topk10_20260803_123807.json",
    "official_scale_experiment_A_S3_20000docs_5db_topk10_20260805_151408.json",
    "official_scale_experiment_A_S4_50000docs_5db_topk10_20260814_154710.json",
    "official_scale_S5_100000docs_5db_topk10_100000docs_5db_topk10_20260820_152847.json",
    "official_scale_S6_200000docs_5db_topk10_200000docs_5db_topk10_20260820_163053.json",
    "official_scale_FINAL_V3_S1_1000docs_5db_topk10_20260822_083807.json",
    "official_scale_FINAL_V3_S2_10000docs_5db_topk10_20260822_084023.json",
    "official_scale_FINAL_V3_S3_20000docs_5db_topk10_20260822_084345.json",
    "official_scale_FINAL_V3_S4_50000docs_5db_topk10_20260822_085010.json",
    "official_scale_SMOKE_metrics_upgrade_1K_1000docs_5db_topk10_20260822_081203.json",
    "official_scale_SMOKE_metrics_upgrade_1K_v2_1000docs_5db_topk10_20260822_083413.json",
    "validation_report_s3_chroma_disk_20260822_073001.json",
    "fixed_queries_20260803_123156.json",
]

EXCLUSION_REASON = {
    "official_scale_experiment_A_S1_1000docs_5db_topk10_20260803_123156.json": "pre-FINAL_4DB pilot run (experiment_A series), superseded by FINAL_4DB",
    "official_scale_experiment_A_S2_10000docs_5db_topk10_20260803_123807.json": "pre-FINAL_4DB pilot run (experiment_A series), superseded by FINAL_4DB",
    "official_scale_experiment_A_S3_20000docs_5db_topk10_20260805_151408.json": "pre-FINAL_4DB pilot run (experiment_A series), superseded by FINAL_4DB",
    "official_scale_experiment_A_S4_50000docs_5db_topk10_20260814_154710.json": "pre-FINAL_4DB pilot run (experiment_A series), superseded by FINAL_4DB",
    "official_scale_S5_100000docs_5db_topk10_100000docs_5db_topk10_20260820_152847.json": "pre-FINAL_4DB S5 run, superseded by FINAL_4DB S5",
    "official_scale_S6_200000docs_5db_topk10_200000docs_5db_topk10_20260820_163053.json": "pre-FINAL_4DB S6 run, superseded by FINAL_4DB S6",
    "official_scale_FINAL_V3_S1_1000docs_5db_topk10_20260822_083807.json": "intermediate FINAL_V3 series (pre-metrics-upgrade), superseded by FINAL_4DB",
    "official_scale_FINAL_V3_S2_10000docs_5db_topk10_20260822_084023.json": "intermediate FINAL_V3 series (pre-metrics-upgrade), superseded by FINAL_4DB",
    "official_scale_FINAL_V3_S3_20000docs_5db_topk10_20260822_084345.json": "intermediate FINAL_V3 series (pre-metrics-upgrade), superseded by FINAL_4DB",
    "official_scale_FINAL_V3_S4_50000docs_5db_topk10_20260822_085010.json": "intermediate FINAL_V3 series (pre-metrics-upgrade), superseded by FINAL_4DB",
    "official_scale_SMOKE_metrics_upgrade_1K_1000docs_5db_topk10_20260822_081203.json": "smoke test of MRR/nDCG metrics upgrade, not a scientific run",
    "official_scale_SMOKE_metrics_upgrade_1K_v2_1000docs_5db_topk10_20260822_083413.json": "smoke test of MRR/nDCG metrics upgrade (v2), not a scientific run",
    "validation_report_s3_chroma_disk_20260822_073001.json": "one-off validation report for Chroma S3 disk-usage measurement bug, not a benchmark result",
    "fixed_queries_20260803_123156.json": "early fixed-query-set artifact from the experiment_A pilot series",
}

THESIS_SECTION = {
    "S1": "4.x Scale Results — 1K", "S2": "4.x Scale Results — 10K", "S3": "4.x Scale Results — 20K",
    "S4": "4.x Scale Results — 50K", "S5": "4.x Scale Results — 100K", "S6": "4.x Scale Results — 200K",
    "bootstrap_ci": "4.y Statistical Confidence (per-DB bootstrap CI)",
    "paired_bootstrap": "4.z Pairwise DB Comparisons (paired bootstrap)",
    "exact_vs_ann_100K": "4.w ANN Fidelity vs Exact Search — 100K",
    "exact_vs_ann_200K": "4.w ANN Fidelity vs Exact Search — 200K",
    "ann_pareto_100K": "4.v Recall–Latency Pareto Sweep — 100K",
}


def load(fn: str) -> dict:
    with open(os.path.join(RESULTS_DIR, fn)) as f:
        return json.load(f)


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. MANIFEST
# ---------------------------------------------------------------------------
manifest_entries = []

for scale_id, scale_label, fn in SCALES:
    d = load(fn)
    r = d["reproducibility"]
    manifest_entries.append({
        "experiment_name": f"FINAL_4DB_{scale_id}",
        "file_path": os.path.relpath(os.path.join(RESULTS_DIR, fn), start=os.path.join(RESULTS_DIR, "..", "..")),
        "scale": scale_label,
        "num_documents": d["num_documents"],
        "num_chunks": d["num_chunks"],
        "db_list": FINAL_DBS,
        "query_count": d["num_queries_evaluated"],
        "query_hash": r["fixed_query_ids_hash"],
        "config_hash": r["config_hash"],
        "git_commit": r["git_commit"],
        "purpose": "Primary scale-point benchmark: indexing + retrieval quality/latency/resource metrics "
                   f"for {scale_label} documents across the 4 final DBs (Chroma also executed internally, "
                   "excluded from thesis metrics).",
        "thesis_section": THESIS_SECTION[scale_id],
        "excluded_from_thesis": False,
    })

d = load(ANALYSIS_FILES["bootstrap_ci"])
manifest_entries.append({
    "experiment_name": "analysis_FINAL_4DB_bootstrap_ci",
    "file_path": f"experiments/results/{ANALYSIS_FILES['bootstrap_ci']}",
    "scale": "S1-S6 (1K-200K)",
    "num_documents": None,
    "num_chunks": None,
    "db_list": FINAL_DBS,
    "query_count": 500,
    "query_hash": None,
    "config_hash": None,
    "git_commit": None,
    "purpose": "Per-DB, per-scale bootstrap 95% CI (10,000 resamples, seed 42) on Recall@10/MRR@10/nDCG@10 "
               "over the 500 fixed queries.",
    "thesis_section": THESIS_SECTION["bootstrap_ci"],
    "excluded_from_thesis": False,
})

d = load(ANALYSIS_FILES["paired_bootstrap"])
manifest_entries.append({
    "experiment_name": "analysis_FINAL_4DB_paired_bootstrap",
    "file_path": f"experiments/results/{ANALYSIS_FILES['paired_bootstrap']}",
    "scale": "S1-S6 (1K-200K)",
    "num_documents": None,
    "num_chunks": None,
    "db_list": FINAL_DBS,
    "query_count": 500,
    "query_hash": None,
    "config_hash": None,
    "git_commit": None,
    "purpose": "Paired-bootstrap mean-delta 95% CI for every DB-pair x scale x metric "
               "(exploratory, no multiple-comparison correction).",
    "thesis_section": THESIS_SECTION["paired_bootstrap"],
    "excluded_from_thesis": False,
})

for key, scale_label in [("exact_vs_ann_100K", "100K"), ("exact_vs_ann_200K", "200K")]:
    d = load(ANALYSIS_FILES[key])
    manifest_entries.append({
        "experiment_name": f"analysis_FINAL_4DB_{key}",
        "file_path": f"experiments/results/{ANALYSIS_FILES[key]}",
        "scale": scale_label,
        "num_documents": d["num_documents"],
        "num_chunks": d["num_chunks"],
        "db_list": FINAL_DBS,
        "query_count": d["num_queries_evaluated"],
        "query_hash": None,
        "config_hash": None,
        "git_commit": None,
        "purpose": "ANN-vs-exact (faiss.IndexFlatIP brute force) agreement@10 for each DB, single index build, "
                   "point estimate (not repeated-build).",
        "thesis_section": THESIS_SECTION[key],
        "excluded_from_thesis": False,
    })

d = load(ANALYSIS_FILES["ann_pareto_100K"])
manifest_entries.append({
    "experiment_name": "analysis_FINAL_4DB_ann_pareto_100K",
    "file_path": f"experiments/results/{ANALYSIS_FILES['ann_pareto_100K']}",
    "scale": "100K",
    "num_documents": d["num_documents"],
    "num_chunks": d["num_chunks"],
    "db_list": ["qdrant", "milvus", "elasticsearch"] + ["weaviate (reference point only)"],
    "query_count": d["num_queries_evaluated"],
    "query_hash": d.get("query_id_hash"),
    "config_hash": None,
    "git_commit": None,
    "purpose": "Query-time ANN search-effort sweep (ef / num_candidates in [16,32,64,128,256]) on ONE build per "
               "DB, isolating the recall-latency trade-off from build-to-build variance. Weaviate excluded from "
               "the sweep (ef is build-time-only in this client) and reported as one reference point.",
    "thesis_section": THESIS_SECTION["ann_pareto_100K"],
    "excluded_from_thesis": False,
})

excluded_entries = []
for fn in EXCLUDED_PATTERNS:
    path = os.path.join(RESULTS_DIR, fn)
    if not os.path.exists(path):
        continue
    excluded_entries.append({
        "experiment_name": fn.rsplit(".json", 1)[0],
        "file_path": f"experiments/results/{fn}",
        "excluded_from_thesis": True,
        "reason": EXCLUSION_REASON[fn],
    })

manifest = {
    "generated_by": "scripts/build_thesis_inventory.py",
    "note": "Read-only inventory over existing result files. No new experiments were run to produce this manifest.",
    "final_thesis_db_scope": FINAL_DBS,
    "excluded_dbs_from_final_comparison": ["chroma", "faiss"],
    "included_experiments": manifest_entries,
    "excluded_from_thesis": excluded_entries,
}

with open(os.path.join(RESULTS_DIR, "THESIS_RESULTS_MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 2. MASTER CSV
# ---------------------------------------------------------------------------
master_rows = []
scale_data = {}  # scale_label -> {db -> row dict}

for scale_id, scale_label, fn in SCALES:
    d = load(fn)
    scale_data[scale_label] = {}
    for db in FINAL_DBS:
        mm = d[f"mean_metrics_{db}"]
        idx = d[db]["indexing"]
        ret = d[db]["retrieval"]
        peak_ram = max(idx.get("container_memory_usage_mb") or 0, ret.get("container_memory_usage_mb") or 0)
        row = {
            "Scale": scale_label,
            "Docs": d["num_documents"],
            "Chunks": d["num_chunks"],
            "DB": DB_LABEL[db],
            "Recall@10": round(mm["recall@10"], 4),
            "MRR@10": round(mm["mrr@10"], 4),
            "nDCG@10": round(mm["ndcg@10"], 4),
            "Search Avg ms": round(d[f"search_only_avg_ms_{db}"], 4),
            "p50": round(d[f"search_only_p50_ms_{db}"], 4),
            "p95": round(d[f"search_only_p95_ms_{db}"], 4),
            "p99": round(d[f"search_only_p99_ms_{db}"], 4),
            "Wall-clock QPS": round(d[f"wall_clock_search_only_qps_mean_{db}"], 2),
            "Index Total sec": round(d[f"{db}_indexing_total_seconds"], 4),
            "Index vectors/sec": round(d[f"total_indexing_vectors_per_second_{db}"], 2),
            "Peak Container RAM": round(peak_ram, 2),
            "Disk MB": d.get(f"disk_size_mb_{db}"),
        }
        master_rows.append(row)
        scale_data[scale_label][db] = row

columns = ["Scale", "Docs", "Chunks", "DB", "Recall@10", "MRR@10", "nDCG@10", "Search Avg ms",
           "p50", "p95", "p99", "Wall-clock QPS", "Index Total sec", "Index vectors/sec",
           "Peak Container RAM", "Disk MB"]

with open(os.path.join(RESULTS_DIR, "THESIS_MASTER_RESULTS.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(master_rows)

# ---------------------------------------------------------------------------
# 3. Bootstrap CI summary table (Scale x DB x metric + CI)
# ---------------------------------------------------------------------------
bci = load(ANALYSIS_FILES["bootstrap_ci"])
scale_label_by_id = {sid: lbl for sid, lbl, _ in SCALES}
bci_rows = []
for scale_id, dbs in bci["results_by_scale_and_db"].items():
    for db, metrics in dbs.items():
        for metric_key, metric_label in [("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]:
            m = metrics[metric_key]
            bci_rows.append({
                "Scale": scale_label_by_id.get(scale_id, scale_id),
                "DB": DB_LABEL[db],
                "Metric": metric_label,
                "Mean": round(m["mean"], 4),
                "CI_low_95": round(m["ci_low"], 4),
                "CI_high_95": round(m["ci_high"], 4),
                "n_queries": m["n_queries"],
            })

with open(os.path.join(RESULTS_DIR, "THESIS_bootstrap_ci_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Scale", "DB", "Metric", "Mean", "CI_low_95", "CI_high_95", "n_queries"])
    w.writeheader()
    w.writerows(bci_rows)

# ---------------------------------------------------------------------------
# 4. Paired bootstrap: only statistically distinguishable pairs
# ---------------------------------------------------------------------------
pb = load(ANALYSIS_FILES["paired_bootstrap"])
pb_rows = []
for scale_id, pairs in pb["results_by_scale_and_pair"].items():
    for pair_key, metrics in pairs.items():
        for metric_key, metric_label in [("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]:
            m = metrics[metric_key]
            if m["statistically_distinguishable_at_95pct"]:
                pb_rows.append({
                    "Scale": scale_label_by_id.get(scale_id, scale_id),
                    "Metric": metric_label,
                    "DB_A": DB_LABEL[m["db_a"]],
                    "DB_B": DB_LABEL[m["db_b"]],
                    "mean_delta_A_minus_B": round(m["mean_delta"], 4),
                    "CI_low_95": round(m["ci_low"], 4),
                    "CI_high_95": round(m["ci_high"], 4),
                    "num_paired_queries": m["num_paired_queries"],
                })

with open(os.path.join(RESULTS_DIR, "THESIS_paired_bootstrap_significant.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Scale", "Metric", "DB_A", "DB_B", "mean_delta_A_minus_B",
                                       "CI_low_95", "CI_high_95", "num_paired_queries"])
    w.writeheader()
    w.writerows(pb_rows)

print(f"manifest entries: {len(manifest_entries)}, excluded: {len(excluded_entries)}")
print(f"master rows: {len(master_rows)}")
print(f"bootstrap CI rows: {len(bci_rows)}")
print(f"significant paired rows: {len(pb_rows)} / total pairs checked")

# ---------------------------------------------------------------------------
# 5. SVG figures (no external plotting deps available in this environment)
# ---------------------------------------------------------------------------
SCALE_LABELS = [lbl for _, lbl, _ in SCALES]
SCALE_X = list(range(len(SCALE_LABELS)))  # categorical x positions, log-ish spacing not needed


def svg_line_chart(path, title, y_label, series, y_fmt="{:.3f}", log_y=False):
    """series: dict[db] -> list of y-values aligned with SCALE_LABELS (None allowed)."""
    W, H = 720, 440
    margin_l, margin_r, margin_t, margin_b = 70, 30, 40, 60
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    all_vals = [v for vals in series.values() for v in vals if v is not None]
    if not all_vals:
        return
    y_min, y_max = min(all_vals), max(all_vals)
    if log_y:
        import math
        y_min = max(y_min, 1e-6)
        ly_min, ly_max = math.log10(y_min), math.log10(y_max)
        if ly_max == ly_min:
            ly_max += 1
        def y_to_px(v):
            v = max(v, y_min)
            return margin_t + plot_h - (math.log10(v) - ly_min) / (ly_max - ly_min) * plot_h
    else:
        pad = (y_max - y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
        def y_to_px(v):
            return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    def x_to_px(i):
        n = len(SCALE_LABELS)
        return margin_l + (i / (n - 1)) * plot_w if n > 1 else margin_l + plot_w / 2

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    parts.append(f'<text x="{W/2}" y="24" font-size="16" text-anchor="middle" fill="#1a1a1a" font-weight="600">{title}</text>')

    # gridlines + y ticks
    n_ticks = 5
    for t in range(n_ticks + 1):
        frac = t / n_ticks
        val = y_min + frac * (y_max - y_min) if not log_y else None
        py = margin_t + plot_h - frac * plot_h
        parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{W-margin_r}" y2="{py:.1f}" '
                     f'stroke="#e5e5e5" stroke-width="1"/>')
        if not log_y:
            parts.append(f'<text x="{margin_l-8}" y="{py+4:.1f}" font-size="10" text-anchor="end" '
                         f'fill="#555">{y_fmt.format(val)}</text>')

    if log_y:
        # ticks at min/max/mid on log scale using actual data extremes
        for v in sorted(set([y_min, y_max] + all_vals))[:0]:
            pass
        for v in [y_min, (y_min*y_max)**0.5, y_max]:
            py = y_to_px(v)
            parts.append(f'<text x="{margin_l-8}" y="{py+4:.1f}" font-size="10" text-anchor="end" '
                         f'fill="#555">{y_fmt.format(v)}</text>')

    # x axis labels
    for i, lbl in enumerate(SCALE_LABELS):
        px = x_to_px(i)
        parts.append(f'<text x="{px:.1f}" y="{H-margin_b+20:.1f}" font-size="11" text-anchor="middle" fill="#333">{lbl}</text>')
    parts.append(f'<text x="{margin_l + plot_w/2:.1f}" y="{H-14}" font-size="11" text-anchor="middle" fill="#333">Document scale</text>')
    parts.append(f'<text x="16" y="{margin_t + plot_h/2:.1f}" font-size="11" text-anchor="middle" '
                 f'fill="#333" transform="rotate(-90 16 {margin_t + plot_h/2:.1f})">{y_label}</text>')

    # axis box
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{W-margin_r}" y2="{margin_t+plot_h}" stroke="#999"/>')

    # series lines
    for db, vals in series.items():
        color = DB_COLOR.get(db, "#333")
        pts = [(x_to_px(i), y_to_px(v)) for i, v in enumerate(vals) if v is not None]
        if len(pts) >= 2:
            path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    # legend
    lx, ly = margin_l + 10, margin_t + 8
    for i, db in enumerate(series.keys()):
        color = DB_COLOR.get(db, "#333")
        yy = ly + i * 16
        parts.append(f'<rect x="{lx}" y="{yy-8}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx+14}" y="{yy+1}" font-size="11" fill="#222">{DB_LABEL[db]}</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def series_for(field_fn):
    return {db: [field_fn(scale_data[lbl][db]) for lbl in SCALE_LABELS] for db in FINAL_DBS}


svg_line_chart(os.path.join(FIG_DIR, "01_recall_at_10_vs_scale.svg"),
                "Recall@10 vs Document Scale", "Recall@10",
                series_for(lambda r: r["Recall@10"]))
svg_line_chart(os.path.join(FIG_DIR, "02_mrr_at_10_vs_scale.svg"),
                "MRR@10 vs Document Scale", "MRR@10",
                series_for(lambda r: r["MRR@10"]))
svg_line_chart(os.path.join(FIG_DIR, "03_ndcg_at_10_vs_scale.svg"),
                "nDCG@10 vs Document Scale", "nDCG@10",
                series_for(lambda r: r["nDCG@10"]))
svg_line_chart(os.path.join(FIG_DIR, "04_search_avg_latency_vs_scale.svg"),
                "Search Avg Latency vs Document Scale", "ms",
                series_for(lambda r: r["Search Avg ms"]), y_fmt="{:.2f}", log_y=True)
svg_line_chart(os.path.join(FIG_DIR, "05_p95_latency_vs_scale.svg"),
                "p95 Search Latency vs Document Scale", "ms",
                series_for(lambda r: r["p95"]), y_fmt="{:.2f}", log_y=True)
svg_line_chart(os.path.join(FIG_DIR, "06_p99_latency_vs_scale.svg"),
                "p99 Search Latency vs Document Scale", "ms",
                series_for(lambda r: r["p99"]), y_fmt="{:.2f}", log_y=True)
svg_line_chart(os.path.join(FIG_DIR, "07_wallclock_qps_vs_scale.svg"),
                "Wall-clock QPS vs Document Scale", "queries/sec",
                series_for(lambda r: r["Wall-clock QPS"]), y_fmt="{:.1f}")
svg_line_chart(os.path.join(FIG_DIR, "08_indexing_time_vs_scale.svg"),
                "Indexing Time vs Document Scale", "seconds",
                series_for(lambda r: r["Index Total sec"]), y_fmt="{:.1f}", log_y=True)
svg_line_chart(os.path.join(FIG_DIR, "09_indexing_throughput_vs_scale.svg"),
                "Indexing Throughput vs Document Scale", "vectors/sec",
                series_for(lambda r: r["Index vectors/sec"]), y_fmt="{:.1f}")
svg_line_chart(os.path.join(FIG_DIR, "10_peak_ram_vs_scale.svg"),
                "Peak Container RAM vs Document Scale", "MB",
                series_for(lambda r: r["Peak Container RAM"]), y_fmt="{:.0f}", log_y=True)
svg_line_chart(os.path.join(FIG_DIR, "11_disk_usage_vs_scale.svg"),
                "Disk Usage vs Document Scale", "MB",
                series_for(lambda r: r["Disk MB"]), y_fmt="{:.0f}", log_y=True)


def svg_grouped_bar(path, title, y_label, categories, series, y_fmt="{:.3f}"):
    """categories: list of x-group labels; series: dict[db] -> list of values aligned with categories."""
    W, H = 720, 440
    margin_l, margin_r, margin_t, margin_b = 70, 30, 40, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    all_vals = [v for vals in series.values() for v in vals if v is not None]
    y_min, y_max = 0, max(all_vals) * 1.15
    n_groups = len(categories)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = group_w / (n_series + 1)

    def y_to_px(v):
        return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Helvetica,Arial,sans-serif">',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{W/2}" y="24" font-size="16" text-anchor="middle" fill="#1a1a1a" font-weight="600">{title}</text>']

    n_ticks = 5
    for t in range(n_ticks + 1):
        frac = t / n_ticks
        val = y_min + frac * (y_max - y_min)
        py = margin_t + plot_h - frac * plot_h
        parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{W-margin_r}" y2="{py:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{margin_l-8}" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{y_fmt.format(val)}</text>')

    parts.append(f'<text x="16" y="{margin_t + plot_h/2:.1f}" font-size="11" text-anchor="middle" fill="#333" '
                 f'transform="rotate(-90 16 {margin_t + plot_h/2:.1f})">{y_label}</text>')

    for gi, cat in enumerate(categories):
        gx0 = margin_l + gi * group_w
        for si, (db, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None:
                continue
            bx = gx0 + (si + 0.5) * bar_w
            by = y_to_px(v)
            h = margin_t + plot_h - by
            parts.append(f'<rect x="{bx - bar_w*0.4:.1f}" y="{by:.1f}" width="{bar_w*0.8:.1f}" height="{h:.1f}" '
                         f'fill="{DB_COLOR.get(db,"#333")}"/>')
        parts.append(f'<text x="{gx0 + group_w/2:.1f}" y="{H-margin_b+20:.1f}" font-size="11" '
                     f'text-anchor="middle" fill="#333">{cat}</text>')

    parts.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{W-margin_r}" y2="{margin_t+plot_h}" stroke="#999"/>')

    lx, ly = margin_l + 10, margin_t + 8
    for i, db in enumerate(series.keys()):
        yy = ly + i * 16
        parts.append(f'<rect x="{lx}" y="{yy-8}" width="10" height="10" fill="{DB_COLOR.get(db,"#333")}"/>')
        parts.append(f'<text x="{lx+14}" y="{yy+1}" font-size="11" fill="#222">{DB_LABEL[db]}</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


ex100 = load(ANALYSIS_FILES["exact_vs_ann_100K"])
ex200 = load(ANALYSIS_FILES["exact_vs_ann_200K"])
ann_vs_exact_series = {
    db: [ex100["results_by_db"][db]["ann_recall_vs_exact@10"]["mean"],
         ex200["results_by_db"][db]["ann_recall_vs_exact@10"]["mean"]]
    for db in FINAL_DBS
}
svg_grouped_bar(os.path.join(FIG_DIR, "12_ann_vs_exact_recall_100k_200k.svg"),
                "ANN-vs-Exact Recall@10 Agreement — 100K & 200K", "Agreement with exact search",
                ["100K", "200K"], ann_vs_exact_series, y_fmt="{:.3f}")


def svg_pareto_scatter(path, title, pareto_series):
    """pareto_series: dict[db] -> list of (p50_ms, qrels_recall) tuples across swept values."""
    W, H = 720, 480
    margin_l, margin_r, margin_t, margin_b = 70, 30, 40, 60
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    xs = [p[0] for pts in pareto_series.values() for p in pts]
    ys = [p[1] for pts in pareto_series.values() for p in pts]
    import math
    x_min, x_max = math.log10(min(xs)), math.log10(max(xs))
    y_min, y_max = min(ys) - 0.005, max(ys) + 0.005

    def x_to_px(v):
        return margin_l + (math.log10(v) - x_min) / (x_max - x_min) * plot_w

    def y_to_px(v):
        return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Helvetica,Arial,sans-serif">',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{W/2}" y="24" font-size="16" text-anchor="middle" fill="#1a1a1a" font-weight="600">{title}</text>']

    for t in range(6):
        frac = t / 5
        val = y_min + frac * (y_max - y_min)
        py = margin_t + plot_h - frac * plot_h
        parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{W-margin_r}" y2="{py:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{margin_l-8}" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{val:.3f}</text>')

    parts.append(f'<text x="{margin_l + plot_w/2:.1f}" y="{H-16}" font-size="11" text-anchor="middle" fill="#333">p50 latency, ms (log scale)</text>')
    parts.append(f'<text x="16" y="{margin_t + plot_h/2:.1f}" font-size="11" text-anchor="middle" fill="#333" '
                 f'transform="rotate(-90 16 {margin_t + plot_h/2:.1f})">Qrels Recall@10</text>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{W-margin_r}" y2="{margin_t+plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#999"/>')

    for db, pts in pareto_series.items():
        color = DB_COLOR.get(db, "#333")
        pts_sorted = sorted(pts, key=lambda p: p[0])
        path_d = "M " + " L ".join(f"{x_to_px(x):.1f},{y_to_px(y):.1f}" for x, y in pts_sorted)
        parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="4,3"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x_to_px(x):.1f}" cy="{y_to_px(y):.1f}" r="4.5" fill="{color}"/>')

    lx, ly = margin_l + 10, margin_t + 8
    for i, db in enumerate(pareto_series.keys()):
        yy = ly + i * 16
        parts.append(f'<rect x="{lx}" y="{yy-8}" width="10" height="10" fill="{DB_COLOR.get(db,"#333")}"/>')
        parts.append(f'<text x="{lx+14}" y="{yy+1}" font-size="11" fill="#222">{DB_LABEL[db]} (ef/num_candidates sweep)</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


pareto_points = {
    "qdrant": [(11.14, 0.9613), (7.77, 0.9573), (3.31, 0.9593), (4.11, 0.9633), (4.86, 0.9653)],
    "milvus": [(4.26, 0.9323), (2.39, 0.9533), (2.16, 0.9553), (2.87, 0.9573), (3.31, 0.9633)],
    "elasticsearch": [(8.29, 0.9653), (6.32, 0.9653), (7.66, 0.9653), (8.03, 0.9653), (9.67, 0.9653)],
    "weaviate": [(4.26, 0.9533)],
}
svg_pareto_scatter(os.path.join(FIG_DIR, "13_ann_pareto_100k.svg"),
                    "100K Recall–Latency Pareto (query-time search-effort sweep)", pareto_points)

print("figures written to", FIG_DIR)
for fn in sorted(os.listdir(FIG_DIR)):
    print(" ", fn)
