#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals from per-query retrieval quality results.

Reads the ``query_level_results_{db}`` list a result JSON now carries (see
scripts/benchmark_db.py, item 5) and bootstraps over the *query axis* — resampling which
queries were drawn, never resampling across the 3 deterministic repeats (those are not
independent quality samples: the same fixed query set, chunking, and embeddings are reused
verbatim in every repeat, so they cannot substitute for a query-level bootstrap).

Usage:
    python scripts/query_level_bootstrap_ci.py --input experiments/results/foo.json --db chroma
    python scripts/query_level_bootstrap_ci.py --input foo.json --db chroma --metric recall@10
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_samples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over ``values``. Returns ``(mean, lo, hi)`` for a
    ``1 - alpha`` confidence interval."""
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)  # drop NaNs
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    resample_idx = rng.integers(0, n, size=(n_samples, n))
    resample_means = arr[resample_idx].mean(axis=1)

    lo = float(np.percentile(resample_means, 100 * (alpha / 2)))
    hi = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def _load_query_level_results(input_path: str, db: str) -> list[dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        report: dict[str, Any] = json.load(f)
    key = f"query_level_results_{db}"
    if key not in report:
        raise KeyError(
            f"'{key}' not found in {input_path}. This result file may predate the "
            "per-query result capture (see CLAUDE.md / item 5) — re-run the benchmark "
            "with the current code to get it."
        )
    rows: list[dict[str, Any]] = report[key]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a benchmark_db.py result JSON.")
    parser.add_argument("--db", required=True, help="DB name, e.g. chroma/qdrant/milvus/elasticsearch/weaviate.")
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Which per-query metric keys to bootstrap (default: all keys found, minus query_id).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None, help="Optional path to write the CI report JSON.")
    args = parser.parse_args()

    rows = _load_query_level_results(args.input, args.db)
    if not rows:
        print(f"No query-level rows for db={args.db!r} in {args.input!r}.", file=sys.stderr)
        sys.exit(1)

    metric_keys = args.metrics or [k for k in rows[0].keys() if k != "query_id"]

    report = {
        "input": args.input,
        "db": args.db,
        "num_queries": len(rows),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "confidence_level": 1 - args.alpha,
        "metrics": {},
    }
    for metric in metric_keys:
        values = [row.get(metric, float("nan")) for row in rows]
        mean, lo, hi = bootstrap_ci(values, args.bootstrap_samples, args.seed, args.alpha)
        report["metrics"][metric] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        print(f"{metric}: mean={mean:.4f}  {int((1 - args.alpha) * 100)}% CI=[{lo:.4f}, {hi:.4f}]")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
