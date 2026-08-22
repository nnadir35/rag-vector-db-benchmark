#!/usr/bin/env python3
"""Paired bootstrap significance test between two DBs on the same query set.

Loads ``query_level_results_{db}`` for two DBs from the same result JSON, inner-joins on
``query_id`` (both DBs are evaluated on the identical ``bench_queries``/``query_ids_used``
by construction — the join is a safety check, not expected to drop rows in practice),
computes the per-query paired difference ``metric_a[q] - metric_b[q]``, and bootstraps
that paired-difference vector (resampling query indices jointly, preserving the pairing)
to a 95% CI on the mean delta. Reports whether the CI excludes 0 — no combined/weighted
"winner" score is computed.

Usage:
    python scripts/paired_bootstrap_delta.py --input foo.json --db-a chroma --db-b qdrant \
        --metric recall@10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.query_level_bootstrap_ci import bootstrap_ci  # noqa: E402


def paired_delta_ci(
    values_a: dict[str, float],
    values_b: dict[str, float],
    n_samples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Inner-join on query id, compute paired deltas, bootstrap the mean delta."""
    common_ids = sorted(set(values_a) & set(values_b))
    dropped_a = sorted(set(values_a) - set(values_b))
    dropped_b = sorted(set(values_b) - set(values_a))

    deltas = [values_a[qid] - values_b[qid] for qid in common_ids]
    mean, lo, hi = bootstrap_ci(deltas, n_samples=n_samples, seed=seed, alpha=alpha)

    return {
        "num_paired_queries": len(common_ids),
        "dropped_only_in_a": dropped_a,
        "dropped_only_in_b": dropped_b,
        "mean_delta": mean,
        "ci_low": lo,
        "ci_high": hi,
        "significant_at_alpha": bool(lo > 0 or hi < 0) if deltas else False,
    }


def _load_query_level_results(input_path: str, db: str) -> list[dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        report: dict[str, Any] = json.load(f)
    key = f"query_level_results_{db}"
    if key not in report:
        raise KeyError(
            f"'{key}' not found in {input_path}. Re-run the benchmark with the current "
            "code to get per-query results (see CLAUDE.md / item 5)."
        )
    rows: list[dict[str, Any]] = report[key]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--db-a", required=True)
    parser.add_argument("--db-b", required=True)
    parser.add_argument("--metric", required=True, help='e.g. "recall@10", "rr", "ndcg@10"')
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows_a = _load_query_level_results(args.input, args.db_a)
    rows_b = _load_query_level_results(args.input, args.db_b)

    values_a = {row["query_id"]: row.get(args.metric, float("nan")) for row in rows_a}
    values_b = {row["query_id"]: row.get(args.metric, float("nan")) for row in rows_b}

    result = paired_delta_ci(values_a, values_b, args.bootstrap_samples, args.seed, args.alpha)
    result.update({"input": args.input, "db_a": args.db_a, "db_b": args.db_b, "metric": args.metric})

    print(
        f"{args.db_a} - {args.db_b} on {args.metric}: "
        f"mean_delta={result['mean_delta']:.4f}  "
        f"{int((1 - args.alpha) * 100)}% CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}]  "
        f"significant={result['significant_at_alpha']}  "
        f"(n={result['num_paired_queries']})"
    )
    if result["dropped_only_in_a"] or result["dropped_only_in_b"]:
        print(
            f"WARNING: query sets were not identical — "
            f"{len(result['dropped_only_in_a'])} only in {args.db_a}, "
            f"{len(result['dropped_only_in_b'])} only in {args.db_b}."
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
