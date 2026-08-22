#!/usr/bin/env python3
"""Validate Chroma's disk-size behavior in the S3/20K official result against the other
official scale experiments (item 12). Read-only: never modifies any official JSON.

Per explicit user direction, S3 (20K docs) is reported as a confirmed
``requires_rerun: true`` regardless of what the automated consistency check below finds —
the check still runs and its evidence/reasoning is recorded, but the report's bottom-line
verdict for S3 is fixed, not left to the heuristic.

Usage:
    python scripts/validate_s3_chroma_disk.py
    python scripts/validate_s3_chroma_disk.py --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any

S3_FILENAME = "official_scale_experiment_A_S3_20000docs_5db_topk10_20260805_151408.json"


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def _chroma_disk_facts(report: dict[str, Any]) -> dict[str, Any]:
    effective_chroma = report.get("effective_config", {}).get("chroma", {})
    return {
        "num_documents": report.get("num_documents"),
        "disk_size_mb": report.get("disk_size_mb_chroma"),
        "disk_size_flag": report.get("disk_size_flag_chroma"),
        "disk_size_error": report.get("disk_size_error_chroma"),
        "disk_size_status": report.get("disk_size_status_chroma"),  # only present post-upgrade
        "in_memory": effective_chroma.get("in_memory"),
        "persist_directory": effective_chroma.get("persist_directory")
        if isinstance(effective_chroma, dict) else None,
    }


def _find_official_files(results_dir: str) -> list[str]:
    pattern = os.path.join(results_dir, "official_scale*.json")
    return sorted(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "experiments", "results"),
    )
    args = parser.parse_args()

    s3_path = os.path.join(args.results_dir, S3_FILENAME)
    if not os.path.exists(s3_path):
        raise FileNotFoundError(f"Expected S3 official result at {s3_path!r} — not found.")

    all_files = _find_official_files(args.results_dir)
    facts_by_file: dict[str, dict[str, Any]] = {}
    for path in all_files:
        try:
            report = _load(path)
        except (OSError, json.JSONDecodeError) as exc:
            facts_by_file[os.path.basename(path)] = {"error": str(exc)}
            continue
        facts_by_file[os.path.basename(path)] = _chroma_disk_facts(report)

    s3_facts = facts_by_file.get(S3_FILENAME, {})
    other_files = {k: v for k, v in facts_by_file.items() if k != S3_FILENAME}

    inconsistencies: list[str] = []

    other_in_memory_values = {v.get("in_memory") for v in other_files.values() if "in_memory" in v}
    if other_in_memory_values and s3_facts.get("in_memory") not in other_in_memory_values:
        inconsistencies.append(
            f"S3 chroma in_memory={s3_facts.get('in_memory')!r} differs from other scales "
            f"({sorted(str(v) for v in other_in_memory_values)})."
        )

    # If persisted (in_memory is False) elsewhere, disk size should be > 0 and roughly
    # scale with num_documents; a 0/near-0 MB figure at 20K docs while smaller/larger
    # scales report meaningfully larger sizes is the specific suspicious pattern this
    # script exists to catch.
    other_disk_sizes: list[tuple[Any, float]] = [
        (v.get("num_documents"), float(v["disk_size_mb"]))
        for v in other_files.values()
        if v.get("disk_size_mb") is not None
    ]
    s3_disk_mb_raw = s3_facts.get("disk_size_mb")
    s3_disk_mb: float | None = float(s3_disk_mb_raw) if isinstance(s3_disk_mb_raw, (int, float)) else None
    if other_disk_sizes and (s3_disk_mb is None or s3_disk_mb < 1.0):
        inconsistencies.append(
            f"S3 (20,000 docs) reports disk_size_mb_chroma={s3_disk_mb!r} (suspiciously "
            f"low/zero), while other scales report: "
            f"{sorted(other_disk_sizes, key=lambda t: str(t[0]))}."
        )

    if s3_facts.get("disk_size_flag") == "in_memory_no_disk" and any(
        v.get("disk_size_flag") != "in_memory_no_disk" for v in other_files.values()
    ):
        inconsistencies.append(
            "S3 chroma disk_size_flag='in_memory_no_disk' but other scales report a real "
            "persisted disk size — Chroma's persist configuration is inconsistent across scales."
        )

    verdict = {
        "s3_file": S3_FILENAME,
        "s3_chroma_facts": s3_facts,
        "other_scale_chroma_facts": other_files,
        "automated_check_inconsistencies": inconsistencies,
        "automated_check_requires_rerun": bool(inconsistencies),
        # Per explicit user direction: S3 is confirmed requires_rerun regardless of the
        # automated check's own verdict above.
        "requires_rerun": True,
        "requires_rerun_reason": (
            "Confirmed by explicit direction, independent of the automated heuristic above. "
            + (
                "Automated check also found inconsistencies: " + "; ".join(inconsistencies)
                if inconsistencies
                else "Automated check did not find an inconsistency on its own, but the "
                "confirmed status stands."
            )
        ),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"validation_report_s3_chroma_disk_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    print("\nOriginal S3 JSON was NOT modified.")


if __name__ == "__main__":
    main()
