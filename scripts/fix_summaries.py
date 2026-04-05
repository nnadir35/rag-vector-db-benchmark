#!/usr/bin/env python3
"""Retroactively fill metrics_summary in experiment result JSON files."""

import json
from pathlib import Path


def _compute_summary(metrics_list: list) -> dict:
    """Per-query metrik listesinden ortalama hesaplar."""
    valid = [m for m in metrics_list if m is not None]
    if not valid:
        return {}
    keys = valid[0].keys()
    return {
        key: round(sum(m[key] for m in valid) / len(valid), 4)
        for key in keys
    }


def main() -> None:
    results_dir = Path(__file__).resolve().parent.parent / "experiments" / "results"
    fixed = 0
    for path in sorted(results_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("metrics_summary") != {}:
            continue
        results = data.get("results", [])
        metrics_list = [r.get("metrics") for r in results]
        data["metrics_summary"] = _compute_summary(metrics_list)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        fixed += 1
    print(f"Fixed {fixed} file(s).")


if __name__ == "__main__":
    main()
