#!/usr/bin/env python3
"""Compare vector DB benchmark results stored in JSON files.

Supports single file view, scale progression view (--scale), and N-file side-by-side comparison (--last N).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.benchmark_db import _print_table


def find_result_files() -> list[str]:
    """Find all result files matching official_*.json sorted by file mtime / update time."""
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "results"))
    pattern = os.path.join(results_dir, "official_*.json")
    files = glob.glob(pattern)

    def get_sort_key(filepath: str) -> str:
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                if "timestamp" in data:
                    return str(data["timestamp"])
                if "updated_at" in data:
                    return str(data["updated_at"])
        except Exception:
            pass
        return str(os.path.getmtime(filepath))

    files.sort(key=get_sort_key)
    return files


def load_json(filepath: str) -> dict[str, Any]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def _format_metric(val: Any, fmt: str = ".3f") -> str:
    if val is None or not isinstance(val, (int, float)) or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:{fmt}}"


def show_scale_mode(files: list[str], metric: str) -> None:
    if not files:
        print("Sonuç dosyası bulunamadı.")
        return

    scale_data: list[tuple[int, int, dict[str, float]]] = []

    for fpath in files:
        data = load_json(fpath)
        num_docs = data.get("num_documents", 0)
        num_chunks = data.get("num_chunks", 0)
        dbs = ["faiss", "qdrant", "chroma", "milvus", "elasticsearch", "weaviate"]
        db_vals: dict[str, float] = {}

        for db in dbs:
            if metric.startswith("recall"):
                val = data.get(f"mean_recall_{db}", float("nan"))
                if val != val or val is None:
                    mm = data.get(f"mean_metrics_{db}", {})
                    val = mm.get(metric, float("nan"))
            elif metric == "mrr":
                mm = data.get(f"mean_metrics_{db}", {})
                val = mm.get("mrr", float("nan"))
            elif metric == "ndcg10" or metric == "ndcg@10":
                mm = data.get(f"mean_metrics_{db}", {})
                val = mm.get("ndcg@10", float("nan"))
            else:
                mm = data.get(f"mean_metrics_{db}", {})
                val = mm.get(metric, float("nan"))
            db_vals[db] = val

        scale_data.append((num_docs, num_chunks, db_vals))

    scale_data.sort(key=lambda x: x[0])

    print(f"\nÖlçek Serisi ({metric}):")
    print(f"{'Ölçek':<12} | {'Vektör':<8} | {'FAISS':>10} | {'Qdrant':>10} | {'Chroma':>10} | {'Milvus':>10} | {'ES':>10} | {'Weaviate':>10}")
    print("-" * 95)
    for num_docs, num_chunks, db_vals in scale_data:
        docs_str = f"{num_docs//1000}K dok" if num_docs >= 1000 else f"{num_docs} dok"
        vec_str = str(num_chunks)
        f_str = _format_metric(db_vals.get("faiss"), ".3f")
        q_str = _format_metric(db_vals.get("qdrant"), ".3f")
        c_str = _format_metric(db_vals.get("chroma"), ".3f")
        m_str = _format_metric(db_vals.get("milvus"), ".3f")
        e_str = _format_metric(db_vals.get("elasticsearch"), ".3f")
        w_str = _format_metric(db_vals.get("weaviate"), ".3f")
        print(f"{docs_str:<12} | {vec_str:<8} | {f_str:>10} | {q_str:>10} | {c_str:>10} | {m_str:>10} | {e_str:>10} | {w_str:>10}")
    print("-" * 95 + "\n")


def show_last_n_mode(files: list[str], n: int) -> None:
    target_files = files[-n:]
    if not target_files:
        print("Karşılaştırılacak dosya bulunamadı.")
        return

    print(f"\nSon {len(target_files)} Deney Karşılaştırması:")
    headers = []
    reports = []
    for fpath in target_files:
        data = load_json(fpath)
        reports.append(data)
        fname = os.path.basename(fpath)
        headers.append(fname[:25])

    col_w = 26
    header_str = " | ".join([f"{h:^{col_w}}" for h in headers])
    print(f"{'Metrik / DB':<30} | {header_str}")
    print("-" * (33 + (col_w + 3) * len(target_files)))

    # num_documents
    doc_vals = [str(r.get("num_documents", "—")) for r in reports]
    print(f"{'num_documents':<30} | " + " | ".join([f"{v:^{col_w}}" for v in doc_vals]))
    print("-" * (33 + (col_w + 3) * len(target_files)))

    dbs = [
        ("ChromaDB", "chroma"),
        ("Qdrant", "qdrant"),
        ("FAISS", "faiss"),
        ("Milvus", "milvus"),
        ("ElasticSearch", "elasticsearch"),
        ("Weaviate", "weaviate"),
    ]

    metrics = [
        ("retrieval_avg_ms", "avg_ms"),
        ("p50", "p50_ms"),
        ("p95", "p95_ms"),
        ("mean_metrics.mrr", "mrr"),
        ("mean_metrics.ndcg@10", "ndcg@10"),
    ]

    for db_label, db_key in dbs:
        for metric_label, metric_key in metrics:
            row_title = f"{db_label} - {metric_label}"
            vals = []
            for r in reports:
                if metric_key == "avg_ms":
                    v = r.get(f"retrieval_avg_ms_{db_key}")
                    vals.append(_format_metric(v, ".2f"))
                elif metric_key == "p50_ms":
                    v = r.get(f"retrieval_p50_ms_{db_key}")
                    vals.append(_format_metric(v, ".2f"))
                elif metric_key == "p95_ms":
                    v = r.get(f"retrieval_p95_ms_{db_key}")
                    vals.append(_format_metric(v, ".2f"))
                elif metric_key == "mrr":
                    mm = r.get(f"mean_metrics_{db_key}", {})
                    v = mm.get("mrr")
                    vals.append(_format_metric(v, ".4f"))
                elif metric_key == "ndcg@10":
                    mm = r.get(f"mean_metrics_{db_key}", {})
                    v = mm.get("ndcg@10")
                    vals.append(_format_metric(v, ".4f"))
            print(f"{row_title:<30} | " + " | ".join([f"{v:^{col_w}}" for v in vals]))
        print("-" * (33 + (col_w + 3) * len(target_files)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector DB Benchmark Sonuç Karşılaştırıcı")
    parser.add_argument("--file", type=str, default=None, help="Belirli bir JSON dosyası göster")
    parser.add_argument("--last", type=int, nargs="?", const=1, default=None, help="Son N sonucu karşılaştır (varsayılan 1)")
    parser.add_argument("--scale", action="store_true", help="Ölçek serisini göster")
    parser.add_argument("--metric", type=str, default="recall@10", help="Ölçek modunda kullanılacak metrik (varsayılan recall@10)")

    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"Hata: Dosya bulunamadı: {args.file}")
            sys.exit(1)
        report = load_json(args.file)
        _print_table(report)
        return

    all_files = find_result_files()

    if args.scale:
        show_scale_mode(all_files, args.metric)
        return

    # If --last is specified or no specific flag is given
    last_n = args.last if args.last is not None else 1

    if last_n == 1:
        if not all_files:
            print("Sonuç dosyası bulunamadı.")
            sys.exit(1)
        latest_file = all_files[-1]
        print(f"Gösterilen dosya: {latest_file}")
        report = load_json(latest_file)
        _print_table(report)
    else:
        show_last_n_mode(all_files, last_n)


if __name__ == "__main__":
    main()
