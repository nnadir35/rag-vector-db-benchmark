#!/usr/bin/env python3
"""Compare per-query retrieved documents between two benchmark result JSON files."""

import argparse
import json
import sys


def compare_results(file1_path: str, file2_path: str) -> None:
    with open(file1_path, encoding="utf-8") as f:
        data1 = json.load(f)

    with open(file2_path, encoding="utf-8") as f:
        data2 = json.load(f)

    qids1 = data1.get("query_ids_used", [])
    qids2 = data2.get("query_ids_used", [])

    print("=" * 80)
    print("PER-QUERY CONSISTENCY CHECK")
    print("=" * 80)
    print(f"File 1: {file1_path} ({len(qids1)} queries)")
    print(f"File 2: {file2_path} ({len(qids2)} queries)")

    common_qids = [q for q in qids1 if q in set(qids2)]
    print(f"Common query count: {len(common_qids)}")

    if len(qids1) >= 20 and len(qids2) >= 20:
        first_20_match = qids1[:20] == qids2[:20]
        print(f"First 20 query IDs identical: {'YES' if first_20_match else 'NO'}")

    db_names = ["chroma", "qdrant", "faiss", "milvus", "elasticsearch", "weaviate"]
    found_any = False
    for db_name in db_names:
        key = f"per_query_details_{db_name}"
        details1 = data1.get(key)
        details2 = data2.get(key)
        if not details1 or not details2:
            continue
        found_any = True

        print("\n" + "-" * 80)
        print(f"DB: {db_name}")
        print("-" * 80)

        # Map query_id -> retrieved info
        map1 = {item["query_id"]: item.get("retrieved", []) for item in details1}
        map2 = {item["query_id"]: item.get("retrieved", []) for item in details2}

        mismatches = 0
        for qid in common_qids:
            r1 = [chunk["chunk_id"] for chunk in map1.get(qid, [])]
            r2 = [chunk["chunk_id"] for chunk in map2.get(qid, [])]
            if r1 != r2:
                mismatches += 1
                print(f"\n[Mismatch] Query ID: {qid}")
                print(f"  File 1 retrieved: {r1}")
                print(f"  File 2 retrieved: {r2}")

        if mismatches == 0:
            print(f"✅ All {len(common_qids)} common queries produced EXACT SAME retrieved chunk sequence!")
        else:
            print(f"❌ Mismatches found in {mismatches}/{len(common_qids)} common queries.")

    if not found_any:
        print("\nNote: Detailed per-query retrieved chunks not stored in both JSON files.")
        print("To enable full per-query document diffing, run benchmark_db.py with --dump-per-query.")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-query retrieved documents between two JSON result files.")
    parser.add_argument("file1", type=str, help="Path to first result JSON")
    parser.add_argument("file2", type=str, help="Path to second result JSON")
    args = parser.parse_args()

    compare_results(args.file1, args.file2)


if __name__ == "__main__":
    main()
