#!/usr/bin/env python3
"""Script to download and setup the MS MARCO passage ranking dataset.

Downloads:
  - queries.dev.small.tsv
  - qrels.dev.small.tsv
  - collection.tar.gz -> extracts collection.tsv

Destination: data/msmarco/
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

MSMARCO_BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
FILES = {
    "qrels.dev.small.tsv": f"{MSMARCO_BASE_URL}/qrels.dev.small.tsv",
    "collectionandqueries.tar.gz": f"{MSMARCO_BASE_URL}/collectionandqueries.tar.gz",
}


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with basic progress display."""
    print(f"Downloading {url} -> {dest_path}...")

    def _progress(count: int, block_size: int, total_size: int) -> None:
        percent = min(100, int(count * block_size * 100 / total_size)) if total_size > 0 else 0
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [{percent:3d}%] {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()


def main() -> None:
    target_dir = Path("data/msmarco")
    target_dir.mkdir(parents=True, exist_ok=True)

    collection_tsv = target_dir / "collection.tsv"
    queries_tsv = target_dir / "queries.dev.small.tsv"

    for filename, url in FILES.items():
        out_file = target_dir / filename

        if filename == "collectionandqueries.tar.gz" and collection_tsv.exists() and queries_tsv.exists():
            print(f"✅ {collection_tsv} and {queries_tsv} already exist, skipping archive download.")
            continue

        if out_file.exists():
            print(f"✅ {out_file} already exists, skipping download.")
            continue

        download_file(url, out_file)

    archive_path = target_dir / "collectionandqueries.tar.gz"
    if archive_path.exists() and not (collection_tsv.exists() and queries_tsv.exists()):
        print(f"Extracting {archive_path} into {target_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"✅ Extracted files into {target_dir}")

    print("\n✅ MS MARCO passage dataset setup complete!")
    print(f"Files in {target_dir}:")
    for f in target_dir.iterdir():
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
