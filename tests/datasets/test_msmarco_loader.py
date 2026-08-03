"""Tests for the streaming local MS MARCO loader."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from src.datasets.config import MSMARCODatasetConfig
from src.datasets.msmarco_loader import MSMARCOLoader


def _write_fixture_files(tmp_path: Path, compressed: bool = False) -> tuple[str, str, str]:
    """Create a small MS MARCO-shaped fixture corpus."""
    suffix = ".tsv.gz" if compressed else ".tsv"
    collection = tmp_path / f"collection{suffix}"
    queries = tmp_path / f"queries{suffix}"
    qrels = tmp_path / f"qrels{suffix}"
    open_file = gzip.open if compressed else open
    with open_file(collection, "wt", encoding="utf-8") as handle:
        handle.write("1\tpassage one\n2\tpassage two\n3\tpassage three\n4\tpassage four\n5\tpassage five\n")
    with open_file(queries, "wt", encoding="utf-8") as handle:
        handle.write("q1\tfirst query\nq2\tsecond query\nq3\tthird query\n")
    with open_file(qrels, "wt", encoding="utf-8") as handle:
        handle.write("q1\t0\t4\t1\nq2\t0\t5\t1\nq3\t0\t3\t1\n")
    return str(collection), str(queries), str(qrels)


def test_s1_loads_exactly_1000_and_contains_all_selected_qrels(tmp_path: Path) -> None:
    """The S1 invariant holds without loading a full collection into memory."""
    collection = tmp_path / "collection.tsv"
    queries = tmp_path / "queries.tsv"
    qrels = tmp_path / "qrels.tsv"
    collection.write_text("".join(f"{pid}\tpassage {pid}\n" for pid in range(1, 1_201)), encoding="utf-8")
    queries.write_text("".join(f"q{i}\tquery {i}\n" for i in range(500)), encoding="utf-8")
    qrels.write_text("".join(f"q{i}\t0\t{700 + i}\t1\n" for i in range(500)), encoding="utf-8")
    loader = MSMARCOLoader(MSMARCODatasetConfig(str(collection), str(queries), str(qrels)))

    selected_queries, ground_truth = loader.load()
    documents = loader.load_documents()

    assert len(documents) == 1_000
    assert len(selected_queries) == 500
    document_ids = {document.id for document in documents}
    assert all(ground_truth[query.id].issubset(document_ids) for query in selected_queries)


def test_larger_scale_has_smaller_scale_as_exact_prefix(tmp_path: Path) -> None:
    """S1 stays a deterministic subset when the target corpus grows."""
    collection, queries, qrels = _write_fixture_files(tmp_path)
    config = dict(collection_path=collection, queries_path=queries, qrels_path=qrels, num_queries=2)
    s1 = MSMARCOLoader(MSMARCODatasetConfig(**config, max_documents=4)).load_documents()
    s2 = MSMARCOLoader(MSMARCODatasetConfig(**config, max_documents=5)).load_documents()

    assert [document.id for document in s2[:4]] == [document.id for document in s1]


def test_loader_reads_gzip_tsv_files(tmp_path: Path) -> None:
    """Gzip-compressed TSV inputs use the same streaming parser."""
    collection, queries, qrels = _write_fixture_files(tmp_path, compressed=True)
    loader = MSMARCOLoader(
        MSMARCODatasetConfig(collection, queries, qrels, max_documents=4, num_queries=2)
    )

    selected_queries, ground_truth = loader.load()

    assert [query.id for query in selected_queries] == ["q1", "q2"]
    assert ground_truth == {"q1": {"4"}, "q2": {"5"}}


def test_config_rejects_more_gold_passages_than_target_scale(tmp_path: Path) -> None:
    """A scale too small for its mandatory qrels passages fails clearly."""
    collection, queries, qrels = _write_fixture_files(tmp_path)
    loader = MSMARCOLoader(
        MSMARCODatasetConfig(collection, queries, qrels, max_documents=1, num_queries=2)
    )

    with pytest.raises(ValueError, match="gold passages exceed"):
        loader.load()
