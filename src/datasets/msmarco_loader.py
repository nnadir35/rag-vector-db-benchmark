"""Streaming loader for the local MS MARCO passage-ranking TSV files."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from ..core.dataset import DatasetLoader
from ..core.types import Document, DocumentMetadata, Query
from .config import MSMARCODatasetConfig


class MSMARCOLoader(DatasetLoader):
    """Load a bounded, prefix-stable MS MARCO corpus from local TSV files.

    A selected query's complete qrels PID set is forced into the smallest
    requested corpus. Remaining slots are filled in collection-file order.
    Reusing the same files and query count therefore makes a smaller scale an
    exact prefix subset of every larger scale.
    """

    def __init__(self, config: MSMARCODatasetConfig) -> None:
        """Initialize the loader with local file paths and scale limits."""
        self._config = config
        self._selection: tuple[list[str], dict[str, set[str]]] | None = None

    @staticmethod
    def _open_text(path: str) -> TextIO:
        """Open a UTF-8 TSV or TSV.GZ file for streaming reads."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"MS MARCO input file not found: {path}")
        if file_path.suffix == ".gz":
            return gzip.open(file_path, mode="rt", encoding="utf-8")
        return file_path.open(mode="rt", encoding="utf-8")

    @classmethod
    def _rows(cls, path: str, fields: int) -> Iterator[list[str]]:
        """Yield validated TSV rows without materializing the source file."""
        with cls._open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                row = line.rstrip("\n").split("\t", maxsplit=fields - 1)
                if len(row) < fields:
                    raise ValueError(f"Malformed TSV row {line_number} in {path}")
                yield row

    def _ensure_selection(self) -> tuple[list[str], dict[str, set[str]]]:
        """Select deterministic query IDs and all of their relevant passage IDs."""
        if self._selection is not None:
            return self._selection

        ordered_query_ids: list[str] = []
        qrels: dict[str, set[str]] = {}
        for row in self._rows(self._config.qrels_path, fields=4):
            query_id, _, passage_id, relevance = row
            if relevance != "1":
                continue
            if query_id not in qrels and len(ordered_query_ids) >= self._config.num_queries:
                continue
            if query_id not in qrels:
                ordered_query_ids.append(query_id)
                qrels[query_id] = set()
            qrels[query_id].add(passage_id)

        if len(ordered_query_ids) < self._config.num_queries:
            raise ValueError(
                f"qrels contains only {len(ordered_query_ids)} relevant queries; "
                f"need {self._config.num_queries}."
            )
        required_count = len(set().union(*qrels.values()))
        if required_count > self._config.max_documents:
            raise ValueError(
                f"{required_count} gold passages exceed max_documents="
                f"{self._config.max_documents}."
            )
        self._selection = ordered_query_ids, qrels
        return self._selection

    def load(self) -> tuple[list[Query], dict[str, set[str]]]:
        """Load the fixed qrels-backed MS MARCO query subset from local TSV."""
        selected_ids, qrels = self._ensure_selection()
        wanted = set(selected_ids)
        query_texts: dict[str, str] = {}
        for query_id, text in self._rows(self._config.queries_path, fields=2):
            if query_id in wanted:
                query_texts[query_id] = text
                if len(query_texts) == len(wanted):
                    break
        missing = wanted.difference(query_texts)
        if missing:
            raise ValueError(f"queries file lacks {len(missing)} qrels query IDs.")
        queries = [
            Query(id=query_id, text=query_texts[query_id], metadata={"source": "msmarco"})
            for query_id in selected_ids
        ]
        return queries, {query_id: set(qrels[query_id]) for query_id in selected_ids}

    def load_documents(self) -> list[Document]:
        """Stream only the selected prefix-sized corpus into memory.

        Required qrels passages are retained wherever they occur in the input;
        non-required passages are admitted in collection order until the target
        scale is reached. Document order is deterministic and prefix-stable.
        """
        _, qrels = self._ensure_selection()
        required_ids = set().union(*qrels.values())
        required_documents: dict[str, Document] = {}
        filler_documents: list[Document] = []
        filler_limit = self._config.max_documents - len(required_ids)

        for passage_id, text in self._rows(self._config.collection_path, fields=2):
            if passage_id in required_ids:
                required_documents[passage_id] = Document(
                    id=passage_id,
                    content=text,
                    metadata=DocumentMetadata(source="msmarco", custom={"pid": passage_id}),
                )
            elif len(filler_documents) < filler_limit:
                filler_documents.append(
                    Document(
                        id=passage_id,
                        content=text,
                        metadata=DocumentMetadata(source="msmarco", custom={"pid": passage_id}),
                    )
                )
            if len(required_documents) == len(required_ids) and len(filler_documents) == filler_limit:
                break

        missing = required_ids.difference(required_documents)
        if missing:
            raise ValueError(f"collection file lacks {len(missing)} qrels passage IDs.")
        documents = list(required_documents.values()) + filler_documents
        if len(documents) != self._config.max_documents:
            raise ValueError(
                f"collection contains only {len(documents)} usable passages; "
                f"need {self._config.max_documents}."
            )
        return documents

    def load_gold_answers(self) -> dict[str, list[str]]:
        """Return empty answer lists because MS MARCO qrels have no short answers."""
        query_ids, _ = self._ensure_selection()
        return {query_id: [] for query_id in query_ids}
