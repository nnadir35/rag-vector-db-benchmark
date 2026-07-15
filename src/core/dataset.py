"""Interface for dataset loading.

Dataset loaders are responsible for fetching a QA (or retrieval-augmented
generation benchmark) dataset from its source and converting it into the
framework's core data types (Document, Query) plus ground-truth mappings.
"""

from abc import ABC, abstractmethod

from .types import Document, Query


class DatasetLoader(ABC):
    """Abstract interface for dataset loading.

    Implementations must handle:
    - Fetching/parsing the raw dataset from its native format or API
    - Converting questions into Query objects
    - Converting source passages/contexts into Document objects
    - Producing a ground-truth mapping from query ID to relevant document IDs
    - Producing gold answer text(s) per query ID, when available, for
      reference-based generation metrics (e.g. Exact Match, F1)

    Implementations must NOT:
    - Perform chunking, embedding, retrieval, or generation
    - Cache results across unrelated experiment runs (each instance should
      reflect one config: one split, one sample limit, one version)
    """

    @abstractmethod
    def load(self) -> tuple[list[Query], dict[str, set[str]]]:
        """Load queries and their ground-truth relevant document ID sets.

        Returns:
            A tuple of (queries, ground_truth) where ground_truth maps each
            query's id to the set of document ids considered relevant/correct.
        """
        pass

    @abstractmethod
    def load_documents(self) -> list[Document]:
        """Load the full corpus of unique source documents.

        Returns:
            A list of Document objects to be chunked, embedded, and indexed.
        """
        pass

    @abstractmethod
    def load_gold_answers(self) -> dict[str, list[str]]:
        """Load acceptable gold answer text(s) per query id, if available.

        Returns:
            A mapping from query id to a list of acceptable answer strings.
            Datasets without extractable short-answer ground truth (e.g.
            long-form or dialogue datasets) may return an empty list per
            query id, signaling that reference-based metrics like EM/F1
            should be skipped for that query.
        """
        pass
