"""SQuAD dataset loader implementation.

This module provides a loader for the Stanford Question Answering Dataset (SQuAD),
fetching it via the HuggingFace datasets library and converting it into the
framework's core Data types (Query, Document).
"""

import hashlib

from ..core.types import Document, DocumentMetadata, Query
from .config import SQuADDatasetConfig


class SQuADLoader:
    """Stanford Question Answering Dataset (SQuAD) loader.

    This loader fetches data from HuggingFace datasets and maps them into
    the internal Document and Query structures. It accurately connects questions
    with their overarching context paragraphs to establish ground truth mappings.
    """

    def __init__(self, config: SQuADDatasetConfig) -> None:
        """Initialize the SQuAD loader.

        Args:
            config: Configuration detailing split, sample limits, and versions.
        """
        self._config = config
        self._dataset = None

    def _ensure_dataset_loaded(self) -> None:
        """Ensure the dataset is loaded from HuggingFace datasets."""
        if self._dataset is not None:
            return

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load SQuAD. "
                "Install it with: pip install datasets"
            )

        try:
            dataset = load_dataset(self._config.version, split=self._config.split)

            if self._config.max_samples is not None:
                # Limit the dataset if max_samples is specified
                dataset = dataset.select(range(min(self._config.max_samples, len(dataset))))

            self._dataset = dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load SQuAD dataset: {e}") from e

    def _generate_context_id(self, context_text: str) -> str:
        """Generate a deterministic ID for a context paragraph.

        Args:
            context_text: The text string of the paragraph.

        Returns:
            A deterministic string ID derived from hashing the context text.
        """
        hash_obj = hashlib.md5(context_text.encode("utf-8"))
        return f"squad_{hash_obj.hexdigest()}"

    def load(self) -> tuple[list[Query], dict[str, set[str]]]:
        """Load queries and their corresponding ground truth context IDs.

        Each question in SQuAD targets a specific context paragraph. The ground
        truth dict maps the question's query ID to a set of valid context IDs.

        Returns:
            A tuple containing:
            - A list of Query objects representing the questions.
            - A dictionary mapping from query ID to a set of context IDs.

        Raises:
            RuntimeError: If dataset loading or processing fails.
        """
        self._ensure_dataset_loaded()

        queries: list[Query] = []
        ground_truth: dict[str, set[str]] = {}

        try:
            for item in self._dataset:
                query_id = str(item["id"])
                question_text = str(item["question"])
                context_text = str(item["context"])

                query = Query(
                    id=query_id,
                    text=question_text,
                    metadata={"source": "squad"}
                )
                queries.append(query)

                context_id = self._generate_context_id(context_text)

                if query_id not in ground_truth:
                    ground_truth[query_id] = set()
                ground_truth[query_id].add(context_id)

            return queries, ground_truth
        except Exception as e:
            raise RuntimeError(f"Error processing SQuAD questions: {e}") from e

    def load_documents(self) -> list[Document]:
        """Load all unique context paragraphs as Document objects.

        Iterates over the dataset to extract unique context paragraphs, converting
        them into Document instances suitable for ingestion.

        Returns:
            A list of Document objects representing SQuAD contexts.

        Raises:
            RuntimeError: If dataset loading or processing fails.
        """
        self._ensure_dataset_loaded()

        documents: list[Document] = []
        seen_contexts: set[str] = set()

        try:
            for item in self._dataset:
                context_text = str(item["context"])

                # SQuAD often repeats contexts across multiple queries. Keep unique ones.
                if context_text in seen_contexts:
                    continue

                seen_contexts.add(context_text)

                context_id = self._generate_context_id(context_text)

                metadata = DocumentMetadata(
                    source="squad",
                    custom={"squad_version": self._config.version}
                )

                doc = Document(
                    id=context_id,
                    content=context_text,
                    metadata=metadata
                )
                documents.append(doc)

            return documents
        except Exception as e:
            raise RuntimeError(f"Error processing SQuAD documents: {e}") from e
