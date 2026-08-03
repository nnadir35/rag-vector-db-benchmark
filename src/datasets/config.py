"""Configuration objects for dataset loaders."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SQuADDatasetConfig:
    """Configuration for loading the SQuAD dataset.

    Attributes:
        split: The dataset split to load (e.g., "train", "validation").
        max_samples: Optional maximum number of samples to load. If None, loads all.
        version: The version/name of the dataset in HuggingFace datasets library.
    """

    split: str = field(default="validation")
    max_samples: int | None = field(default=None)
    version: str = field(default="squad_v2")

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.split:
            raise ValueError("split cannot be empty.")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")

        if not self.version:
            raise ValueError("version cannot be empty.")


@dataclass(frozen=True)
class MSMARCODatasetConfig:
    """Configuration for a local MS MARCO passage-ranking dataset.

    The three input files may be either plain TSV or gzip-compressed TSV files.
    ``max_documents`` is deliberately required: the full MS MARCO collection is
    too large to materialize in memory for a benchmark run.
    """

    collection_path: str
    queries_path: str
    qrels_path: str
    max_documents: int = 1_000
    num_queries: int = 500

    def __post_init__(self) -> None:
        """Validate paths and bounded corpus/query sizes."""
        if not self.collection_path or not self.queries_path or not self.qrels_path:
            raise ValueError("collection_path, queries_path, and qrels_path cannot be empty.")
        if self.max_documents <= 0:
            raise ValueError("max_documents must be positive.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")
