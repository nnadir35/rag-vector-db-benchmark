"""Configuration for SQuAD dataset loader."""

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
