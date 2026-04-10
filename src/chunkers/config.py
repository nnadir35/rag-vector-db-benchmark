"""Configuration classes for chunker implementations.

This module defines configuration dataclasses for various chunker
implementations.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FixedSizeChunkerConfig:
    """Configuration for FixedSizeChunker.

    This configuration defines the parameters for fixed-size character chunking.

    Attributes:
        chunk_size: The number of characters in each chunk. Defaults to 512.
        overlap: The number of characters to overlap with the previous chunk. Defaults to 50.
    """

    chunk_size: int = field(default=512)
    overlap: int = field(default=50)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )
