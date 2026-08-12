from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkDTO:
    """Data Transfer Object for a text chunk."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[ChunkDTO]:
        """
        Splits text into chunks.
        
        Args:
            text: The raw text to chunk.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            A list of ChunkDTO objects.
        """
        pass
