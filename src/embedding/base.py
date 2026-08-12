from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingDTO:
    """Data Transfer Object for a vector embedding."""
    vector: Sequence[float]
    dimension: int

class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingDTO:
        """
        Embeds a single text string.
        
        Args:
            text: The text to embed.
            
        Returns:
            An EmbeddingDTO object containing the vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingDTO]:
        """
        Embeds a batch of text strings efficiently.
        
        Args:
            texts: A list of text strings to embed.
            
        Returns:
            A list of EmbeddingDTO objects.
        """
        pass
