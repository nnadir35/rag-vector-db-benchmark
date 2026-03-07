"""Configuration classes for embedder implementations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SentenceTransformersEmbedderConfig:
    """Configuration for SentenceTransformersEmbedder.
    
    Attributes:
        model_name: The HuggingFace model identifier or local path.
        device: The device to run the model on ('cpu', 'cuda', etc.).
        batch_size: Batch size used for encoding multiple chunks.
        normalize_embeddings: Whether to normalize embeddings to unit length.
    """
    
    model_name: str = field(default="all-MiniLM-L6-v2")
    device: str = field(default="cpu")
    batch_size: int = field(default=32)
    normalize_embeddings: bool = field(default=True)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")
            
        if not self.device:
            raise ValueError("device cannot be empty.")
            
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
