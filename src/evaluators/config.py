"""Configuration classes for evaluator implementations."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RetrievalEvaluatorConfig:
    """Configuration for RetrievalEvaluator.
    
    Attributes:
        k_values: A list of K cutoffs to calculate metrics like Precision@K and Recall@K.
    """
    
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    def __post_init__(self) -> None:
        """Validate k_values."""
        if not self.k_values:
            raise ValueError("k_values cannot be empty.")
            
        for k in self.k_values:
            if k <= 0:
                raise ValueError(f"All k values must be positive, got {k}")
