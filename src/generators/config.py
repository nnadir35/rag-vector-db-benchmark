"""Configuration classes for generator implementations.

This module defines configuration dataclasses for various generator
implementations, designed to be serializable and loadable from
configuration files.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class UniversalGeneratorConfig:
    """Configuration for UniversalGenerator.
    
    This configuration encapsulates all settings needed to generate
    responses using an LLM via the litellm library, which supports
    various backends like Ollama, OpenAI, and Groq.
    
    Attributes:
        model_name: The target model name (e.g., "ollama/llama3.1", "gpt-4o").
        temperature: Sampling temperature for reproducible outputs (default: 0.0).
        max_tokens: Maximum number of tokens to generate.
        api_base: Optional base URL for API requests (useful for local models).
        api_key: Optional API key for external providers.
        system_prompt: System prompt instructions.
    """
    
    model_name: str = field(default="ollama/llama3.1")
    temperature: float = field(default=0.0)
    max_tokens: int = field(default=512)
    api_base: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    system_prompt: str = field(
        default="Answer based only on the provided context."
    )
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")
            
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
            
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
