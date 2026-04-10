"""Universal generator implementation using LiteLLM.

This module provides a flexible implementation of the Generator interface
that uses the LiteLLM library to support a wide variety of LLM providers.
"""

import time
from collections.abc import Sequence

from ..core.generation import Generator
from ..core.types import GenerationResult, Query, RetrievedChunk
from .config import UniversalGeneratorConfig


class UniversalGenerator(Generator):
    """LiteLLM-based universal generator implementation.

    This generator formats queries and retrieved contexts into prompts
    and uses LiteLLM to call any supported local or remote LLM (e.g.,
    Ollama, OpenAI, Groq).
    """

    def __init__(self, config: UniversalGeneratorConfig) -> None:
        """Initialize the UniversalGenerator.

        Args:
            config: Configuration for model, temperature, endpoints, etc.
        """
        self._config = config

    def _ensure_litellm_imported(self) -> None:
        """Ensure litellm is installed and imported lazily."""
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm package is required for UniversalGenerator. "
                "Install it with: pip install litellm"
            )

    def _format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string."""
        formatted_chunks = []
        for i, rc in enumerate(chunks, 1):
            formatted_chunks.append(f"[{i}] {rc.chunk.content}")
        return "\n".join(formatted_chunks)

    def generate(
        self,
        query: Query,
        retrieved_chunks: Sequence[RetrievedChunk],
        **kwargs: object,
    ) -> GenerationResult:
        """Generate a response using the configured LLM.

        Args:
            query: The user query.
            retrieved_chunks: The context chunks.
            **kwargs: Extra parameters (merged with config).

        Returns:
            GenerationResult wrapper containing response and usage metadata.

        Raises:
            ImportError: If litellm is missing.
            RuntimeError: If API call fails.
        """
        self._ensure_litellm_imported()
        from litellm import completion

        context_str = self._format_context(retrieved_chunks)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query.text}"

        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        provider = self._config.model_name.split("/")[0] if "/" in self._config.model_name else "unknown"

        start_time = time.time()

        try:
            response = completion(
                model=self._config.model_name,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed via litellm: {e}") from e

        latency = time.time() - start_time

        content = response.choices[0].message.content or ""

        # Extract usage stats Safely
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        metadata = {
            "model_name": self._config.model_name,
            "provider": provider,
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        return GenerationResult(
            query=query,
            response=content,
            retrieved_chunks=retrieved_chunks,
            metadata=metadata
        )
