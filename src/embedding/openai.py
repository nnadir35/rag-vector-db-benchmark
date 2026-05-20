import os
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from .base import BaseEmbedder, EmbeddingDTO

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding implementation with rate-limit protection."""
    
    def __init__(self, api_key: str | None = None, model_name: str = "text-embedding-3-small") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via constructor or OPENAI_API_KEY env var.")
        
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI, RateLimitError
                self._client = OpenAI(api_key=self.api_key)
                self.RateLimitError = RateLimitError
            except ImportError as e:
                raise ImportError("OpenAIEmbedder requires openai package. Install with: uv add openai") from e
        return self._client

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _call_api_with_retry(self, input_data: list[str]) -> list[list[float]]:
        client = self._get_client()
        # Retry only on specific RateLimitError implicitly handled by Tenacity if we specified retry_if_exception_type, 
        # but to avoid import errors at module load, we retry generally for OpenAI API failures that bubble up.
        response = client.embeddings.create(
            input=input_data,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> EmbeddingDTO:
        vectors = self._call_api_with_retry([text])
        return EmbeddingDTO(vector=vectors[0], dimension=len(vectors[0]))

    def embed_batch(self, texts: List[str]) -> List[EmbeddingDTO]:
        if not texts:
            return []
        vectors = self._call_api_with_retry(texts)
        return [EmbeddingDTO(vector=vec, dimension=len(vec)) for vec in vectors]
