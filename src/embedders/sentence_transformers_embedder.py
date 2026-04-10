"""SentenceTransformers embedder implementation.

This module provides a concrete implementation of the Embedder interface
using the SentenceTransformers library.
"""

from collections.abc import Sequence
from typing import Any

from ..core.embedding import Embedder
from ..core.types import Chunk, Embedding, Query
from .config import SentenceTransformersEmbedderConfig


class SentenceTransformersEmbedder(Embedder):
    """SentenceTransformers-based embedder implementation.

    This embedder uses the sentence_transformers library to encode text
    into dense vector embeddings. It supports batch processing and lazy
    loading of the models.
    """

    def __init__(self, config: SentenceTransformersEmbedderConfig) -> None:
        """Initialize the embedder.

        Args:
            config: Configuration for the embedder.
        """
        self._config = config
        self._model: Any | None = None
        self._dimension: int | None = None

    def _ensure_imported(self) -> None:
        """Ensure sentence_transformers is installed and imported."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentence_transformers package is required for SentenceTransformersEmbedder. "
                "Install it with: pip install sentence-transformers"
            )

    def _get_model(self) -> Any:
        """Get or lazily load the SentenceTransformer model.

        Returns:
            The loaded SentenceTransformer model instance.

        Raises:
            ImportError: If sentence_transformers is not installed.
            RuntimeError: If model loading fails.
        """
        if self._model is None:
            self._ensure_imported()
            import sentence_transformers

            try:
                self._model = sentence_transformers.SentenceTransformer(
                    self._config.model_name,
                    device=self._config.device
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self._config.model_name}': {e}"
                ) from e

        return self._model

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder.

        Returns:
            The dimension of embedding vectors.
        """
        if self._dimension is None:
            model = self._get_model()
            try:
                self._dimension = model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(f"Failed to get model dimension: {e}") from e

        return self._dimension

    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """Embed a single chunk.

        Args:
            chunk: The chunk to embed.

        Returns:
            Embedding vector for the chunk.

        Raises:
            ValueError: If chunk is invalid.
            RuntimeError: If embedding fails.
        """
        if not chunk or not chunk.content:
            raise ValueError("Invalid chunk or chunk content is empty.")

        embeddings = self.embed_chunks([chunk])
        return embeddings[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        """Embed multiple chunks in batch.

        Args:
            chunks: Sequence of chunks to embed.

        Returns:
            Sequence of embeddings, one per chunk, in the same order.

        Raises:
            ValueError: If any chunk is invalid.
            RuntimeError: If embedding fails.
        """
        if not chunks:
            return []

        texts = []
        for chunk in chunks:
            if not chunk or not chunk.content:
                raise ValueError(f"Invalid chunk found in batch: {chunk}")
            texts.append(chunk.content)

        model = self._get_model()
        dimension = self.get_dimension()

        try:
            vectors = model.encode(
                texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize_embeddings,
                device=self._config.device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to embed chunks: {e}") from e

        result = []
        for vector in vectors:
            float_vector = [float(v) for v in vector]
            result.append(Embedding(vector=float_vector, dimension=dimension))

        return result

    def embed_query(self, query: Query) -> Embedding:
        """Embed a query.

        Args:
            query: The query to embed.

        Returns:
            Embedding vector for the query.

        Raises:
            ValueError: If query is invalid.
            RuntimeError: If embedding fails.
        """
        if not query or not query.text:
            raise ValueError("Invalid query or query text is empty.")

        model = self._get_model()
        dimension = self.get_dimension()

        try:
            vector = model.encode(
                [query.text],
                batch_size=1,
                normalize_embeddings=self._config.normalize_embeddings,
                device=self._config.device,
            )[0]
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}") from e

        float_vector = [float(v) for v in vector]
        return Embedding(vector=float_vector, dimension=dimension)
