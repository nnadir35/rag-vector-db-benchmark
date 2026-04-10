"""Tests for SentenceTransformersEmbedder."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import Chunk, ChunkMetadata
from src.embedders.config import SentenceTransformersEmbedderConfig
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder


@pytest.fixture
def mock_sentence_transformers():
    """Mock the sentence_transformers module."""
    mock_st = MagicMock()
    mock_model = MagicMock()

    mock_model.get_sentence_embedding_dimension.return_value = 384

    def mock_encode(sentences, **kwargs):
        if isinstance(sentences, str):
            return [[0.1] * 384]
        return [[0.1] * 384 for _ in sentences]

    mock_model.encode.side_effect = mock_encode
    mock_st.SentenceTransformer.return_value = mock_model

    with patch.dict(sys.modules, {'sentence_transformers': mock_st}):
        yield mock_st, mock_model


def test_embed_chunk_returns_correct_dimension(mock_sentence_transformers):
    """Test that embedding a single chunk works and returns correct dimension."""
    mock_st, mock_model = mock_sentence_transformers
    config = SentenceTransformersEmbedderConfig()
    embedder = SentenceTransformersEmbedder(config)

    chunk = Chunk(
        id="c1",
        content="Test chunk content",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=18)
    )

    embedding = embedder.embed_chunk(chunk)

    assert embedding.dimension == 384
    assert len(embedding.vector) == 384


def test_embed_chunks_batch_returns_same_count_as_input(mock_sentence_transformers):
    """Test that embedding multiple chunks passes arguments correctly downstream."""
    mock_st, mock_model = mock_sentence_transformers
    config = SentenceTransformersEmbedderConfig(batch_size=2)
    embedder = SentenceTransformersEmbedder(config)

    chunks = [
        Chunk(
            id=f"c{i}",
            content=f"content {i}",
            metadata=ChunkMetadata(document_id="d1", chunk_index=i, start_char=0, end_char=1)
        )
        for i in range(5)
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert len(embeddings) == 5
    args, kwargs = mock_model.encode.call_args
    assert len(args[0]) == 5
    assert kwargs.get("batch_size") == 2
    assert kwargs.get("normalize_embeddings") is True


def test_get_dimension_is_consistent_with_embeddings(mock_sentence_transformers):
    """Test dimension fetch is cached after first read."""
    mock_st, mock_model = mock_sentence_transformers
    config = SentenceTransformersEmbedderConfig()
    embedder = SentenceTransformersEmbedder(config)

    dim = embedder.get_dimension()
    assert dim == 384

    # Change mock setup to ensure cache is used
    mock_model.get_sentence_embedding_dimension.return_value = 999
    assert embedder.get_dimension() == 384  # still 384


def test_model_is_lazy_loaded_on_first_use(mock_sentence_transformers):
    """Test that SentenceTransformer initialization only runs when actually needed."""
    mock_st, mock_model = mock_sentence_transformers
    config = SentenceTransformersEmbedderConfig(model_name="test-model", device="cuda")

    # Instantiation shouldn't load
    embedder = SentenceTransformersEmbedder(config)
    mock_st.SentenceTransformer.assert_not_called()

    # Usage should load
    embedder.get_dimension()
    mock_st.SentenceTransformer.assert_called_once_with("test-model", device="cuda")


def test_import_error_when_sentence_transformers_not_installed():
    """Test behavior when the optional sentence_transformers dependency is missing."""
    with patch.dict(sys.modules, {'sentence_transformers': None}):
        config = SentenceTransformersEmbedderConfig()
        embedder = SentenceTransformersEmbedder(config)
        with pytest.raises(ImportError, match="sentence_transformers package is required"):
            embedder.get_dimension()
