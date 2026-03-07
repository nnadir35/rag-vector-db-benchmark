"""Tests for UniversalGenerator."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import Chunk, ChunkMetadata, Query, RetrievedChunk
from src.generators.config import UniversalGeneratorConfig
from src.generators.universal_generator import UniversalGenerator


@pytest.fixture
def sample_query():
    return Query(id="q1", text="Who is John?")


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            chunk=Chunk(
                id="c1",
                content="John is a software engineer.",
                metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=28)
            ),
            score=0.9,
            rank=0
        )
    ]


@pytest.fixture
def mock_litellm():
    """Mock the litellm library and completion call."""
    mock_llm = MagicMock()
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "John is an engineer."
    mock_response.choices = [mock_choice]
    
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 5
    mock_response.usage = mock_usage
    
    mock_llm.completion.return_value = mock_response
    
    with patch.dict(sys.modules, {'litellm': mock_llm}):
        yield mock_llm


def test_ollama_provider_call(mock_litellm, sample_query, sample_chunks):
    """Test using UniversalGenerator with Ollama model format."""
    config = UniversalGeneratorConfig(
        model_name="ollama/llama3.1",
        api_base="http://localhost:11434"
    )
    generator = UniversalGenerator(config)
    
    res = generator.generate(sample_query, sample_chunks)
    
    # Check if mock was called with corresponding model info
    mock_litellm.completion.assert_called_once()
    kwargs = mock_litellm.completion.call_args[1]
    
    assert kwargs["model"] == "ollama/llama3.1"
    assert kwargs["api_base"] == "http://localhost:11434"
    
    # Check that metadata has correct provider
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["prompt_tokens"] == 15
    assert res.metadata["completion_tokens"] == 5
    assert "John is a software engineer." in kwargs["messages"][1]["content"]


def test_groq_provider_call(mock_litellm, sample_query, sample_chunks):
    """Test using UniversalGenerator with Groq model format."""
    config = UniversalGeneratorConfig(
        model_name="groq/llama-3.1-8b",
        api_key="fake-groq-key"
    )
    generator = UniversalGenerator(config)
    
    res = generator.generate(sample_query, sample_chunks)
    
    kwargs = mock_litellm.completion.call_args[1]
    assert kwargs["model"] == "groq/llama-3.1-8b"
    assert kwargs["api_key"] == "fake-groq-key"
    assert res.metadata["provider"] == "groq"


def test_import_error_when_litellm_not_installed(sample_query, sample_chunks):
    """Test that missing litellm throws correct ImportError."""
    with patch.dict(sys.modules, {'litellm': None}):
        config = UniversalGeneratorConfig()
        generator = UniversalGenerator(config)
        with pytest.raises(ImportError, match="litellm package is required"):
            generator.generate(sample_query, sample_chunks)
