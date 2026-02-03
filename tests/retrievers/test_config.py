"""Tests for retriever configuration classes."""

import os
from unittest.mock import patch

import pytest

from src.retrievers.config import PineconeRetrieverConfig


class TestPineconeRetrieverConfig:
    """Test PineconeRetrieverConfig."""
    
    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        config = PineconeRetrieverConfig(
            api_key="test-key",
            index_name="test-index",
            dimension=384,
        )
        assert config.api_key == "test-key"
        assert config.index_name == "test-index"
        assert config.dimension == 384
        assert config.metric == "cosine"  # Default
    
    @patch.dict(os.environ, {"PINECONE_API_KEY": "env-api-key"})
    def test_init_from_environment(self) -> None:
        """Test initialization reads API key from environment."""
        config = PineconeRetrieverConfig(
            index_name="test-index",
            dimension=384,
        )
        assert config.api_key == "env-api-key"
    
    def test_init_no_api_key(self) -> None:
        """Test initialization fails when no API key provided."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="api_key must be provided"):
                PineconeRetrieverConfig(
                    index_name="test-index",
                    dimension=384,
                )
    
    def test_init_invalid_metric(self) -> None:
        """Test initialization fails with invalid metric."""
        with pytest.raises(ValueError, match="metric must be one of"):
            PineconeRetrieverConfig(
                api_key="test-key",
                index_name="test-index",
                dimension=384,
                metric="invalid-metric",
            )
    
    def test_init_invalid_dimension(self) -> None:
        """Test initialization fails with invalid dimension."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            PineconeRetrieverConfig(
                api_key="test-key",
                index_name="test-index",
                dimension=-1,
            )
    
    def test_init_with_namespace(self) -> None:
        """Test initialization with namespace."""
        config = PineconeRetrieverConfig(
            api_key="test-key",
            index_name="test-index",
            dimension=384,
            namespace="test-namespace",
        )
        assert config.namespace == "test-namespace"
    
    def test_init_with_timeout(self) -> None:
        """Test initialization with timeout."""
        config = PineconeRetrieverConfig(
            api_key="test-key",
            index_name="test-index",
            dimension=384,
            timeout=30.0,
        )
        assert config.timeout == 30.0
    
    def test_init_all_metrics(self) -> None:
        """Test initialization with all valid metrics."""
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        for metric in valid_metrics:
            config = PineconeRetrieverConfig(
                api_key="test-key",
                index_name="test-index",
                dimension=384,
                metric=metric,
            )
            assert config.metric == metric
