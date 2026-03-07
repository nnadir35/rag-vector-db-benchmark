"""Tests for SQuADLoader."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.datasets.config import SQuADDatasetConfig
from src.datasets.squad_loader import SQuADLoader


@pytest.fixture
def mock_datasets():
    """Mock the HuggingFace datasets library."""
    mock_datasets_module = MagicMock()
    
    # Create a mock dataset
    mock_dataset = MagicMock()
    mock_data = [
        {"id": "q1", "question": "What is AI?", "context": "AI is artificial intelligence."},
        {"id": "q2", "question": "What does AI stand for?", "context": "AI is artificial intelligence."},
        {"id": "q3", "question": "Are cats cute?", "context": "Yes, cats are very cute."},
    ]
    
    # Make the mock dataset iterable
    mock_dataset.__iter__.return_value = iter(mock_data)
    mock_dataset.__len__.return_value = len(mock_data)
    
    # Mock the select logic for max_samples
    def mock_select(indices):
        limited_data = [mock_data[i] for i in indices]
        subds = MagicMock()
        subds.__iter__.return_value = iter(limited_data)
        subds.__len__.return_value = len(limited_data)
        return subds
        
    mock_dataset.select.side_effect = mock_select
    mock_datasets_module.load_dataset.return_value = mock_dataset
    
    with patch.dict(sys.modules, {'datasets': mock_datasets_module}):
        yield mock_datasets_module, mock_dataset


def test_load_documents_extracts_unique_contexts(mock_datasets):
    """Test that duplicate contexts are deduplicated and mapped correctly."""
    mock_module, mock_dataset = mock_datasets
    config = SQuADDatasetConfig()
    loader = SQuADLoader(config)
    
    documents = loader.load_documents()
    
    # 3 total questions in mock, but 2 unique contexts.
    assert len(documents) == 2
    
    contexts = [doc.content for doc in documents]
    assert "AI is artificial intelligence." in contexts
    assert "Yes, cats are very cute." in contexts
    
    # IDs should start with squad_
    assert all(doc.id.startswith("squad_") for doc in documents)


def test_load_maps_questions_and_ground_truth(mock_datasets):
    """Test that questions are converted to Query objects and mapped to contexts."""
    mock_module, mock_dataset = mock_datasets
    config = SQuADDatasetConfig()
    loader = SQuADLoader(config)
    
    queries, ground_truth = loader.load()
    
    assert len(queries) == 3
    assert queries[0].id == "q1"
    assert queries[0].text == "What is AI?"
    
    assert "q1" in ground_truth
    assert "q2" in ground_truth
    assert "q3" in ground_truth
    
    # Questions 1 and 2 share the same context ID
    q1_truth = ground_truth["q1"].pop()
    q2_truth = ground_truth["q2"].pop()
    assert q1_truth == q2_truth
    assert q1_truth.startswith("squad_")


def test_max_samples_limits_returned_data(mock_datasets):
    """Test that max_samples properly restricts dataset size."""
    mock_module, mock_dataset = mock_datasets
    config = SQuADDatasetConfig(max_samples=2)
    loader = SQuADLoader(config)
    
    queries, ground_truth = loader.load()
    
    assert len(queries) == 2
    assert "q1" in ground_truth
    assert "q2" in ground_truth
    assert "q3" not in ground_truth


def test_import_error_when_datasets_not_installed():
    """Test behavior when the datasets dependency is missing."""
    with patch.dict(sys.modules, {'datasets': None}):
        config = SQuADDatasetConfig()
        loader = SQuADLoader(config)
        with pytest.raises(ImportError, match="The 'datasets' package is required"):
            loader.load()


def test_config_validates_split_and_version():
    """Test that configuration rejects invalid values."""
    with pytest.raises(ValueError, match="split cannot be empty"):
        SQuADDatasetConfig(split="")
        
    with pytest.raises(ValueError, match="version cannot be empty"):
        SQuADDatasetConfig(version="")
        
    with pytest.raises(ValueError, match="max_samples must be positive"):
        SQuADDatasetConfig(max_samples=0)
