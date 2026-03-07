"""Tests for config parsing and object creation."""

import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.config_loader import build_component_configs, load_yaml


@pytest.fixture
def sample_yaml_content():
    return """
experiment:
  name: test_run
generator:
  model_name: "test/model"
  temperature: 0.5
pipeline:
  top_k: 10
evaluator:
  k_values: [1, 5]
"""


def test_load_yaml_success(sample_yaml_content):
    """Test standard YAML file parsing."""
    with patch("builtins.open", mock_open(read_data=sample_yaml_content)):
        data = load_yaml("fake_path.yaml")
        assert data["experiment"]["name"] == "test_run"
        assert data["generator"]["temperature"] == 0.5


def test_load_yaml_missing_pyyaml():
    """Test behavior when PyYAML is missing."""
    with patch.dict(sys.modules, {'yaml': None}):
        with pytest.raises(ImportError, match="PyYAML"):
            load_yaml("fake.yaml")


def test_build_component_configs():
    """Test mapping raw dict dicts to valid frozen dataclasses."""
    raw_dict = {
        "experiment": {"name": "hello_rag"},
        "generator": {"model_name": "ollama/llama3.1", "temperature": 0.0, "max_tokens": 100},
        "pipeline": {"top_k": 3},
        "dataset": {"max_samples": 5}
    }
    
    exp_config = build_component_configs(raw_dict)
    
    # Check experiment name
    assert exp_config.name == "hello_rag"
    
    # Check populated class fields
    assert exp_config.generator.model_name == "ollama/llama3.1"
    assert exp_config.generator.max_tokens == 100
    assert exp_config.pipeline.top_k == 3
    assert exp_config.dataset.max_samples == 5
    
    # Check defaults handled properly where missing
    assert exp_config.evaluator.k_values == [1, 3, 5, 10]
    assert exp_config.chunker.chunk_size == 512
