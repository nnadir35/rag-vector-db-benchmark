"""Configuration loader utility.

This module is responsible for loading YAML configuration files and converting
them into appropriate configuration dataclasses for various components.
"""

from dataclasses import dataclass
from typing import Any

from src.chunkers.config import FixedSizeChunkerConfig
from src.datasets.config import SQuADDatasetConfig
from src.embedders.config import SentenceTransformersEmbedderConfig
from src.evaluators.config import RetrievalEvaluatorConfig
from src.generators.config import UniversalGeneratorConfig
from src.pipeline.config import RAGPipelineConfig


@dataclass
class ExperimentConfig:
    """Holistic configuration for a complete experiment run.

    Contains all instantiated configuration dataclasses for individual
    components, as well as high-level experiment metadata.
    """

    name: str
    chunker: FixedSizeChunkerConfig
    embedder: SentenceTransformersEmbedderConfig
    dataset: SQuADDatasetConfig
    generator: UniversalGeneratorConfig
    evaluator: RetrievalEvaluatorConfig
    pipeline: RAGPipelineConfig


def load_yaml(file_path: str) -> dict[str, Any]:
    """Load and parse a YAML file into a dictionary.

    Uses lazy importing for pyyaml to prioritize fast startup.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Dictionary representation of the YAML.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML formatting is invalid.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "The 'PyYAML' package is required but not installed. "
            "Install it with: pip install PyYAML"
        )

    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file '{file_path}': {e}") from e


def build_component_configs(config_dict: dict[str, Any]) -> ExperimentConfig:
    """Convert raw configuration dictionary into typed Dataclass objects.

    This function acts as a mapping factory between raw JSON/YAML dicts
    and strict, immutable @dataclasses enforcing validation logic.

    Args:
        config_dict: Dictionary from parsed configuration file.

    Returns:
        An ExperimentConfig instance housing parsed configurations.
    """
    exp_name = config_dict.get("experiment", {}).get("name", "unnamed_experiment")

    return ExperimentConfig(
        name=exp_name,
        chunker=FixedSizeChunkerConfig(**config_dict.get("chunker", {})),
        embedder=SentenceTransformersEmbedderConfig(**config_dict.get("embedder", {})),
        dataset=SQuADDatasetConfig(**config_dict.get("dataset", {})),
        generator=UniversalGeneratorConfig(**config_dict.get("generator", {})),
        evaluator=RetrievalEvaluatorConfig(**config_dict.get("evaluator", {})),
        pipeline=RAGPipelineConfig(**config_dict.get("pipeline", {})),
    )
