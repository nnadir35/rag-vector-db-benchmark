"""Tests for LLM-based generation evaluation."""

from unittest.mock import MagicMock

import pytest

from src.core.types import (
    Chunk,
    ChunkMetadata,
    GenerationResult,
    Query,
    RAGResponse,
    RetrievedChunk,
)
from src.evaluators.generation_evaluator import GenerationEvaluator


@pytest.fixture
def sample_rag_response():
    chunk = Chunk(
        id="c1",
        content="The capital of France is Paris.",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=30)
    )
    r_chunk = RetrievedChunk(chunk=chunk, score=0.9, rank=0)

    return RAGResponse(
        query=Query(id="q1", text="What is the capital of France?"),
        retrieved_chunks=[r_chunk],
        response="Paris is the capital."
    )


def test_evaluate_faithfulness_and_relevancy(sample_rag_response):
    """Test standard evaluation parsing returning JSON correctly."""
    mock_generator = MagicMock()

    def mock_generate(query, retrieved_chunks):
        if "eval_faith" in query.id:
            # Simulated JSON output (raw)
            raw = '{"score": 9, "reason": "Accurate to context"}'
            return GenerationResult(query=query, response=raw, retrieved_chunks=[])
        else:
            # Simulated JSON output wrapped in markdown
            raw = '```json\n{"score": 10, "reason": "Direct"}\n```'
            return GenerationResult(query=query, response=raw, retrieved_chunks=[])

    mock_generator.generate.side_effect = mock_generate

    evaluator = GenerationEvaluator(judge_generator=mock_generator)
    metrics = evaluator.evaluate(sample_rag_response)

    assert "faithfulness" in metrics
    assert "relevancy" in metrics

    assert metrics["faithfulness"] == 0.9  # 9 / 10
    assert metrics["relevancy"] == 1.0     # 10 / 10


def test_evaluate_handles_dirty_json(sample_rag_response):
    """Test that failed json parsing defaults to 0 safely."""
    mock_generator = MagicMock()

    def mock_generate(query, retrieved_chunks):
        # Invalid JSON
        raw = 'Score is 8 because... wait I cannot output JSON'
        return GenerationResult(query=query, response=raw, retrieved_chunks=[])

    mock_generator.generate.side_effect = mock_generate

    evaluator = GenerationEvaluator(judge_generator=mock_generator)
    metrics = evaluator.evaluate(sample_rag_response)

    assert metrics["faithfulness"] == 0.0
    assert metrics["relevancy"] == 0.0
