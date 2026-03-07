"""Tests for RAG Pipeline execution."""

import pytest
from unittest.mock import MagicMock
from typing import Dict

from src.core.types import (
    GenerationResult,
    Query,
    RetrievedChunk,
    RetrievalResult,
    Chunk,
    ChunkMetadata
)
from src.pipeline.config import RAGPipelineConfig
from src.pipeline.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_components():
    """Mock the core components needed by pipeline."""
    retriever = MagicMock()
    generator = MagicMock()
    evaluator = MagicMock()
    
    # Setup standard mock responses
    chunk = Chunk(
        id="c1", 
        content="mock content", 
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, start_char=0, end_char=10)
    )
    
    r_chunk = RetrievedChunk(chunk=chunk, score=0.9, rank=0)
    
    def retrieve_side_effect(query, top_k):
        return RetrievalResult(query=query, chunks=[r_chunk], metadata={})
        
    retriever.retrieve.side_effect = retrieve_side_effect
    
    def generate_side_effect(query, chunks):
        return GenerationResult(
            query=query, 
            response="A generated answer.", 
            retrieved_chunks=chunks,
            metadata={}
        )
        
    generator.generate.side_effect = generate_side_effect
    
    def evaluate_side_effect(result, truths):
        return {"precision@1": 1.0}
        
    evaluator.evaluate.side_effect = evaluate_side_effect
    
    return retriever, generator, evaluator


@pytest.mark.asyncio
async def test_pipeline_execution_order_and_metrics(mock_components):
    """Test that pipeline executes retrieve->evaluate->generate properly."""
    retriever, generator, evaluator = mock_components
    
    config = RAGPipelineConfig(top_k=2, evaluate_retrieval=True)
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        config=config,
        retrieval_evaluator=evaluator
    )
    
    result = await pipeline.run(query="What is AI?", ground_truth_ids=["d1"])
    
    assert result.query.text == "What is AI?"
    assert result.rag_response.response == "A generated answer."
    assert len(result.rag_response.retrieved_chunks) == 1
    
    # Assert call tracks
    retriever.retrieve.assert_called_once()
    generator.generate.assert_called_once()
    evaluator.evaluate.assert_called_once()
    
    assert result.retrieval_metrics == {"precision@1": 1.0}
    assert result.total_latency_seconds > 0.0


@pytest.mark.asyncio
async def test_pipeline_skips_evaluation_if_no_ground_truth(mock_components):
    """Test that missing ground truth skips the evaluation phase."""
    retriever, generator, evaluator = mock_components
    
    config = RAGPipelineConfig(top_k=2, evaluate_retrieval=True)
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        config=config,
        retrieval_evaluator=evaluator
    )
    
    result = await pipeline.run(query="Question with no truth?", ground_truth_ids=None)
    
    assert result.retrieval_metrics is None
    evaluator.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_batch_execution(mock_components):
    """Test running multiple queries asynchronously in a batch."""
    retriever, generator, evaluator = mock_components
    
    config = RAGPipelineConfig(top_k=1, evaluate_retrieval=False)
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        config=config
    )
    
    queries = ["Question 1?", "Question 2?"]
    
    results = await pipeline.run_batch(queries)
    
    assert len(results) == 2
    assert results[0].query.text == "Question 1?"
    assert results[1].query.text == "Question 2?"
    
    assert retriever.retrieve.call_count == 2
    assert generator.generate.call_count == 2
