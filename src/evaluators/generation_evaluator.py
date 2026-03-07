"""LLM-based evaluator for Generation quality.

This module uses an LLM acting as a "judge" to score pipeline performance
metrics such as Faithfulness (hallucination reduction) and Relevancy.
"""

import json
import logging
import re
from typing import Any, Dict

from ..core.evaluation import Evaluator
from ..core.generation import Generator
from ..core.types import Query, RAGResponse, RetrievedChunk
from .judge_prompts import FAITHFULNESS_PROMPT, RELEVANCY_PROMPT

logger = logging.getLogger(__name__)


class GenerationEvaluator(Evaluator):
    """LLM-as-a-judge Generation Evaluator.
    
    This class leverages a secondary LLM pipeline (via UniversalGenerator)
    to perform reference-free evaluation of generated answers.
    """
    
    def __init__(self, judge_generator: Generator) -> None:
        """Initialize the evaluator.
        
        Args:
            judge_generator: A Generator instance (like UniversalGenerator)
                             acting as the impartial judge.
        """
        self._judge_generator = judge_generator
        
    def _parse_json_score(self, text: str) -> float:
        """Extract and normalize score from LLM JSON output.
        
        Args:
            text: The raw text generation from the LLM.
            
        Returns:
            Normalized float score between 0.0 and 1.0. Returns 0.0 on failure.
        """
        # Try raw loading first
        try:
            data = json.loads(text)
            score = float(data.get("score", 0))
            return min(max(score / 10.0, 0.0), 1.0)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
            
        # Fallback to regex extraction (often LLMs wrap JSON in ```json...```)
        try:
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0))
                return min(max(score / 10.0, 0.0), 1.0)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
            
        logger.warning(f"Failed to parse judge JSON. Defaulting score to 0. Raw output: {text}")
        return 0.0

    def evaluate(self, result: RAGResponse, ground_truth: Any = None) -> Dict[str, float]:
        """Evaluate Generation Quality.
        
        Args:
            result: End-to-end RAGResponse containing context and answer
            ground_truth: Unused since this is reference-free LLM evaluation.
            
        Returns:
            Dictionary mapped to generic float metrics [0.0 - 1.0]
        """
        if not isinstance(result, RAGResponse):
            raise ValueError("GenerationEvaluator expects a RAGResponse target.")
            
        # Format context tightly
        context_str = "\n".join([c.chunk.content for c in result.retrieved_chunks])
        
        # 1. Faithfulness
        faith_content = FAITHFULNESS_PROMPT.format(
            context=context_str,
            response=result.response
        )
        try:
            faith_gen = self._judge_generator.generate(
                query=Query(id="eval_faith", text=faith_content),
                retrieved_chunks=[]
            )
            faithfulness_score = self._parse_json_score(faith_gen.response)
        except Exception as e:
            logger.error(f"Faithfulness eval failed: {e}")
            faithfulness_score = 0.0
            
        # 2. Relevancy
        relev_content = RELEVANCY_PROMPT.format(
            question=result.query.text,
            response=result.response
        )
        try:
            relev_gen = self._judge_generator.generate(
                query=Query(id="eval_relev", text=relev_content),
                retrieved_chunks=[]
            )
            relevancy_score = self._parse_json_score(relev_gen.response)
        except Exception as e:
            logger.error(f"Relevancy eval failed: {e}")
            relevancy_score = 0.0
            
        return {
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score
        }
