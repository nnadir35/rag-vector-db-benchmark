"""Answer quality evaluator implementation.

This module provides the evaluator class designed to score RAGResponse
outputs against SQuAD ground truth answers.
"""

from ..core.evaluation import Evaluator
from ..core.types import RAGResponse
from .config import AnswerQualityEvaluatorConfig
from .metrics import exact_match_score, f1_score


class AnswerQualityEvaluator(Evaluator):
    """SQuAD'ın resmi referans-tabanlı metrikleri (EM, F1).

    RAGAS-tarzı Faithfulness/Relevancy'nin aksine bu evaluator referanssız
    değil — altın cevap metinlerine ihtiyaç duyar. Model-bağımsız ve
    deterministiktir (Rajpurkar ve ark., 2016).
    """

    def __init__(self, config: AnswerQualityEvaluatorConfig | None = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Configuration (optional, provided for consistency).
        """
        self._config = config or AnswerQualityEvaluatorConfig()

    def evaluate(self, result: RAGResponse, ground_truth: list[str]) -> dict[str, float]:
        """Compute exact match and F1 scores.
        
        Args:
            result: The generation response containing the model's answer.
            ground_truth: o sorunun kabul edilebilir altın cevap metinleri listesi.
        
        Returns:
            A dictionary with 'exact_match' and 'f1' float values.
        """
        em = exact_match_score(result.answer, ground_truth)
        f1 = f1_score(result.answer, ground_truth)
        
        return {
            "exact_match": em,
            "f1": f1
        }
