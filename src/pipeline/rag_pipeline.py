"""RAG Pipeline execution engine.

This module coordinates retrievers, generators, and evaluators to process
end-to-end question answering requests.
"""

import asyncio
import time
from collections.abc import Sequence

from ..core.evaluation import Evaluator
from ..core.generation import Generator
from ..core.retrieval import Retriever
from ..core.types import Query, RAGResponse
from .config import RAGPipelineConfig
from .result import PipelineResult


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    This class combines a Retriever and a Generator, and optionally routes
    through an Evaluator to produce metrics alongside answers.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        config: RAGPipelineConfig,
        retrieval_evaluator: Evaluator | None = None,
        generation_evaluator: Evaluator | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            retriever: Component yielding relevant chunks from a vector DB.
            generator: Component generating the answer from retrieved context.
            config: Pipeline settings (top_k, evaluation toggles).
            retrieval_evaluator: Optional evaluator to compute MRR/nDCG/Precision metrics.
        """
        self._retriever = retriever
        self._generator = generator
        self._config = config
        self._retrieval_evaluator = retrieval_evaluator
        self._generation_evaluator = generation_evaluator

    async def run(
        self,
        query: str,
        ground_truth_ids: list[str] | None = None
    ) -> PipelineResult:
        """Execute the pipeline asynchronously for a single query.

        Args:
            query: The user's input string.
            ground_truth_ids: Valid document ids used if evaluation is enabled.

        Returns:
            PipelineResult structure mapping latency, metrics, and answers.
        """
        start_time = time.time()

        # Structure the query
        # ID could be deterministic or just random natively, but typically
        # it might be assigned upstream. We'll use a time-based ID if none exists.
        q_obj = Query(id=f"q_{time.time_ns()}", text=query)

        # 1. Retrieve
        # Assuming retrieve may eventually be async, but currently
        # standard interface is sync. We will wrap it in asyncio.to_thread if necessary,
        # but for now we follow the regular synchronous interface within async def.
        # Future-proofing by using async structure where possible.
        retrieval_result = await asyncio.to_thread(
            self._retriever.retrieve,
            q_obj,
            top_k=self._config.top_k
        )

        # 2. Evaluate Retrieval (if enabled and truths exist)
        metrics: dict[str, float] | None = None
        if self._config.evaluate_retrieval and self._retrieval_evaluator and ground_truth_ids:
            metrics = await asyncio.to_thread(
                self._retrieval_evaluator.evaluate,
                retrieval_result,
                set(ground_truth_ids)
            )

        # 3. Generate
        generation_result = await asyncio.to_thread(
            self._generator.generate,
            q_obj,
            retrieval_result.chunks
        )

        # Structure final RAG Response
        rag_response = RAGResponse(
            query=q_obj,
            retrieved_chunks=retrieval_result.chunks,
            response=generation_result.response,
            metadata={
                "retrieval_metadata": retrieval_result.metadata,
                "generation_metadata": generation_result.metadata,
            }
        )

        # 4. Evaluate Generation
        if getattr(self._config, "evaluate_generation", False) and getattr(self, "_generation_evaluator", None):
            gen_metrics = await asyncio.to_thread(
                self._generation_evaluator.evaluate,
                rag_response
            )
            if metrics is None:
                metrics = {}
            metrics.update(gen_metrics)

        total_time = time.time() - start_time

        return PipelineResult(
            query=q_obj,
            rag_response=rag_response,
            retrieval_metrics=metrics,
            total_latency_seconds=total_time
        )

    async def run_batch(
        self,
        queries: Sequence[str],
        ground_truths: Sequence[list[str]] | None = None
    ) -> list[PipelineResult]:
        """Execute the pipeline across multiple queries concurrently.

        Args:
            queries: Sequence of string queries.
            ground_truths: Sequence of ground truth ID lists (matching queries).

        Returns:
            List of PipelineResult objects preserving input order.
        """
        if ground_truths and len(queries) != len(ground_truths):
            raise ValueError(
                f"Count of queries ({len(queries)}) must match "
                f"count of ground_truths ({len(ground_truths)})"
            )

        tasks = []
        for i, q in enumerate(queries):
            truth = ground_truths[i] if ground_truths else None
            task = self.run(q, truth)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return list(results)
