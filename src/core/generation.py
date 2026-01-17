"""Interface for text generation.

Generation is the process of producing a response to a query using
retrieved context and a language model.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .types import Query, GenerationResult, RetrievedChunk


class Generator(ABC):
    """Abstract interface for text generation.
    
    Generators are responsible for producing responses to queries using
    retrieved context chunks. They encapsulate the LLM, prompt template,
    and generation parameters.
    
    Implementations must handle:
    - Formatting queries and context into prompts
    - Calling the language model
    - Post-processing generated responses
    - Managing generation parameters (temperature, max_tokens, etc.)
    
    Implementations must NOT:
    - Know about retrieval or vector databases
    - Know about evaluation or benchmarking
    - Perform retrieval operations
    """
    
    @abstractmethod
    def generate(
        self,
        query: Query,
        retrieved_chunks: Sequence[RetrievedChunk],
        **kwargs: object,
    ) -> GenerationResult:
        """Generate a response for a query using retrieved context.
        
        This method takes a query and retrieved chunks, formats them into
        a prompt, calls the language model, and returns the generated response.
        
        Args:
            query: The query to generate a response for
            retrieved_chunks: Context chunks retrieved for the query
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
                Implementation-specific, but should be configurable
        
        Returns:
            GenerationResult containing the response and metadata
            
        Raises:
            ValueError: If query or chunks are invalid
            RuntimeError: If generation fails
        """
        pass
    
    def generate_batch(
        self,
        queries: Sequence[Query],
        retrieved_chunks_list: Sequence[Sequence[RetrievedChunk]],
        **kwargs: object,
    ) -> Sequence[GenerationResult]:
        """Generate responses for multiple queries in batch.
        
        This method allows implementations to optimize batch processing
        if supported. The default implementation calls generate() for each
        query, but implementations may override for efficiency.
        
        Args:
            queries: Sequence of queries to generate responses for
            retrieved_chunks_list: Sequence of retrieved chunk lists,
                one per query (must match length of queries)
            **kwargs: Additional generation parameters
        
        Returns:
            Sequence of GenerationResults, one per query, in the same order
            
        Raises:
            ValueError: If inputs are invalid or mismatched
            RuntimeError: If generation fails
        """
        if len(queries) != len(retrieved_chunks_list):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match "
                f"number of retrieved chunk lists ({len(retrieved_chunks_list)})"
            )
        
        results: list[GenerationResult] = []
        for query, retrieved_chunks in zip(queries, retrieved_chunks_list):
            result = self.generate(query, retrieved_chunks, **kwargs)
            results.append(result)
        return results
