from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestionDocumentDTO:
    """DTO representing a single document to be ingested."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class IngestionBatchDTO:
    """DTO representing a batch of documents to be processed by a worker."""
    batch_id: str
    documents: list[IngestionDocumentDTO]
    chunker_name: str  # 'recursive' or 'semantic'
    embedder_name: str  # 'openai' or 'bgem3'
    chunker_params: dict[str, Any] = field(default_factory=dict)
    embedder_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedChunkDTO:
    """DTO representing a processed and vectorized text chunk."""
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]

@dataclass
class ProcessedBatchResultDTO:
    """DTO representing the final vectorized batch payload."""
    batch_id: str
    processed_chunks: list[ProcessedChunkDTO]
    success: bool
    error_message: str | None = None
