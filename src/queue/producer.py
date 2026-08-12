import logging
import uuid
from typing import Any

from rq import Queue, Retry

from src.core.types import Document

from .config import QueueConfig
from .dto import IngestionBatchDTO, IngestionDocumentDTO
from .worker_tasks import process_batch

logger = logging.getLogger("rq.worker")

class IngestionProducer:
    """Producer class responsible for batching documents and enqueuing them to Redis Queue."""

    def __init__(self, config: QueueConfig | None = None) -> None:
        self.config = config or QueueConfig()
        self.redis_conn = self.config.get_redis_connection()
        self.queue = Queue(self.config.queue_name, connection=self.redis_conn)

    def enqueue_documents(
        self,
        documents: list[Document | IngestionDocumentDTO],
        chunker_name: str,
        embedder_name: str,
        chunker_params: dict[str, Any] | None = None,
        embedder_params: dict[str, Any] | None = None,
        batch_size: int = 500
    ) -> list[str]:
        """Convert documents to DTOs, batch them, and enqueue to RQ.

        Args:
            documents: List of Document (core) or IngestionDocumentDTO objects.
            chunker_name: Name of the chunker to use ('recursive' or 'semantic').
            embedder_name: Name of the embedder to use ('openai' or 'bgem3').
            chunker_params: Chunker initialization parameters.
            embedder_params: Embedder initialization parameters.
            batch_size: Number of documents to process in a single worker job.

        Returns:
            A list of enqueued RQ Job IDs.
        """
        chunker_params = chunker_params or {}
        embedder_params = embedder_params or {}

        dto_documents: list[IngestionDocumentDTO] = []
        for doc in documents:
            if isinstance(doc, Document):
                metadata_dict = {}
                if doc.metadata:
                    metadata_dict = {
                        "source": doc.metadata.source,
                        "title": doc.metadata.title,
                        "author": doc.metadata.author,
                        "created_at": doc.metadata.created_at,
                        "custom": doc.metadata.custom
                    }
                dto_documents.append(
                    IngestionDocumentDTO(
                        id=doc.id,
                        content=doc.content,
                        metadata=metadata_dict
                    )
                )
            elif isinstance(doc, IngestionDocumentDTO):
                dto_documents.append(doc)
            else:
                raise TypeError("Documents must be of type src.core.types.Document or IngestionDocumentDTO")

        job_ids: list[str] = []

        for i in range(0, len(dto_documents), batch_size):
            batch_docs = dto_documents[i:i + batch_size]
            batch_id = str(uuid.uuid4())

            batch_dto = IngestionBatchDTO(
                batch_id=batch_id,
                documents=batch_docs,
                chunker_name=chunker_name,
                chunker_params=chunker_params,
                embedder_name=embedder_name,
                embedder_params=embedder_params
            )

            retry_config = Retry(max=self.config.max_retries, interval=self.config.retry_delay)

            job = self.queue.enqueue(
                process_batch,
                batch_dto,
                retry=retry_config,
                job_id=f"ingest_batch_{batch_id}"
            )

            job_ids.append(job.id)
            logger.info(f"Enqueued batch {batch_id} (size={len(batch_docs)}) as Job ID: {job.id}")

        return job_ids
