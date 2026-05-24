import json
from unittest.mock import MagicMock, patch

import pytest
from src.core.types import Document, DocumentMetadata
from src.queue.config import QueueConfig
from src.queue.dto import IngestionBatchDTO, IngestionDocumentDTO
from src.queue.producer import IngestionProducer
from src.queue.worker import dlq_exception_handler
from src.queue.worker_tasks import process_batch


def test_producer_batching_and_enqueue():
    # Mock Redis ve RQ Queue
    mock_redis = MagicMock()
    mock_queue = MagicMock()

    with patch("src.queue.config.QueueConfig.get_redis_connection", return_value=mock_redis), \
         patch("src.queue.producer.Queue", return_value=mock_queue):
        
        producer = IngestionProducer()
        
        # Test dökümanları oluştur
        docs = [
            Document(id=f"doc_{i}", content=f"Content of document {i}", metadata=DocumentMetadata(source="squad"))
            for i in range(10)
        ]
        
        # Batch size 3 olacak şekilde kuyruğa ekle
        job_ids = producer.enqueue_documents(
            documents=docs,
            chunker_name="recursive",
            embedder_name="openai",
            chunker_params={"chunk_size": 100, "chunk_overlap": 10},
            embedder_params={"model_name": "text-embedding-3-small"},
            batch_size=3
        )
        
        # 10 döküman, batch_size=3 -> 4 batch oluşturmalı (3, 3, 3, 1)
        assert len(job_ids) == 4
        assert mock_queue.enqueue.call_count == 4
        
        # İlk enqueue çağrısının argümanlarını doğrula
        first_call_args = mock_queue.enqueue.call_args_list[0]
        called_fn = first_call_args[0][0]
        batch_dto = first_call_args[0][1]
        
        assert called_fn == process_batch
        assert isinstance(batch_dto, IngestionBatchDTO)
        assert len(batch_dto.documents) == 3
        assert batch_dto.chunker_name == "recursive"
        assert batch_dto.embedder_name == "openai"
        assert batch_dto.chunker_params["chunk_size"] == 100
        assert batch_dto.embedder_params["model_name"] == "text-embedding-3-small"


def test_worker_task_success():
    # Model indirmesini engellemek için chunker ve embedder'ı mock'la
    mock_chunk = MagicMock()
    mock_chunk.content = "Hello worl"
    mock_chunk.metadata = {"chunk_index": 0}
    
    mock_chunker = MagicMock()
    mock_chunker.chunk_text.return_value = [mock_chunk]
    
    mock_embedding = MagicMock(vector=[0.1, 0.2, 0.3])
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [mock_embedding]

    with patch("src.queue.worker_tasks.get_chunker", return_value=mock_chunker), \
         patch("src.queue.worker_tasks.get_embedder", return_value=mock_embedder), \
         patch("src.queue.worker_tasks.MockVectorDB.insert_chunks") as mock_vdb_insert:
        
        batch = IngestionBatchDTO(
            batch_id="test_batch_123",
            documents=[
                IngestionDocumentDTO(id="doc_1", content="Hello world", metadata={"source": "test"})
            ],
            chunker_name="recursive",
            chunker_params={},
            embedder_name="openai",
            embedder_params={}
        )
        
        result = process_batch(batch)
        
        assert result.success is True
        assert len(result.processed_chunks) == 1
        assert result.processed_chunks[0].id == "doc_1_chunk_0"
        assert result.processed_chunks[0].content == "Hello worl"
        assert result.processed_chunks[0].embedding == [0.1, 0.2, 0.3]
        mock_vdb_insert.assert_called_once()


def test_dlq_routing_on_final_failure():
    mock_job = MagicMock()
    mock_job.id = "failed_job_999"
    mock_job.origin = "ingestion_queue"
    mock_job.args = (IngestionBatchDTO(
        batch_id="test_batch_failed",
        documents=[],
        chunker_name="recursive",
        embedder_name="openai"
    ),)
    mock_job.kwargs = {}
    
    mock_redis = MagicMock()
    mock_job.connection = mock_redis
    
    # 1. Henüz deneme hakları bitmemişse (retries_left > 0) DLQ'ya eklememeli
    mock_job.retries_left = 2
    propagate = dlq_exception_handler(mock_job, ValueError, ValueError("API Error"), None)
    assert propagate is True
    mock_redis.rpush.assert_not_called()
    
    # 2. Deneme hakları bittiğinde (retries_left == 0) DLQ'ya eklemeli
    mock_job.retries_left = 0
    propagate = dlq_exception_handler(mock_job, ValueError, ValueError("API Error Final"), None)
    assert propagate is True
    mock_redis.rpush.assert_called_once()
    
    # Redis listesine yazılan JSON çıktısını doğrula
    call_args = mock_redis.rpush.call_args[0]
    queue_name = call_args[0]
    payload = json.loads(call_args[1])
    
    assert queue_name == "ingestion_dlq"
    assert payload["job_id"] == "failed_job_999"
    assert payload["exception_type"] == "ValueError"
    assert payload["exception_message"] == "API Error Final"
