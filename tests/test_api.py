import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from src.core.types import Document, Chunk, ChunkMetadata, Query, RAGResponse, RetrievedChunk
from src.pipeline.result import PipelineResult

@pytest.fixture
def mock_components():
    with patch("api.FixedSizeChunker") as mock_chunker_cls, \
         patch("api.SentenceTransformersEmbedder") as mock_embedder_cls, \
         patch("api.UniversalGenerator") as mock_generator_cls, \
         patch("api.build_retriever_from_yaml") as mock_build_retriever, \
         patch("api.RAGPipeline") as mock_pipeline_cls:
        
        # Setup mock instances
        mock_chunker = mock_chunker_cls.return_value
        mock_embedder = mock_embedder_cls.return_value
        mock_generator = mock_generator_cls.return_value
        mock_retriever = mock_build_retriever.return_value
        mock_pipeline = mock_pipeline_cls.return_value

        # Set default behaviors
        mock_chunker.chunk.return_value = [
            Chunk(id="chunk1", content="chunk 1 content", metadata=ChunkMetadata(document_id="doc1", start_char=0, end_char=10, chunk_index=0))
        ]
        
        mock_embedder.embed_chunks.return_value = [[0.1, 0.2, 0.3]]
        mock_retriever.add_chunks.return_value = None

        mock_pipeline.run = AsyncMock()
        mock_pipeline.run.return_value = PipelineResult(
            query=Query(id="q1", text="What is this?"),
            rag_response=RAGResponse(
                query=Query(id="q1", text="What is this?"),
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk=Chunk(id="chunk1", content="context snippet", metadata=ChunkMetadata(document_id="doc1", start_char=0, end_char=10, chunk_index=0)),
                        score=0.9,
                        rank=0
                    )
                ],
                response="This is an answer.",
                metadata={}
            ),
            retrieval_metrics=None,
            total_latency_seconds=0.1
        )
        
        yield {
            "chunker": mock_chunker,
            "embedder": mock_embedder,
            "generator": mock_generator,
            "retriever": mock_retriever,
            "pipeline": mock_pipeline
        }

@pytest.fixture
def client(mock_components):
    # Import inside to ensure patches are applied correctly when FastAPI app lifespan triggers
    from api import app
    with TestClient(app) as c:
        yield c

def test_upload_txt_success(client, mock_components):
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"Hello world! This is a test file.", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 'test.txt'"
    assert "document_id" in data
    assert data["chunks_added"] == 1
    
    mock_components["chunker"].chunk.assert_called_once()
    mock_components["embedder"].embed_chunks.assert_called_once()
    mock_components["retriever"].add_chunks.assert_called_once()

def test_upload_empty_chunks(client, mock_components):
    mock_components["chunker"].chunk.return_value = []
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Document is empty or could not be chunked."
    assert data["chunks_added"] == 0

def test_upload_pdf_success(client, mock_components):
    with patch("pypdf.PdfReader") as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF text content"
        mock_pdf_reader.return_value.pages = [mock_page]

        response = client.post(
            "/upload",
            files={"file": ("test.pdf", b"fake_pdf_content", "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 'test.pdf'"
        assert data["chunks_added"] == 1

def test_upload_unsupported_file(client, mock_components):
    response = client.post(
        "/upload",
        files={"file": ("test.csv", b"a,b,c", "text/csv")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only .txt and .pdf files are supported for now."

def test_ask_question_success(client, mock_components):
    response = client.post(
        "/ask",
        json={"question": "What is this?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is this?"
    assert data["answer"] == "This is an answer."
    assert "latency_seconds" in data
    assert len(data["retrieved_context"]) == 1
    assert data["retrieved_context"][0]["content"] == "context snippet"
    assert data["retrieved_context"][0]["document_id"] == "doc1"
    
    mock_components["pipeline"].run.assert_called_once_with("What is this?")

def test_ask_empty_question(client, mock_components):
    response = client.post(
        "/ask",
        json={"question": "   "}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."
