import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.types import Document
from src.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.generators.universal_generator import UniversalGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrievers.factory import build_retriever_from_yaml
from src.utils.config_loader import build_component_configs, load_yaml

# Global variables for the components
chunker = None
embedder = None
retriever = None
generator = None
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunker, embedder, retriever, generator, pipeline
    print("Loading RAG configurations and models...")
    config_path = os.getenv("RAG_CONFIG_PATH", "experiments/configs/baseline_ollama.yaml")

    if os.path.exists(config_path):
        raw_config = load_yaml(config_path)
    else:
        # Provide a fallback default configuration if the yaml is missing
        raw_config = {
            "chunker": {"chunk_size": 512, "overlap": 50},
            "embedder": {"model_name": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 32},
            "generator": {"model_name": "ollama/llama3:8b", "temperature": 0.0, "max_tokens": 512, "api_base": "http://localhost:11434"},
            "retriever": {"collection_name": "api_collection", "persist_directory": "./chroma_api_db"},
            "pipeline": {"top_k": 3, "evaluate_retrieval": False, "evaluate_generation": False}
        }
        print(f"Config file not found at {config_path}. Using fallback defaults.")

    exp_config = build_component_configs(raw_config)

    # Initialize components
    chunker = FixedSizeChunker(exp_config.chunker)
    embedder = SentenceTransformersEmbedder(exp_config.embedder)
    generator = UniversalGenerator(exp_config.generator)

    retriever = build_retriever_from_yaml(raw_config, embedder)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        config=exp_config.pipeline,
        retrieval_evaluator=None
    )
    print("RAG Pipeline initialized!")
    yield
    print("Shutting down API...")

app = FastAPI(title="RAG Vector DB Benchmark API", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported for now.")

    try:
        content = await file.read()
        if file.filename.endswith(".pdf"):
            import io

            from pypdf import PdfReader
            pdf_reader = PdfReader(io.BytesIO(content))
            text_content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        else:
            text_content = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    doc_id = f"doc_{time.time_ns()}"
    document = Document(
        id=doc_id,
        content=text_content
    )

    # Chunk the document
    chunks = chunker.chunk(document)
    if not chunks:
        return {"message": "Document is empty or could not be chunked.", "chunks_added": 0}

    # Embed the chunks
    embeddings = embedder.embed_chunks(chunks)

    # Add to retriever (ChromaDb)
    retriever.add_chunks(chunks, embeddings)

    return {
        "message": f"Successfully processed '{file.filename}'",
        "document_id": doc_id,
        "chunks_added": len(chunks)
    }

@app.post("/ask")
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = await pipeline.run(request.question)

    return {
        "question": request.question,
        "answer": result.rag_response.response,
        "latency_seconds": result.total_latency_seconds,
        "retrieved_context": [
            {
                "content": rc.chunk.content,
                "score": rc.score,
                "document_id": rc.chunk.metadata.document_id
            }
            for rc in result.rag_response.retrieved_chunks
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
