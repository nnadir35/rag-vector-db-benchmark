import logging
from typing import Any

from src.chunking.recursive import RecursiveCharacterChunker
from src.chunking.semantic import SemanticChunker
from src.embedding.bgem3 import BGEM3Embedder
from src.embedding.openai import OpenAIEmbedder

from .dto import IngestionBatchDTO, ProcessedBatchResultDTO, ProcessedChunkDTO

logger = logging.getLogger("rq.worker")

class MockVectorDB:
    """Mock Vector Database adapter to simulate embedding ingestion."""

    @staticmethod
    def insert_chunks(chunks: list[ProcessedChunkDTO]) -> None:
        logger.info(f"[MockVectorDB] Successfully ingested {len(chunks)} chunks into vector database.")
        for i, chunk in enumerate(chunks[:3]):
            logger.info(
                f"[MockVectorDB] Chunk Sample {i}: ID={chunk.id}, "
                f"Content='{chunk.content[:60]}...', VectorDim={len(chunk.embedding)}"
            )
        if len(chunks) > 3:
            logger.info(f"[MockVectorDB] ... and {len(chunks) - 3} more chunks.")

def get_chunker(name: str, params: dict[str, Any]):
    """Factory function to resolve chunkers."""
    if name == "recursive":
        return RecursiveCharacterChunker(**params)
    elif name == "semantic":
        return SemanticChunker(**params)
    else:
        raise ValueError(f"Unknown chunker name: {name}")

def get_embedder(name: str, params: dict[str, Any]):
    """Factory function to resolve embedders."""
    if name == "openai":
        return OpenAIEmbedder(**params)
    elif name == "bgem3":
        return BGEM3Embedder(**params)
    else:
        raise ValueError(f"Unknown embedder name: {name}")

def process_batch(batch: IngestionBatchDTO) -> ProcessedBatchResultDTO:
    """RQ Task to process a batch of documents (chunking + embedding)."""
    logger.info(f"Starting processing for batch {batch.batch_id} with {len(batch.documents)} documents.")
    try:
        chunker = get_chunker(batch.chunker_name, batch.chunker_params)
        embedder = get_embedder(batch.embedder_name, batch.embedder_params)

        all_processed_chunks: list[ProcessedChunkDTO] = []

        # 1. Her bir dokümanı parçalara ayır
        for doc in batch.documents:
            doc_metadata = doc.metadata.copy() if doc.metadata else {}
            doc_metadata["source_document_id"] = doc.id

            chunks = chunker.chunk_text(doc.content, metadata=doc_metadata)
            if not chunks:
                continue

            # 2. Parçaları vektörleştir
            chunk_contents = [c.content for c in chunks]
            embeddings = embedder.embed_batch(chunk_contents)

            for idx, (chunk, emb_dto) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc.id}_chunk_{chunk.metadata.get('chunk_index', idx)}"
                processed_chunk = ProcessedChunkDTO(
                    id=chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=list(emb_dto.vector)
                )
                all_processed_chunks.append(processed_chunk)

        # 3. Vektör veritabanı mock ingestion
        MockVectorDB.insert_chunks(all_processed_chunks)

        logger.info(f"Finished processing batch {batch.batch_id} successfully. Total chunks: {len(all_processed_chunks)}")
        return ProcessedBatchResultDTO(
            batch_id=batch.batch_id,
            processed_chunks=all_processed_chunks,
            success=True
        )

    except Exception as e:
        logger.error(f"Error processing batch {batch.batch_id}: {str(e)}", exc_info=True)
        # Hata durumunda istisnayı fırlatıyoruz ki RQ retry ve DLQ mekanizması tetiklenebilsin
        raise e
