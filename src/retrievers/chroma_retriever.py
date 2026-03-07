"""ChromaDB retriever implementation.

This module provides a concrete implementation of the Retriever interface
that uses ChromaDB as the vector database backend.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    import chromadb
else:
    chromadb = None

from ..core.embedding import Embedder
from ..core.retrieval import Retriever
from ..core.types import (
    Chunk,
    Embedding,
    Query,
    RetrievedChunk,
    RetrievalResult,
)
from .config import ChromaRetrieverConfig


class ChromaRetriever(Retriever):
    """ChromaDB-based retriever implementation."""
    
    def __init__(
        self,
        config: ChromaRetrieverConfig,
        embedder: Embedder,
    ) -> None:
        """Initialize ChromaDB retriever.
        
        Args:
            config: Configuration for Chroma usage
            embedder: Embedder instance to use for queries
        """
        self._config = config
        self._embedder = embedder
        
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        
    def _ensure_chroma_imported(self) -> None:
        """Ensure chromadb module is lazily imported."""
        global chromadb
        if chromadb is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb package is required for ChromaRetriever. "
                    "Install it with: pip install chromadb"
                )
                
    @property
    def _chroma_collection(self) -> "chromadb.Collection":
        """Get or create Chroma collection."""
        self._ensure_chroma_imported()
        
        if self._collection is None:
            try:
                if self._config.persist_directory:
                    self._client = chromadb.PersistentClient(path=self._config.persist_directory)
                else:
                    self._client = chromadb.EphemeralClient()
                    
                self._collection = self._client.get_or_create_collection(
                    name=self._config.collection_name,
                    metadata={"hnsw:space": self._config.distance_metric}
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to ChromaDB collection: {e}"
                ) from e
                
        return self._collection

    def _chunk_metadata_to_dict(self, chunk: Chunk) -> dict:
        """Convert Chunk metadata to Chroma-compatible dictionary."""
        metadata = {
            "document_id": chunk.metadata.document_id,
            "chunk_index": chunk.metadata.chunk_index,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
        }
        
        for key, value in chunk.metadata.custom.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"custom_{key}"] = value
            else:
                try:
                    metadata[f"custom_{key}"] = json.dumps(value)
                except (TypeError, ValueError):
                    pass
        return metadata

    def _dict_to_chunk(self, chunk_id: str, content: str, metadata: dict) -> Chunk:
        """Reconstruct Chunk from Chroma metadata."""
        from ..core.types import ChunkMetadata
        
        custom_metadata = {}
        for key, value in metadata.items():
            if key.startswith("custom_"):
                custom_key = key[7:]
                # Try loading back objects or let it stay string
                try:
                    parsed = json.loads(value)
                    custom_metadata[custom_key] = parsed
                except (TypeError, ValueError):
                    custom_metadata[custom_key] = value
                    
        chunk_metadata = ChunkMetadata(
            document_id=metadata["document_id"],
            chunk_index=metadata["chunk_index"],
            start_char=metadata["start_char"],
            end_char=metadata["end_char"],
            custom=custom_metadata,
        )
        
        return Chunk(
            id=chunk_id,
            content=content,
            metadata=chunk_metadata,
        )

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Add chunks and embeddings to Chroma."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
            
        try:
            collection = self._chroma_collection
            
            ids = [c.id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [self._chunk_metadata_to_dict(c) for c in chunks]
            vecs = [list(e.vector) for e in embeddings]
            
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    embeddings=vecs[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to Chroma: {e}") from e

    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks for a query using specific embedder."""
        start_time = time.time()
        query_embedding = self._embedder.embed_query(query)
        embedding_time = time.time() - start_time
        
        result = self.retrieve_with_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            query_id=query.id,
        )
        
        metadata = dict(result.metadata)
        metadata["embedding_latency_seconds"] = embedding_time
        
        return RetrievalResult(
            query=query,
            chunks=result.chunks,
            metadata=metadata,
        )

    def retrieve_with_embedding(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        query_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using a pre-computed query embedding."""
        try:
            start_time = time.time()
            collection = self._chroma_collection
            
            # ChromaDB output mapping
            query_response = collection.query(
                query_embeddings=[list(query_embedding.vector)],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieval_time = time.time() - start_time
            
            retrieved_chunks = []
            
            if query_response["ids"] and len(query_response["ids"][0]) > 0:
                ids = query_response["ids"][0]
                documents = query_response["documents"][0]
                metadatas = query_response["metadatas"][0]
                distances = query_response["distances"][0]
                
                for rank in range(len(ids)):
                    chunk = self._dict_to_chunk(
                        chunk_id=ids[rank],
                        content=documents[rank],
                        metadata=metadatas[rank],
                    )
                    
                    distance = distances[rank]
                    # Score interpretation for Chroma: 
                    # If cosine, distance is cosine distance, score = 1 - distance
                    # If l2, distance is Euclidean squared, score = -distance
                    score = 1.0 - distance if self._config.distance_metric == "cosine" else -distance
                    
                    retrieved_chunks.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=score,
                            rank=rank,
                        )
                    )
            
            query = Query(id=query_id or "unknown", text="")
            
            metadata = {
                "retrieval_latency_seconds": retrieval_time,
                "num_results": len(retrieved_chunks),
                "collection_name": self._config.collection_name,
            }
            
            return RetrievalResult(
                query=query,
                chunks=retrieved_chunks,
                metadata=metadata,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve from Chroma: {e}") from e

    def clear(self) -> None:
        """Clear all items in the collection without destroying definition."""
        try:
            # Recreate an empty collection
            if self._client and self._collection:
                self._client.delete_collection(self._config.collection_name)
                self._collection = None
                # Property _chroma_collection will recreate it on next call
        except Exception as e:
            raise RuntimeError(f"Failed to clear ChromaDB: {e}") from e
