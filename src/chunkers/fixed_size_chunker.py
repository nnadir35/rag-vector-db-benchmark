"""Fixed-size chunker implementation.

This module provides a concrete implementation of the Chunker interface
using a fixed-size character strategy with optional overlap.
"""

from typing import Sequence

from ..core.chunking import Chunker
from ..core.types import Chunk, ChunkMetadata, Document
from .config import FixedSizeChunkerConfig


class FixedSizeChunker(Chunker):
    """Fixed-size character chunker implementation.
    
    This chunker splits document content into fixed-size character chunks,
    optionally overlapping sequentially to preserve context across boundaries.
    
    Example:
        ```python
        from src.chunkers import FixedSizeChunker, FixedSizeChunkerConfig
        from src.core.types import Document
        
        config = FixedSizeChunkerConfig(chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)
        
        chunks = chunker.chunk(document)
        ```
    """
    
    def __init__(self, config: FixedSizeChunkerConfig) -> None:
        """Initialize fixed-size chunker.
        
        Args:
            config: Configuration defining chunk size and overlap settings.
        """
        self._config = config
        
    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Chunk a document into fixed-size segments.
        
        This method splits the document content based on the configured chunk_size
        and overlap. If the document is empty, returns an empty list. If the document
        is shorter than chunk_size, returns a single chunk.
        
        Args:
            document: The document to chunk.
            
        Returns:
            Sequence of Chunk objects.
        """
        if not document.content:
            return []
            
        content = document.content
        content_length = len(content)
        chunk_size = self._config.chunk_size
        overlap = self._config.overlap
        step = chunk_size - overlap
        
        chunks = []
        chunk_index = 0
        start_char = 0
        
        while start_char < content_length:
            end_char = min(start_char + chunk_size, content_length)
            chunk_content = content[start_char:end_char]
            
            chunk_id = f"{document.id}_chunk_{chunk_index}"
            
            # Copy over document custom metadata
            custom_metadata = dict(document.metadata.custom) if document.metadata else {}
            
            chunk_metadata = ChunkMetadata(
                document_id=document.id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                custom=custom_metadata,
            )
            
            chunk = Chunk(
                id=chunk_id,
                content=chunk_content,
                metadata=chunk_metadata,
            )
            
            chunks.append(chunk)
            
            if end_char == content_length:
                break
                
            start_char += step
            chunk_index += 1
            
        return chunks
