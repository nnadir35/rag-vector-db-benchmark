"""Interface for document ingestion.

Document ingestion is the process of loading raw documents from various sources
and converting them into the framework's Document format.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .types import Document


class DocumentIngester(ABC):
    """Abstract interface for document ingestion.
    
    Document ingesters are responsible for loading documents from various sources
    (files, databases, APIs, etc.) and converting them into the framework's
    standardized Document format.
    
    Implementations must handle:
    - Reading from different source formats
    - Extracting text content
    - Extracting and preserving metadata
    - Generating unique document IDs
    
    Implementations must NOT:
    - Perform chunking (use Chunker interface)
    - Perform embedding (use Embedder interface)
    - Know about retrieval or generation
    """
    
    @abstractmethod
    def ingest(self, source: str, **kwargs: object) -> Iterator[Document]:
        """Ingest documents from a source.
        
        This method reads documents from the specified source and yields
        Document objects. The source format is implementation-specific
        (e.g., file path, database connection string, API endpoint).
        
        Args:
            source: Source identifier (format depends on implementation)
            **kwargs: Additional implementation-specific parameters
            
        Yields:
            Document objects from the source
            
        Raises:
            ValueError: If source format is invalid
            IOError: If source cannot be read
            RuntimeError: If ingestion fails for implementation-specific reasons
        """
        pass
    
    @abstractmethod
    def can_ingest(self, source: str) -> bool:
        """Check if this ingester can handle the given source.
        
        This method allows the framework to select the appropriate ingester
        for a given source without attempting ingestion.
        
        Args:
            source: Source identifier to check
            
        Returns:
            True if this ingester can handle the source, False otherwise
        """
        pass
