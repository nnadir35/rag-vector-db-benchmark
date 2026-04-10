"""Interface for document chunking.

Chunking is the process of splitting documents into smaller text segments
that are suitable for embedding and retrieval.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .types import Chunk, Document


class Chunker(ABC):
    """Abstract interface for document chunking.

    Chunkers are responsible for splitting documents into smaller chunks
    that can be embedded and stored in vector databases. The chunking strategy
    (fixed-size, semantic, sentence-based, etc.) is implementation-specific.

    Implementations must handle:
    - Splitting document content into chunks
    - Preserving document metadata in chunk metadata
    - Generating unique chunk IDs
    - Tracking chunk position within source document

    Implementations must NOT:
    - Perform embedding (use Embedder interface)
    - Know about retrieval or generation
    - Modify document content
    """

    @abstractmethod
    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Chunk a document into smaller segments.

        This method takes a Document and splits it into Chunk objects.
        The chunking strategy (size, overlap, boundaries, etc.) is
        determined by the implementation.

        Args:
            document: The document to chunk

        Returns:
            Sequence of Chunk objects, ordered by position in source document

        Raises:
            ValueError: If document is invalid or cannot be chunked
        """
        pass

    def chunk_batch(self, documents: Sequence[Document]) -> Sequence[Chunk]:
        """Chunk multiple documents in batch.

        This method allows implementations to optimize batch processing
        if supported. The default implementation calls chunk() for each
        document, but implementations may override for efficiency.

        Args:
            documents: Sequence of documents to chunk

        Returns:
            Sequence of all Chunk objects from all documents, ordered by
            document and position within document
        """
        all_chunks: list[Chunk] = []
        for document in documents:
            chunks = self.chunk(document)
            all_chunks.extend(chunks)
        return all_chunks
