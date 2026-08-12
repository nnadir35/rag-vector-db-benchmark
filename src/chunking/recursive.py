from typing import Any

from .base import BaseChunker, ChunkDTO


class RecursiveCharacterChunker(BaseChunker):
    """Chunks text with a fixed size and overlap using recursive character splitting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, separators: list[str] | None = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap cannot be greater than or equal to chunk_size")

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text based on separators."""
        final_chunks: list[str] = []
        if not separators:
            return [text]

        separator = separators[0]
        splits = text.split(separator) if separator else list(text)

        current_chunk = ""
        for split in splits:
            item = split + (separator if separator and split != splits[-1] else "")
            if len(current_chunk) + len(item) <= self.chunk_size:
                current_chunk += item
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)

                # Recursively split the oversized item if we have more separators
                if len(item) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text(item, separators[1:])
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Overlap handling for the next chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + item if current_chunk else item

        if current_chunk:
            final_chunks.append(current_chunk)

        return [c.strip() for c in final_chunks if c.strip()]

    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[ChunkDTO]:
        base_metadata = metadata or {}
        chunks_text = self._split_text(text, self.separators)

        return [
            ChunkDTO(
                content=chunk,
                metadata={**base_metadata, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks_text)
        ]
