"""Tests for fixed-size chunker implementation."""

import pytest

from src.chunkers.config import FixedSizeChunkerConfig
from src.chunkers.fixed_size_chunker import FixedSizeChunker
from src.core.types import Document


def test_chunk_splits_document_into_correct_number_of_chunks():
    """Test that a document is split into the expected number of chunks based on size and overlap."""
    config = FixedSizeChunkerConfig(chunk_size=10, overlap=5)
    chunker = FixedSizeChunker(config)

    # 25 chars -> step is 5
    # chunk 0: 0:10
    # chunk 1: 5:15
    # chunk 2: 10:20
    # chunk 3: 15:25
    doc = Document(id="doc1", content="0123456789012345678901234")

    chunks = chunker.chunk(doc)
    assert len(chunks) == 4


def test_chunk_overlap_is_applied_correctly():
    """Test that the overlap is correctly applied to contiguous chunks."""
    config = FixedSizeChunkerConfig(chunk_size=10, overlap=4)
    chunker = FixedSizeChunker(config)

    # "abcdefghij" is 10 chars
    # "klmnopqrst" is 10 chars
    doc = Document(id="doc1", content="abcdefghijklmnopqrst")

    chunks = chunker.chunk(doc)

    # Expected chunks with size=10, overlap=4, step=6:
    # chunk 0: 0-10 -> abcdefghij
    # chunk 1: 6-16 -> ghijklmnop
    # chunk 2: 12-20 -> mnopqrst

    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content == "ghijklmnop"
    assert chunks[2].content == "mnopqrst"


def test_chunk_ids_are_unique_and_deterministic():
    """Test that chunk IDs match the required format and are distinct."""
    config = FixedSizeChunkerConfig(chunk_size=5, overlap=0)
    chunker = FixedSizeChunker(config)

    doc = Document(id="test-doc", content="1234567890")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].id == "test-doc_chunk_0"
    assert chunks[1].id == "test-doc_chunk_1"


def test_chunk_metadata_has_correct_start_and_end_chars():
    """Test that start_char and end_char in ChunkMetadata correctly map back to the slice."""
    config = FixedSizeChunkerConfig(chunk_size=6, overlap=2)
    chunker = FixedSizeChunker(config)

    content = "Hello there world"
    doc = Document(id="d", content=content)
    chunks = chunker.chunk(doc)

    for chunk in chunks:
        # Verify the slice from the original content matches the chunk content
        expected_content = content[chunk.metadata.start_char:chunk.metadata.end_char]
        assert chunk.content == expected_content
        assert chunk.metadata.document_id == "d"


def test_empty_document_returns_empty_list():
    """Test that chunking an empty document correctly returns an empty list."""
    config = FixedSizeChunkerConfig()
    chunker = FixedSizeChunker(config)

    doc = Document(id="empty", content="")
    chunks = chunker.chunk(doc)

    assert chunks == []


def test_document_shorter_than_chunk_size_returns_single_chunk():
    """Test that chunking a short document returns exactly one chunk containing all text."""
    config = FixedSizeChunkerConfig(chunk_size=100)
    chunker = FixedSizeChunker(config)

    doc = Document(id="short", content="Short")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].content == "Short"
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char == 5


def test_config_raises_when_overlap_exceeds_chunk_size():
    """Test that FixedSizeChunkerConfig validates overlap strictly less than chunk_size."""
    with pytest.raises(ValueError, match="must be strictly less than"):
        FixedSizeChunkerConfig(chunk_size=10, overlap=15)

    with pytest.raises(ValueError, match="must be strictly less than"):
        FixedSizeChunkerConfig(chunk_size=10, overlap=10)
