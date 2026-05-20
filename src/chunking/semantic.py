from typing import Any, Dict, List, Optional
from .base import BaseChunker, ChunkDTO

class SemanticChunker(BaseChunker):
    """Chunks text preserving semantic boundaries like sentences.
    
    Note: Requires 'nltk' to be installed via uv (uv add nltk).
    """

    def __init__(self, language: str = "english") -> None:
        self.language = language
        self._ensure_dependencies()

    def _ensure_dependencies(self) -> None:
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
        except ImportError as e:
            raise ImportError(
                "SemanticChunker requires nltk. Please install it using: uv add nltk"
            ) from e

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkDTO]:
        from nltk.tokenize import sent_tokenize
        
        base_metadata = metadata or {}
        sentences = sent_tokenize(text, language=self.language)
        
        return [
            ChunkDTO(
                content=sentence.strip(),
                metadata={**base_metadata, "chunk_index": i, "strategy": "semantic"}
            )
            for i, sentence in enumerate(sentences) if sentence.strip()
        ]
