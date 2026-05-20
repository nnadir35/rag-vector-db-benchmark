from typing import List
from .base import BaseEmbedder, EmbeddingDTO

class BGEM3Embedder(BaseEmbedder):
    """Local embedding implementation using the BGE-m3 model via HuggingFace."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            # Using BGE-M3 model natively with SentenceTransformers
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError as e:
            raise ImportError(
                "BGEM3Embedder requires sentence-transformers. Install with: uv add sentence-transformers"
            ) from e

    def embed_text(self, text: str) -> EmbeddingDTO:
        self._ensure_model_loaded()
        # encode returns numpy array by default
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        return EmbeddingDTO(vector=vector, dimension=len(vector))

    def embed_batch(self, texts: List[str]) -> List[EmbeddingDTO]:
        if not texts:
            return []
            
        self._ensure_model_loaded()
        vectors = self._model.encode(texts, normalize_embeddings=True).tolist()
        
        return [EmbeddingDTO(vector=vec, dimension=len(vec)) for vec in vectors]
