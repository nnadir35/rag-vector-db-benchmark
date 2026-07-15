"""Factory for constructing retriever instances from YAML-loaded configuration."""

from typing import Any

from ..core.embedding import Embedder
from ..core.retrieval import Retriever


def build_retriever_from_yaml(raw_config: dict[str, Any], embedder: Embedder) -> Retriever:
    """Instantiate the retriever specified by ``retriever.type`` in the YAML config.

    Args:
        raw_config: Full experiment dict (must contain optional ``retriever`` section).
        embedder: Embedder used for query encoding.

    Returns:
        A concrete ``Retriever`` implementation.

    Raises:
        ValueError: If ``retriever.type`` is unknown.
    """
    section = raw_config.get("retriever", {})
    rtype = section.get("type", "chroma")
    kwargs = {k: v for k, v in section.items() if k != "type"}
    if rtype == "pinecone":
        from .config import PineconeRetrieverConfig
        from .pinecone_retriever import PineconeRetriever

        return PineconeRetriever(PineconeRetrieverConfig(**kwargs), embedder)
    if rtype == "chroma":
        from .chroma_retriever import ChromaRetriever
        from .config import ChromaRetrieverConfig

        return ChromaRetriever(ChromaRetrieverConfig(**kwargs), embedder)
    if rtype == "qdrant":
        from .config import QdrantRetrieverConfig
        from .qdrant_retriever import QdrantRetriever

        return QdrantRetriever(QdrantRetrieverConfig(**kwargs), embedder)
    if rtype == "faiss":
        from .config import FAISSRetrieverConfig
        from .faiss_retriever import FAISSRetriever

        return FAISSRetriever(FAISSRetrieverConfig(**kwargs), embedder)
    if rtype == "milvus":
        from .config import MilvusRetrieverConfig
        from .milvus_retriever import MilvusRetriever

        return MilvusRetriever(MilvusRetrieverConfig(**kwargs), embedder)
    if rtype == "weaviate":
        from .config import WeaviateRetrieverConfig
        from .weaviate_retriever import WeaviateRetriever

        return WeaviateRetriever(WeaviateRetrieverConfig(**kwargs), embedder)
    raise ValueError(
        f"Unknown retriever type '{rtype}'. Expected one of: pinecone, chroma, qdrant, faiss, milvus, weaviate."
    )

