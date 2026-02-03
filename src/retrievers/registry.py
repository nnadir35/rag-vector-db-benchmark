"""Registry pattern for retriever discovery and instantiation.

This module provides a registry system that allows retrievers to be
registered and retrieved by name, enabling configuration-driven
component selection without hardcoding.
"""

from typing import Dict, List, Type

from ..core.retrieval import Retriever


# Global registry mapping retriever names to their classes
RETRIEVER_REGISTRY: Dict[str, Type[Retriever]] = {}


def register_retriever(name: str, retriever_class: Type[Retriever]) -> None:
    """Register a retriever class in the global registry.
    
    This function allows retriever implementations to register themselves
    with a unique identifier, enabling them to be instantiated by name
    from configuration files.
    
    Args:
        name: Unique identifier for the retriever (e.g., 'pinecone', 'weaviate')
        retriever_class: The retriever class to register (must be a subclass of Retriever)
        
    Raises:
        ValueError: If name is already registered or retriever_class is not a Retriever subclass
    """
    if not issubclass(retriever_class, Retriever):
        raise ValueError(
            f"retriever_class must be a subclass of Retriever, got {retriever_class}"
        )
    
    if name in RETRIEVER_REGISTRY:
        raise ValueError(
            f"Retriever '{name}' is already registered. "
            f"Use a different name or unregister the existing one first."
        )
    
    RETRIEVER_REGISTRY[name] = retriever_class


def get_retriever(name: str) -> Type[Retriever]:
    """Get a retriever class by name from the registry.
    
    Args:
        name: The registered name of the retriever
        
    Returns:
        The retriever class
        
    Raises:
        KeyError: If the retriever name is not registered
    """
    if name not in RETRIEVER_REGISTRY:
        available = ", ".join(sorted(RETRIEVER_REGISTRY.keys()))
        raise KeyError(
            f"Retriever '{name}' is not registered. "
            f"Available retrievers: {available if available else 'none'}"
        )
    
    return RETRIEVER_REGISTRY[name]


def list_retrievers() -> List[str]:
    """List all registered retriever names.
    
    Returns:
        List of registered retriever names, sorted alphabetically
    """
    return sorted(RETRIEVER_REGISTRY.keys())


def unregister_retriever(name: str) -> None:
    """Unregister a retriever from the registry.
    
    This is primarily useful for testing or dynamic reconfiguration.
    
    Args:
        name: The name of the retriever to unregister
        
    Raises:
        KeyError: If the retriever name is not registered
    """
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Retriever '{name}' is not registered")
    
    del RETRIEVER_REGISTRY[name]
