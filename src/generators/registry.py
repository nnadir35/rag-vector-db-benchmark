"""Registry pattern for generator discovery and instantiation.

This module provides a registry system that allows generators to be
registered and retrieved by name.
"""

from typing import Dict, List, Type

from ..core.generation import Generator


GENERATOR_REGISTRY: Dict[str, Type[Generator]] = {}


def register_generator(name: str, generator_class: Type[Generator]) -> None:
    """Register a generator class in the global registry.
    
    Args:
        name: Unique identifier for the generator
        generator_class: The generator class to register
        
    Raises:
        ValueError: If name is registered or class is not a Generator
    """
    if not issubclass(generator_class, Generator):
        raise ValueError(
            f"generator_class must be a subclass of Generator, got {generator_class}"
        )
    
    if name in GENERATOR_REGISTRY:
        raise ValueError(f"Generator '{name}' is already registered.")
    
    GENERATOR_REGISTRY[name] = generator_class


def get_generator(name: str) -> Type[Generator]:
    """Get a generator class by name from the registry."""
    if name not in GENERATOR_REGISTRY:
        available = ", ".join(sorted(GENERATOR_REGISTRY.keys()))
        raise KeyError(
            f"Generator '{name}' is not registered. "
            f"Available generators: {available if available else 'none'}"
        )
    
    return GENERATOR_REGISTRY[name]


def list_generators() -> List[str]:
    """List all registered generator names."""
    return sorted(GENERATOR_REGISTRY.keys())


def unregister_generator(name: str) -> None:
    """Unregister a generator from the registry."""
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Generator '{name}' is not registered")
    
    del GENERATOR_REGISTRY[name]
