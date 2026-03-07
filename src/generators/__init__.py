"""Generator implementations.

This package provides concrete implementations of the Generator interface.
"""

from .config import UniversalGeneratorConfig
from .registry import (
    get_generator,
    list_generators,
    register_generator,
    unregister_generator,
)
from .universal_generator import UniversalGenerator

# Register default generators
register_generator("universal", UniversalGenerator)

__all__ = [
    "UniversalGeneratorConfig",
    "UniversalGenerator",
    "get_generator",
    "list_generators",
    "register_generator",
    "unregister_generator",
]
