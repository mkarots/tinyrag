"""tinyrag - Portable memory for small text corpora."""

__version__ = "0.1.0"

from tinyrag.config.config import (
    ChunkingConfig,
    EmbeddingConfig,
    SearchConfig,
    TinyRAGConfig,
)
from tinyrag.core.chunk import Chunk
from tinyrag.core.rag import TinyRAG

__all__ = [
    "TinyRAG",
    "Chunk",
    "TinyRAGConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "SearchConfig",
]
