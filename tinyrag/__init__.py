"""tinyrag - Portable memory for small text corpora."""

__version__ = "0.1.0"

from tinyrag.core.rag import TinyRAG
from tinyrag.core.chunk import Chunk
from tinyrag.config.config import TinyRAGConfig

__all__ = ["TinyRAG", "Chunk", "TinyRAGConfig"]
