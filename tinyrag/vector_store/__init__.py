"""Vector store module."""

from tinyrag.vector_store.faiss_store import FAISSVectorStore
from tinyrag.vector_store.interfaces import VectorStore

__all__ = ["VectorStore", "FAISSVectorStore"]
