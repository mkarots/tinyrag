"""Interfaces for vector storage and search."""

from abc import ABC, abstractmethod

import numpy as np

from raglet.core.chunk import Chunk


class VectorStore(ABC):
    """Interface for storing and searching vectors."""

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors to the store with associated chunks.

        Args:
            vectors: NumPy array of shape (n_vectors, embedding_dim)
            chunks: List of Chunk objects corresponding to vectors

        Raises:
            ValueError: If vectors and chunks have different lengths
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[Chunk]:
        """Search for similar vectors.

        Args:
            query_vector: NumPy array of shape (embedding_dim,)
            top_k: Number of results to return

        Returns:
            List of Chunk objects with score attribute set, sorted by similarity
            (most similar first). Returns empty list if store is empty.
        """
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get the number of vectors stored.

        Returns:
            Number of vectors in the store
        """
        pass

    @abstractmethod
    def get_all_vectors(self) -> np.ndarray:
        """Retrieve all indexed vectors.

        Returns a contiguous float32 array of shape ``(n_vectors, dim)``.
        Implementations may reconstruct from the underlying index; callers
        should not assume the returned array shares memory with the index.

        Returns:
            NumPy array of shape (n_vectors, embedding_dim), dtype float32.
            Empty array with shape (0, dim) when the store is empty.
        """
        pass
