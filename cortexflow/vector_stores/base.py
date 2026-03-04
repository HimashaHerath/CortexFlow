"""Abstract base class and shared types for vector store backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorSearchResult:
    """A single vector search result.

    Attributes:
        id: Unique identifier for the stored embedding.
        text: The original text associated with the embedding.
        score: Similarity score (higher is more similar).
        type: Item type, e.g. ``"knowledge"`` or ``"fact"``.
        metadata: Arbitrary extra metadata attached to the item.
    """

    id: str | int
    text: str
    score: float
    type: str = "knowledge"
    metadata: dict[str, Any] | None = field(default=None, repr=False)


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends.

    Concrete implementations must provide storage, similarity search,
    deletion, and count operations for embedding vectors.
    """

    @abstractmethod
    def add_embedding(
        self,
        id: str | int,
        text: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store an embedding vector.

        Args:
            id: Unique identifier for the item.
            text: The original text.
            embedding: A 1-D numpy array of floats.
            metadata: Optional metadata dict.
        """

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, max_results: int = 10
    ) -> list[VectorSearchResult]:
        """Search for the most similar vectors.

        Args:
            query_embedding: A 1-D numpy query vector.
            max_results: Maximum number of results to return.

        Returns:
            A list of :class:`VectorSearchResult` ordered by descending score.
        """

    @abstractmethod
    def delete(self, id: str | int) -> bool:
        """Delete an embedding by ID.

        Args:
            id: Identifier of the embedding to remove.

        Returns:
            ``True`` if an item was deleted, ``False`` otherwise.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored embeddings."""

    def close(self) -> None:
        """Clean up resources.  Override if needed."""
        pass
