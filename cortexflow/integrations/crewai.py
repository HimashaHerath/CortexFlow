"""
CortexFlow <-> CrewAI integration.

Provides :class:`CortexFlowCrewStorage`, a storage backend that satisfies the
CrewAI storage protocol by delegating to a
:class:`~cortexflow.manager.CortexFlowManager` instance.

When ``crewai`` is not installed the module can still be imported -- the class
simply does not inherit from any CrewAI base.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cortexflow.integrations.crewai")

# ---------------------------------------------------------------------------
# Optional CrewAI imports
# ---------------------------------------------------------------------------

try:
    from crewai.memory.storage.interface import Storage as CrewAIStorage

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    CrewAIStorage = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Determine the appropriate base class
# ---------------------------------------------------------------------------

_Base: type = (
    CrewAIStorage if CREWAI_AVAILABLE and CrewAIStorage is not None else object
)


# ---------------------------------------------------------------------------
# CortexFlowCrewStorage
# ---------------------------------------------------------------------------


class CortexFlowCrewStorage(_Base):  # type: ignore[misc]
    """CrewAI-compatible storage backend backed by CortexFlow.

    This adapter maps CrewAI's ``save`` / ``search`` / ``reset`` protocol
    onto the CortexFlow knowledge store so that CrewAI agents can
    transparently benefit from CortexFlow's multi-tier memory, hybrid
    retrieval, and knowledge-graph capabilities.

    Parameters
    ----------
    manager:
        An initialised :class:`~cortexflow.manager.CortexFlowManager` instance.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    # -- CrewAI Storage protocol --------------------------------------------

    def save(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a key/value pair in the CortexFlow knowledge store.

        The *key* is used as the ``source`` identifier and *value* is
        persisted as knowledge text.  Any *metadata* is included as part
        of the source tag so it can be retrieved later.

        Parameters
        ----------
        key:
            A unique key for the stored item (used as ``source``).
        value:
            The text content to store.
        metadata:
            Optional metadata dict.  Currently serialised into the source
            string for downstream retrieval.
        """
        knowledge_store = getattr(self.manager, "knowledge_store", None)
        if knowledge_store is None:
            logger.warning(
                "CortexFlowCrewStorage.save: manager has no knowledge_store."
            )
            return

        source = key
        if metadata:
            # Encode lightweight metadata into the source tag.
            meta_parts = [f"{k}={v}" for k, v in metadata.items()]
            source = f"{key} [{', '.join(meta_parts)}]"

        knowledge_store.add_knowledge(value, source=source)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search the CortexFlow knowledge store.

        Parameters
        ----------
        query:
            Natural-language search query.
        limit:
            Maximum number of results to return.
        score_threshold:
            Minimum relevance score (0.0-1.0).  Results below this
            threshold are excluded.

        Returns
        -------
        list[dict]:
            Each dict contains at minimum ``"text"`` (or ``"content"``)
            and ``"score"`` keys.
        """
        knowledge_store = getattr(self.manager, "knowledge_store", None)
        if knowledge_store is None:
            logger.warning(
                "CortexFlowCrewStorage.search: manager has no knowledge_store."
            )
            return []

        raw_results: list[dict[str, Any]] = knowledge_store.retrieve(
            query, max_results=limit
        )

        # Apply score threshold filtering.
        filtered: list[dict[str, Any]] = []
        for item in raw_results:
            score = item.get("score", item.get("relevance", 1.0))
            if score >= score_threshold:
                filtered.append(item)

        return filtered

    def reset(self) -> None:
        """Clear all stored knowledge."""
        knowledge_store = getattr(self.manager, "knowledge_store", None)
        if knowledge_store is None:
            logger.warning(
                "CortexFlowCrewStorage.reset: manager has no knowledge_store."
            )
            return

        knowledge_store.clear()
        logger.info("CortexFlowCrewStorage: knowledge store cleared.")
