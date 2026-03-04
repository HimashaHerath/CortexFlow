"""
CortexFlow Graph Store package.

Provides graph-based knowledge representation for CortexFlow, split into
focused modules:

- ``_deps``              -- optional dependency imports and feature flags
- ``relation_extractor`` -- RelationExtractor and shared SVO helpers
- ``schema``             -- DDL validation and database schema management
- ``traversal``          -- path finding, bidirectional search, graph abstraction
- ``merger``             -- GraphMerger for intelligent graph merging
- ``store``              -- main GraphStore class (entity/relation CRUD, NER, search)

Backward-compatible usage::

    from cortexflow.graph_store import GraphStore, RelationExtractor, GraphMerger
"""

from __future__ import annotations

# Re-export dependency flags so existing code that does
#     from cortexflow.graph_store import SPACY_ENABLED
# continues to work.
from ._deps import (
    FLAIR_ENABLED,
    FUZZY_MATCHING_ENABLED,
    NETWORKX_ENABLED,
    ONTOLOGY_ENABLED,
    SPACY_ENABLED,
    SPANBERT_ENABLED,
)
from .merger import GraphMerger
from .relation_extractor import RelationExtractor, _extract_svo_triples_from_sentence
from .store import GraphStore

__all__ = [
    # Primary public classes
    "GraphStore",
    "RelationExtractor",
    "GraphMerger",
    # Feature flags
    "NETWORKX_ENABLED",
    "SPACY_ENABLED",
    "FLAIR_ENABLED",
    "SPANBERT_ENABLED",
    "FUZZY_MATCHING_ENABLED",
    "ONTOLOGY_ENABLED",
    # Shared helper
    "_extract_svo_triples_from_sentence",
]
