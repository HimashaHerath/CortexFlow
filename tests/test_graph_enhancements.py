#!/usr/bin/env python
"""
Test script for CortexFlow enhanced entity recognition and relation extraction.

These tests require the spaCy en_core_web_sm model to be installed.
"""

import logging

import pytest

from cortexflow.config import CortexFlowConfig
from cortexflow.graph_store import GraphStore, RelationExtractor

logger = logging.getLogger(__name__)

# Check if spaCy model is available
try:
    import spacy
    spacy.load("en_core_web_sm")
    HAS_SPACY_MODEL = True
except Exception:
    HAS_SPACY_MODEL = False

pytestmark = pytest.mark.skipif(
    not HAS_SPACY_MODEL, reason="spaCy en_core_web_sm model not installed"
)


def test_entity_recognition():
    """Test enhanced entity recognition capabilities."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"

    graph = GraphStore(config)

    text = """
    Albert Einstein developed the theory of relativity while working in Berlin.
    Apple Inc. is headquartered in Cupertino and was founded by Steve Jobs and Steve Wozniak.
    Python is a popular programming language used in machine learning and data science.
    The film Inception, directed by Christopher Nolan, was released on July 16, 2010.
    """

    entities = graph.extract_entities(text)
    assert len(entities) > 0, "Should extract at least one entity"


def test_entity_linking():
    """Test entity linking and fuzzy matching."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"

    graph = GraphStore(config)

    einstein_id = graph.add_entity(
        entity="Albert Einstein",
        entity_type="PERSON",
        metadata={
            "aliases": ["Einstein", "A. Einstein"],
            "birth_year": 1879,
            "death_year": 1955
        }
    )

    graph.add_entity(
        entity="Python",
        entity_type="PROGRAMMING_LANGUAGE",
        metadata={
            "creator": "Guido van Rossum",
            "year": 1991
        }
    )

    graph.add_entity_alias(einstein_id, "Father of Relativity")

    text = """
    Einstein's work on relativity revolutionized physics.
    The Father of Relativity received the Nobel Prize in 1921.
    Many data scientists prefer Python programming for AI development.
    """

    entities = graph.extract_entities(text)
    assert isinstance(entities, list)


def test_relation_extraction():
    """Test enhanced relation extraction capabilities."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"

    graph = GraphStore(config)

    text = """
    Albert Einstein developed the theory of relativity.
    Marie Curie discovered radium and polonium.
    Alan Turing invented the Turing machine.
    Amazon is headquartered in Seattle.
    Bill Gates founded Microsoft with Paul Allen.
    """

    relations = graph.extract_relations(text)
    assert isinstance(relations, list)
    assert len(relations) > 0, "Should extract at least one relation"

    for subj, pred, obj in relations:
        assert len(subj) > 0
        assert len(pred) > 0
        assert len(obj) > 0

    extractor = RelationExtractor()
    more_relations = extractor.extract_relations(text)
    assert isinstance(more_relations, list)


@pytest.mark.timeout(180)
def test_process_text_to_graph():
    """Test end-to-end processing of text to graph."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"

    graph = GraphStore(config)

    text = """
    Albert Einstein developed the theory of relativity while working in Berlin.
    He was born in Ulm, Germany and won the Nobel Prize in Physics in 1921.
    """

    relations_added = graph.process_text_to_graph(text, source="test")
    assert isinstance(relations_added, int)
    assert relations_added >= 0

    subgraph = graph.build_knowledge_subgraph("Albert Einstein")
    assert isinstance(subgraph, dict)
    assert "nodes" in subgraph
    assert "edges" in subgraph
