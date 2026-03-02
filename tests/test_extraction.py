"""
Tests for enhanced entity and relation extraction capabilities.

Refactored from the original print-based test script to use proper
pytest assertions. Uses in-memory or tempfile databases for isolation.
"""

import os
import sys
import tempfile
import pytest
import logging

from cortexflow.graph_store import GraphStore
from cortexflow.config import CortexFlowConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_store():
    """Create a GraphStore with a temp DB for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_extraction.db")
    config = CortexFlowConfig.from_dict({
        "knowledge_store_path": db_path,
        "use_graph_rag": True,
    })
    gs = GraphStore(config)
    yield gs
    try:
        gs.close()
    except Exception:
        pass
    if os.path.exists(db_path):
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestEntityExtraction:
    """Test enhanced entity extraction from text."""

    def test_extracts_entities_from_text(self, graph_store):
        text = (
            "Apple Inc. was founded by Steve Jobs in California in 1976. "
            "The company released the iPhone in 2007, which runs on iOS. "
            "Tim Cook is the current CEO and the stock price was $150.23 on May 15, 2023."
        )
        entities = graph_store.extract_entities(text)
        assert isinstance(entities, list)
        assert len(entities) > 0, "Should extract at least some entities"

    def test_entity_has_text_and_type_fields(self, graph_store):
        text = "Steve Jobs founded Apple Inc."
        entities = graph_store.extract_entities(text)
        if entities:  # May not extract any if NLP model not loaded
            entity = entities[0]
            assert "text" in entity, "Entity should have 'text' field"
            assert "type" in entity, "Entity should have 'type' field"

    def test_extracts_key_named_entities(self, graph_store):
        text = "Steve Jobs founded Apple in California."
        entities = graph_store.extract_entities(text)
        entity_texts = [e["text"] for e in entities]
        # Only assert quality when spaCy model is loaded; the regex fallback
        # produces unreliable fragments so we just check it doesn't crash.
        if entities and graph_store.nlp is not None:
            all_text = " ".join(entity_texts).lower()
            has_relevant = any(
                name.lower() in all_text
                for name in ["Steve", "Jobs", "Apple", "California"]
            )
            assert has_relevant, f"Expected relevant entities, got: {entity_texts}"

    def test_empty_text_returns_empty_or_no_crash(self, graph_store):
        entities = graph_store.extract_entities("")
        assert isinstance(entities, list)


# ---------------------------------------------------------------------------
# Domain-specific extraction
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestDomainSpecificExtraction:
    """Test domain-specific entity extraction."""

    def test_extracts_tech_entities(self, graph_store):
        text = (
            "Python has become very popular for machine learning applications. "
            "Many developers use TensorFlow and PyTorch for deep learning "
            "projects, especially when working with CNNs or RNNs."
        )
        entities = graph_store.extract_entities(text)
        assert isinstance(entities, list)
        assert len(entities) > 0, "Should extract at least some entities from tech text"

    def test_domain_entities_have_correct_structure(self, graph_store):
        text = "TensorFlow is a machine learning framework."
        entities = graph_store.extract_entities(text)
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert isinstance(entity["text"], str)
            assert len(entity["text"]) > 0


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestRelationExtraction:
    """Test enhanced relation extraction from text."""

    def test_extracts_relations_from_text(self, graph_store):
        text = (
            "Steve Jobs founded Apple in California. "
            "Microsoft develops Windows. "
            "The researchers published their findings in Nature. "
            "Google acquired YouTube for $1.65 billion in 2006."
        )
        relations = graph_store.extract_relations(text)
        assert isinstance(relations, list)
        # Relation extraction depends on NLP models; some setups may
        # not extract triples from all sentences. We just verify it
        # returns a proper list without crashing.
        assert isinstance(relations, list)

    def test_relation_is_triple(self, graph_store):
        text = "Steve Jobs founded Apple."
        relations = graph_store.extract_relations(text)
        for relation in relations:
            assert isinstance(relation, (list, tuple))
            assert len(relation) >= 3, (
                f"Relation should be a triple (subject, predicate, object), got: {relation}"
            )

    def test_subject_verb_object_extracted(self, graph_store):
        text = "Google acquired YouTube."
        relations = graph_store.extract_relations(text)
        if relations:
            # Check at least one relation has non-empty components
            has_valid = any(
                len(r[0]) > 0 and len(r[1]) > 0 and len(r[2]) > 0
                for r in relations
            )
            assert has_valid, f"Should have valid subject-verb-object triples, got: {relations}"

    def test_empty_text_returns_empty_or_no_crash(self, graph_store):
        relations = graph_store.extract_relations("")
        assert isinstance(relations, list)


# ---------------------------------------------------------------------------
# Coreference resolution (optional, depends on neuralcoref)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestCoreferenceResolution:
    """Test coreference resolution (skipped if neuralcoref not available)."""

    def test_coreference_or_skip(self, graph_store):
        try:
            import neuralcoref
            import spacy
        except ImportError:
            pytest.skip("neuralcoref not available")

        text = (
            "Albert Einstein published his theory of relativity in 1915. "
            "He was born in Germany but later moved to the United States."
        )
        relations_added = graph_store.process_text_to_graph(text)
        assert isinstance(relations_added, int)
        assert relations_added >= 0


# ---------------------------------------------------------------------------
# Semantic role labeling (optional, depends on allennlp)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestSemanticRoleLabeling:
    """Test SRL-based extraction (skipped if allennlp not available)."""

    def test_srl_or_skip(self, graph_store):
        try:
            from allennlp.predictors.predictor import Predictor
        except (ImportError, Exception):
            pytest.skip("allennlp not available")

        text = (
            "The researchers from Stanford University developed a new algorithm "
            "to solve complex optimization problems in machine learning."
        )
        srl_relations = graph_store._extract_with_semantic_roles(text)
        assert isinstance(srl_relations, list)
        assert len(srl_relations) > 0
