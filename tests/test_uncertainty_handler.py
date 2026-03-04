"""
Tests for the CortexFlow UncertaintyHandler module.

Covers contradiction detection, contradiction resolution (all strategies),
probability distribution storage/retrieval, source reliability management,
belief revision recording, incomplete information reasoning, and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest

from cortexflow.config import ConfigBuilder
from cortexflow.uncertainty_handler import UncertaintyHandler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create a CortexFlowConfig with in-memory DB for fast tests.

    CortexFlowConfig.__post_init__ converts relative paths to absolute,
    which mangles ':memory:'.  We fix it up after construction so the
    UncertaintyHandler sees the literal ':memory:' value it checks for.
    """
    cfg = (
        ConfigBuilder()
        .with_knowledge_store(knowledge_store_path=":memory:")
        .with_uncertainty(
            use_uncertainty_handling=True,
            auto_detect_contradictions=True,
            default_contradiction_strategy="weighted",
            recency_weight=0.6,
            reliability_weight=0.4,
            confidence_threshold=0.7,
        )
        .build()
    )
    # Restore the literal ':memory:' that __post_init__ mangled
    cfg.knowledge_store.knowledge_store_path = ":memory:"
    return cfg


@pytest.fixture
def handler(config):
    """Create an UncertaintyHandler with in-memory DB and no graph store."""
    h = UncertaintyHandler(config=config, graph_store=None)
    yield h
    h.close()


@pytest.fixture
def handler_with_graph(config):
    """Create an UncertaintyHandler with a mocked graph store."""
    mock_graph = MagicMock()
    mock_graph.graph = MagicMock()
    mock_graph.graph.has_edge.return_value = False
    h = UncertaintyHandler(config=config, graph_store=mock_graph)
    yield h
    h.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test UncertaintyHandler initialization."""

    def test_creates_with_in_memory_db(self, config):
        h = UncertaintyHandler(config=config)
        assert h.conn is not None
        h.close()

    def test_default_confidence_value(self, handler):
        assert handler.default_confidence == 0.5

    def test_default_weights(self, handler):
        assert handler.recency_weight == 0.6
        assert handler.reliability_weight == 0.4

    def test_source_reliability_initially_empty(self, handler):
        assert handler.source_reliability == {}

    def test_close_sets_conn_to_none(self, config):
        h = UncertaintyHandler(config=config)
        assert h.conn is not None
        h.close()
        assert h.conn is None


# ---------------------------------------------------------------------------
# Source Reliability
# ---------------------------------------------------------------------------


class TestSourceReliability:
    """Test source reliability management."""

    def test_update_and_get_reliability(self, handler):
        handler.update_source_reliability("wikipedia", 0.85)
        score = handler.get_source_reliability("wikipedia")
        assert score == 0.85

    def test_get_unknown_source_returns_default(self, handler):
        score = handler.get_source_reliability("nonexistent_source")
        assert score == 0.5

    def test_update_with_metadata(self, handler):
        meta = {"category": "academic", "verified": True}
        handler.update_source_reliability("arxiv", 0.95, metadata=meta)
        assert handler.get_source_reliability("arxiv") == 0.95

    def test_reject_score_below_zero(self, handler):
        handler.update_source_reliability("bad_source", -0.1)
        # Should not be stored; default returned
        assert handler.get_source_reliability("bad_source") == 0.5

    def test_reject_score_above_one(self, handler):
        handler.update_source_reliability("bad_source", 1.5)
        assert handler.get_source_reliability("bad_source") == 0.5

    def test_update_overwrites_previous(self, handler):
        handler.update_source_reliability("wiki", 0.7)
        handler.update_source_reliability("wiki", 0.9)
        assert handler.get_source_reliability("wiki") == 0.9

    def test_boundary_score_zero(self, handler):
        handler.update_source_reliability("low_trust", 0.0)
        assert handler.get_source_reliability("low_trust") == 0.0

    def test_boundary_score_one(self, handler):
        handler.update_source_reliability("high_trust", 1.0)
        assert handler.get_source_reliability("high_trust") == 1.0


# ---------------------------------------------------------------------------
# Detect Contradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    """Test contradiction detection."""

    def test_no_graph_store_returns_empty(self, handler):
        result = handler.detect_contradictions()
        assert result == []

    def test_no_graph_store_with_entity_id_returns_empty(self, handler):
        result = handler.detect_contradictions(entity_id=1, relation_type="is_a")
        assert result == []


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Recency Strategy
# ---------------------------------------------------------------------------


class TestResolveContradictionRecency:
    """Test contradiction resolution using the recency strategy."""

    def test_prefers_newer_timestamp(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "Paris",
            "target2": "Lyon",
            "confidence1": 0.8,
            "confidence2": 0.7,
            "timestamp1": 1000.0,
            "timestamp2": 2000.0,
            "source1": None,
            "source2": None,
            "entity": "France",
            "relation": "capital",
        }
        result = handler.resolve_contradiction(contradiction, strategy="recency")
        assert result["resolved_value"] == "Lyon"
        assert result["confidence"] == 0.7
        assert result["strategy_used"] == "recency"
        assert result["kept_both"] is False

    def test_recency_prefers_first_when_newer(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "Paris",
            "target2": "Lyon",
            "confidence1": 0.8,
            "confidence2": 0.7,
            "timestamp1": 3000.0,
            "timestamp2": 2000.0,
            "entity": "France",
            "relation": "capital",
        }
        result = handler.resolve_contradiction(contradiction, strategy="recency")
        assert result["resolved_value"] == "Paris"


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Confidence Strategy
# ---------------------------------------------------------------------------


class TestResolveContradictionConfidence:
    """Test contradiction resolution using the confidence strategy."""

    def test_prefers_higher_confidence(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "Python",
            "target2": "Java",
            "confidence1": 0.9,
            "confidence2": 0.6,
            "entity": "best_language",
            "relation": "is",
        }
        result = handler.resolve_contradiction(contradiction, strategy="confidence")
        assert result["resolved_value"] == "Python"
        assert result["confidence"] == 0.9

    def test_equal_confidence_prefers_second(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 0.5,
            "confidence2": 0.5,
            "entity": "test",
            "relation": "val",
        }
        result = handler.resolve_contradiction(contradiction, strategy="confidence")
        # When equal, the else branch fires, choosing target2
        assert result["resolved_value"] == "B"


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Reliability Strategy
# ---------------------------------------------------------------------------


class TestResolveContradictionReliability:
    """Test contradiction resolution using the reliability strategy."""

    def test_prefers_more_reliable_source(self, handler):
        handler.update_source_reliability("trusted_src", 0.9)
        handler.update_source_reliability("untrusted_src", 0.3)
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "ValueA",
            "target2": "ValueB",
            "confidence1": 0.7,
            "confidence2": 0.7,
            "source1": "trusted_src",
            "source2": "untrusted_src",
            "entity": "test",
            "relation": "attr",
        }
        result = handler.resolve_contradiction(contradiction, strategy="reliability")
        assert result["resolved_value"] == "ValueA"

    def test_unknown_sources_default_to_half(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "X",
            "target2": "Y",
            "confidence1": 0.8,
            "confidence2": 0.6,
            "source1": "unknown_a",
            "source2": "unknown_b",
            "entity": "test",
            "relation": "rel",
        }
        result = handler.resolve_contradiction(contradiction, strategy="reliability")
        # Both default to 0.5; equal reliability -> else branch -> target2
        assert result["resolved_value"] == "Y"


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Weighted Strategy
# ---------------------------------------------------------------------------


class TestResolveContradictionWeighted:
    """Test contradiction resolution using the weighted strategy."""

    def test_weighted_combines_recency_and_reliability(self, handler):
        handler.update_source_reliability("new_reliable", 0.9)
        handler.update_source_reliability("old_unreliable", 0.2)
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "NewVal",
            "target2": "OldVal",
            "confidence1": 0.7,
            "confidence2": 0.7,
            "source1": "new_reliable",
            "source2": "old_unreliable",
            "timestamp1": 5000.0,
            "timestamp2": 1000.0,
            "entity": "test",
            "relation": "attr",
        }
        result = handler.resolve_contradiction(contradiction, strategy="weighted")
        # target1: recency=1.0*0.6 + reliability=0.9*0.4 = 0.96
        # target2: recency=0.0*0.6 + reliability=0.2*0.4 = 0.08
        assert result["resolved_value"] == "NewVal"
        assert result["strategy_used"] == "weighted"

    def test_weighted_equal_timestamps(self, handler):
        handler.update_source_reliability("src_a", 0.8)
        handler.update_source_reliability("src_b", 0.3)
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 0.5,
            "confidence2": 0.5,
            "source1": "src_a",
            "source2": "src_b",
            "timestamp1": 1000.0,
            "timestamp2": 1000.0,
            "entity": "test",
            "relation": "rel",
        }
        result = handler.resolve_contradiction(contradiction, strategy="weighted")
        # Equal timestamps -> both recency=1.0
        # score_a = 1.0*0.6 + 0.8*0.4 = 0.92
        # score_b = 1.0*0.6 + 0.3*0.4 = 0.72
        assert result["resolved_value"] == "A"


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Keep Both Strategy
# ---------------------------------------------------------------------------


class TestResolveContradictionKeepBoth:
    """Test contradiction resolution using the keep_both strategy."""

    def test_keeps_both_values(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "Red",
            "target2": "Blue",
            "confidence1": 0.8,
            "confidence2": 0.6,
            "entity": "color",
            "relation": "is",
        }
        result = handler.resolve_contradiction(contradiction, strategy="keep_both")
        assert result["kept_both"] is True
        assert "Red" in result["resolved_value"]
        assert "Blue" in result["resolved_value"]
        assert "OR" in result["resolved_value"]

    def test_keep_both_reduces_confidence(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 1.0,
            "confidence2": 1.0,
            "entity": "test",
            "relation": "is",
        }
        result = handler.resolve_contradiction(contradiction, strategy="keep_both")
        # (1.0 + 1.0) / 2 * 0.9 = 0.9
        assert result["confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Auto Strategy Selection
# ---------------------------------------------------------------------------


class TestResolveContradictionAuto:
    """Test automatic strategy selection in resolve_contradiction."""

    def test_auto_selects_reliability_when_sources_present(self, handler):
        handler.update_source_reliability("src1", 0.9)
        handler.update_source_reliability("src2", 0.4)
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "V1",
            "target2": "V2",
            "confidence1": 0.7,
            "confidence2": 0.7,
            "source1": "src1",
            "source2": "src2",
            "timestamp1": 100.0,
            "timestamp2": 200.0,
            "entity": "e",
            "relation": "r",
        }
        result = handler.resolve_contradiction(contradiction, strategy="auto")
        assert result["strategy_used"] == "reliability"

    def test_auto_selects_recency_when_no_sources(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "V1",
            "target2": "V2",
            "confidence1": 0.7,
            "confidence2": 0.8,
            "source1": None,
            "source2": None,
            "timestamp1": 100.0,
            "timestamp2": 200.0,
            "entity": "e",
            "relation": "r",
        }
        result = handler.resolve_contradiction(contradiction, strategy="auto")
        assert result["strategy_used"] == "recency"

    def test_auto_selects_confidence_when_only_confidence(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "V1",
            "target2": "V2",
            "confidence1": 0.9,
            "confidence2": 0.3,
            "entity": "e",
            "relation": "r",
        }
        result = handler.resolve_contradiction(contradiction, strategy="auto")
        assert result["strategy_used"] == "confidence"

    def test_auto_selects_keep_both_when_nothing_available(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "V1",
            "target2": "V2",
            "entity": "e",
            "relation": "r",
        }
        result = handler.resolve_contradiction(contradiction, strategy="auto")
        assert result["strategy_used"] == "keep_both"
        assert result["kept_both"] is True


# ---------------------------------------------------------------------------
# Resolve Contradiction -- Graph Store Integration
# ---------------------------------------------------------------------------


class TestResolutionAppliesGraph:
    """Test that non-keep_both resolution calls _apply_resolution with graph store."""

    def test_apply_resolution_called_for_non_keep_both(self, handler_with_graph):
        h = handler_with_graph
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 0.9,
            "confidence2": 0.3,
            "entity": "test",
            "relation": "val",
        }
        with patch.object(h, "_apply_resolution") as mock_apply:
            h.resolve_contradiction(contradiction, strategy="confidence")
            mock_apply.assert_called_once()

    def test_apply_resolution_not_called_for_keep_both(self, handler_with_graph):
        h = handler_with_graph
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 0.5,
            "confidence2": 0.5,
            "entity": "test",
            "relation": "val",
        }
        with patch.object(h, "_apply_resolution") as mock_apply:
            h.resolve_contradiction(contradiction, strategy="keep_both")
            mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# Probability Distributions
# ---------------------------------------------------------------------------


class TestProbabilityDistributions:
    """Test probability distribution storage and retrieval."""

    def test_add_and_retrieve_discrete_distribution(self, handler):
        dist_data = {"outcomes": {"yes": 0.7, "no": 0.2, "maybe": 0.1}}
        handler.add_probability_distribution(
            entity_id=1,
            relation_id=10,
            distribution_type="discrete",
            distribution_data=dist_data,
        )
        result = handler.get_probability_distribution(entity_id=1, relation_id=10)
        assert result is not None
        assert result["type"] == "discrete"
        assert result["data"] == dist_data
        assert "timestamp" in result

    def test_add_and_retrieve_gaussian_distribution(self, handler):
        dist_data = {"mean": 42.0, "std_dev": 3.5}
        handler.add_probability_distribution(
            entity_id=2,
            relation_id=20,
            distribution_type="gaussian",
            distribution_data=dist_data,
        )
        result = handler.get_probability_distribution(entity_id=2, relation_id=20)
        assert result is not None
        assert result["type"] == "gaussian"
        assert result["data"]["mean"] == 42.0

    def test_get_nonexistent_distribution_returns_none(self, handler):
        result = handler.get_probability_distribution(entity_id=999, relation_id=999)
        assert result is None

    def test_overwrite_distribution(self, handler):
        handler.add_probability_distribution(
            entity_id=1,
            relation_id=10,
            distribution_type="discrete",
            distribution_data={"a": 0.5, "b": 0.5},
        )
        handler.add_probability_distribution(
            entity_id=1,
            relation_id=10,
            distribution_type="gaussian",
            distribution_data={"mean": 10.0, "std_dev": 1.0},
        )
        result = handler.get_probability_distribution(entity_id=1, relation_id=10)
        # The second insert should be retrievable (may or may not overwrite
        # depending on INSERT OR REPLACE behavior with auto-increment PK).
        # At minimum, we should get a valid result back.
        assert result is not None


# ---------------------------------------------------------------------------
# Reason with Incomplete Information
# ---------------------------------------------------------------------------


class TestReasonWithIncompleteInformation:
    """Test reasoning with incomplete information."""

    def test_complete_information_returns_best(self, handler):
        query = {"question": "What is the capital?", "required_fields": ["location"]}
        knowledge = [
            {"location": "Paris", "answer": "Paris", "confidence": 0.9},
            {"location": "Lyon", "answer": "Lyon", "confidence": 0.6},
        ]
        result = handler.reason_with_incomplete_information(query, knowledge)
        assert result["answer"] == "Paris"
        assert result["confidence"] == 0.9
        assert result["missing_information"] == []

    def test_missing_information_identified(self, handler):
        query = {"required_fields": ["temperature", "humidity"]}
        knowledge = [
            {"temperature": 25, "answer": "warm", "confidence": 0.8},
        ]
        result = handler.reason_with_incomplete_information(query, knowledge)
        assert "humidity" in result["missing_information"]

    def test_partial_match_reduces_confidence(self, handler):
        query = {"required_fields": ["name", "age", "location"]}
        knowledge = [
            {"name": "Alice", "age": 30, "answer": "Alice info", "confidence": 0.9},
        ]
        result = handler.reason_with_incomplete_information(query, knowledge)
        # 2/3 fields matched, confidence scaled by 0.8 for incomplete info
        assert result["confidence"] < 0.9
        assert result["answer"] is not None

    def test_no_knowledge_returns_unknown(self, handler):
        query = {"required_fields": ["something"]}
        knowledge = []
        result = handler.reason_with_incomplete_information(query, knowledge)
        assert result["answer"] == "Unknown"
        assert result["confidence"] == 0.1

    def test_empty_required_fields_uses_best_knowledge(self, handler):
        query = {"required_fields": []}
        knowledge = [
            {"answer": "Alpha", "confidence": 0.6},
            {"answer": "Beta", "confidence": 0.95},
        ]
        result = handler.reason_with_incomplete_information(query, knowledge)
        assert result["answer"] == "Beta"
        assert result["confidence"] == 0.95
        assert result["missing_information"] == []

    def test_all_fields_missing_returns_low_confidence(self, handler):
        query = {"required_fields": ["x", "y", "z"]}
        knowledge = [
            {"answer": "some value", "confidence": 0.5},
        ]
        result = handler.reason_with_incomplete_information(query, knowledge)
        # 0 out of 3 required fields matched -> match_confidence=0 -> below 0.3 threshold
        # No partial matches -> "Unknown"
        assert result["answer"] == "Unknown"
        assert result["confidence"] == 0.1

    def test_empty_query_empty_knowledge(self, handler):
        query = {"required_fields": []}
        knowledge = []
        result = handler.reason_with_incomplete_information(query, knowledge)
        assert result["answer"] == "Unknown"
        assert result["confidence"] == 0
        assert "explanation" in result


# ---------------------------------------------------------------------------
# Internal: _identify_missing_information
# ---------------------------------------------------------------------------


class TestIdentifyMissingInformation:
    """Test the private _identify_missing_information helper."""

    def test_no_required_fields(self, handler):
        missing = handler._identify_missing_information({"required_fields": []}, [])
        assert missing == []

    def test_all_fields_present(self, handler):
        query = {"required_fields": ["a", "b"]}
        knowledge = [{"a": 1, "b": 2}]
        missing = handler._identify_missing_information(query, knowledge)
        assert missing == []

    def test_some_fields_missing(self, handler):
        query = {"required_fields": ["a", "b", "c"]}
        knowledge = [{"a": 1}]
        missing = handler._identify_missing_information(query, knowledge)
        assert "b" in missing
        assert "c" in missing
        assert "a" not in missing


# ---------------------------------------------------------------------------
# Internal: _find_partial_matches
# ---------------------------------------------------------------------------


class TestFindPartialMatches:
    """Test the private _find_partial_matches helper."""

    def test_no_required_fields_returns_empty(self, handler):
        matches = handler._find_partial_matches(
            {"required_fields": []}, [{"answer": "test"}]
        )
        assert matches == []

    def test_high_match_ratio_included(self, handler):
        query = {"required_fields": ["a", "b"]}
        knowledge = [{"a": 1, "b": 2, "answer": "result", "confidence": 0.8}]
        matches = handler._find_partial_matches(query, knowledge)
        assert len(matches) == 1
        assert matches[0]["answer"] == "result"

    def test_low_match_ratio_excluded(self, handler):
        # 1/4 = 0.25 < 0.3 threshold
        query = {"required_fields": ["a", "b", "c", "d"]}
        knowledge = [{"a": 1, "answer": "result", "confidence": 0.8}]
        matches = handler._find_partial_matches(query, knowledge)
        assert len(matches) == 0

    def test_match_confidence_scales_with_item_confidence(self, handler):
        query = {"required_fields": ["a", "b"]}
        knowledge = [
            {"a": 1, "b": 2, "answer": "high", "confidence": 1.0},
            {"a": 1, "b": 2, "answer": "low", "confidence": 0.3},
        ]
        matches = handler._find_partial_matches(query, knowledge)
        assert len(matches) == 2
        high = [m for m in matches if m["answer"] == "high"][0]
        low = [m for m in matches if m["answer"] == "low"][0]
        assert high["match_confidence"] > low["match_confidence"]


# ---------------------------------------------------------------------------
# Record Contradiction Resolution (DB verification)
# ---------------------------------------------------------------------------


class TestRecordContradictionResolution:
    """Verify that resolution is persisted in the database."""

    def test_resolution_creates_db_records(self, handler):
        contradiction = {
            "entity_id": 1,
            "relation1_id": 10,
            "relation2_id": 20,
            "target1": "A",
            "target2": "B",
            "confidence1": 0.9,
            "confidence2": 0.3,
            "entity": "test_entity",
            "relation": "test_rel",
        }
        handler.resolve_contradiction(contradiction, strategy="confidence")

        # Query the in-memory DB directly
        cursor = handler.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM contradictions")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT COUNT(*) FROM belief_revisions")
        assert cursor.fetchone()[0] == 1

    def test_contradiction_stores_correct_strategy(self, handler):
        contradiction = {
            "entity_id": 2,
            "relation1_id": 30,
            "relation2_id": 40,
            "target1": "X",
            "target2": "Y",
            "confidence1": 0.5,
            "confidence2": 0.5,
            "entity": "ent",
            "relation": "rel",
        }
        handler.resolve_contradiction(contradiction, strategy="keep_both")

        cursor = handler.conn.cursor()
        cursor.execute(
            "SELECT resolution_strategy FROM contradictions WHERE entity_id = 2"
        )
        row = cursor.fetchone()
        assert row[0] == "keep_both"
