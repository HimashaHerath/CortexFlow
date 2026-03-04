"""
Tests for the CortexFlow Performance Optimizer module.

Covers ReasoningPattern data class, PerformanceOptimizer initialization,
graph partitioning (Louvain, connected-components fallback), hop index creation,
query plan generation, reasoning pattern caching, cache pruning, cache clearing,
statistics, partition pruning, close/cleanup, and persistence of patterns.
"""

import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import community as community_louvain  # noqa: F401
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False

from cortexflow.config import CortexFlowConfig
from cortexflow.performance_optimizer import (
    PerformanceOptimizer,
    ReasoningPattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> CortexFlowConfig:
    """Create a CortexFlowConfig with sensible defaults for testing."""
    defaults = {
        "use_graph_partitioning": False,
        "use_multihop_indexing": False,
        "max_graph_hops": 3,
        "max_indexed_hops": 2,
    }
    defaults.update(overrides)
    return CortexFlowConfig.from_dict(defaults)


def _build_small_graph():
    """Build a small directed networkx graph for testing.

    Graph topology (6 nodes, 7 edges):
        A -[related_to]-> B -[related_to]-> C
        A -[causes]-> D -[causes]-> E -[causes]-> F
        B -[causes]-> E
        D -[related_to]-> B
    """
    G = nx.DiGraph()
    G.add_node("A", type="concept", label="Alpha")
    G.add_node("B", type="concept", label="Beta")
    G.add_node("C", type="concept", label="Gamma")
    G.add_node("D", type="process", label="Delta")
    G.add_node("E", type="process", label="Epsilon")
    G.add_node("F", type="result", label="Zeta")

    G.add_edge("A", "B", type="related_to")
    G.add_edge("B", "C", type="related_to")
    G.add_edge("A", "D", type="causes")
    G.add_edge("D", "E", type="causes")
    G.add_edge("E", "F", type="causes")
    G.add_edge("B", "E", type="causes")
    G.add_edge("D", "B", type="related_to")
    return G


def _mock_graph_store(graph=None):
    """Return a mock graph store with a real or empty networkx graph."""
    store = MagicMock()
    store.graph = graph if graph is not None else nx.DiGraph()
    # By default no index-creation methods exist (tests add them when needed)
    if not hasattr(store, "create_index"):
        del store.create_index
    if not hasattr(store, "create_composite_index"):
        del store.create_composite_index
    if not hasattr(store, "create_path_index"):
        del store.create_path_index
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Basic config with optimizations disabled."""
    return _make_config()


@pytest.fixture
def small_graph():
    """Small directed graph for partitioning / indexing tests."""
    return _build_small_graph()


@pytest.fixture
def optimizer(config):
    """Optimizer without a graph store."""
    return PerformanceOptimizer(config, graph_store=None)


@pytest.fixture
def optimizer_with_graph(config, small_graph):
    """Optimizer attached to a mock graph store holding a small graph."""
    store = _mock_graph_store(small_graph)
    return PerformanceOptimizer(config, graph_store=store)


# ===================================================================
# ReasoningPattern
# ===================================================================

class TestReasoningPattern:
    """Tests for the ReasoningPattern data class."""

    def test_init_defaults(self):
        pattern = ReasoningPattern("key1")
        assert pattern.pattern_key == "key1"
        assert pattern.hop_count == 0
        assert pattern.entities == []
        assert pattern.path == []
        assert pattern.hit_count == 0
        assert isinstance(pattern.created_at, datetime)
        assert isinstance(pattern.last_accessed, datetime)

    def test_init_with_values(self):
        pattern = ReasoningPattern(
            "key2", hop_count=3,
            entities=["Person", "Place"],
            path=["A", "rel1", "B", "rel2", "C"]
        )
        assert pattern.hop_count == 3
        assert pattern.entities == ["Person", "Place"]
        assert len(pattern.path) == 5

    def test_update_stats_increments_hit_count(self):
        pattern = ReasoningPattern("k")
        assert pattern.hit_count == 0
        pattern.update_stats()
        assert pattern.hit_count == 1
        pattern.update_stats()
        assert pattern.hit_count == 2

    def test_update_stats_refreshes_last_accessed(self):
        pattern = ReasoningPattern("k")
        old_ts = pattern.last_accessed
        # Force a tiny delay so the timestamp can differ
        time.sleep(0.01)
        pattern.update_stats()
        assert pattern.last_accessed >= old_ts

    def test_to_dict_and_from_dict_roundtrip(self):
        original = ReasoningPattern(
            "roundtrip", hop_count=2,
            entities=["X", "Y"],
            path=["X", "edge", "Y"]
        )
        original.update_stats()  # hit_count = 1
        d = original.to_dict()

        restored = ReasoningPattern.from_dict(d)
        assert restored.pattern_key == "roundtrip"
        assert restored.hop_count == 2
        assert restored.entities == ["X", "Y"]
        assert restored.path == ["X", "edge", "Y"]
        assert restored.hit_count == 1
        assert isinstance(restored.last_accessed, datetime)
        assert isinstance(restored.created_at, datetime)

    def test_to_dict_contains_expected_keys(self):
        d = ReasoningPattern("k").to_dict()
        expected_keys = {
            "pattern_key", "hop_count", "entities", "path",
            "hit_count", "last_accessed", "created_at"
        }
        assert set(d.keys()) == expected_keys


# ===================================================================
# PerformanceOptimizer — Initialization
# ===================================================================

class TestOptimizerInit:
    """Tests for PerformanceOptimizer construction."""

    def test_init_sets_empty_caches(self, optimizer):
        assert optimizer.query_cache == {}
        assert optimizer.reasoning_cache == {}
        assert optimizer.reasoning_patterns == {}

    def test_init_sets_default_stats(self, optimizer):
        s = optimizer.stats
        assert s["query_cache_hits"] == 0
        assert s["query_cache_misses"] == 0
        assert s["reasoning_cache_hits"] == 0
        assert s["reasoning_cache_misses"] == 0
        assert s["partitions_created"] == 0

    def test_init_without_graph_store(self, config):
        opt = PerformanceOptimizer(config, graph_store=None)
        assert opt.graph_store is None
        assert opt.partitions == {}

    def test_init_does_not_partition_when_disabled(self, config, small_graph):
        store = _mock_graph_store(small_graph)
        opt = PerformanceOptimizer(config, graph_store=store)
        assert opt.partitions == {}

    def test_init_partitions_when_enabled(self, small_graph):
        cfg = _make_config(use_graph_partitioning=True)
        store = _mock_graph_store(small_graph)
        opt = PerformanceOptimizer(cfg, graph_store=store)
        # Should have created at least one partition
        assert len(opt.partitions) >= 1


# ===================================================================
# Graph Partitioning
# ===================================================================

@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
class TestGraphPartitioning:
    """Tests for partition_graph method."""

    def test_partition_with_no_graph_store(self, optimizer):
        result = optimizer.partition_graph()
        assert result["status"] == "failed"
        assert result["partitions"] == 0

    def test_partition_empty_graph(self, config):
        store = _mock_graph_store(nx.DiGraph())
        opt = PerformanceOptimizer(config, graph_store=store)
        result = opt.partition_graph()
        assert result["status"] == "failed"
        assert "Empty graph" in result["reason"]

    def test_partition_default_method_connected_components(self, optimizer_with_graph):
        result = optimizer_with_graph.partition_graph(method="components")
        assert result["status"] == "success"
        assert result["partitions"] >= 1
        assert "partition_stats" in result

    @pytest.mark.skipif(not HAS_COMMUNITY, reason="python-louvain not installed")
    def test_partition_louvain(self, optimizer_with_graph):
        result = optimizer_with_graph.partition_graph(method="louvain")
        assert result["status"] == "success"
        assert result["method"] == "louvain"
        assert result["partitions"] >= 1
        # Every node should be in the partition mapping
        graph = optimizer_with_graph.graph_store.graph
        for node in graph.nodes():
            assert node in optimizer_with_graph.partition_mapping

    def test_partition_modularity(self, optimizer_with_graph):
        result = optimizer_with_graph.partition_graph(method="modularity")
        assert result["status"] == "success"
        assert result["partitions"] >= 1

    def test_partition_stats_updated(self, optimizer_with_graph):
        optimizer_with_graph.partition_graph(method="components")
        assert optimizer_with_graph.stats["partitions_created"] >= 1
        assert optimizer_with_graph.stats["optimization_time"] > 0


# ===================================================================
# Hop Indexes
# ===================================================================

@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
class TestHopIndexes:
    """Tests for create_hop_indexes and helpers."""

    def test_create_hop_indexes_no_graph_store(self, optimizer):
        result = optimizer.create_hop_indexes()
        assert result["status"] == "failed"

    def test_create_hop_indexes_empty_graph(self, config):
        store = _mock_graph_store(nx.DiGraph())
        opt = PerformanceOptimizer(config, graph_store=store)
        result = opt.create_hop_indexes()
        assert result["status"] == "failed"
        assert "Empty graph" in result["reason"]

    def test_create_hop_indexes_populates_node_and_relation_types(self, config, small_graph):
        store = _mock_graph_store(small_graph)
        # Provide all index-creation methods as no-ops
        store.create_index = MagicMock()
        store.create_composite_index = MagicMock()
        store.create_path_index = MagicMock()
        opt = PerformanceOptimizer(config, graph_store=store)
        result = opt.create_hop_indexes(max_hops=3)
        assert result["status"] == "success"
        # The small graph has node types: concept, process, result
        assert set(result["indexed_node_types"]) == {"concept", "process", "result"}
        # Relation types: related_to, causes
        assert set(result["indexed_relation_types"]) == {"related_to", "causes"}

    def test_create_direct_indexes_calls_create_index(self, config, small_graph):
        store = _mock_graph_store(small_graph)
        store.create_index = MagicMock()
        opt = PerformanceOptimizer(config, graph_store=store)
        count = opt._create_direct_indexes(small_graph)
        # Should have called create_index at least once (for edge types and node attrs)
        assert count > 0
        assert store.create_index.call_count == count

    def test_create_direct_indexes_without_method(self, config, small_graph):
        store = _mock_graph_store(small_graph)
        # create_index not present -> should return 0
        if hasattr(store, "create_index"):
            del store.create_index
        opt = PerformanceOptimizer(config, graph_store=store)
        count = opt._create_direct_indexes(small_graph)
        assert count == 0


# ===================================================================
# Query Plan Generation
# ===================================================================

class TestQueryPlanGeneration:
    """Tests for generate_query_plan."""

    def test_path_query_default_bidirectional(self, optimizer):
        query = {
            "type": "path",
            "start_entity": "A",
            "end_entity": "C",
            "max_hops": 3,
        }
        plan = optimizer.generate_query_plan(query)
        assert plan["strategy"] == "bidirectional"
        assert plan["estimated_cost"] == 1.0
        assert len(plan["steps"]) > 0

    def test_path_query_uses_hop_index_when_enabled(self):
        cfg = _make_config(use_multihop_indexing=True, max_indexed_hops=3)
        opt = PerformanceOptimizer(cfg, graph_store=None)
        query = {
            "type": "path",
            "start_entity": "A",
            "end_entity": "B",
            "max_hops": 2,
        }
        plan = opt.generate_query_plan(query)
        assert plan["strategy"] == "hop_index"
        assert plan["estimated_cost"] == 0.5

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_path_query_partition_based_same_partition(self, small_graph):
        cfg = _make_config()
        store = _mock_graph_store(small_graph)
        opt = PerformanceOptimizer(cfg, graph_store=store)
        # Manually set up partition mapping so A and B are in the same partition
        opt.partitions = {0: small_graph}
        opt.partition_mapping = {n: 0 for n in small_graph.nodes()}

        query = {
            "type": "path",
            "start_entity": "A",
            "end_entity": "B",
            "max_hops": 3,
        }
        plan = opt.generate_query_plan(query)
        assert plan["strategy"] == "partition_based"
        assert plan["estimated_cost"] == 0.3

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_path_query_cross_partition(self, small_graph):
        cfg = _make_config()
        store = _mock_graph_store(small_graph)
        opt = PerformanceOptimizer(cfg, graph_store=store)
        # Put A and C in different partitions
        sub_a = small_graph.subgraph(["A", "B"]).copy()
        sub_c = small_graph.subgraph(["C", "D", "E", "F"]).copy()
        opt.partitions = {0: sub_a, 1: sub_c}
        opt.partition_mapping = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 1, "F": 1}

        query = {
            "type": "path",
            "start_entity": "A",
            "end_entity": "C",
            "max_hops": 3,
        }
        plan = opt.generate_query_plan(query)
        assert plan["strategy"] == "cross_partition"
        assert plan["estimated_cost"] == 0.7

    def test_subgraph_query_radius_based(self, optimizer):
        query = {
            "type": "subgraph",
            "start_entity": "A",
            "radius": 2,
        }
        plan = optimizer.generate_query_plan(query)
        assert plan["strategy"] == "radius_based"

    def test_query_plan_cached_on_repeat(self, optimizer):
        query = {"type": "path", "start_entity": "X", "end_entity": "Y", "max_hops": 2}
        plan1 = optimizer.generate_query_plan(query)
        plan2 = optimizer.generate_query_plan(query)
        # Second call should be a cache hit
        assert optimizer.stats["query_cache_hits"] >= 1
        assert plan1["strategy"] == plan2["strategy"]

    def test_stats_updated_after_plan_generation(self, optimizer):
        query = {"type": "path", "start_entity": "X", "end_entity": "Y", "max_hops": 2}
        optimizer.generate_query_plan(query)
        assert optimizer.stats["query_plans_generated"] >= 1
        assert optimizer.stats["query_cache_misses"] >= 1


# ===================================================================
# Reasoning Pattern Caching
# ===================================================================

class TestReasoningCaching:
    """Tests for cache_reasoning_pattern and get_cached_reasoning."""

    def test_cache_and_retrieve_pattern(self, optimizer):
        pattern_data = {
            "entity_types": ["Person", "Place"],
            "relation_types": ["lives_in"],
            "hop_count": 1,
        }
        result = {"answer": "Alice lives in Wonderland"}
        optimizer.cache_reasoning_pattern(pattern_data, result)

        hit, cached = optimizer.get_cached_reasoning(pattern_data)
        assert hit is True
        assert cached == result

    def test_cache_miss_returns_none(self, optimizer):
        pattern_data = {"entity_types": ["Unknown"], "hop_count": 0}
        hit, cached = optimizer.get_cached_reasoning(pattern_data)
        assert hit is False
        assert cached is None

    def test_cache_hit_increments_stats(self, optimizer):
        pattern_data = {"entity_types": ["A"], "hop_count": 1}
        optimizer.cache_reasoning_pattern(pattern_data, "result_value")
        optimizer.get_cached_reasoning(pattern_data)
        assert optimizer.stats["reasoning_cache_hits"] == 1

    def test_cache_miss_increments_stats(self, optimizer):
        pattern_data = {"entity_types": ["Missing"], "hop_count": 1}
        optimizer.get_cached_reasoning(pattern_data)
        assert optimizer.stats["reasoning_cache_misses"] == 1

    def test_cache_updates_hit_count_on_existing_pattern(self, optimizer):
        pattern_data = {"entity_types": ["A"], "hop_count": 1}
        key = optimizer.cache_reasoning_pattern(pattern_data, "r1")
        # Cache the same pattern again — should update stats
        optimizer.cache_reasoning_pattern(pattern_data, "r2")
        pattern = optimizer.reasoning_patterns[key]
        assert pattern.hit_count == 1  # update_stats called on second cache call


# ===================================================================
# Cache Pruning
# ===================================================================

class TestCachePruning:
    """Tests for _prune_reasoning_cache."""

    def test_no_prune_when_under_limit(self, optimizer):
        optimizer.max_cache_size = 100
        for i in range(50):
            key = f"pattern_{i}"
            optimizer.reasoning_cache[key] = f"result_{i}"
            optimizer.reasoning_patterns[key] = ReasoningPattern(key)
        optimizer._prune_reasoning_cache()
        assert len(optimizer.reasoning_cache) == 50

    def test_prune_removes_expired_entries(self, optimizer):
        optimizer.max_cache_size = 5
        optimizer.cache_expiry = 1  # 1 second expiry

        # Add 10 entries, all with old timestamps
        for i in range(10):
            key = f"pattern_{i}"
            optimizer.reasoning_cache[key] = f"result_{i}"
            p = ReasoningPattern(key)
            p.last_accessed = datetime.now() - timedelta(seconds=60)
            optimizer.reasoning_patterns[key] = p

        optimizer._prune_reasoning_cache()
        # All should be expired and removed
        assert len(optimizer.reasoning_cache) == 0

    def test_prune_lru_when_still_over_limit(self, optimizer):
        optimizer.max_cache_size = 3
        optimizer.cache_expiry = 9999  # nothing expires

        # Add 6 entries with staggered access times
        for i in range(6):
            key = f"pattern_{i}"
            optimizer.reasoning_cache[key] = f"result_{i}"
            p = ReasoningPattern(key)
            p.last_accessed = datetime.now() - timedelta(seconds=100 - i * 10)
            optimizer.reasoning_patterns[key] = p

        optimizer._prune_reasoning_cache()
        assert len(optimizer.reasoning_cache) <= 3


# ===================================================================
# Pattern Key & Extraction
# ===================================================================

class TestPatternKeyAndExtraction:
    """Tests for generate_pattern_key and extract_pattern_from_query."""

    def test_generate_pattern_key_deterministic(self, optimizer):
        pattern = {"entity_types": ["A", "B"], "relation_types": ["r1"], "hop_count": 2}
        key1 = optimizer.generate_pattern_key(pattern)
        key2 = optimizer.generate_pattern_key(pattern)
        assert key1 == key2

    def test_generate_pattern_key_different_for_different_patterns(self, optimizer):
        p1 = {"entity_types": ["A"], "hop_count": 1}
        p2 = {"entity_types": ["B"], "hop_count": 1}
        assert optimizer.generate_pattern_key(p1) != optimizer.generate_pattern_key(p2)

    def test_generate_pattern_key_sorts_entity_types(self, optimizer):
        p1 = {"entity_types": ["B", "A"]}
        p2 = {"entity_types": ["A", "B"]}
        assert optimizer.generate_pattern_key(p1) == optimizer.generate_pattern_key(p2)

    def test_extract_pattern_from_query_extracts_entities(self, optimizer):
        query = {
            "entities": [{"type": "Person"}, {"type": "Place"}],
        }
        pattern = optimizer.extract_pattern_from_query(query)
        assert set(pattern["entity_types"]) == {"Person", "Place"}

    def test_extract_pattern_from_query_extracts_relations(self, optimizer):
        query = {
            "relations": [{"type": "lives_in"}, {"type": "works_at"}],
        }
        pattern = optimizer.extract_pattern_from_query(query)
        assert set(pattern["relation_types"]) == {"lives_in", "works_at"}

    def test_extract_pattern_calculates_hop_count(self, optimizer):
        query = {
            "path": ["A", "rel1", "B", "rel2", "C"],
        }
        pattern = optimizer.extract_pattern_from_query(query)
        assert pattern["hop_count"] == 2


# ===================================================================
# Optimize Query Execution
# ===================================================================

class TestOptimizeQueryExecution:
    """Tests for optimize_query_execution."""

    def test_optimize_returns_plan(self, optimizer):
        query = {"type": "path", "start_entity": "X", "end_entity": "Y", "max_hops": 2}
        result = optimizer.optimize_query_execution(query)
        assert "execution_plan" in result
        assert result["execution_plan"]["strategy"] in (
            "bidirectional", "hop_index", "partition_based", "fallback"
        )

    def test_optimize_caches_result(self, optimizer):
        query = {"type": "path", "start_entity": "X", "end_entity": "Y", "max_hops": 2}
        optimizer.optimize_query_execution(query)
        # Second call hits cache
        optimizer.optimize_query_execution(query)
        assert optimizer.stats["query_cache_hits"] >= 1


# ===================================================================
# Cache Clearing
# ===================================================================

class TestCacheClearing:
    """Tests for clear_caches."""

    def test_clear_caches_empties_all(self, optimizer):
        optimizer.query_cache["k1"] = "v1"
        optimizer.reasoning_cache["k2"] = "v2"
        optimizer.reasoning_patterns["k3"] = ReasoningPattern("k3")

        result = optimizer.clear_caches()
        assert result["query_cache_cleared"] == 1
        assert result["reasoning_cache_cleared"] == 1
        assert optimizer.query_cache == {}
        assert optimizer.reasoning_cache == {}
        assert optimizer.reasoning_patterns == {}


# ===================================================================
# Statistics
# ===================================================================

class TestGetStats:
    """Tests for get_stats."""

    def test_stats_contain_expected_keys(self, optimizer):
        stats = optimizer.get_stats()
        assert "query_cache_hits" in stats
        assert "reasoning_cache_hits" in stats
        assert "query_hit_rate" in stats
        assert "reasoning_hit_rate" in stats
        assert "query_cache_size" in stats
        assert "common_patterns" in stats

    def test_hit_rate_zero_when_no_queries(self, optimizer):
        stats = optimizer.get_stats()
        assert stats["query_hit_rate"] == 0.0
        assert stats["reasoning_hit_rate"] == 0.0

    def test_hit_rate_calculation(self, optimizer):
        optimizer.stats["query_cache_hits"] = 3
        optimizer.stats["query_cache_misses"] = 7
        stats = optimizer.get_stats()
        assert stats["query_hit_rate"] == pytest.approx(30.0)


# ===================================================================
# Partition Pruning
# ===================================================================

@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
class TestPartitionPruning:
    """Tests for prune_partitions."""

    def test_prune_removes_single_node_partitions(self):
        cfg = _make_config()
        store = _mock_graph_store(_build_small_graph())
        opt = PerformanceOptimizer(cfg, graph_store=store)

        # Create partitions: one with a single node, one with multiple
        single_node_graph = nx.DiGraph()
        single_node_graph.add_node("lonely")
        multi_node_graph = nx.DiGraph()
        multi_node_graph.add_nodes_from(["a", "b", "c"])
        multi_node_graph.add_edge("a", "b")
        multi_node_graph.add_edge("b", "c")

        opt.partitions = {0: single_node_graph, 1: multi_node_graph}
        opt.partition_mapping = {"lonely": 0, "a": 1, "b": 1, "c": 1}

        pruned = opt.prune_partitions(density_threshold=0.0)
        assert pruned >= 1
        assert 0 not in opt.partitions
        assert "lonely" not in opt.partition_mapping

    def test_prune_removes_low_density_partitions(self):
        cfg = _make_config()
        store = _mock_graph_store(_build_small_graph())
        opt = PerformanceOptimizer(cfg, graph_store=store)

        # Create a sparse partition (density < threshold)
        sparse = nx.DiGraph()
        sparse.add_nodes_from(["x", "y", "z", "w"])
        sparse.add_edge("x", "y")  # density = 1/12 ~ 0.083

        dense = nx.DiGraph()
        dense.add_nodes_from(["a", "b"])
        dense.add_edge("a", "b")
        dense.add_edge("b", "a")  # density = 1.0

        opt.partitions = {0: sparse, 1: dense}
        opt.partition_mapping = {"x": 0, "y": 0, "z": 0, "w": 0, "a": 1, "b": 1}

        pruned = opt.prune_partitions(density_threshold=0.5)
        assert pruned == 1
        assert 0 not in opt.partitions
        assert 1 in opt.partitions

    def test_prune_returns_zero_when_no_partitions(self, optimizer):
        assert optimizer.prune_partitions() == 0


# ===================================================================
# Close / Cleanup
# ===================================================================

class TestClose:
    """Tests for close and __del__."""

    def test_close_clears_internal_state(self, optimizer):
        optimizer.query_cache["a"] = 1
        optimizer.reasoning_cache["b"] = 2
        optimizer.partitions["c"] = "data"
        optimizer.partition_mapping["d"] = 0

        optimizer.close()

        assert optimizer.query_cache == {}
        assert optimizer.reasoning_cache == {}
        assert optimizer.partitions == {}
        assert optimizer.partition_mapping == {}

    def test_close_handles_no_conn_attribute(self, optimizer):
        # Should not raise even without a conn attribute
        optimizer.close()

    def test_close_closes_db_connection_if_present(self, optimizer):
        mock_conn = MagicMock()
        optimizer.conn = mock_conn
        optimizer.close()
        mock_conn.close.assert_called_once()
        assert optimizer.conn is None


# ===================================================================
# Persistence of Reasoning Patterns
# ===================================================================

class TestPatternPersistence:
    """Tests for _load_cached_patterns and _save_cached_patterns."""

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "patterns.json")

            # Create a config that enables persistence
            cfg = _make_config()
            cfg.custom_config["persist_reasoning_cache"] = True
            cfg.custom_config["reasoning_cache_file"] = cache_file
            # Monkey-patch so hasattr/getattr checks work
            cfg.persist_reasoning_cache = True
            cfg.reasoning_cache_file = cache_file

            opt = PerformanceOptimizer(cfg, graph_store=None)
            # Cache a pattern
            opt.reasoning_patterns["test_key"] = ReasoningPattern(
                "test_key", hop_count=2,
                entities=["E1"], path=["E1", "rel", "E2"]
            )
            opt._save_cached_patterns()

            # Verify the file was written
            assert os.path.exists(cache_file)

            # Create a new optimizer that should load the saved patterns
            opt2 = PerformanceOptimizer(cfg, graph_store=None)
            opt2._load_cached_patterns()
            assert "test_key" in opt2.reasoning_patterns
            loaded = opt2.reasoning_patterns["test_key"]
            assert loaded.hop_count == 2
            assert loaded.entities == ["E1"]

    def test_load_ignores_missing_file(self, optimizer):
        optimizer.config.persist_reasoning_cache = True
        optimizer.config.reasoning_cache_file = "/nonexistent/path/patterns.json"
        # Should not raise
        optimizer._load_cached_patterns()

    def test_load_handles_corrupt_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "bad.json")
            with open(cache_file, "w") as f:
                f.write("{corrupt json!!!")

            cfg = _make_config()
            cfg.persist_reasoning_cache = True
            cfg.reasoning_cache_file = cache_file

            opt = PerformanceOptimizer(cfg, graph_store=None)
            # Should not raise, just log the error
            opt._load_cached_patterns()
            assert opt.reasoning_patterns == {}


# ===================================================================
# _generate_cache_key
# ===================================================================

class TestGenerateCacheKey:
    """Tests for _generate_cache_key."""

    def test_same_query_same_key(self, optimizer):
        q = {"type": "path", "start": "A", "end": "B"}
        assert optimizer._generate_cache_key(q) == optimizer._generate_cache_key(q)

    def test_key_order_independent(self, optimizer):
        q1 = {"a": 1, "b": 2}
        q2 = {"b": 2, "a": 1}
        assert optimizer._generate_cache_key(q1) == optimizer._generate_cache_key(q2)

    def test_different_queries_different_keys(self, optimizer):
        q1 = {"type": "path", "start": "A"}
        q2 = {"type": "path", "start": "B"}
        assert optimizer._generate_cache_key(q1) != optimizer._generate_cache_key(q2)


# ===================================================================
# _calculate_hit_rate
# ===================================================================

class TestCalculateHitRate:
    """Tests for _calculate_hit_rate helper."""

    def test_zero_total_returns_zero(self, optimizer):
        assert optimizer._calculate_hit_rate(0, 0) == 0.0

    def test_all_hits(self, optimizer):
        assert optimizer._calculate_hit_rate(10, 0) == 100.0

    def test_mixed(self, optimizer):
        assert optimizer._calculate_hit_rate(1, 3) == pytest.approx(25.0)
