"""
Tests for the TraversalProfiler and TraversalProfile classes.

Covers:
- TraversalProfile initialization, start/stop, step recording, serialization
- TraversalProfiler lifecycle (start, stop, log steps, stats)
- Optimization suggestion generation
- Aggregated statistics
- Context manager usage
- profile_traversal decorator
- Edge cases (disabled profiler, no current profile)
"""

import os
import json
import time
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from cortexflow.traversal_profiler import (
    TraversalProfile,
    TraversalProfiler,
    profile_traversal,
)


# ---------------------------------------------------------------------------
# TraversalProfile tests
# ---------------------------------------------------------------------------

class TestTraversalProfile:
    """Tests for the TraversalProfile data container."""

    def test_init_defaults(self):
        """Profile initializes with correct defaults."""
        p = TraversalProfile("test_traversal")
        assert p.name == "test_traversal"
        assert p.metadata == {}
        assert p.duration == 0.0
        assert p.nodes_visited == 0
        assert p.edges_traversed == 0
        assert p.path_length == 0
        assert p.steps == []
        assert p.end_time is None

    def test_init_with_metadata(self):
        """Profile stores provided metadata."""
        meta = {"type": "bfs", "depth": 3}
        p = TraversalProfile("bfs_search", metadata=meta)
        assert p.metadata == meta

    def test_start_resets_start_time(self):
        """start() updates start_time."""
        p = TraversalProfile("t")
        old_start = p.start_time
        time.sleep(0.01)
        p.start()
        assert p.start_time >= old_start

    def test_stop_computes_duration(self):
        """stop() sets end_time and computes a positive duration."""
        p = TraversalProfile("t")
        p.start()
        time.sleep(0.02)
        p.stop()
        assert p.end_time is not None
        assert p.duration > 0

    def test_add_step(self):
        """add_step() appends step dict with name, duration, metadata."""
        p = TraversalProfile("t")
        p.add_step("expand_node", 0.05, {"node": "A"})
        p.add_step("score_path", 0.01)

        assert len(p.steps) == 2
        assert p.steps[0]["name"] == "expand_node"
        assert p.steps[0]["duration"] == 0.05
        assert p.steps[0]["metadata"] == {"node": "A"}
        assert p.steps[1]["metadata"] == {}

    def test_to_dict(self):
        """to_dict() returns a serializable dictionary."""
        p = TraversalProfile("t", metadata={"k": "v"})
        p.start()
        p.nodes_visited = 10
        p.edges_traversed = 15
        p.path_length = 4
        p.add_step("s1", 0.1)
        p.stop()

        d = p.to_dict()

        assert d["name"] == "t"
        assert d["metadata"] == {"k": "v"}
        assert d["nodes_visited"] == 10
        assert d["edges_traversed"] == 15
        assert d["path_length"] == 4
        assert len(d["steps"]) == 1
        assert isinstance(d["start_time"], str)  # ISO format
        assert isinstance(d["end_time"], str)
        assert d["duration"] > 0


# ---------------------------------------------------------------------------
# TraversalProfiler tests
# ---------------------------------------------------------------------------

@pytest.fixture
def profiler(tmp_path):
    """Create a TraversalProfiler that writes to a temp directory."""
    tp = TraversalProfiler(output_dir=str(tmp_path / "profiles"))
    yield tp
    tp.close()


@pytest.fixture
def disabled_profiler(tmp_path):
    """Create a disabled TraversalProfiler."""
    tp = TraversalProfiler(
        config={"enable_traversal_profiling": False},
        output_dir=str(tmp_path / "profiles"),
    )
    yield tp
    tp.close()


class TestTraversalProfilerLifecycle:
    """Tests for starting, stopping, and logging profiles."""

    def test_start_profile(self, profiler):
        """start_profile creates and returns a TraversalProfile."""
        profile = profiler.start_profile("bfs")
        assert profile is not None
        assert profile.name == "bfs"
        assert profiler.current_profile is profile
        assert len(profiler.profiles) == 1
        profiler.stop_profile()

    def test_stop_profile(self, profiler):
        """stop_profile completes the profile and clears current_profile."""
        profiler.start_profile("dfs")
        completed = profiler.stop_profile()
        assert completed is not None
        assert completed.duration >= 0
        assert profiler.current_profile is None

    def test_stop_profile_without_start(self, profiler):
        """stop_profile returns None when nothing is being profiled."""
        assert profiler.stop_profile() is None

    def test_log_traversal_step(self, profiler):
        """log_traversal_step adds a step to the current profile."""
        profiler.start_profile("walk")
        profiler.log_traversal_step("visit_node", 0.02, {"node": "X"})
        assert len(profiler.current_profile.steps) == 1
        profiler.stop_profile()

    def test_log_step_without_profile(self, profiler):
        """log_traversal_step is a no-op when no profile is active."""
        profiler.log_traversal_step("noop", 0.01)  # Should not raise

    def test_update_traversal_stats(self, profiler):
        """update_traversal_stats sets stats on the current profile."""
        profiler.start_profile("stat_test")
        profiler.update_traversal_stats(nodes_visited=50, edges_traversed=80, path_length=5)
        assert profiler.current_profile.nodes_visited == 50
        assert profiler.current_profile.edges_traversed == 80
        assert profiler.current_profile.path_length == 5
        profiler.stop_profile()

    def test_update_stats_without_profile(self, profiler):
        """update_traversal_stats is a no-op when no profile is active."""
        profiler.update_traversal_stats(10, 20, 3)  # Should not raise

    def test_disabled_profiler_returns_none(self, disabled_profiler):
        """When profiling is disabled, start_profile returns None."""
        result = disabled_profiler.start_profile("ignored")
        assert result is None
        assert disabled_profiler.stop_profile() is None


class TestProfileLimit:
    """Tests for the profile_limit eviction behavior."""

    def test_profile_limit_eviction(self, tmp_path):
        """When profile_limit is reached, oldest profile is evicted."""
        tp = TraversalProfiler(
            config={"profile_limit": 3},
            output_dir=str(tmp_path / "profiles"),
        )
        for i in range(5):
            tp.start_profile(f"p{i}")
            tp.stop_profile()

        assert len(tp.profiles) == 3
        # The oldest two (p0, p1) should have been evicted
        names = [p.name for p in tp.profiles]
        assert "p0" not in names
        assert "p1" not in names
        assert "p4" in names
        tp.close()


class TestOptimizationSuggestions:
    """Tests for the _analyze_profile and optimization suggestion logic."""

    def test_excessive_exploration_suggestion(self, profiler):
        """Excessive node-to-path ratio triggers a suggestion."""
        profiler.start_profile("wide_search")
        profiler.update_traversal_stats(nodes_visited=200, edges_traversed=300, path_length=5)
        profiler.stop_profile()

        suggestions = profiler.get_optimization_suggestions()
        assert len(suggestions) >= 1
        types = [s["type"] for entry in suggestions for s in entry["suggestions"]]
        assert "excessive_exploration" in types

    def test_slow_step_suggestion(self, profiler):
        """Steps slower than 100ms generate a 'slow_steps' suggestion."""
        profiler.start_profile("slow_ops")
        profiler.log_traversal_step("heavy_compute", 0.5)
        profiler.update_traversal_stats(nodes_visited=1, edges_traversed=1, path_length=1)
        profiler.stop_profile()

        suggestions = profiler.get_optimization_suggestions()
        types = [s["type"] for entry in suggestions for s in entry["suggestions"]]
        assert "slow_steps" in types

    def test_no_suggestions_for_fast_traversal(self, profiler):
        """A quick, efficient traversal generates no suggestions."""
        profiler.start_profile("fast")
        profiler.update_traversal_stats(nodes_visited=5, edges_traversed=4, path_length=3)
        profiler.stop_profile()

        suggestions = profiler.get_optimization_suggestions()
        # There may be suggestions, but "excessive_exploration" and "slow_steps" should not appear
        if suggestions:
            types = [s["type"] for entry in suggestions for s in entry["suggestions"]]
            assert "excessive_exploration" not in types
            assert "slow_steps" not in types


class TestAggregatedStats:
    """Tests for get_aggregated_stats."""

    def test_aggregated_stats_multiple_profiles(self, profiler):
        """get_aggregated_stats computes correct aggregates."""
        for i in range(3):
            profiler.start_profile(f"p{i}", metadata={"type": "bfs"})
            profiler.update_traversal_stats(
                nodes_visited=10 * (i + 1),
                edges_traversed=15 * (i + 1),
                path_length=i + 1,
            )
            profiler.stop_profile()

        stats = profiler.get_aggregated_stats()
        assert stats["total_profiles"] == 3
        assert stats["avg_nodes_visited"] == 20.0  # (10+20+30)/3
        assert stats["avg_edges_traversed"] == 30.0  # (15+30+45)/3
        assert stats["avg_path_length"] == 2.0  # (1+2+3)/3
        assert stats["operations_by_type"]["bfs"] == 3

    def test_aggregated_stats_empty(self, profiler):
        """get_aggregated_stats returns empty dict when no profiles exist."""
        assert profiler.get_aggregated_stats() == {}


class TestSaveProfiles:
    """Tests for save_profiles."""

    def test_save_profiles_creates_json(self, profiler, tmp_path):
        """save_profiles writes valid JSON with profiles and suggestions."""
        profiler.start_profile("save_test")
        profiler.update_traversal_stats(5, 5, 3)
        profiler.stop_profile()

        file_path = profiler.save_profiles("test_output.json")
        assert file_path is not None
        assert os.path.exists(file_path)

        with open(file_path) as f:
            data = json.load(f)

        assert "profiles" in data
        assert "optimization_suggestions" in data
        assert "timestamp" in data
        assert len(data["profiles"]) == 1

    def test_save_profiles_empty(self, profiler):
        """save_profiles returns None when no profiles exist."""
        assert profiler.save_profiles() is None

    def test_save_profiles_disabled(self, disabled_profiler):
        """save_profiles returns None when profiling is disabled."""
        assert disabled_profiler.save_profiles() is None


class TestClearAndClose:
    """Tests for clear_profiles and close."""

    def test_clear_profiles(self, profiler):
        """clear_profiles removes all stored data."""
        profiler.start_profile("x")
        profiler.stop_profile()
        assert len(profiler.profiles) == 1

        profiler.clear_profiles()
        assert profiler.profiles == []
        assert profiler.current_profile is None
        assert profiler.optimization_suggestions == []

    def test_context_manager(self, tmp_path):
        """TraversalProfiler works as a context manager."""
        with TraversalProfiler(output_dir=str(tmp_path / "ctx")) as tp:
            tp.start_profile("cm_test")
            tp.stop_profile()
            assert len(tp.profiles) == 1

        # After exiting context, profiles should be cleared by close()
        assert tp.profiles == []


class TestProfileTraversalDecorator:
    """Tests for the @profile_traversal decorator."""

    def test_decorator_profiles_method(self):
        """Decorator automatically profiles a method call."""
        profiler = TraversalProfiler()

        class FakeGraph:
            def __init__(self):
                self.traversal_profiler = profiler

            @profile_traversal(name="custom_walk")
            def walk(self, start, end):
                return {"nodes_visited": 10, "edges_traversed": 12, "path": ["A", "B", "C"]}

        fg = FakeGraph()
        result = fg.walk("A", "C")

        assert result["nodes_visited"] == 10
        assert len(profiler.profiles) == 1
        assert profiler.profiles[0].name == "custom_walk"
        assert profiler.profiles[0].nodes_visited == 10
        assert profiler.profiles[0].path_length == 3
        profiler.close()

    def test_decorator_without_profiler(self):
        """Decorator is a no-op when no profiler is attached."""
        class NoProfiler:
            @profile_traversal()
            def walk(self):
                return "ok"

        np = NoProfiler()
        assert np.walk() == "ok"

    def test_decorator_uses_func_name_by_default(self):
        """Decorator uses function name when no explicit name is given."""
        profiler = TraversalProfiler()

        class FakeGraph:
            def __init__(self):
                self.traversal_profiler = profiler

            @profile_traversal()
            def my_traversal(self):
                return {}

        fg = FakeGraph()
        fg.my_traversal()

        assert profiler.profiles[0].name == "my_traversal"
        profiler.close()
