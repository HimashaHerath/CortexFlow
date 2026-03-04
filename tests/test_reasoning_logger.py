"""
Tests for the ReasoningLogger and ReasoningContext classes.

Covers:
- Logger initialization (with and without DB)
- Reasoning session lifecycle (start, step, path, end)
- Database persistence and retrieval
- Session analysis
- Export functionality
- Context manager usage
- Edge cases (no active session, empty DB)
"""

import json
import os
import sqlite3

import pytest

from cortexflow.reasoning_logger import ReasoningLogger


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "reasoning_test.db")


@pytest.fixture
def logger_with_db(db_path):
    """Create a ReasoningLogger with database logging enabled, file logging disabled."""
    rl = ReasoningLogger(
        db_path=db_path,
        log_level="WARNING",
        enable_file_logging=False,
    )
    return rl


@pytest.fixture
def logger_no_db():
    """Create a ReasoningLogger without database logging."""
    rl = ReasoningLogger(
        db_path=None,
        log_level="WARNING",
        enable_file_logging=False,
    )
    return rl


class TestReasoningLoggerInit:
    """Tests for ReasoningLogger initialization."""

    def test_init_without_db(self, logger_no_db):
        """Logger initializes without a database path."""
        assert logger_no_db.db_path is None
        assert logger_no_db.current_path is None
        assert logger_no_db.current_reasoning_id is None

    def test_init_with_db_creates_tables(self, logger_with_db, db_path):
        """Logger creates required tables when db_path is provided."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "reasoning_sessions" in tables
        assert "reasoning_steps" in tables
        assert "reasoning_paths" in tables

    def test_max_path_length_default(self, logger_no_db):
        """Default max_path_length is 1000."""
        assert logger_no_db.max_path_length == 1000

    def test_custom_max_path_length(self, tmp_path):
        """Custom max_path_length is respected."""
        rl = ReasoningLogger(
            db_path=None,
            log_level="WARNING",
            enable_file_logging=False,
            max_path_length=50,
        )
        assert rl.max_path_length == 50


class TestReasoningSessionLifecycle:
    """Tests for start_reasoning / log_reasoning_step / log_path / end_reasoning."""

    def test_start_reasoning_returns_id(self, logger_with_db):
        """start_reasoning returns a non-empty session ID string."""
        session_id = logger_with_db.start_reasoning("What is AI?")
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_start_reasoning_sets_current_state(self, logger_with_db):
        """start_reasoning sets current_reasoning_id and current_path."""
        session_id = logger_with_db.start_reasoning("test query")
        assert logger_with_db.current_reasoning_id == session_id
        assert logger_with_db.current_path == []

    def test_log_step_appends_to_path(self, logger_with_db):
        """log_reasoning_step adds step to current_path."""
        logger_with_db.start_reasoning("query")
        logger_with_db.log_reasoning_step(
            step_type="entity_extraction",
            description="Extracted entity: AI",
            entities=["AI"],
            confidence=0.95,
        )
        assert len(logger_with_db.current_path) == 1
        step = logger_with_db.current_path[0]
        assert step["step_type"] == "entity_extraction"
        assert step["entities"] == ["AI"]
        assert step["confidence"] == 0.95

    def test_log_step_without_active_session(self, logger_with_db):
        """log_reasoning_step without an active session does nothing."""
        # Should not raise
        logger_with_db.log_reasoning_step("test", "no session active")
        assert logger_with_db.current_path is None

    def test_log_path(self, logger_with_db):
        """log_path records a reasoning path."""
        logger_with_db.start_reasoning("path test query")
        logger_with_db.log_path(["A", "B", "C"], score=0.8)
        # Verify via DB retrieval
        session_id = logger_with_db.current_reasoning_id
        logger_with_db.end_reasoning(success=True)

        session = logger_with_db.get_reasoning_session(session_id)
        assert len(session["paths"]) == 1
        assert session["paths"][0]["path"] == ["A", "B", "C"]
        assert session["paths"][0]["score"] == 0.8
        assert session["paths"][0]["hop_count"] == 2  # len(path) - 1

    def test_log_path_without_active_session(self, logger_with_db):
        """log_path without an active session does nothing."""
        logger_with_db.log_path(["X", "Y"])  # Should not raise

    def test_end_reasoning_resets_state(self, logger_with_db):
        """end_reasoning clears current_path and current_reasoning_id."""
        logger_with_db.start_reasoning("q")
        logger_with_db.end_reasoning(success=True)
        assert logger_with_db.current_reasoning_id is None
        assert logger_with_db.current_path is None

    def test_end_reasoning_without_active_session(self, logger_with_db):
        """end_reasoning without an active session does nothing."""
        logger_with_db.end_reasoning(success=False)  # Should not raise


class TestDatabaseRetrieval:
    """Tests for get_reasoning_session and get_recent_sessions."""

    def test_get_reasoning_session_full(self, logger_with_db):
        """Retrieve a complete session with steps and paths."""
        sid = logger_with_db.start_reasoning("full session test", metadata={"k": "v"})
        logger_with_db.log_reasoning_step(
            "inference", "step 1", entities=["E1"], relations=["R1"], confidence=0.9
        )
        logger_with_db.log_reasoning_step("synthesis", "step 2", confidence=0.85)
        logger_with_db.log_path(["N1", "N2", "N3"], score=0.7, hop_count=2)
        logger_with_db.end_reasoning(success=True, metadata={"result": "ok"})

        session = logger_with_db.get_reasoning_session(sid)

        assert session["query"] == "full session test"
        assert session["success"] == 1  # SQLite stores booleans as int
        assert len(session["steps"]) == 2
        assert session["steps"][0]["step_type"] == "inference"
        assert session["steps"][0]["entities"] == ["E1"]
        assert session["steps"][0]["relations"] == ["R1"]
        assert len(session["paths"]) == 1
        assert session["paths"][0]["hop_count"] == 2

    def test_get_nonexistent_session(self, logger_with_db):
        """Retrieving a nonexistent session returns empty dict."""
        result = logger_with_db.get_reasoning_session("nonexistent-id")
        assert result == {}

    def test_get_session_without_db(self, logger_no_db):
        """get_reasoning_session without DB returns empty dict."""
        result = logger_no_db.get_reasoning_session("any-id")
        assert result == {}

    def test_get_recent_sessions(self, logger_with_db):
        """get_recent_sessions returns correct count and structure."""
        for i in range(3):
            logger_with_db.start_reasoning(f"query {i}")
            logger_with_db.log_reasoning_step("step", f"desc {i}")
            logger_with_db.end_reasoning(success=True)

        sessions = logger_with_db.get_recent_sessions(limit=10)
        assert len(sessions) == 3
        # Each session should have step_count and path_count
        for s in sessions:
            assert "step_count" in s
            assert "path_count" in s

    def test_get_recent_sessions_without_db(self, logger_no_db):
        """get_recent_sessions without DB returns empty list."""
        assert logger_no_db.get_recent_sessions() == []

    def test_get_recent_sessions_limit(self, logger_with_db):
        """get_recent_sessions respects the limit parameter."""
        for i in range(5):
            logger_with_db.start_reasoning(f"q{i}")
            logger_with_db.end_reasoning()

        sessions = logger_with_db.get_recent_sessions(limit=2)
        assert len(sessions) == 2


class TestSessionAnalysis:
    """Tests for analyze_session."""

    def test_analyze_session_with_steps(self, logger_with_db):
        """analyze_session computes correct statistics."""
        sid = logger_with_db.start_reasoning("analysis test")
        logger_with_db.log_reasoning_step(
            "extraction", "e1", entities=["A", "B"], confidence=0.9
        )
        logger_with_db.log_reasoning_step(
            "inference", "i1", entities=["A"], relations=["R1"], confidence=0.7
        )
        logger_with_db.log_path(["A", "B", "C"], score=0.8)
        logger_with_db.end_reasoning(success=True)

        analysis = logger_with_db.analyze_session(sid)

        assert analysis["query"] == "analysis test"
        assert analysis["step_count"] == 2
        assert analysis["path_count"] == 1
        assert analysis["max_confidence"] == 0.9
        assert analysis["min_confidence"] == 0.7
        assert abs(analysis["avg_confidence"] - 0.8) < 0.01
        assert analysis["entity_frequencies"]["A"] == 2
        assert analysis["entity_frequencies"]["B"] == 1
        assert analysis["relation_frequencies"]["R1"] == 1
        assert analysis["step_type_distribution"]["extraction"] == 1
        assert analysis["step_type_distribution"]["inference"] == 1

    def test_analyze_nonexistent_session(self, logger_with_db):
        """analyze_session with unknown ID returns empty dict."""
        assert logger_with_db.analyze_session("unknown") == {}


class TestExportAndClear:
    """Tests for export_session_data and clear_session."""

    def test_export_session_to_file(self, logger_with_db, tmp_path):
        """export_session_data writes valid JSON."""
        sid = logger_with_db.start_reasoning("export test")
        logger_with_db.log_reasoning_step("s", "d")
        logger_with_db.end_reasoning(success=True)

        out_file = str(tmp_path / "export.json")
        result = logger_with_db.export_session_data(sid, out_file)
        assert result is True
        assert os.path.exists(out_file)

        with open(out_file) as f:
            data = json.load(f)
        assert data["query"] == "export test"

    def test_export_nonexistent_session(self, logger_with_db, tmp_path):
        """export_session_data with unknown ID returns False."""
        out_file = str(tmp_path / "nope.json")
        result = logger_with_db.export_session_data("no-such-id", out_file)
        assert result is False

    def test_clear_session(self, logger_with_db):
        """clear_session removes session data from DB."""
        sid = logger_with_db.start_reasoning("clear test")
        logger_with_db.log_reasoning_step("s", "d")
        logger_with_db.log_path(["X", "Y"])
        logger_with_db.end_reasoning()

        result = logger_with_db.clear_session(sid)
        assert result is True

        # Verify data is gone
        session = logger_with_db.get_reasoning_session(sid)
        assert session == {}

    def test_clear_session_without_db(self, logger_no_db):
        """clear_session without DB returns False."""
        assert logger_no_db.clear_session("any") is False


class TestReasoningContext:
    """Tests for the ReasoningContext context manager."""

    def test_context_manager_normal_flow(self, logger_with_db):
        """Context manager starts and ends a session automatically."""
        with logger_with_db.create_context_manager("ctx query") as ctx:
            assert ctx.session_id is not None
            logger_with_db.log_reasoning_step("step", "in context")

        # After exiting, session should be ended
        assert logger_with_db.current_reasoning_id is None

        session = logger_with_db.get_reasoning_session(ctx.session_id)
        assert session["query"] == "ctx query"
        assert session["success"] == 1

    def test_context_manager_on_exception(self, logger_with_db):
        """Context manager records failure on exception and re-raises it."""
        with pytest.raises(ValueError):
            with logger_with_db.create_context_manager("error query") as ctx:
                session_id = ctx.session_id
                raise ValueError("test error")

        session = logger_with_db.get_reasoning_session(session_id)
        assert session["success"] == 0  # marked as failed

    def test_context_manager_with_metadata(self, logger_with_db):
        """Context manager passes metadata through."""
        meta = {"source": "unit_test"}
        with logger_with_db.create_context_manager("meta query", metadata=meta) as ctx:
            pass

        session = logger_with_db.get_reasoning_session(ctx.session_id)
        # The end_reasoning updates metadata; it should contain exception key
        assert "exception" in session.get("metadata", {})
