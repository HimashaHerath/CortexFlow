"""
Tests for the CortexFlow CLI module (cortexflow/cli.py).

Covers:
- Argument parsing for the main parser and subcommands
- Version flag output
- run_chat and run_analyze with mocked CortexFlowManager and CortexFlowConfig
- Default argument values
- Unknown command behavior
"""

import pytest
from unittest.mock import patch, MagicMock, call
import argparse

from cortexflow.cli import main, run_chat, run_analyze


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    """Tests for CLI argument parsing behavior."""

    @patch("cortexflow.cli.run_chat")
    def test_chat_default_args(self, mock_run_chat):
        """'chat' subcommand uses correct defaults for model, host, db."""
        with patch("sys.argv", ["cortexflow", "chat"]):
            main()
        args = mock_run_chat.call_args[0][0]
        assert args.model == "llama3"
        assert args.host == "http://localhost:11434"
        assert args.db == ":memory:"

    @patch("cortexflow.cli.run_chat")
    def test_chat_custom_args(self, mock_run_chat):
        """'chat' subcommand passes custom arguments correctly."""
        with patch("sys.argv", [
            "cortexflow", "chat",
            "--model", "gemma3",
            "--host", "http://myhost:11434",
            "--db", "/tmp/test.db"
        ]):
            main()
        args = mock_run_chat.call_args[0][0]
        assert args.model == "gemma3"
        assert args.host == "http://myhost:11434"
        assert args.db == "/tmp/test.db"

    @patch("cortexflow.cli.run_analyze")
    def test_analyze_requires_db(self, mock_run_analyze):
        """'analyze' subcommand requires --db argument."""
        with patch("sys.argv", ["cortexflow", "analyze", "--db", "/data/store.db"]):
            main()
        args = mock_run_analyze.call_args[0][0]
        assert args.db == "/data/store.db"

    def test_analyze_missing_db_exits(self):
        """'analyze' without --db causes SystemExit (argparse error)."""
        with patch("sys.argv", ["cortexflow", "analyze"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 on missing required args
            assert exc_info.value.code == 2


class TestVersionFlag:
    """Tests for --version output."""

    def test_version_flag(self):
        """--version flag prints version and exits."""
        with patch("sys.argv", ["cortexflow", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestNoCommand:
    """Tests for behavior when no subcommand is given."""

    def test_no_command_returns_one(self):
        """Running with no subcommand returns exit code 1."""
        with patch("sys.argv", ["cortexflow"]):
            result = main()
        assert result == 1


# ---------------------------------------------------------------------------
# run_chat tests
# ---------------------------------------------------------------------------

class TestRunChat:
    """Tests for the run_chat function."""

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager")
    @patch("builtins.input", side_effect=["exit"])
    @patch("builtins.print")
    def test_chat_exit_immediately(self, mock_print, mock_input, mock_manager_cls, mock_config_cls):
        """Typing 'exit' immediately ends the chat loop."""
        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager

        args = argparse.Namespace(model="llama3", host="http://localhost:11434", db=":memory:")
        run_chat(args)

        mock_manager.close.assert_called_once()

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager")
    @patch("builtins.input", side_effect=["stats", "quit"])
    @patch("builtins.print")
    def test_chat_stats_command(self, mock_print, mock_input, mock_manager_cls, mock_config_cls):
        """Typing 'stats' calls manager.get_stats()."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "memory": {
                "message_count": 5,
                "tiers": {
                    "active": {"used": 100, "limit": 4096, "fullness": 0.02}
                }
            }
        }
        mock_manager_cls.return_value = mock_manager

        args = argparse.Namespace(model="llama3", host="http://localhost:11434", db=":memory:")
        run_chat(args)

        mock_manager.get_stats.assert_called_once()

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager")
    @patch("builtins.input", side_effect=["hello world", "q"])
    @patch("builtins.print")
    def test_chat_generates_response(self, mock_print, mock_input, mock_manager_cls, mock_config_cls):
        """A normal message triggers generate_response_stream or generate_response."""
        mock_manager = MagicMock()
        mock_manager.generate_response_stream.return_value = iter(["Hello", " back!"])
        mock_manager_cls.return_value = mock_manager

        args = argparse.Namespace(model="llama3", host="http://localhost:11434", db=":memory:")
        run_chat(args)

        mock_manager.generate_response_stream.assert_called_once_with("hello world")

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager")
    @patch("builtins.input", side_effect=["hello", "exit"])
    @patch("builtins.print")
    def test_chat_fallback_to_non_streaming(self, mock_print, mock_input, mock_manager_cls, mock_config_cls):
        """Falls back to generate_response when streaming raises AttributeError."""
        mock_manager = MagicMock()
        mock_manager.generate_response_stream.side_effect = AttributeError("no streaming")
        mock_manager.generate_response.return_value = "Fallback response"
        mock_manager_cls.return_value = mock_manager

        args = argparse.Namespace(model="llama3", host="http://localhost:11434", db=":memory:")
        run_chat(args)

        mock_manager.generate_response.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# run_analyze tests
# ---------------------------------------------------------------------------

class TestRunAnalyze:
    """Tests for the run_analyze function."""

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager")
    @patch("builtins.print")
    def test_analyze_prints_stats(self, mock_print, mock_manager_cls, mock_config_cls):
        """run_analyze retrieves and prints stats from the manager."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "memory": {
                "message_count": 42,
                "tiers": {
                    "active": {"used": 500, "limit": 4096, "segment_count": 3, "fullness": 0.12}
                }
            }
        }
        # Ensure hasattr checks work on mock
        mock_manager.knowledge_store = MagicMock()
        mock_manager.knowledge_store.get_stats.return_value = {
            "fact_count": 10,
            "knowledge_count": 5,
            "vector_enabled": True,
            "bm25_enabled": False,
        }
        mock_manager_cls.return_value = mock_manager

        args = argparse.Namespace(db="/tmp/test.db")
        result = run_analyze(args)

        assert result == 0
        mock_manager.get_stats.assert_called_once()
        mock_manager.close.assert_called_once()

    @patch("cortexflow.cli.CortexFlowConfig")
    @patch("cortexflow.cli.CortexFlowManager", side_effect=Exception("DB not found"))
    @patch("builtins.print")
    def test_analyze_handles_error(self, mock_print, mock_manager_cls, mock_config_cls):
        """run_analyze handles errors gracefully and returns 1."""
        args = argparse.Namespace(db="/nonexistent/path.db")
        result = run_analyze(args)
        assert result == 1
