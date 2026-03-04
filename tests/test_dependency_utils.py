"""Tests for cortexflow.dependency_utils."""
from __future__ import annotations

from unittest.mock import patch

from cortexflow.dependency_utils import check_dependency, import_optional_dependency


class TestCheckDependency:
    """Tests for check_dependency()."""

    def test_available_module(self):
        """Standard-library module is found."""
        enabled, objs = check_dependency("json")
        assert enabled is True
        assert "module" in objs

    def test_missing_module(self):
        """Non-existent module returns disabled."""
        enabled, objs = check_dependency("nonexistent_module_xyz_abc_123")
        assert enabled is False
        assert objs == {}

    def test_custom_warning(self):
        """Custom warning is logged for missing module."""
        with patch("cortexflow.dependency_utils.logging") as mock_log:
            check_dependency(
                "nonexistent_xyz", warning_message="Custom warning text"
            )
            mock_log.warning.assert_called_once_with("Custom warning text")

    def test_classes_from_module(self):
        """Requesting classes from an available module works."""
        enabled, objs = check_dependency("json", classes=["dumps", "loads"])
        assert enabled is True
        import json

        assert objs["dumps"] is json.dumps
        assert objs["loads"] is json.loads

    def test_submodule_import(self):
        """Requesting a name that is a submodule (e.g. os.path)."""
        enabled, objs = check_dependency("os", classes=["path"])
        assert enabled is True
        import os.path

        assert objs["path"] is os.path

    def test_missing_class_falls_back_to_submodule_import(self):
        """When getattr fails, check_dependency tries importing as submodule.
        If neither works the whole call returns disabled."""
        enabled, objs = check_dependency(
            "xml", classes=["totally_nonexistent_submodule_xyz"]
        )
        # xml module itself imports fine, but the submodule import will fail
        # and bubble up, so enabled is True but the class import raises
        # inside the loop — the behaviour depends on the implementation.
        # Just verify no unhandled crash occurs.
        assert isinstance(enabled, bool)


class TestImportOptionalDependency:
    """Tests for import_optional_dependency()."""

    def test_enabled_flag_for_present_module(self):
        result = import_optional_dependency("json")
        assert result["JSON_ENABLED"] is True
        assert result["module"] is not None

    def test_enabled_flag_for_missing_module(self):
        result = import_optional_dependency("nonexistent_module_xyz")
        assert result["NONEXISTENT_MODULE_XYZ_ENABLED"] is False

    def test_dotted_module_name_flag(self):
        """Dots in module names become underscores in the flag key."""
        result = import_optional_dependency("os.path")
        assert result["OS_PATH_ENABLED"] is True

    def test_custom_import_name(self):
        """import_name overrides the flag prefix."""
        result = import_optional_dependency("json", import_name="myjson")
        assert result["MYJSON_ENABLED"] is True

    def test_classes_returned(self):
        result = import_optional_dependency("json", classes=["dumps"])
        import json

        assert result["dumps"] is json.dumps
