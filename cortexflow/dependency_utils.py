"""
CortexFlow Dependency Utilities module.

This module provides utilities for handling optional dependencies in CortexFlow.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any


def check_dependency(
    module_name: str,
    import_name: str | None = None,
    warning_message: str | None = None,
    classes: list[str] | None = None
) -> tuple[bool, dict[str, Any]]:
    """
    Check if a dependency is available and return its status along with any requested classes.

    Args:
        module_name: The name of the module to import
        import_name: Optional alternative name to import the module as
        warning_message: Optional custom warning message if the module is not available
        classes: Optional list of class names to import from the module

    Returns:
        Tuple of (is_enabled, imported_objects) where imported_objects is a dict of
        imported classes or the module itself under the key 'module'
    """
    import_name = import_name or module_name
    warning_message = warning_message or f"{module_name} not found. Related functionality will be limited."
    imported_objects = {}

    try:
        module = importlib.import_module(module_name)
        imported_objects['module'] = module

        # Import requested classes if specified
        if classes:
            for class_name in classes:
                try:
                    imported_objects[class_name] = getattr(module, class_name)
                except AttributeError:
                    # Try importing as a submodule (e.g., thefuzz.fuzz)
                    imported_objects[class_name] = importlib.import_module(f"{module_name}.{class_name}")

        return True, imported_objects
    except ImportError:
        logging.warning(warning_message)
        return False, imported_objects

def import_optional_dependency(
    module_name: str,
    import_name: str | None = None,
    warning_message: str | None = None,
    classes: list[str] | None = None
) -> dict[str, Any]:
    """
    Import an optional dependency and return a dictionary with an ENABLED flag and imported objects.

    Args:
        module_name: The name of the module to import
        import_name: Optional alternative name to import the module as
        warning_message: Optional custom warning message if the module is not available
        classes: Optional list of class names to import from the module

    Returns:
        Dict with keys:
        - '{import_name}_ENABLED': Boolean flag indicating if the module is available
        - 'module': The imported module if available, None otherwise
        - Plus any requested class names pointing to their imported classes
    """
    is_enabled, imported_objects = check_dependency(
        module_name, import_name, warning_message, classes
    )

    # Add the ENABLED flag to the dictionary (replace dots with underscores for dotted module names)
    raw_name = import_name if import_name else module_name
    flag_name = f"{raw_name.upper().replace('.', '_')}_ENABLED"
    imported_objects[flag_name] = is_enabled

    return imported_objects
