"""
CortexFlow Dependency Utilities module.

This module provides utilities for handling optional dependencies in CortexFlow.
"""

import logging
import importlib
from typing import Dict, Tuple, Optional, List, Any

def check_dependency(
    module_name: str, 
    import_name: Optional[str] = None, 
    warning_message: Optional[str] = None,
    classes: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, Any]]:
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
                imported_objects[class_name] = getattr(module, class_name)
                
        return True, imported_objects
    except ImportError:
        logging.warning(warning_message)
        return False, imported_objects

def import_optional_dependency(
    module_name: str, 
    import_name: Optional[str] = None,
    warning_message: Optional[str] = None,
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
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
    
    # Add the ENABLED flag to the dictionary
    flag_name = f"{import_name.upper()}_ENABLED" if import_name else f"{module_name.upper()}_ENABLED"
    imported_objects[flag_name] = is_enabled
    
    return imported_objects 