#!/usr/bin/env python3
"""
Fix Python path to ensure AdaptiveContext module is accessible.
"""
import os
import sys

def fix_python_path():
    """Add the project root directory to Python path."""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Add to Python path if not already there
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to Python path")
    else:
        print(f"{parent_dir} already in Python path")
    
    # Try importing AdaptiveContextManager to verify it works
    try:
        from adaptive_context import AdaptiveContextManager
        print("Successfully imported AdaptiveContextManager")
        return True
    except ImportError as e:
        print(f"Failed to import AdaptiveContextManager: {e}")
        print(f"Current Python path: {sys.path}")
        return False

if __name__ == "__main__":
    success = fix_python_path()
    sys.exit(0 if success else 1) 