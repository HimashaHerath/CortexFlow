"""
Traversal Profiler for CortexFlow.

This module provides functionality for profiling and optimizing graph traversal operations
in the CortexFlow knowledge graph.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import json
import os
from collections import defaultdict
import heapq
from functools import wraps

# Configure logging
logger = logging.getLogger('cortexflow.traversal')

class TraversalProfile:
    """
    Represents a profile of a graph traversal operation.
    """
    
    def __init__(self, name: str, metadata: Dict[str, Any] = None):
        """
        Initialize a traversal profile.
        
        Args:
            name: Name of the traversal operation
            metadata: Optional metadata about the traversal
        """
        self.name = name
        self.metadata = metadata or {}
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = 0.0
        self.nodes_visited = 0
        self.edges_traversed = 0
        self.path_length = 0
        self.steps = []
    
    def start(self):
        """Start profiling the traversal."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop profiling the traversal."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
    
    def add_step(self, step_name: str, duration: float, metadata: Dict[str, Any] = None):
        """
        Add a step to the traversal profile.
        
        Args:
            step_name: Name of the step
            duration: Duration of the step in seconds
            metadata: Optional metadata about the step
        """
        self.steps.append({
            "name": step_name,
            "duration": duration,
            "metadata": metadata or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "nodes_visited": self.nodes_visited,
            "edges_traversed": self.edges_traversed,
            "path_length": self.path_length,
            "steps": self.steps
        }

class TraversalProfiler:
    """
    Profiler for graph traversal operations.
    """
    
    def __init__(self, config=None, output_dir: str = None):
        """
        Initialize the traversal profiler.
        
        Args:
            config: Optional configuration
            output_dir: Directory for output files
        """
        self.config = config or {}
        self.output_dir = output_dir or "traversal_profiles"
        self.profiles = []
        self.current_profile = None
        self.enabled = self.config.get("enable_traversal_profiling", True)
        self.profile_limit = self.config.get("profile_limit", 1000)
        
        # Optimization suggestions based on profiles
        self.optimization_suggestions = []
        
        # Create output directory if it doesn't exist
        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.close()
    
    def close(self):
        """Close and clean up resources."""
        try:
            # Stop any ongoing profiling
            if self.current_profile:
                self.stop_profile()
            
            # Clear profiles to free memory
            self.profiles = []
            self.optimization_suggestions = []
            
            logger.info("TraversalProfiler closed")
        except Exception as e:
            logger.error(f"Error closing TraversalProfiler: {e}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()
    
    def start_profile(self, name: str, metadata: Dict[str, Any] = None) -> Optional[TraversalProfile]:
        """
        Start profiling a traversal operation.
        
        Args:
            name: Name of the traversal operation
            metadata: Optional metadata about the traversal
            
        Returns:
            TraversalProfile object or None if profiling is disabled
        """
        if not self.enabled:
            return None
        
        # Limit the number of profiles
        if len(self.profiles) >= self.profile_limit:
            # Remove oldest profile
            self.profiles.pop(0)
        
        # Create and start profile
        profile = TraversalProfile(name, metadata)
        profile.start()
        
        self.current_profile = profile
        self.profiles.append(profile)
        
        return profile
    
    def stop_profile(self) -> Optional[TraversalProfile]:
        """
        Stop profiling the current traversal operation.
        
        Returns:
            Completed TraversalProfile object or None if no current profile
        """
        if not self.enabled or not self.current_profile:
            return None
        
        self.current_profile.stop()
        
        # Generate optimization suggestions
        self._analyze_profile(self.current_profile)
        
        # Reset current profile
        profile = self.current_profile
        self.current_profile = None
        
        return profile
    
    def log_traversal_step(self, step_name: str, duration: float, metadata: Dict[str, Any] = None):
        """
        Log a step in the current traversal profile.
        
        Args:
            step_name: Name of the step
            duration: Duration of the step in seconds
            metadata: Optional metadata about the step
        """
        if not self.enabled or not self.current_profile:
            return
        
        self.current_profile.add_step(step_name, duration, metadata)
    
    def update_traversal_stats(self, nodes_visited: int, edges_traversed: int, path_length: int):
        """
        Update statistics for the current traversal profile.
        
        Args:
            nodes_visited: Number of nodes visited
            edges_traversed: Number of edges traversed
            path_length: Length of the path found
        """
        if not self.enabled or not self.current_profile:
            return
        
        self.current_profile.nodes_visited = nodes_visited
        self.current_profile.edges_traversed = edges_traversed
        self.current_profile.path_length = path_length
    
    def save_profiles(self, filename: str = None) -> str:
        """
        Save traversal profiles to a file.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to the saved file
        """
        if not self.enabled or not self.profiles:
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traversal_profiles_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Convert profiles to dictionaries
        profiles_data = [profile.to_dict() for profile in self.profiles]
        
        # Add optimization suggestions
        data = {
            "profiles": profiles_data,
            "optimization_suggestions": self.optimization_suggestions,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.profiles)} traversal profiles to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving traversal profiles to {file_path}: {e}")
            return None
    
    def _analyze_profile(self, profile: TraversalProfile):
        """
        Analyze a traversal profile and generate optimization suggestions.
        
        Args:
            profile: TraversalProfile to analyze
        """
        suggestions = []
        
        # Check for excessive node visitation
        nodes_to_path_ratio = profile.nodes_visited / max(1, profile.path_length)
        if nodes_to_path_ratio > 10:
            suggestions.append({
                "type": "excessive_exploration",
                "description": f"Traversal visited {profile.nodes_visited} nodes for a path of length {profile.path_length}",
                "improvement": "Consider adding more specific constraints or using bidirectional search"
            })
        
        # Check for slow steps
        slow_steps = []
        for step in profile.steps:
            if step["duration"] > 0.1:  # More than 100ms
                slow_steps.append(step)
        
        if slow_steps:
            suggestions.append({
                "type": "slow_steps",
                "description": f"Found {len(slow_steps)} slow steps in traversal",
                "steps": [step["name"] for step in slow_steps],
                "improvement": "Consider optimizing these specific traversal steps"
            })
        
        # Check for excessive total duration
        if profile.duration > 1.0:  # More than 1 second
            suggestions.append({
                "type": "slow_traversal",
                "description": f"Traversal took {profile.duration:.2f} seconds to complete",
                "improvement": "Consider adding indexes or partitioning the graph"
            })
        
        # Add suggestions to the list
        if suggestions:
            self.optimization_suggestions.append({
                "profile_name": profile.name,
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions
            })
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions based on traversal profiles.
        
        Returns:
            List of optimization suggestions
        """
        return self.optimization_suggestions
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all profiles.
        
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.profiles:
            return {}
        
        stats = {
            "total_profiles": len(self.profiles),
            "total_duration": sum(p.duration for p in self.profiles),
            "avg_duration": sum(p.duration for p in self.profiles) / len(self.profiles),
            "max_duration": max(p.duration for p in self.profiles),
            "min_duration": min(p.duration for p in self.profiles),
            "avg_nodes_visited": sum(p.nodes_visited for p in self.profiles) / len(self.profiles),
            "avg_edges_traversed": sum(p.edges_traversed for p in self.profiles) / len(self.profiles),
            "avg_path_length": sum(p.path_length for p in self.profiles) / len(self.profiles),
            "operations_by_type": defaultdict(int)
        }
        
        # Count operations by type
        for profile in self.profiles:
            operation_type = profile.metadata.get("type", "unknown")
            stats["operations_by_type"][operation_type] += 1
        
        # Convert defaultdict to regular dict for serialization
        stats["operations_by_type"] = dict(stats["operations_by_type"])
        
        return stats
    
    def clear_profiles(self):
        """Clear all stored profiles."""
        self.profiles = []
        self.current_profile = None
        self.optimization_suggestions = []
        logger.info("Cleared all traversal profiles")

def profile_traversal(name: str = None):
    """
    Decorator for profiling traversal methods.
    
    Args:
        name: Optional name for the traversal profile
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if traversal profiler is available
            profiler = getattr(self, "traversal_profiler", None)
            if not profiler or not profiler.enabled:
                return func(self, *args, **kwargs)
            
            # Start profiling
            profile_name = name or func.__name__
            metadata = {
                "type": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            profiler.start_profile(profile_name, metadata)
            
            # Call the function
            start_time = time.time()
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            # Update traversal stats if result has them
            if isinstance(result, dict):
                nodes_visited = result.get("nodes_visited", 0)
                edges_traversed = result.get("edges_traversed", 0)
                path_length = len(result.get("path", []))
                profiler.update_traversal_stats(nodes_visited, edges_traversed, path_length)
            
            # Stop profiling
            profiler.stop_profile()
            
            return result
        
        return wrapper
    
    return decorator 