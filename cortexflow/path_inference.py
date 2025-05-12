"""
CortexFlow Path Inference module.

This module provides advanced path-based inference capabilities for the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Generator
import json
import copy

logger = logging.getLogger('cortexflow')

class BidirectionalSearch:
    """Implements bidirectional search for efficient path finding in knowledge graphs."""
    
    def __init__(self, graph_store):
        """
        Initialize bidirectional search.
        
        Args:
            graph_store: Graph store interface
        """
        self.graph_store = graph_store
    
    def search(
        self, 
        start_entity: str,
        end_entity: str,
        max_hops: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between start and end entities using bidirectional search.
        
        Args:
            start_entity: Source entity name or ID
            end_entity: Target entity name or ID
            max_hops: Maximum number of hops
            
        Returns:
            List of paths, where each path is a list of steps
        """
        # Early return if entities are the same
        if start_entity == end_entity:
            return [[{"source": start_entity, "relation": "is_same_as", "target": end_entity, "confidence": 1.0}]]
        
        # Initialize forward and backward frontiers
        forward_frontier = {start_entity: []}
        backward_frontier = {end_entity: []}
        
        # Keep track of visited nodes
        forward_visited = {start_entity}
        backward_visited = {end_entity}
        
        # Maximum hops per direction
        max_direction_hops = (max_hops + 1) // 2
        
        # Iterate until max hops or frontiers are empty
        for i in range(max_direction_hops):
            # Check for intersection
            intersection = set(forward_frontier.keys()) & set(backward_frontier.keys())
            if intersection:
                # Found meeting points - construct paths
                return self._construct_paths(
                    meeting_points=intersection,
                    forward_paths=forward_frontier,
                    backward_paths=backward_frontier
                )
            
            # Expand forward frontier
            if i < max_direction_hops - 1:
                forward_frontier = self._expand_frontier(
                    frontier=forward_frontier,
                    visited=forward_visited,
                    direction="outgoing"
                )
            
            # Expand backward frontier
            if i < max_direction_hops - 1:
                backward_frontier = self._expand_frontier(
                    frontier=backward_frontier,
                    visited=backward_visited,
                    direction="incoming"
                )
            
            # Check for intersection again
            intersection = set(forward_frontier.keys()) & set(backward_frontier.keys())
            if intersection:
                # Found meeting points - construct paths
                return self._construct_paths(
                    meeting_points=intersection,
                    forward_paths=forward_frontier,
                    backward_paths=backward_frontier
                )
        
        # No paths found
        return []
    
    def _expand_frontier(
        self,
        frontier: Dict[str, List[Dict[str, Any]]],
        visited: Set[str],
        direction: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expand a frontier in the search.
        
        Args:
            frontier: Current frontier as a mapping from entity to path
            visited: Set of visited entities
            direction: Direction of expansion ("outgoing" or "incoming")
            
        Returns:
            New frontier as a mapping from entity to path
        """
        new_frontier = {}
        
        for entity, path in frontier.items():
            # Get neighbors based on direction
            if direction == "outgoing":
                neighbors = self.graph_store.get_entity_neighbors(
                    entity=entity,
                    direction="outgoing"
                )
            else:
                neighbors = self.graph_store.get_entity_neighbors(
                    entity=entity,
                    direction="incoming"
                )
            
            for neighbor in neighbors:
                if direction == "outgoing":
                    # For outgoing, neighbor is the target
                    target_entity = neighbor.get("entity", "")
                    relation = neighbor.get("relation", "")
                    confidence = neighbor.get("confidence", 1.0)
                    
                    # Create the step
                    step = {
                        "source": entity,
                        "relation": relation,
                        "target": target_entity,
                        "confidence": confidence
                    }
                    
                    # Add to new frontier if not visited
                    if target_entity not in visited:
                        new_frontier[target_entity] = path + [step]
                        visited.add(target_entity)
                else:
                    # For incoming, neighbor is the source
                    source_entity = neighbor.get("entity", "")
                    relation = neighbor.get("relation", "")
                    confidence = neighbor.get("confidence", 1.0)
                    
                    # Create the step
                    step = {
                        "source": source_entity,
                        "relation": relation,
                        "target": entity,
                        "confidence": confidence
                    }
                    
                    # Add to new frontier if not visited
                    if source_entity not in visited:
                        new_frontier[source_entity] = [step] + path
                        visited.add(source_entity)
        
        return new_frontier
    
    def _construct_paths(
        self,
        meeting_points: Set[str],
        forward_paths: Dict[str, List[Dict[str, Any]]],
        backward_paths: Dict[str, List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Construct full paths from meeting points.
        
        Args:
            meeting_points: Set of entities where forward and backward searches meet
            forward_paths: Mapping from entity to forward path
            backward_paths: Mapping from entity to backward path
            
        Returns:
            List of full paths
        """
        paths = []
        
        for meeting_point in meeting_points:
            # Get forward and backward paths to the meeting point
            forward_path = forward_paths.get(meeting_point, [])
            backward_path = backward_paths.get(meeting_point, [])
            
            # Combine the paths
            full_path = forward_path + backward_path
            
            # Add to result
            if full_path:
                paths.append(full_path)
        
        return paths


class WeightedPathSearch:
    """Implements weighted path search for knowledge graphs."""
    
    def __init__(self, graph_store):
        """
        Initialize weighted path search.
        
        Args:
            graph_store: Graph store interface
        """
        self.graph_store = graph_store
    
    def search(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 3,
        importance_weight: float = 0.5,
        confidence_weight: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Find weighted paths between start and end entities.
        
        Args:
            start_entity: Source entity name or ID
            end_entity: Target entity name or ID
            max_hops: Maximum number of hops
            importance_weight: Weight for importance score
            confidence_weight: Weight for confidence score
            
        Returns:
            List of paths sorted by combined weight
        """
        # Early return if entities are the same
        if start_entity == end_entity:
            return [[{"source": start_entity, "relation": "is_same_as", "target": end_entity, "confidence": 1.0}]]
        
        # Initialize frontier
        frontier = [{
            "entity": start_entity,
            "path": [],
            "visited": {start_entity},
            "total_weight": 0.0
        }]
        
        # List to store complete paths
        complete_paths = []
        
        # Maximum number of paths to return
        max_results = 5
        
        # Explore until maximum hops or frontier is empty
        for _ in range(max_hops):
            if not frontier:
                break
                
            # Sort frontier by total weight (descending)
            frontier.sort(key=lambda x: x["total_weight"], reverse=True)
            
            # Limit frontier size
            if len(frontier) > 100:
                frontier = frontier[:100]
                
            new_frontier = []
            
            for item in frontier:
                current_entity = item["entity"]
                current_path = item["path"]
                visited = item["visited"]
                total_weight = item["total_weight"]
                
                # Get neighbors
                neighbors = self.graph_store.get_entity_neighbors(
                    entity=current_entity,
                    direction="outgoing"
                )
                
                for neighbor in neighbors:
                    target_entity = neighbor.get("entity", "")
                    relation = neighbor.get("relation", "")
                    confidence = neighbor.get("confidence", 1.0)
                    importance = neighbor.get("weight", 0.5)
                    
                    # Skip if already visited
                    if target_entity in visited:
                        continue
                        
                    # Calculate combined weight
                    combined_weight = (
                        importance * importance_weight +
                        confidence * confidence_weight
                    )
                    
                    # Create the step
                    step = {
                        "source": current_entity,
                        "relation": relation,
                        "target": target_entity,
                        "confidence": confidence,
                        "importance": importance
                    }
                    
                    # Create new path
                    new_path = current_path + [step]
                    
                    # Create new visited set
                    new_visited = visited.copy()
                    new_visited.add(target_entity)
                    
                    # Update total weight
                    new_total_weight = total_weight + combined_weight
                    
                    # Check if we've reached the target
                    if target_entity == end_entity:
                        complete_paths.append({
                            "path": new_path,
                            "total_weight": new_total_weight
                        })
                        
                        # Sort and limit complete paths
                        if len(complete_paths) > max_results:
                            complete_paths.sort(key=lambda x: x["total_weight"], reverse=True)
                            complete_paths = complete_paths[:max_results]
                    else:
                        # Add to new frontier
                        new_frontier.append({
                            "entity": target_entity,
                            "path": new_path,
                            "visited": new_visited,
                            "total_weight": new_total_weight
                        })
            
            # Update frontier
            frontier = new_frontier
        
        # Sort complete paths by total weight
        complete_paths.sort(key=lambda x: x["total_weight"], reverse=True)
        
        # Return paths
        return [item["path"] for item in complete_paths[:max_results]]


class ConstrainedPathSearch:
    """Implements constrained path search for knowledge graphs."""
    
    def __init__(self, graph_store):
        """
        Initialize constrained path search.
        
        Args:
            graph_store: Graph store interface
        """
        self.graph_store = graph_store
    
    def search(
        self,
        start_entity: str,
        end_entity: str,
        allowed_relations: Optional[List[str]] = None,
        excluded_relations: Optional[List[str]] = None,
        max_hops: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths with relation constraints.
        
        Args:
            start_entity: Source entity name or ID
            end_entity: Target entity name or ID
            allowed_relations: List of allowed relation types (None means all allowed)
            excluded_relations: List of excluded relation types
            max_hops: Maximum number of hops
            
        Returns:
            List of paths that satisfy the constraints
        """
        # Early return if entities are the same
        if start_entity == end_entity:
            return [[{"source": start_entity, "relation": "is_same_as", "target": end_entity, "confidence": 1.0}]]
        
        # Initialize frontier
        frontier = [{
            "entity": start_entity,
            "path": [],
            "visited": {start_entity}
        }]
        
        # List to store complete paths
        complete_paths = []
        
        # Maximum number of paths to return
        max_results = 5
        
        # Prepare sets for fast membership testing
        allowed_set = set(allowed_relations) if allowed_relations else None
        excluded_set = set(excluded_relations) if excluded_relations else set()
        
        # Explore until maximum hops or frontier is empty
        for _ in range(max_hops):
            if not frontier:
                break
                
            new_frontier = []
            
            for item in frontier:
                current_entity = item["entity"]
                current_path = item["path"]
                visited = item["visited"]
                
                # Get neighbors
                neighbors = self.graph_store.get_entity_neighbors(
                    entity=current_entity,
                    direction="outgoing"
                )
                
                for neighbor in neighbors:
                    target_entity = neighbor.get("entity", "")
                    relation = neighbor.get("relation", "")
                    confidence = neighbor.get("confidence", 1.0)
                    
                    # Skip if already visited
                    if target_entity in visited:
                        continue
                        
                    # Check if relation is allowed
                    if allowed_set and relation not in allowed_set:
                        continue
                        
                    # Check if relation is excluded
                    if relation in excluded_set:
                        continue
                        
                    # Create the step
                    step = {
                        "source": current_entity,
                        "relation": relation,
                        "target": target_entity,
                        "confidence": confidence
                    }
                    
                    # Create new path
                    new_path = current_path + [step]
                    
                    # Create new visited set
                    new_visited = visited.copy()
                    new_visited.add(target_entity)
                    
                    # Check if we've reached the target
                    if target_entity == end_entity:
                        complete_paths.append(new_path)
                        
                        # Limit complete paths
                        if len(complete_paths) > max_results:
                            complete_paths = complete_paths[:max_results]
                    else:
                        # Add to new frontier
                        new_frontier.append({
                            "entity": target_entity,
                            "path": new_path,
                            "visited": new_visited
                        })
            
            # Update frontier
            frontier = new_frontier
        
        # Return paths
        return complete_paths[:max_results]


class PathExplainer:
    """Generates human-readable explanations for paths."""
    
    def __init__(self, graph_store):
        """
        Initialize path explainer.
        
        Args:
            graph_store: Graph store interface
        """
        self.graph_store = graph_store
    
    def explain_path(self, path: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation of a path.
        
        Args:
            path: Path as a list of steps
            
        Returns:
            Human-readable explanation
        """
        if not path:
            return "No path available to explain."
            
        explanation = []
        
        for i, step in enumerate(path):
            source = step.get("source", "unknown")
            relation = step.get("relation", "unknown relation")
            target = step.get("target", "unknown")
            confidence = step.get("confidence", 0.0)
            
            explanation.append(f"{source} {relation} {target} (confidence: {confidence:.2f})")
        
        return " → ".join(explanation)


def register_path_inference(graph_store):
    """
    Register path inference capabilities with a graph store.
    
    Args:
        graph_store: Graph store to augment with path inference
    """
    # Create components
    bidirectional_search = BidirectionalSearch(graph_store)
    weighted_path_search = WeightedPathSearch(graph_store)
    constrained_path_search = ConstrainedPathSearch(graph_store)
    path_explainer = PathExplainer(graph_store)
    
    # Register bidirectional search
    setattr(graph_store, "bidirectional_search", bidirectional_search.search)
    
    # Register weighted path search
    setattr(graph_store, "weighted_path_query", weighted_path_search.search)
    
    # Register constrained path search
    setattr(graph_store, "constrained_path_search", constrained_path_search.search)
    
    # Register path explainer
    setattr(graph_store, "explain_path", path_explainer.explain_path)
    
    logger.info("Path inference capabilities registered")
    
    return {
        "bidirectional_search": bidirectional_search,
        "weighted_path_search": weighted_path_search,
        "constrained_path_search": constrained_path_search,
        "path_explainer": path_explainer
    } 