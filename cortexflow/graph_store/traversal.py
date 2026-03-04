"""
Graph traversal and path-finding methods.

Includes path_query, weighted_path_query, bidirectional_search,
constrained_path_search, and graph abstraction utilities.
"""
from __future__ import annotations

import itertools
import logging
import sqlite3
from typing import Any

from ._deps import NETWORKX_ENABLED, nx

# Safety limit for graph path enumeration to prevent DoS on dense graphs
MAX_PATHS_TO_ENUMERATE = 100


# ---------------------------------------------------------------------------
# Mixin class -- these methods are injected into GraphStore at import time
# via multiple inheritance (see store.py).
# ---------------------------------------------------------------------------

class TraversalMixin:
    """Methods for path finding, bidirectional search, and graph abstraction.

    This mixin is designed to be combined with the main :class:`GraphStore`
    class.  It expects the following attributes / methods on ``self``:

    * ``self.graph`` -- NetworkX DiGraph (or ``None``)
    * ``self.conn`` / ``self.db_path`` -- database connection info
    * ``self.get_entity_metadata(entity_id)``
    """

    # The constant is also available as a class attribute for backward compat
    MAX_PATHS_TO_ENUMERATE = MAX_PATHS_TO_ENUMERATE

    def path_query(self, start_entity: str, end_entity: str, max_hops: int = 3) -> list[list[dict[str, Any]]]:
        """
        Find paths between two entities in the graph.

        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for path queries")
            return []

        paths = []

        try:
            # Get entity IDs
            source_id = None
            target_id = None

            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()

            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()

            if source_row:
                source_id = source_row[0]

            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()

            if target_row:
                target_id = target_row[0]

            if self.conn is None:
                conn.close()

            # Return empty if entities not found
            if not source_id or not target_id:
                return []

            # Use NetworkX for path finding
            try:
                # Warn if graph is very large -- enumeration may be expensive
                if self.graph.number_of_nodes() > 10_000:
                    logging.warning(
                        "Graph has %d nodes; enumerating simple paths may be slow. "
                        "Limiting to %d candidate paths.",
                        self.graph.number_of_nodes(),
                        self.MAX_PATHS_TO_ENUMERATE,
                    )

                # Find simple paths with a hard cap via islice to avoid
                # unbounded enumeration on dense graphs.
                simple_paths = itertools.islice(
                    nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops),
                    self.MAX_PATHS_TO_ENUMERATE,
                )

                # Convert paths to our format
                for path in list(simple_paths)[:5]:  # Return at most 5 paths
                    formatted_path = []

                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)

                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }

                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)

                            if edge_data:
                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_data.get('weight', 1.0),
                                    "confidence": edge_data.get('confidence', 0.5)
                                }
                                node_info["next_relation"] = relation_info

                        formatted_path.append(node_info)

                    paths.append(formatted_path)

            except nx.NetworkXNoPath:
                # No path exists
                pass

            return paths

        except Exception as e:
            logging.error(f"Error in path query: {e}")
            return []

    def weighted_path_query(self, start_entity: str, end_entity: str,
                          max_hops: int = 3, importance_weight: float = 0.6,
                          confidence_weight: float = 0.4) -> list[list[dict[str, Any]]]:
        """
        Find weighted paths between entities considering relation importance and confidence.

        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length
            importance_weight: Weight factor for relation importance (0-1)
            confidence_weight: Weight factor for relation confidence (0-1)

        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for weighted path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for weighted path queries")
            return []

        weighted_paths = []

        try:
            # Get entity IDs
            source_id = None
            target_id = None

            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()

            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()

            if source_row:
                source_id = source_row[0]

            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()

            if target_row:
                target_id = target_row[0]

            if self.conn is None:
                conn.close()

            # Return empty if entities not found
            if not source_id or not target_id:
                return []

            # Create a copy of the graph with calculated weights
            weighted_graph = nx.DiGraph()

            # Copy nodes
            for node in self.graph.nodes():
                weighted_graph.add_node(node)

            # Copy edges with inverted weights
            for u, v, data in self.graph.edges(data=True):
                # Calculate combined weight based on importance and confidence
                edge_weight = data.get('weight', 0.5)
                edge_confidence = data.get('confidence', 0.5)

                # Normalize weights to 0-1 range
                norm_weight = min(max(edge_weight, 0.1), 1.0)
                norm_confidence = min(max(edge_confidence, 0.1), 1.0)

                # Calculate combined weight
                combined_weight = (importance_weight * norm_weight) + (confidence_weight * norm_confidence)

                # Invert weight for shortest path algorithm (higher weight/confidence = shorter path)
                inverted_weight = 1.0 / combined_weight if combined_weight > 0 else float('inf')

                # Create a copy of the data without the weight to avoid conflict
                edge_data = data.copy()
                if 'weight' in edge_data:
                    del edge_data['weight']

                # Add edge with inverted weight
                weighted_graph.add_edge(u, v, weight=inverted_weight, **edge_data)

            # Find k shortest paths
            try:
                # Get k-shortest paths using Dijkstra
                for path in nx.shortest_simple_paths(weighted_graph, source_id, target_id, weight='weight'):
                    # Check max hops
                    if len(path) > max_hops + 1:
                        break

                    formatted_path = []
                    path_total_weight = 0
                    path_min_confidence = 1.0

                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)

                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }

                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)

                            if edge_data:
                                edge_weight = edge_data.get('weight', 0.5)
                                edge_confidence = edge_data.get('confidence', 0.5)

                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_weight,
                                    "confidence": edge_confidence
                                }
                                node_info["next_relation"] = relation_info

                                path_total_weight += edge_weight
                                path_min_confidence = min(path_min_confidence, edge_confidence)

                        formatted_path.append(node_info)

                    # Add path metadata
                    formatted_path_with_meta = {
                        "path": formatted_path,
                        "avg_weight": path_total_weight / (len(path) - 1) if len(path) > 1 else 0,
                        "min_confidence": path_min_confidence,
                        "path_length": len(path) - 1
                    }

                    weighted_paths.append(formatted_path_with_meta)

                    # Limit to top 5 paths
                    if len(weighted_paths) >= 5:
                        break

            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                # No path exists
                logging.warning(f"No path found in weighted search: {e}")
                pass

            # Format return value to match existing path_query
            return [wp["path"] for wp in weighted_paths]

        except Exception as e:
            logging.error(f"Error in weighted path query: {e}")
            return []

    def bidirectional_search(self, start_entity: str, end_entity: str, max_hops: int = 3) -> list[list[dict[str, Any]]]:
        """
        Find paths between entities using bidirectional search for efficiency.

        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for bidirectional search")
            return []

        paths = []

        try:
            # Get entity IDs
            source_id = None
            target_id = None

            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()

            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()

            if source_row:
                source_id = source_row[0]

            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()

            if target_row:
                target_id = target_row[0]

            if self.conn is None:
                conn.close()

            # Return empty if entities not found
            if not source_id or not target_id:
                logging.warning(f"Could not find entities: {start_entity} or {end_entity}")
                return []

            # Check for direct connection through a common node
            # First, look for common neighbors
            try:
                source_neighbors = set(nx.all_neighbors(self.graph, source_id))
                target_neighbors = set(nx.all_neighbors(self.graph, target_id))
                common = source_neighbors.intersection(target_neighbors)

                if common:
                    for connector in common:
                        # Get details about the connector node
                        connector_details = self.get_entity_metadata(connector)
                        connector_name = connector_details.get('entity', 'Unknown')

                        # Create a simple path through the common node
                        formatted_path = []

                        # Add source node
                        source_details = self.get_entity_metadata(source_id)
                        source_node = {
                            "id": source_id,
                            "entity": source_details.get('entity', 'Unknown'),
                            "type": source_details.get('entity_type', 'unknown')
                        }

                        # Get edge data for source to connector
                        if self.graph.has_edge(source_id, connector):
                            edge_data = self.graph.get_edge_data(source_id, connector)
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            source_node["next_relation"] = relation_info
                        else:
                            # Check if the edge is in the reverse direction
                            edge_data = self.graph.get_edge_data(connector, source_id)
                            relation_info = {
                                "type": f"inverse_{edge_data.get('relation', 'is_related_to')}",
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            source_node["next_relation"] = relation_info

                        formatted_path.append(source_node)

                        # Add connector node
                        connector_node = {
                            "id": connector,
                            "entity": connector_name,
                            "type": connector_details.get('entity_type', 'unknown')
                        }

                        # Get edge data for connector to target
                        if self.graph.has_edge(connector, target_id):
                            edge_data = self.graph.get_edge_data(connector, target_id)
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            connector_node["next_relation"] = relation_info
                        else:
                            # Check if the edge is in the reverse direction
                            edge_data = self.graph.get_edge_data(target_id, connector)
                            relation_info = {
                                "type": f"inverse_{edge_data.get('relation', 'is_related_to')}",
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            connector_node["next_relation"] = relation_info

                        formatted_path.append(connector_node)

                        # Add target node
                        target_details = self.get_entity_metadata(target_id)
                        target_node = {
                            "id": target_id,
                            "entity": target_details.get('entity', 'Unknown'),
                            "type": target_details.get('entity_type', 'unknown')
                        }
                        formatted_path.append(target_node)

                        paths.append(formatted_path)

                        # Only use the first common node for simplicity
                        break

                if paths:
                    return paths
            except Exception as e:
                logging.error(f"Error checking for common neighbors: {e}")

            # If no direct connection through common neighbors, use bidirectional BFS
            # Implementation of bidirectional BFS
            max_distance = max_hops // 2 + max_hops % 2  # Split max hops between forward and backward searches

            # Forward search from source
            forward_paths = {source_id: [[source_id]]}
            forward_visited = {source_id}

            # Backward search from target
            backward_paths = {target_id: [[target_id]]}
            backward_visited = {target_id}

            # Intersection of paths
            intersection = set()

            # Bidirectional BFS
            for _ in range(max_distance):
                # If no paths to expand, break
                if not forward_paths and not backward_paths:
                    break

                # Expand forward paths
                new_forward_paths = {}
                for node, paths_to_node in forward_paths.items():
                    try:
                        for neighbor in self.graph.successors(node):
                            if neighbor not in forward_visited:
                                new_forward_paths.setdefault(neighbor, [])
                                for path in paths_to_node:
                                    new_forward_paths[neighbor].append(path + [neighbor])
                                forward_visited.add(neighbor)

                                # Check for intersection
                                if neighbor in backward_visited:
                                    intersection.add(neighbor)
                    except Exception as e:
                        logging.error(f"Error expanding forward from node {node}: {e}")

                forward_paths = new_forward_paths

                # Expand backward paths
                new_backward_paths = {}
                for node, paths_to_node in backward_paths.items():
                    try:
                        for neighbor in self.graph.predecessors(node):
                            if neighbor not in backward_visited:
                                new_backward_paths.setdefault(neighbor, [])
                                for path in paths_to_node:
                                    new_backward_paths[neighbor].append([neighbor] + path)
                                backward_visited.add(neighbor)

                                # Check for intersection
                                if neighbor in forward_visited:
                                    intersection.add(neighbor)
                    except Exception as e:
                        logging.error(f"Error expanding backward from node {node}: {e}")

                backward_paths = new_backward_paths

                # If we have intersections, construct the paths
                if intersection:
                    break

            # Construct complete paths
            complete_paths = []
            for node in intersection:
                # Get all forward paths to the intersection node
                if node in forward_paths:
                    forward_to_node = forward_paths[node]
                else:
                    # Check if this is the source
                    forward_to_node = [[source_id]] if node == source_id else []

                # Get all backward paths from the intersection node
                if node in backward_paths:
                    backward_from_node = backward_paths[node]
                else:
                    # Check if this is the target
                    backward_from_node = [[target_id]] if node == target_id else []

                # Connect forward and backward paths
                for f_path in forward_to_node:
                    for b_path in backward_from_node:
                        if node == f_path[-1] and node == b_path[0]:
                            # Merge paths, avoiding duplicate intersection node
                            complete_path = f_path + b_path[1:]
                            if len(complete_path) <= max_hops + 1:
                                complete_paths.append(complete_path)

            # If no paths found, try a direct connection search
            if not complete_paths:
                logging.info("No paths found using bidirectional BFS, trying direct path search")
                try:
                    # Look for direct paths using a higher max_hops
                    for path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops):
                        complete_paths.append(path)
                        # Only take the first few paths
                        if len(complete_paths) >= 3:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    logging.warning(f"No direct path exists: {e}")

            # Format paths
            for path in complete_paths[:5]:  # Limit to top 5 paths
                formatted_path = []

                for i, node_id in enumerate(path):
                    # Get node details
                    node_details = self.get_entity_metadata(node_id)

                    node_info = {
                        "id": node_id,
                        "entity": node_details.get('entity', 'Unknown'),
                        "type": node_details.get('entity_type', 'unknown')
                    }

                    # Add relation to next node if not the last node
                    if i < len(path) - 1:
                        next_node = path[i + 1]
                        edge_data = self.graph.get_edge_data(node_id, next_node)

                        if edge_data:
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            node_info["next_relation"] = relation_info

                    formatted_path.append(node_info)

                paths.append(formatted_path)

            # If we still haven't found a path, try a common connection through intermediate nodes
            if not paths:
                logging.info("No direct paths found, looking for connections through intermediate nodes")
                # Find all nodes that connect to the source
                source_connections = set()
                try:
                    for node in self.graph.nodes():
                        if nx.has_path(self.graph, source_id, node) or nx.has_path(self.graph, node, source_id):
                            source_connections.add(node)
                except Exception as e:
                    logging.error(f"Error finding source connections: {e}")

                # Find all nodes that connect to the target
                target_connections = set()
                try:
                    for node in self.graph.nodes():
                        if nx.has_path(self.graph, target_id, node) or nx.has_path(self.graph, node, target_id):
                            target_connections.add(node)
                except Exception as e:
                    logging.error(f"Error finding target connections: {e}")

                # Find common connections
                common_connections = source_connections.intersection(target_connections)
                if common_connections:
                    # Use the first common connection for simplicity
                    connector = next(iter(common_connections))

                    # Try to find a path from source to connector
                    source_to_connector = None
                    try:
                        source_to_connector = next(nx.all_simple_paths(
                            self.graph, source_id, connector, cutoff=max_hops//2
                        ))
                    except (nx.NetworkXNoPath, StopIteration):
                        try:
                            source_to_connector = next(nx.all_simple_paths(
                                self.graph, connector, source_id, cutoff=max_hops//2
                            ))
                            # Reverse the path
                            source_to_connector = list(reversed(source_to_connector))
                        except (nx.NetworkXNoPath, StopIteration):
                            pass

                    # Try to find a path from connector to target
                    connector_to_target = None
                    try:
                        connector_to_target = next(nx.all_simple_paths(
                            self.graph, connector, target_id, cutoff=max_hops//2
                        ))
                    except (nx.NetworkXNoPath, StopIteration):
                        try:
                            connector_to_target = next(nx.all_simple_paths(
                                self.graph, target_id, connector, cutoff=max_hops//2
                            ))
                            # Reverse the path
                            connector_to_target = list(reversed(connector_to_target))
                        except (nx.NetworkXNoPath, StopIteration):
                            pass

                    # If we found both paths, combine them
                    if source_to_connector and connector_to_target:
                        # Combine paths, avoiding duplicate connector node
                        complete_path = source_to_connector
                        if connector_to_target[0] == complete_path[-1]:
                            complete_path.extend(connector_to_target[1:])
                        else:
                            complete_path.extend(connector_to_target)

                        # Format the path
                        formatted_path = []
                        for i, node_id in enumerate(complete_path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)

                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }

                            # Add relation to next node if not the last node
                            if i < len(complete_path) - 1:
                                next_node = complete_path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)

                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info

                            formatted_path.append(node_info)

                        paths.append(formatted_path)

            return paths

        except Exception as e:
            logging.error(f"Error in bidirectional search: {e}")
            return []

    def constrained_path_search(self, start_entity: str, end_entity: str,
                              allowed_relations: list[str] = None,
                              forbidden_relations: list[str] = None,
                              max_hops: int = 3) -> list[list[dict[str, Any]]]:
        """
        Find paths with constraints on relation types.

        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            allowed_relations: List of relation types to allow (if None, all are allowed)
            forbidden_relations: List of relation types to forbid (if None, none are forbidden)
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for constrained path search")
            return []

        paths = []

        try:
            # Get entity IDs
            source_id = None
            target_id = None

            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()

            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()

            if source_row:
                source_id = source_row[0]

            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()

            if target_row:
                target_id = target_row[0]

            if self.conn is None:
                conn.close()

            # Return empty if entities not found
            if not source_id or not target_id:
                return []

            # Create a subgraph with only allowed relations
            constrained_graph = nx.DiGraph()

            # Copy nodes
            for node in self.graph.nodes():
                constrained_graph.add_node(node)

            # Copy edges that meet constraints
            for u, v, data in self.graph.edges(data=True):
                relation = data.get('relation', '')

                # Skip forbidden relations
                if forbidden_relations and relation in forbidden_relations:
                    continue

                # Check if relation is allowed
                if allowed_relations is None or relation in allowed_relations:
                    constrained_graph.add_edge(u, v, **data)

            # Find paths in constrained graph
            try:
                # Find all simple paths (can be slow for large graphs)
                simple_paths = nx.all_simple_paths(constrained_graph, source_id, target_id, cutoff=max_hops)

                # Convert paths to our format
                for path in list(simple_paths)[:5]:  # Limit to top 5 paths
                    formatted_path = []

                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)

                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }

                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)

                            if edge_data:
                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_data.get('weight', 1.0),
                                    "confidence": edge_data.get('confidence', 0.5)
                                }
                                node_info["next_relation"] = relation_info

                        formatted_path.append(node_info)

                    paths.append(formatted_path)

            except nx.NetworkXNoPath:
                # No path exists
                pass

            return paths

        except Exception as e:
            logging.error(f"Error in constrained path search: {e}")
            return []

    def contract_graph(self, min_edge_weight: float = 0.2,
                      min_confidence: float = 0.3,
                      combine_parallel_edges: bool = True) -> dict[str, Any]:
        """
        Contract the graph to handle large knowledge graphs efficiently.
        Removes low-weight/confidence edges and combines parallel edges.

        Args:
            min_edge_weight: Minimum edge weight to keep
            min_confidence: Minimum confidence to keep
            combine_parallel_edges: Whether to combine parallel edges between nodes

        Returns:
            Dictionary with statistics about the contraction
        """
        # NetworkX is required for graph contraction
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for graph contraction")
            return {"success": False, "reason": "NetworkX not available"}

        stats = {
            "original_nodes": self.graph.number_of_nodes(),
            "original_edges": self.graph.number_of_edges(),
            "removed_edges": 0,
            "combined_edges": 0,
            "success": True
        }

        try:
            # Create a new graph for the contracted result
            contracted_graph = nx.DiGraph()

            # Copy all nodes
            for node, data in self.graph.nodes(data=True):
                contracted_graph.add_node(node, **data)

            # Filter edges by weight and confidence
            for u, v, data in self.graph.edges(data=True):
                edge_weight = data.get('weight', 0.5)
                edge_confidence = data.get('confidence', 0.5)

                if edge_weight >= min_edge_weight and edge_confidence >= min_confidence:
                    contracted_graph.add_edge(u, v, **data)
                else:
                    stats["removed_edges"] += 1

            # Combine parallel edges if requested
            if combine_parallel_edges:
                # Find all node pairs with multiple edges
                multi_edges = {}

                for u, v, key, data in contracted_graph.edges(data=True, keys=True):
                    multi_edges.setdefault((u, v), []).append((key, data))

                # Combine parallel edges
                for (u, v), edges in multi_edges.items():
                    if len(edges) > 1:
                        # Combine edges between the same nodes
                        combined_data = {
                            "relations": [],
                            "weight": 0,
                            "confidence": 0,
                            "is_combined": True
                        }

                        for key, data in edges:
                            relation = data.get('relation', '')
                            if relation and relation not in combined_data["relations"]:
                                combined_data["relations"].append(relation)

                            combined_data["weight"] += data.get('weight', 0.5)
                            combined_data["confidence"] += data.get('confidence', 0.5)

                        # Average the weight and confidence
                        combined_data["weight"] /= len(edges)
                        combined_data["confidence"] /= len(edges)

                        # Create a combined relation description
                        if combined_data["relations"]:
                            combined_data["relation"] = " & ".join(combined_data["relations"])
                        else:
                            combined_data["relation"] = "related_to"

                        # Remove old edges
                        for key, _ in edges:
                            contracted_graph.remove_edge(u, v, key)

                        # Add combined edge
                        contracted_graph.add_edge(u, v, **combined_data)
                        stats["combined_edges"] += len(edges) - 1

            # Update the graph
            self.graph = contracted_graph

            # Update stats
            stats["final_nodes"] = self.graph.number_of_nodes()
            stats["final_edges"] = self.graph.number_of_edges()

            return stats

        except Exception as e:
            logging.error(f"Error contracting graph: {e}")
            return {"success": False, "reason": str(e)}

    def create_graph_abstraction(self, community_resolution: float = 1.0,
                               min_community_size: int = 3) -> dict[str, Any]:
        """
        Create a hierarchical abstraction of the graph using community detection.
        Useful for navigating and querying large knowledge graphs.

        Args:
            community_resolution: Resolution parameter for community detection (higher values create smaller communities)
            min_community_size: Minimum size for a community to be represented as a supernode

        Returns:
            Dictionary with abstraction details and statistics
        """
        # NetworkX is required for graph abstraction
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for graph abstraction")
            return {"success": False, "reason": "NetworkX not available"}

        # Additional libraries required
        try:
            import community as community_louvain
        except ImportError:
            logging.warning("python-louvain package not found. Install it for graph abstraction.")
            return {"success": False, "reason": "Required package 'python-louvain' not installed"}

        abstraction_stats = {
            "original_nodes": self.graph.number_of_nodes(),
            "original_edges": self.graph.number_of_edges(),
            "communities": 0,
            "supernodes": 0,
            "success": True
        }

        try:
            # Create an undirected copy of the graph for community detection
            undirected_graph = self.graph.to_undirected()

            # Detect communities using Louvain method
            partition = community_louvain.best_partition(undirected_graph,
                                                        resolution=community_resolution,
                                                        random_state=42)

            # Count communities
            communities = {}
            for node, community_id in partition.items():
                communities.setdefault(community_id, []).append(node)

            abstraction_stats["communities"] = len(communities)

            # Create abstracted graph
            abstracted_graph = nx.DiGraph()

            # Track community metadata
            community_metadata = {}

            # Create supernodes for sufficiently large communities
            for community_id, nodes in communities.items():
                if len(nodes) >= min_community_size:
                    # Create a supernode
                    supernode_id = f"community_{community_id}"

                    # Determine the most representative entity for the community
                    # (highest degree or highest confidence)
                    representative = max(nodes, key=lambda n: self.graph.degree(n))
                    rep_metadata = self.get_entity_metadata(representative)

                    # Get types of entities in this community
                    entity_types = {}
                    for node in nodes:
                        node_metadata = self.get_entity_metadata(node)
                        node_type = node_metadata.get('entity_type', 'unknown')
                        entity_types[node_type] = entity_types.get(node_type, 0) + 1

                    # Get most common entity type
                    common_type = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else "mixed"

                    supernode_attrs = {
                        "is_supernode": True,
                        "community_id": community_id,
                        "size": len(nodes),
                        "representative": rep_metadata.get('entity', 'Unknown'),
                        "entity_type": common_type,
                        "members": nodes  # Store member nodes for expansion
                    }

                    abstracted_graph.add_node(supernode_id, **supernode_attrs)
                    community_metadata[community_id] = supernode_attrs
                    abstraction_stats["supernodes"] += 1
                else:
                    # Add individual nodes
                    for node in nodes:
                        node_data = self.graph.nodes[node]
                        abstracted_graph.add_node(node, **node_data)

            # Add edges between supernodes and regular nodes
            for u, v, data in self.graph.edges(data=True):
                u_community = partition.get(u)
                v_community = partition.get(v)

                if u_community == v_community:
                    # Skip intra-community edges unless the community is too small
                    if len(communities[u_community]) >= min_community_size:
                        continue

                # Determine source node or supernode
                if u_community is not None and len(communities[u_community]) >= min_community_size:
                    source = f"community_{u_community}"
                else:
                    source = u

                # Determine target node or supernode
                if v_community is not None and len(communities[v_community]) >= min_community_size:
                    target = f"community_{v_community}"
                else:
                    target = v

                # Add or update edge
                if abstracted_graph.has_edge(source, target):
                    # Update existing edge
                    edge_data = abstracted_graph.get_edge_data(source, target)
                    edge_data["weight"] = edge_data.get("weight", 0) + data.get("weight", 1.0)
                    edge_data["count"] = edge_data.get("count", 0) + 1
                else:
                    # Add new edge
                    abstracted_graph.add_edge(source, target,
                                             weight=data.get("weight", 1.0),
                                             relation=data.get("relation", "related_to"),
                                             count=1)

            # Normalize edge weights for abstracted graph
            for u, v, data in abstracted_graph.edges(data=True):
                if "count" in data:
                    data["weight"] = data["weight"] / data["count"]

            # Store abstracted graph and metadata
            self.abstracted_graph = abstracted_graph
            self.community_metadata = community_metadata

            # Update stats
            abstraction_stats["abstracted_nodes"] = abstracted_graph.number_of_nodes()
            abstraction_stats["abstracted_edges"] = abstracted_graph.number_of_edges()
            abstraction_stats["compression_ratio"] = (abstraction_stats["original_nodes"] /
                                                     abstraction_stats["abstracted_nodes"])

            return abstraction_stats

        except Exception as e:
            logging.error(f"Error creating graph abstraction: {e}")
            return {"success": False, "reason": str(e)}

    def path_query_with_abstraction(self, start_entity: str, end_entity: str,
                                  max_hops: int = 5) -> list[list[dict[str, Any]]]:
        """
        Find paths between entities using graph abstraction for efficiency.

        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # Check if abstraction is available
        if not hasattr(self, 'abstracted_graph') or self.abstracted_graph is None:
            logging.warning("Graph abstraction not available. Creating one with default settings.")
            abstraction_result = self.create_graph_abstraction()
            if not abstraction_result.get("success", False):
                logging.warning("Failed to create abstraction. Falling back to regular path query.")
                return self.path_query(start_entity, end_entity, max_hops)

        paths = []

        try:
            # Get entity IDs
            source_id = None
            target_id = None

            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()

            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()

            if source_row:
                source_id = source_row[0]

            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()

            if target_row:
                target_id = target_row[0]

            if self.conn is None:
                conn.close()

            # Return empty if entities not found
            if not source_id or not target_id:
                return []

            # Find which community each entity belongs to
            source_community = None
            target_community = None

            for community_id, metadata in self.community_metadata.items():
                if source_id in metadata.get("members", []):
                    source_community = community_id
                if target_id in metadata.get("members", []):
                    target_community = community_id

            # Determine starting and ending nodes in abstracted graph
            if source_community is not None:
                abstracted_source = f"community_{source_community}"
            else:
                abstracted_source = source_id

            if target_community is not None:
                abstracted_target = f"community_{target_community}"
            else:
                abstracted_target = target_id

            # If source and target are in the same community, use regular path query
            if source_community is not None and source_community == target_community:
                # Filter the graph to only include nodes in this community
                community_nodes = self.community_metadata[source_community]["members"]
                subgraph = self.graph.subgraph(community_nodes)

                # Find paths in this community
                try:
                    simple_paths = nx.all_simple_paths(subgraph, source_id, target_id, cutoff=max_hops)

                    # Convert paths to our format
                    for path in list(simple_paths)[:5]:  # Limit to top 5 paths
                        formatted_path = []

                        for i, node_id in enumerate(path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)

                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }

                            # Add relation to next node if not the last node
                            if i < len(path) - 1:
                                next_node = path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)

                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info

                            formatted_path.append(node_info)

                        paths.append(formatted_path)

                except nx.NetworkXNoPath:
                    # No path exists
                    pass

                return paths

            # Find paths in abstracted graph
            try:
                abstracted_paths = list(nx.all_simple_paths(
                    self.abstracted_graph,
                    abstracted_source,
                    abstracted_target,
                    cutoff=max_hops//2
                ))[:3]  # Limit to top 3 abstracted paths

                # Expand abstracted paths to detailed paths
                for abst_path in abstracted_paths:
                    # Build a list of segments to find paths for
                    segments = []

                    for i in range(len(abst_path) - 1):
                        current = abst_path[i]
                        next_node = abst_path[i + 1]

                        # Determine actual nodes to connect
                        if isinstance(current, str) and current.startswith("community_"):
                            if i == 0:  # Source community
                                start_node = source_id
                            else:
                                # Use representative node or random member
                                comm_id = int(current.split("_")[1])
                                start_node = self.community_metadata[comm_id].get("members", [])[0]
                        else:
                            start_node = current

                        if isinstance(next_node, str) and next_node.startswith("community_"):
                            if i == len(abst_path) - 2:  # Target community
                                end_node = target_id
                            else:
                                # Use representative node or random member
                                comm_id = int(next_node.split("_")[1])
                                end_node = self.community_metadata[comm_id].get("members", [])[0]
                        else:
                            end_node = next_node

                        segments.append((start_node, end_node))

                    # Find paths for each segment and connect them
                    segment_paths = []
                    for start, end in segments:
                        try:
                            # Find a single path for this segment
                            segment_path = next(nx.all_simple_paths(self.graph, start, end, cutoff=2))
                            segment_paths.append(segment_path)
                        except (nx.NetworkXNoPath, StopIteration):
                            # No path exists for this segment
                            # Try using neighbors as connectors
                            try:
                                # Find common neighbors
                                start_neighbors = set(self.graph.successors(start))
                                end_neighbors = set(self.graph.predecessors(end))
                                common_neighbors = start_neighbors.intersection(end_neighbors)

                                if common_neighbors:
                                    # Use first common neighbor
                                    connector = next(iter(common_neighbors))
                                    segment_path = [start, connector, end]
                                    segment_paths.append(segment_path)
                                else:
                                    # No common neighbor, skip this abstracted path
                                    segment_paths = []
                                    break
                            except Exception as e:
                                logging.warning(f"Error finding segment path via common neighbors: {e}")
                                segment_paths = []
                                break

                    # If all segments have paths, connect them
                    if segment_paths:
                        complete_path = segment_paths[0]

                        for i in range(1, len(segment_paths)):
                            # Skip the first node of subsequent segments (already included)
                            complete_path.extend(segment_paths[i][1:])

                        # Format the complete path
                        formatted_path = []

                        for i, node_id in enumerate(complete_path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)

                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }

                            # Add relation to next node if not the last node
                            if i < len(complete_path) - 1:
                                next_node = complete_path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)

                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info

                            formatted_path.append(node_info)

                        paths.append(formatted_path)

            except nx.NetworkXNoPath:
                # No path exists in abstracted graph
                pass

            # If no paths found, fall back to regular path query
            if not paths:
                logging.info("No paths found using abstraction, falling back to regular path query")
                return self.path_query(start_entity, end_entity, max_hops)

            return paths

        except Exception as e:
            logging.error(f"Error in abstracted path query: {e}")
            # Fall back to regular path query
            return self.path_query(start_entity, end_entity, max_hops)
