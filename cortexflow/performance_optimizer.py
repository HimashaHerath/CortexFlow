"""
CortexFlow Performance Optimizer module.

This module provides performance optimization capabilities for the knowledge graph in CortexFlow.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import sqlite3
from collections import defaultdict, Counter
import heapq

# Try importing graph libraries
try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    NETWORKX_ENABLED = False
    logging.warning("networkx not found. Graph optimization functionality will be limited.")

try:
    import community as community_louvain
    COMMUNITY_DETECTION_ENABLED = True
except ImportError:
    COMMUNITY_DETECTION_ENABLED = False
    logging.warning("python-louvain not found. Community detection for graph partitioning will be limited.")

# Configure logging
logger = logging.getLogger('cortexflow.performance')

class PerformanceOptimizer:
    """
    Performance optimization for CortexFlow knowledge graph.
    Provides features for graph partitioning, indexing, query planning, and caching.
    """
    
    def __init__(self, config, graph_store=None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: CortexFlow configuration
            graph_store: Optional graph store instance to optimize
        """
        self.config = config
        self.graph_store = graph_store
        
        # Cache for query results
        self.query_cache = {}
        
        # Cache for common reasoning patterns
        self.reasoning_cache = {}
        
        # Cache hit/miss statistics
        self.stats = {
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "reasoning_cache_hits": 0,
            "reasoning_cache_misses": 0,
            "partitions_created": 0,
            "index_operations": 0,
            "query_plans_generated": 0,
            "optimization_time": 0.0,
        }
        
        # Create graph partitions
        self.partitions = {}
        self.partition_mapping = {}
        
        # Query pattern statistics for adaptive optimization
        self.query_patterns = Counter()
        
        # Initialize optimizations based on config
        self._init_optimizations()
        
        logger.info("Performance optimizer initialized")
    
    def _init_optimizations(self):
        """Initialize optimization features based on configuration."""
        # Apply partitioning if enabled
        if hasattr(self.config, "use_graph_partitioning") and self.config.use_graph_partitioning:
            if self.graph_store and NETWORKX_ENABLED:
                try:
                    self.partition_graph()
                except Exception as e:
                    logger.error(f"Failed to partition graph: {e}")
        
        # Create indexes if enabled
        if hasattr(self.config, "use_multihop_indexing") and self.config.use_multihop_indexing:
            if self.graph_store:
                try:
                    self.create_hop_indexes()
                except Exception as e:
                    logger.error(f"Failed to create hop indexes: {e}")
    
    def partition_graph(self, method: str = "louvain", partition_count: int = None) -> Dict[str, Any]:
        """
        Partition the knowledge graph to improve query performance.
        
        Args:
            method: Partitioning method, one of "louvain", "spectral", "modularity"
            partition_count: Optional target number of partitions
            
        Returns:
            Dictionary with partition statistics
        """
        if not NETWORKX_ENABLED:
            logger.warning("NetworkX not available, graph partitioning disabled")
            return {"partitions": 0, "status": "failed", "reason": "NetworkX not available"}
        
        if not self.graph_store or not hasattr(self.graph_store, "graph"):
            logger.warning("No graph store available for partitioning")
            return {"partitions": 0, "status": "failed", "reason": "No graph store available"}
        
        start_time = time.time()
        
        try:
            # Get the graph from graph store
            G = self.graph_store.graph
            
            if len(G.nodes()) == 0:
                logger.warning("Graph is empty, cannot partition")
                return {"partitions": 0, "status": "failed", "reason": "Empty graph"}
            
            # Apply the chosen partitioning method
            if method == "louvain" and COMMUNITY_DETECTION_ENABLED:
                # Convert directed graph to undirected for community detection
                G_undirected = G.to_undirected()
                communities = community_louvain.best_partition(G_undirected)
                
                # Community info is returned as a dict mapping node -> community_id
                partition_ids = set(communities.values())
                
                # Create partition mapping
                self.partition_mapping = communities
                
                # Create subgraphs for each partition
                for partition_id in partition_ids:
                    nodes = [node for node, comm_id in communities.items() if comm_id == partition_id]
                    self.partitions[partition_id] = G.subgraph(nodes).copy()
                
            elif method == "spectral":
                if partition_count is None:
                    partition_count = min(10, len(G.nodes()) // 5) if len(G.nodes()) > 10 else 1
                
                # Use spectral clustering for partitioning
                try:
                    import numpy as np
                    from sklearn.cluster import SpectralClustering
                    
                    # Convert graph to adjacency matrix
                    adj_matrix = nx.to_numpy_array(G)
                    
                    # Apply spectral clustering
                    clustering = SpectralClustering(
                        n_clusters=partition_count,
                        affinity='precomputed',
                        n_init=10,
                        assign_labels='discretize'
                    ).fit(adj_matrix)
                    
                    # Get clusters
                    labels = clustering.labels_
                    
                    # Create partition mapping
                    nodes = list(G.nodes())
                    self.partition_mapping = {nodes[i]: labels[i] for i in range(len(nodes))}
                    
                    # Create subgraphs for each partition
                    partition_ids = set(labels)
                    for partition_id in partition_ids:
                        partition_nodes = [nodes[i] for i in range(len(nodes)) if labels[i] == partition_id]
                        self.partitions[partition_id] = G.subgraph(partition_nodes).copy()
                        
                except ImportError:
                    logger.warning("scikit-learn not available, falling back to connected components")
                    # Fall back to connected components
                    components = list(nx.weakly_connected_components(G))
                    for i, component in enumerate(components):
                        self.partitions[i] = G.subgraph(component).copy()
                        for node in component:
                            self.partition_mapping[node] = i
            
            elif method == "modularity":
                # Modularity-based partitioning using Girvan-Newman algorithm
                if partition_count is None:
                    partition_count = min(5, len(G.nodes()) // 10) if len(G.nodes()) > 5 else 1
                
                # Convert directed graph to undirected
                G_undirected = G.to_undirected()
                
                # Generate partitions
                comp = nx.community.girvan_newman(G_undirected)
                
                # Take the first n partitions
                limited_comp = []
                for communities in comp:
                    limited_comp.append(communities)
                    if len(limited_comp) >= partition_count:
                        break
                
                if limited_comp:
                    communities = limited_comp[-1]  # Get the last one
                    
                    # Create partition mapping and subgraphs
                    for i, community in enumerate(communities):
                        for node in community:
                            self.partition_mapping[node] = i
                        self.partitions[i] = G.subgraph(community).copy()
                else:
                    # Fall back to connected components
                    components = list(nx.weakly_connected_components(G))
                    for i, component in enumerate(components):
                        self.partitions[i] = G.subgraph(component).copy()
                        for node in component:
                            self.partition_mapping[node] = i
            
            else:
                # Default: use connected components
                components = list(nx.weakly_connected_components(G))
                for i, component in enumerate(components):
                    self.partitions[i] = G.subgraph(component).copy()
                    for node in component:
                        self.partition_mapping[node] = i
            
            # Update stats
            self.stats["partitions_created"] = len(self.partitions)
            
            # Calculate partition statistics
            partition_stats = {}
            for partition_id, subgraph in self.partitions.items():
                partition_stats[partition_id] = {
                    "nodes": len(subgraph.nodes()),
                    "edges": len(subgraph.edges()),
                    "density": nx.density(subgraph) if len(subgraph.nodes()) > 1 else 0
                }
            
            end_time = time.time()
            execution_time = end_time - start_time
            self.stats["optimization_time"] += execution_time
            
            return {
                "partitions": len(self.partitions),
                "method": method,
                "partition_stats": partition_stats,
                "execution_time": execution_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during graph partitioning: {e}")
            return {"partitions": 0, "status": "failed", "reason": str(e)}
    
    def create_hop_indexes(self, max_hops: int = 2) -> Dict[str, Any]:
        """
        Create indexes for multi-hop queries to speed up traversal.
        
        Args:
            max_hops: Maximum number of hops to index
            
        Returns:
            Dictionary with indexing statistics
        """
        if not self.graph_store:
            logger.warning("No graph store available for indexing")
            return {"status": "failed", "reason": "No graph store available"}
        
        start_time = time.time()
        
        try:
            # Access the database connection from graph store
            if hasattr(self.graph_store, "conn") and self.graph_store.conn:
                conn = self.graph_store.conn
            else:
                # Create a new connection to the database
                conn = sqlite3.connect(self.graph_store.db_path)
            
            cursor = conn.cursor()
            
            # Create multi-hop index table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_hop_indexes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                hop_count INTEGER NOT NULL,
                path_metadata TEXT,
                timestamp REAL,
                UNIQUE(source_entity_id, target_entity_id, hop_count)
            )
            ''')
            
            # Create indexes for the table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hop_source ON graph_hop_indexes(source_entity_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hop_target ON graph_hop_indexes(target_entity_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hop_count ON graph_hop_indexes(hop_count)')
            
            # Commit the schema changes
            conn.commit()
            
            # Index 1-hop connections (direct relationships from graph_relationships)
            cursor.execute('''
            INSERT OR REPLACE INTO graph_hop_indexes (source_entity_id, target_entity_id, hop_count, path_metadata, timestamp)
            SELECT source_id, target_id, 1, 
                   json_object('relation_type', relation_type, 'weight', weight, 'confidence', confidence),
                   strftime('%s', 'now')
            FROM graph_relationships
            ''')
            
            # Commit the 1-hop indexes
            conn.commit()
            
            # For 2+ hops, we need to join the tables
            index_count = 0
            
            # Only build additional hop indexes if max_hops > 1
            if max_hops > 1 and NETWORKX_ENABLED and hasattr(self.graph_store, "graph"):
                G = self.graph_store.graph
                
                # Get all nodes
                nodes = list(G.nodes())
                
                # For each node, find paths up to max_hops away
                for source_node in nodes:
                    # Get the entity ID for the source node
                    cursor.execute("SELECT id FROM graph_entities WHERE entity = ?", (source_node,))
                    source_id_result = cursor.fetchone()
                    
                    if not source_id_result:
                        continue
                    
                    source_id = source_id_result[0]
                    
                    # Use BFS to find nodes within max_hops
                    for hop in range(2, max_hops + 1):
                        # Find all nodes exactly hop steps away
                        paths = nx.single_source_shortest_path_length(G, source_node, cutoff=hop)
                        
                        # Filter to only nodes exactly 'hop' steps away
                        hop_nodes = [node for node, length in paths.items() if length == hop]
                        
                        for target_node in hop_nodes:
                            # Get the entity ID for the target node
                            cursor.execute("SELECT id FROM graph_entities WHERE entity = ?", (target_node,))
                            target_id_result = cursor.fetchone()
                            
                            if not target_id_result:
                                continue
                                
                            target_id = target_id_result[0]
                            
                            # Find shortest path between source and target
                            path = nx.shortest_path(G, source=source_node, target=target_node)
                            
                            # Create path metadata
                            path_metadata = {
                                "path": path,
                                "length": hop,
                                "intermediate_nodes": path[1:-1]
                            }
                            
                            # Insert into index
                            cursor.execute('''
                            INSERT OR REPLACE INTO graph_hop_indexes 
                            (source_entity_id, target_entity_id, hop_count, path_metadata, timestamp)
                            VALUES (?, ?, ?, ?, strftime('%s', 'now'))
                            ''', (source_id, target_id, hop, json.dumps(path_metadata)))
                            
                            index_count += 1
                            
                        # Commit after each hop level
                        conn.commit()
            
            # Update stats
            self.stats["index_operations"] += (index_count + 1)  # +1 for the 1-hop batch operation
            
            end_time = time.time()
            execution_time = end_time - start_time
            self.stats["optimization_time"] += execution_time
            
            return {
                "indexes_created": index_count + 1,  # Include 1-hop indexes
                "max_hops": max_hops,
                "execution_time": execution_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during hop indexing: {e}")
            return {"status": "failed", "reason": str(e)}
        finally:
            # Close the connection if we created a new one
            if not hasattr(self.graph_store, "conn") and conn:
                conn.close()
    
    def generate_query_plan(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an optimized query plan for reasoning over the knowledge graph.
        
        Args:
            query: Dictionary containing query parameters
            
        Returns:
            Optimized query plan
        """
        start_time = time.time()
        
        # Extract query parameters
        query_type = query.get("type", "path")
        start_entity = query.get("start_entity")
        end_entity = query.get("end_entity")
        relation_constraints = query.get("relation_constraints", [])
        max_hops = query.get("max_hops", 3)
        
        # Default plan
        plan = {
            "steps": [],
            "estimated_cost": float('inf'),
            "strategy": "default"
        }
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                self.stats["query_cache_hits"] += 1
                return self.query_cache[cache_key]
            
            self.stats["query_cache_misses"] += 1
            
            # Update query pattern statistics
            pattern_key = f"{query_type}:{max_hops}"
            self.query_patterns[pattern_key] += 1
            
            # Determine if we should use partitioning
            use_partitioning = len(self.partitions) > 0
            
            # Determine if we should use hop indexes
            use_hop_indexes = hasattr(self.config, "use_multihop_indexing") and self.config.use_multihop_indexing
            
            # Path query optimization
            if query_type == "path" and start_entity and end_entity:
                # Check if start and end are in the same partition
                if use_partitioning and start_entity in self.partition_mapping and end_entity in self.partition_mapping:
                    start_partition = self.partition_mapping[start_entity]
                    end_partition = self.partition_mapping[end_entity]
                    
                    if start_partition == end_partition:
                        # Entities in same partition, use subgraph search
                        plan["steps"] = [
                            {"operation": "get_partition", "partition_id": start_partition},
                            {"operation": "path_search", "algorithm": "bidirectional", "max_hops": max_hops}
                        ]
                        plan["strategy"] = "partition_based"
                        plan["estimated_cost"] = 0.3  # Much lower cost when in same partition
                    else:
                        # Entities in different partitions
                        # Use hop indexes if available and within range
                        if use_hop_indexes and max_hops <= self.config.max_indexed_hops:
                            plan["steps"] = [
                                {"operation": "lookup_hop_index", "max_hops": max_hops},
                                {"operation": "verify_constraints", "constraints": relation_constraints}
                            ]
                            plan["strategy"] = "hop_index"
                            plan["estimated_cost"] = 0.5
                        else:
                            # Cross-partition search with graph abstraction for efficiency
                            plan["steps"] = [
                                {"operation": "abstract_graph", "level": "partition"},
                                {"operation": "path_search_abstract", "algorithm": "bidirectional", "max_hops": max_hops},
                                {"operation": "expand_abstract_path"}
                            ]
                            plan["strategy"] = "cross_partition"
                            plan["estimated_cost"] = 0.7
                else:
                    # No partitioning or entities not in partition mapping
                    # Use hop indexes if available and within range
                    if use_hop_indexes and max_hops <= self.config.max_indexed_hops:
                        plan["steps"] = [
                            {"operation": "lookup_hop_index", "max_hops": max_hops},
                            {"operation": "verify_constraints", "constraints": relation_constraints}
                        ]
                        plan["strategy"] = "hop_index"
                        plan["estimated_cost"] = 0.5
                    else:
                        # Default to bidirectional search
                        plan["steps"] = [
                            {"operation": "path_search", "algorithm": "bidirectional", "max_hops": max_hops}
                        ]
                        plan["strategy"] = "bidirectional"
                        plan["estimated_cost"] = 1.0
            
            # Subgraph query optimization
            elif query_type == "subgraph" and start_entity:
                # Determine radius of subgraph
                radius = query.get("radius", 2)
                
                # Use partitioning if available
                if use_partitioning and start_entity in self.partition_mapping:
                    partition_id = self.partition_mapping[start_entity]
                    partition_size = len(self.partitions[partition_id].nodes())
                    
                    if partition_size < 1000:  # Small partition, explore whole partition
                        plan["steps"] = [
                            {"operation": "get_partition", "partition_id": partition_id},
                            {"operation": "filter_by_constraints", "constraints": relation_constraints}
                        ]
                        plan["strategy"] = "whole_partition"
                        plan["estimated_cost"] = 0.4
                    else:
                        # Large partition, use radius-based exploration
                        plan["steps"] = [
                            {"operation": "get_partition", "partition_id": partition_id},
                            {"operation": "neighborhood_search", "radius": radius},
                            {"operation": "filter_by_constraints", "constraints": relation_constraints}
                        ]
                        plan["strategy"] = "partition_neighborhood"
                        plan["estimated_cost"] = 0.6
                else:
                    # No partitioning, use radius-based exploration of whole graph
                    plan["steps"] = [
                        {"operation": "neighborhood_search", "radius": radius},
                        {"operation": "filter_by_constraints", "constraints": relation_constraints}
                    ]
                    plan["strategy"] = "radius_based"
                    plan["estimated_cost"] = 0.8
            
            # Store in cache
            self.query_cache[cache_key] = plan
            
            end_time = time.time()
            execution_time = end_time - start_time
            self.stats["optimization_time"] += execution_time
            self.stats["query_plans_generated"] += 1
            
            # Add execution time to plan
            plan["plan_generation_time"] = execution_time
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating query plan: {e}")
            # Return a simple fallback plan
            return {
                "steps": [
                    {"operation": "path_search", "algorithm": "bidirectional", "max_hops": max_hops}
                ],
                "strategy": "fallback",
                "estimated_cost": 1.0,
                "error": str(e)
            }
    
    def cache_reasoning_pattern(self, pattern_key: str, pattern_result: Any) -> None:
        """
        Cache a common reasoning pattern for reuse.
        
        Args:
            pattern_key: Unique identifier for the reasoning pattern
            pattern_result: Result of the reasoning pattern
        """
        self.reasoning_cache[pattern_key] = {
            "result": pattern_result,
            "timestamp": time.time(),
            "usage_count": 0
        }
    
    def get_cached_reasoning(self, pattern_key: str) -> Optional[Any]:
        """
        Retrieve a cached reasoning pattern result.
        
        Args:
            pattern_key: Unique identifier for the reasoning pattern
            
        Returns:
            Cached result if available, None otherwise
        """
        if pattern_key in self.reasoning_cache:
            cache_entry = self.reasoning_cache[pattern_key]
            cache_entry["usage_count"] += 1
            self.stats["reasoning_cache_hits"] += 1
            return cache_entry["result"]
        
        self.stats["reasoning_cache_misses"] += 1
        return None
    
    def optimize_query_execution(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all optimization strategies to a query.
        
        Args:
            query: Query parameters
            
        Returns:
            Optimized query plan and execution strategy
        """
        # Step 1: Generate query plan
        plan = self.generate_query_plan(query)
        
        # Step 2: Try to get from reasoning cache
        cache_key = self._generate_cache_key(query)
        cached_result = self.get_cached_reasoning(cache_key)
        
        if cached_result is not None:
            return {
                "plan": plan,
                "result": cached_result,
                "source": "cache"
            }
        
        # Step 3: Return optimized plan for execution
        return {
            "plan": plan,
            "source": "plan_only"
        }
    
    def _generate_cache_key(self, query: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a query.
        
        Args:
            query: Query parameters
            
        Returns:
            String cache key
        """
        relevant_parts = []
        
        if "type" in query:
            relevant_parts.append(f"type:{query['type']}")
        
        if "start_entity" in query:
            relevant_parts.append(f"start:{query['start_entity']}")
        
        if "end_entity" in query:
            relevant_parts.append(f"end:{query['end_entity']}")
        
        if "max_hops" in query:
            relevant_parts.append(f"hops:{query['max_hops']}")
        
        if "relation_constraints" in query and query["relation_constraints"]:
            constraints = sorted(query["relation_constraints"])
            relevant_parts.append(f"constraints:{','.join(constraints)}")
        
        return "|".join(relevant_parts)
    
    def clear_caches(self) -> Dict[str, int]:
        """
        Clear all caches and return statistics.
        
        Returns:
            Statistics about cleared items
        """
        query_cache_size = len(self.query_cache)
        reasoning_cache_size = len(self.reasoning_cache)
        
        self.query_cache = {}
        self.reasoning_cache = {}
        
        return {
            "query_cache_cleared": query_cache_size,
            "reasoning_cache_cleared": reasoning_cache_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            "caching": {
                "query_cache_size": len(self.query_cache),
                "query_cache_hits": self.stats["query_cache_hits"],
                "query_cache_misses": self.stats["query_cache_misses"],
                "reasoning_cache_size": len(self.reasoning_cache),
                "reasoning_cache_hits": self.stats["reasoning_cache_hits"],
                "reasoning_cache_misses": self.stats["reasoning_cache_misses"],
                "hit_rate": self._calculate_hit_rate()
            },
            "partitioning": {
                "partition_count": len(self.partitions),
                "partitions_created": self.stats["partitions_created"]
            },
            "indexing": {
                "index_operations": self.stats["index_operations"]
            },
            "query_planning": {
                "plans_generated": self.stats["query_plans_generated"]
            },
            "common_patterns": dict(self.query_patterns.most_common(5)),
            "total_optimization_time": self.stats["optimization_time"]
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        query_hits = self.stats["query_cache_hits"]
        query_misses = self.stats["query_cache_misses"]
        reasoning_hits = self.stats["reasoning_cache_hits"]
        reasoning_misses = self.stats["reasoning_cache_misses"]
        
        total_requests = query_hits + query_misses + reasoning_hits + reasoning_misses
        total_hits = query_hits + reasoning_hits
        
        if total_requests == 0:
            return 0.0
        
        return (total_hits / total_requests) * 100.0
    
    def prune_partitions(self, density_threshold: float = 0.1) -> int:
        """
        Prune partitions by merging sparse partitions.
        
        Args:
            density_threshold: Minimum density for a partition to remain independent
            
        Returns:
            Number of partitions after pruning
        """
        if not NETWORKX_ENABLED or not self.partitions:
            return 0
        
        # Find sparse partitions
        sparse_partitions = []
        for partition_id, subgraph in self.partitions.items():
            if len(subgraph.nodes()) > 1:
                density = nx.density(subgraph)
                if density < density_threshold:
                    sparse_partitions.append(partition_id)
        
        if not sparse_partitions:
            return len(self.partitions)
        
        # Find optimal merging strategy
        if len(sparse_partitions) > 1:
            # Merge all sparse partitions together
            new_partition_id = max(self.partitions.keys()) + 1
            nodes_to_merge = []
            
            for partition_id in sparse_partitions:
                nodes_to_merge.extend(self.partitions[partition_id].nodes())
                
            # Create the merged partition
            if hasattr(self.graph_store, "graph"):
                self.partitions[new_partition_id] = self.graph_store.graph.subgraph(nodes_to_merge).copy()
                
                # Update partition mappings
                for node in nodes_to_merge:
                    self.partition_mapping[node] = new_partition_id
                
                # Remove old partitions
                for partition_id in sparse_partitions:
                    if partition_id in self.partitions:
                        del self.partitions[partition_id]
        
        return len(self.partitions)
    
    def close(self):
        """Clean up resources."""
        # Clear caches
        self.query_cache = {}
        self.reasoning_cache = {}
        self.partitions = {}
        self.partition_mapping = {} 