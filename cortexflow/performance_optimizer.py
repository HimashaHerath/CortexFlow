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
import hashlib
import pickle
from datetime import datetime, timedelta
import os

# Import dependency utilities
from cortexflow.dependency_utils import import_optional_dependency

# Import graph libraries
nx_deps = import_optional_dependency(
    'networkx',
    warning_message="networkx not found. Graph optimization functionality will be limited."
)
NETWORKX_ENABLED = nx_deps['NETWORKX_ENABLED']
if NETWORKX_ENABLED:
    nx = nx_deps['module']

# Import community detection library
community_deps = import_optional_dependency(
    'community',
    import_name='community_louvain',
    warning_message="python-louvain not found. Community detection for graph partitioning will be limited."
)
COMMUNITY_DETECTION_ENABLED = community_deps['COMMUNITY_LOUVAIN_ENABLED']
if COMMUNITY_DETECTION_ENABLED:
    community_louvain = community_deps['module']

# Configure logging
logger = logging.getLogger('cortexflow.performance')

class ReasoningPattern:
    """
    Represents a reasoning pattern for caching purposes.
    """
    
    def __init__(self, pattern_key: str, hop_count: int = 0, entities: List[str] = None, path: List[str] = None):
        """
        Initialize a reasoning pattern.
        
        Args:
            pattern_key: Unique key for the pattern
            hop_count: Number of hops in the pattern
            entities: List of entity types involved
            path: List of path components
        """
        self.pattern_key = pattern_key
        self.hop_count = hop_count
        self.entities = entities or []
        self.path = path or []
        self.hit_count = 0
        self.last_accessed = datetime.now()
        self.created_at = datetime.now()
    
    def update_stats(self):
        """Update usage statistics."""
        self.hit_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_key": self.pattern_key,
            "hop_count": self.hop_count,
            "entities": self.entities,
            "path": self.path,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningPattern':
        """Create from dictionary."""
        pattern = cls(
            pattern_key=data["pattern_key"],
            hop_count=data["hop_count"],
            entities=data["entities"],
            path=data["path"]
        )
        pattern.hit_count = data["hit_count"]
        pattern.last_accessed = datetime.fromisoformat(data["last_accessed"])
        pattern.created_at = datetime.fromisoformat(data["created_at"])
        return pattern

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
        self.reasoning_patterns = {}
        
        # LRU cache for reasoning patterns
        self.max_cache_size = getattr(config, "max_reasoning_cache_size", 1000)
        self.cache_expiry = getattr(config, "reasoning_cache_expiry", 3600)  # in seconds
        
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
        
        # Load cached reasoning patterns if persistence is enabled
        self._load_cached_patterns()
        
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
    
    def _load_cached_patterns(self):
        """Load cached reasoning patterns from disk if available."""
        if not hasattr(self.config, "persist_reasoning_cache") or not self.config.persist_reasoning_cache:
            return
        
        cache_file = getattr(self.config, "reasoning_cache_file", "reasoning_patterns.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_data in patterns_data:
                    pattern = ReasoningPattern.from_dict(pattern_data)
                    self.reasoning_patterns[pattern.pattern_key] = pattern
                
                logger.info(f"Loaded {len(self.reasoning_patterns)} reasoning patterns from {cache_file}")
        except IOError as e:
            logger.error(f"IO error loading reasoning patterns: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in reasoning patterns file: {e}")
        except Exception as e:
            logger.error(f"Failed to load reasoning patterns: {e}")
    
    def _save_cached_patterns(self):
        """Save cached reasoning patterns to disk if enabled."""
        if not hasattr(self.config, "persist_reasoning_cache") or not self.config.persist_reasoning_cache:
            return
        
        cache_file = getattr(self.config, "reasoning_cache_file", "reasoning_patterns.json")
        
        try:
            patterns_data = [pattern.to_dict() for pattern in self.reasoning_patterns.values()]
            
            # Create the directory if it doesn't exist
            cache_dir = os.path.dirname(cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                
            with open(cache_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            logger.info(f"Saved {len(self.reasoning_patterns)} reasoning patterns to {cache_file}")
        except IOError as e:
            logger.error(f"IO error saving reasoning patterns: {e}")
        except Exception as e:
            logger.error(f"Failed to save reasoning patterns: {e}")
    
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
    
    def create_hop_indexes(self, max_hops: int = 2, index_frequent_paths: bool = True) -> Dict[str, Any]:
        """
        Create indexes for multi-hop queries.
        
        Args:
            max_hops: Maximum number of hops to index
            index_frequent_paths: Whether to index frequent paths based on query patterns
            
        Returns:
            Dictionary with indexing statistics
        """
        if not self.graph_store or not hasattr(self.graph_store, "graph"):
            logger.warning("No graph store available for indexing")
            return {"status": "failed", "reason": "No graph store available"}
        
        start_time = time.time()
        
        try:
            # Get the graph from graph store
            G = self.graph_store.graph
            
            if len(G.nodes()) == 0:
                logger.warning("Graph is empty, cannot create indexes")
                return {"status": "failed", "reason": "Empty graph"}
            
            # Initialize statistics
            stats = {
                "direct_indexes": 0,
                "two_hop_indexes": 0,
                "multi_hop_indexes": 0,
                "frequent_path_indexes": 0,
                "indexed_node_types": set(),
                "indexed_relation_types": set(),
                "execution_time": 0.0
            }
            
            # Create direct relationship indexes (1-hop)
            logger.info("Creating direct relationship indexes")
            direct_indexes = self._create_direct_indexes(G)
            stats["direct_indexes"] = direct_indexes
            
            # Create 2-hop indexes if max_hops >= 2
            if max_hops >= 2:
                logger.info("Creating 2-hop indexes")
                two_hop_indexes = self._create_two_hop_indexes(G)
                stats["two_hop_indexes"] = two_hop_indexes
            
            # Create multi-hop indexes if max_hops > 2
            if max_hops > 2:
                logger.info(f"Creating multi-hop indexes (up to {max_hops} hops)")
                multi_hop_indexes = self._create_multi_hop_indexes(G, max_hops)
                stats["multi_hop_indexes"] = multi_hop_indexes
            
            # Create indexes for frequent paths if requested
            if index_frequent_paths and self.query_patterns:
                logger.info("Creating indexes for frequent query patterns")
                frequent_path_indexes = self._create_frequent_path_indexes()
                stats["frequent_path_indexes"] = frequent_path_indexes
            
            # Update indexed types statistics
            for node in G.nodes():
                node_type = G.nodes[node].get("type")
                if node_type:
                    stats["indexed_node_types"].add(node_type)
            
            for _, _, edge_attrs in G.edges(data=True):
                edge_type = edge_attrs.get("type")
                if edge_type:
                    stats["indexed_relation_types"].add(edge_type)
            
            # Convert sets to lists for JSON serialization
            stats["indexed_node_types"] = list(stats["indexed_node_types"])
            stats["indexed_relation_types"] = list(stats["indexed_relation_types"])
            
            # Update optimization statistics
            self.stats["index_operations"] += 1
            end_time = time.time()
            execution_time = end_time - start_time
            stats["execution_time"] = execution_time
            self.stats["optimization_time"] += execution_time
            
            logger.info(f"Indexing completed in {execution_time:.2f} seconds")
            return {"status": "success", **stats}
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def _create_direct_indexes(self, G) -> int:
        """
        Create indexes for direct relationships (1-hop).
        
        Args:
            G: NetworkX graph
            
        Returns:
            Number of indexes created
        """
        if not hasattr(self.graph_store, "create_index"):
            return 0
        
        indexes_created = 0
        
        # Group edges by type
        edge_types = {}
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "unknown")
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((u, v))
        
        # Create indexes for each edge type
        for edge_type, edges in edge_types.items():
            try:
                self.graph_store.create_index(edge_type)
                indexes_created += 1
            except Exception as e:
                logger.warning(f"Failed to create index for edge type {edge_type}: {e}")
        
        # Create indexes for common node attributes
        node_attrs = set()
        for node, attrs in G.nodes(data=True):
            for attr in attrs:
                if attr not in ("id", "type"):
                    node_attrs.add(attr)
        
        for attr in node_attrs:
            try:
                self.graph_store.create_index(attr)
                indexes_created += 1
            except Exception as e:
                logger.warning(f"Failed to create index for node attribute {attr}: {e}")
        
        return indexes_created
    
    def _create_two_hop_indexes(self, G) -> int:
        """
        Create indexes for 2-hop paths.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Number of indexes created
        """
        if not hasattr(self.graph_store, "create_composite_index"):
            return 0
        
        indexes_created = 0
        
        # Find common 2-hop paths
        two_hop_paths = {}
        
        for node in G.nodes():
            # Get all neighbors (1-hop)
            neighbors = list(G.neighbors(node))
            
            # For each neighbor, get its neighbors (2-hop from original node)
            for neighbor in neighbors:
                # Get edge type from node to neighbor
                edge1_type = G.get_edge_data(node, neighbor).get("type", "unknown")
                
                # Get all 2-hop neighbors
                two_hop_neighbors = list(G.neighbors(neighbor))
                
                for two_hop_neighbor in two_hop_neighbors:
                    if two_hop_neighbor == node:  # Skip cycles
                        continue
                    
                    # Get edge type from neighbor to 2-hop neighbor
                    edge2_type = G.get_edge_data(neighbor, two_hop_neighbor).get("type", "unknown")
                    
                    # Create path signature
                    path_key = f"{edge1_type}|{edge2_type}"
                    
                    if path_key not in two_hop_paths:
                        two_hop_paths[path_key] = 0
                    
                    two_hop_paths[path_key] += 1
        
        # Create indexes for common 2-hop paths
        for path_key, count in two_hop_paths.items():
            # Only index paths that appear frequently
            if count < 10:
                continue
            
            edge_types = path_key.split("|")
            try:
                self.graph_store.create_composite_index(edge_types)
                indexes_created += 1
            except Exception as e:
                logger.warning(f"Failed to create 2-hop index for path {path_key}: {e}")
        
        return indexes_created
    
    def _create_multi_hop_indexes(self, G, max_hops: int) -> int:
        """
        Create indexes for multi-hop paths (3+ hops).
        
        Args:
            G: NetworkX graph
            max_hops: Maximum number of hops to index
            
        Returns:
            Number of indexes created
        """
        if not hasattr(self.graph_store, "create_path_index"):
            return 0
        
        indexes_created = 0
        
        # Sample nodes to analyze paths
        sample_size = min(100, len(G.nodes()))
        sample_nodes = list(G.nodes())[:sample_size]
        
        # Find common multi-hop paths
        path_counts = Counter()
        
        for source in sample_nodes:
            # Use BFS to find paths up to max_hops
            visited = {source}
            queue = [(source, [])]
            
            while queue:
                node, path = queue.pop(0)
                
                # Skip if path is already at max length
                if len(path) // 2 >= max_hops:
                    continue
                
                # Get neighbors
                for neighbor in G.neighbors(node):
                    if neighbor in visited:
                        continue
                    
                    # Get edge type
                    edge_type = G.get_edge_data(node, neighbor).get("type", "unknown")
                    
                    # Create new path
                    new_path = path + [edge_type, neighbor]
                    
                    # Record path signature if it's at least 3 hops
                    if len(new_path) // 2 >= 3:
                        # Extract only edge types for signature
                        edge_types = [new_path[i] for i in range(0, len(new_path), 2)]
                        path_key = "|".join(edge_types)
                        path_counts[path_key] += 1
                    
                    # Add to queue for further exploration
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        # Create indexes for common multi-hop paths
        for path_key, count in path_counts.most_common(20):  # Limit to top 20 paths
            edge_types = path_key.split("|")
            try:
                self.graph_store.create_path_index(edge_types)
                indexes_created += 1
            except Exception as e:
                logger.warning(f"Failed to create multi-hop index for path {path_key}: {e}")
        
        return indexes_created
    
    def _create_frequent_path_indexes(self) -> int:
        """
        Create indexes for frequent query patterns.
        
        Returns:
            Number of indexes created
        """
        if not hasattr(self.graph_store, "create_path_index"):
            return 0
        
        indexes_created = 0
        
        # Extract patterns from query history
        common_patterns = self.query_patterns.most_common(10)
        
        for pattern_key, count in common_patterns:
            # Skip patterns with low frequency
            if count < 5:
                continue
            
            # Try to find pattern in reasoning patterns
            path = []
            if pattern_key in self.reasoning_patterns:
                pattern = self.reasoning_patterns[pattern_key]
                path = pattern.path
            
            # Skip if no path found
            if not path:
                continue
            
            # Extract edge types from path
            edge_types = []
            for i in range(1, len(path), 2):
                if i < len(path):
                    edge_types.append(path[i])
            
            # Skip if no edge types found
            if not edge_types:
                continue
            
            # Create index for this path
            try:
                self.graph_store.create_path_index(edge_types)
                indexes_created += 1
            except Exception as e:
                logger.warning(f"Failed to create index for frequent path {edge_types}: {e}")
        
        return indexes_created
    
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
    
    def generate_pattern_key(self, pattern: Dict[str, Any]) -> str:
        """
        Generate a unique key for a reasoning pattern.
        
        Args:
            pattern: Reasoning pattern data
            
        Returns:
            String key for the pattern
        """
        # Extract key components
        components = []
        
        # Add entity types
        if "entity_types" in pattern:
            components.append("entities:" + ",".join(sorted(pattern["entity_types"])))
        
        # Add relation types
        if "relation_types" in pattern:
            components.append("relations:" + ",".join(sorted(pattern["relation_types"])))
        
        # Add path structure
        if "path" in pattern:
            components.append("path:" + ",".join(pattern["path"]))
        
        # Add hop count
        if "hop_count" in pattern:
            components.append(f"hops:{pattern['hop_count']}")
        
        # Join all components and create a hash
        key_string = "|".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def extract_pattern_from_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a reasoning pattern from a query.
        
        Args:
            query: Query to analyze
            
        Returns:
            Extracted pattern data
        """
        pattern = {}
        
        # Extract entity types
        if "entities" in query:
            entity_types = set()
            for entity in query["entities"]:
                if "type" in entity:
                    entity_types.add(entity["type"])
            
            if entity_types:
                pattern["entity_types"] = list(entity_types)
        
        # Extract relation types
        if "relations" in query:
            relation_types = set()
            for relation in query["relations"]:
                if "type" in relation:
                    relation_types.add(relation["type"])
            
            if relation_types:
                pattern["relation_types"] = list(relation_types)
        
        # Extract path structure
        if "path" in query:
            pattern["path"] = query["path"]
        
        # Calculate hop count
        if "path" in query:
            pattern["hop_count"] = max(0, len(query["path"]) // 2)
        
        return pattern
    
    def cache_reasoning_pattern(self, pattern_data: Dict[str, Any], result: Any) -> str:
        """
        Cache a reasoning pattern and its result.
        
        Args:
            pattern_data: Pattern data
            result: Query result
            
        Returns:
            Pattern key
        """
        # Generate pattern key
        pattern_key = self.generate_pattern_key(pattern_data)
        
        # Create or update pattern
        if pattern_key in self.reasoning_patterns:
            pattern = self.reasoning_patterns[pattern_key]
            pattern.update_stats()
        else:
            pattern = ReasoningPattern(
                pattern_key=pattern_key,
                hop_count=pattern_data.get("hop_count", 0),
                entities=pattern_data.get("entity_types", []),
                path=pattern_data.get("path", [])
            )
            self.reasoning_patterns[pattern_key] = pattern
        
        # Cache result
        self.reasoning_cache[pattern_key] = result
        
        # Prune cache if needed
        self._prune_reasoning_cache()
        
        # Update stats
        self.stats["optimization_time"] += 0.001  # Negligible time
        
        # Save cached patterns if persistence is enabled
        if (pattern.hit_count % 10 == 0 and  # Save every 10 hits
            hasattr(self.config, "persist_reasoning_cache") and 
            self.config.persist_reasoning_cache):
            self._save_cached_patterns()
        
        return pattern_key
    
    def get_cached_reasoning(self, pattern_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Get cached reasoning result for a pattern.
        
        Args:
            pattern_data: Pattern data
            
        Returns:
            Tuple of (cache_hit, result)
        """
        # Generate pattern key
        pattern_key = self.generate_pattern_key(pattern_data)
        
        # Check if pattern exists in cache
        if pattern_key in self.reasoning_cache:
            # Update pattern stats
            if pattern_key in self.reasoning_patterns:
                self.reasoning_patterns[pattern_key].update_stats()
            
            # Update hit stats
            self.stats["reasoning_cache_hits"] += 1
            
            return True, self.reasoning_cache[pattern_key]
        
        # Cache miss
        self.stats["reasoning_cache_misses"] += 1
        return False, None
    
    def _prune_reasoning_cache(self):
        """Prune the reasoning cache if it exceeds the maximum size."""
        if len(self.reasoning_cache) <= self.max_cache_size:
            return
        
        # First, remove expired entries
        current_time = datetime.now()
        expired_keys = []
        
        for pattern_key, pattern in self.reasoning_patterns.items():
            if current_time - pattern.last_accessed > timedelta(seconds=self.cache_expiry):
                expired_keys.append(pattern_key)
        
        for key in expired_keys:
            if key in self.reasoning_cache:
                del self.reasoning_cache[key]
            if key in self.reasoning_patterns:
                del self.reasoning_patterns[key]
        
        # If still too large, remove least recently used entries
        if len(self.reasoning_cache) > self.max_cache_size:
            # Sort patterns by last access time
            sorted_patterns = sorted(
                self.reasoning_patterns.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest entries
            to_remove = len(self.reasoning_cache) - self.max_cache_size
            for i in range(to_remove):
                if i < len(sorted_patterns):
                    key = sorted_patterns[i][0]
                    if key in self.reasoning_cache:
                        del self.reasoning_cache[key]
                    if key in self.reasoning_patterns:
                        del self.reasoning_patterns[key]
    
    def optimize_query_execution(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a query for execution.
        
        Args:
            query: Query to optimize
            
        Returns:
            Optimized query with execution plan
        """
        start_time = time.time()
        
        # Check if the query is in the cache
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            self.stats["query_cache_hits"] += 1
            return self.query_cache[cache_key]
        
        self.stats["query_cache_misses"] += 1
        
        # Update query pattern statistics
        pattern = self.extract_pattern_from_query(query)
        pattern_key = self.generate_pattern_key(pattern)
        self.query_patterns[pattern_key] += 1
        
        # Generate query plan
        plan = self.generate_query_plan(query)
        
        # Update optimized query
        optimized_query = query.copy()
        optimized_query["execution_plan"] = plan
        
        # Cache the optimized query
        self.query_cache[cache_key] = optimized_query
        
        # Update stats
        end_time = time.time()
        self.stats["optimization_time"] += (end_time - start_time)
        self.stats["query_plans_generated"] += 1
        
        return optimized_query
    
    def _generate_cache_key(self, query: Dict[str, Any]) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Query to generate key for
            
        Returns:
            Cache key string
        """
        # Normalize query by sorting keys
        normalized = json.dumps(query, sort_keys=True)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def clear_caches(self) -> Dict[str, int]:
        """
        Clear all caches.
        
        Returns:
            Dictionary with number of entries cleared
        """
        query_cache_size = len(self.query_cache)
        reasoning_cache_size = len(self.reasoning_cache)
        
        self.query_cache = {}
        self.reasoning_cache = {}
        self.reasoning_patterns = {}
        
        logger.info("All caches cleared")
        
        return {
            "query_cache_cleared": query_cache_size,
            "reasoning_cache_cleared": reasoning_cache_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance optimization statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate cache hit rates
        query_hit_rate = self._calculate_hit_rate(
            self.stats["query_cache_hits"],
            self.stats["query_cache_misses"]
        )
        
        reasoning_hit_rate = self._calculate_hit_rate(
            self.stats["reasoning_cache_hits"],
            self.stats["reasoning_cache_misses"]
        )
        
        # Get most common query patterns
        common_patterns = [
            {"pattern": pattern, "count": count}
            for pattern, count in self.query_patterns.most_common(10)
        ]
        
        # Get general stats
        stats = self.stats.copy()
        stats.update({
            "query_cache_size": len(self.query_cache),
            "reasoning_cache_size": len(self.reasoning_cache),
            "reasoning_patterns": len(self.reasoning_patterns),
            "query_hit_rate": query_hit_rate,
            "reasoning_hit_rate": reasoning_hit_rate,
            "common_patterns": common_patterns
        })
        
        return stats
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """
        Calculate cache hit rate.
        
        Args:
            hits: Number of cache hits
            misses: Number of cache misses
            
        Returns:
            Hit rate as a percentage
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100.0
    
    def prune_partitions(self, density_threshold: float = 0.1) -> int:
        """
        Prune graph partitions to optimize memory usage.
        
        Args:
            density_threshold: Minimum density threshold for keeping partitions
            
        Returns:
            Number of partitions pruned
        """
        if not NETWORKX_ENABLED or not self.partitions:
            return 0
        
        partitions_before = len(self.partitions)
        
        # Identify partitions to prune
        to_prune = []
        for partition_id, subgraph in self.partitions.items():
            if len(subgraph.nodes()) <= 1:
                to_prune.append(partition_id)
                continue
            
            density = nx.density(subgraph)
            if density < density_threshold:
                to_prune.append(partition_id)
        
        # Prune identified partitions
        for partition_id in to_prune:
            del self.partitions[partition_id]
        
        # Update partition mapping
        if to_prune:
            self.partition_mapping = {
                node: partition_id
                for node, partition_id in self.partition_mapping.items()
                if partition_id not in to_prune
            }
        
        partitions_after = len(self.partitions)
        pruned = partitions_before - partitions_after
        
        logger.info(f"Pruned {pruned} partitions with density below {density_threshold}")
        
        return pruned
    
    def close(self):
        """Clean up resources and save state."""
        try:
            # Save cached patterns if enabled
            if hasattr(self.config, "persist_reasoning_cache") and self.config.persist_reasoning_cache:
                self._save_cached_patterns()
            
            # Clear caches to free memory
            self.query_cache = {}
            self.reasoning_cache = {}
            
            # Clear partitions and indexes
            self.partitions = {}
            self.partition_mapping = {}
            self.direct_indexes = {}
            self.two_hop_indexes = {}
            self.multi_hop_indexes = {}
            
            # Close any database connections
            if hasattr(self, 'conn') and self.conn is not None:
                self.conn.close()
                self.conn = None
            
            logger.info("Performance optimizer resources cleaned up")
        except Exception as e:
            logger.error(f"Error during performance optimizer cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error in performance optimizer __del__: {e}") 