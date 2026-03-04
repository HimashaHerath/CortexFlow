"""
Reasoning / performance facade for CortexFlow.

Extracts performance-optimizer and reasoning-cache operations from
CortexFlowManager into a focused class.  CortexFlowManager delegates to
an instance of ReasoningFacade for query optimization, graph
partitioning, hop indexing, and caching.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('cortexflow')


class ReasoningFacade:
    """Wraps the PerformanceOptimizer with the same public API that
    CortexFlowManager previously exposed directly."""

    def __init__(self, config, performance_optimizer):
        """
        Args:
            config: CortexFlowConfig instance.
            performance_optimizer: PerformanceOptimizer or None.
        """
        self.config = config
        self.performance_optimizer = performance_optimizer

    # ------------------------------------------------------------------
    # Query optimization
    # ------------------------------------------------------------------

    def optimize_query(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Generate an optimized query plan for knowledge graph operations.

        Args:
            query: Dictionary with query parameters

        Returns:
            Optimized query plan
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}

        try:
            return self.performance_optimizer.generate_query_plan(query)
        except Exception as e:
            logger.error(f"Error generating optimized query plan: {e}")
            return {"status": "error", "message": str(e)}

    def optimize_path_query(self, start_entity: str, end_entity: str,
                       max_hops: int = 3,
                       relation_constraints: list[str] = None) -> dict[str, Any]:
        """
        Optimize a path query between entities using the query planning system.

        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length
            relation_constraints: Optional list of allowed relation types

        Returns:
            Optimized query plan
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}

        try:
            query = {
                "type": "path",
                "start_entity": start_entity,
                "end_entity": end_entity,
                "max_hops": max_hops,
                "relation_constraints": relation_constraints or []
            }

            return self.performance_optimizer.optimize_query_execution(query)
        except Exception as e:
            logger.error(f"Error optimizing path query: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Graph partitioning / indexing
    # ------------------------------------------------------------------

    def partition_graph(self, method: str = None, partition_count: int = None) -> dict[str, Any]:
        """
        Partition the knowledge graph for improved performance.

        Args:
            method: Partitioning method (louvain, spectral, modularity)
            partition_count: Target number of partitions

        Returns:
            Partition statistics
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}

        try:
            # Use default method from config if not specified
            if method is None:
                method = self.config.graph_partition_method

            # Use default count from config if not specified
            if partition_count is None:
                partition_count = self.config.target_partition_count

            return self.performance_optimizer.partition_graph(method, partition_count)
        except Exception as e:
            logger.error(f"Error partitioning graph: {e}")
            return {"status": "error", "message": str(e)}

    def create_hop_indexes(self, max_hops: int = None) -> dict[str, Any]:
        """
        Create indexes for multi-hop queries to speed up traversal.

        Args:
            max_hops: Maximum number of hops to index

        Returns:
            Indexing statistics
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}

        try:
            # Use default from config if not specified
            if max_hops is None:
                max_hops = self.config.max_indexed_hops

            return self.performance_optimizer.create_hop_indexes(max_hops)
        except Exception as e:
            logger.error(f"Error creating hop indexes: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Performance stats / caching
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics from the optimizer.

        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}

        try:
            return self.performance_optimizer.get_stats()
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"status": "error", "message": str(e)}

    def clear_performance_caches(self) -> dict[str, Any]:
        """
        Clear all performance optimization caches.

        Returns:
            Dictionary with cache clearing statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}

        try:
            return self.performance_optimizer.clear_caches()
        except Exception as e:
            logger.error(f"Error clearing performance caches: {e}")
            return {"status": "error", "message": str(e)}

    def cache_reasoning_pattern(self, pattern_key: str, pattern_result: Any) -> bool:
        """
        Cache a common reasoning pattern for reuse.

        Args:
            pattern_key: Unique identifier for the reasoning pattern
            pattern_result: Result of the reasoning pattern

        Returns:
            True if successful, False otherwise
        """
        if not self.performance_optimizer:
            return False

        try:
            self.performance_optimizer.cache_reasoning_pattern(pattern_key, pattern_result)
            return True
        except Exception as e:
            logger.error(f"Error caching reasoning pattern: {e}")
            return False

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics including hit rates.

        Returns:
            Dictionary with cache statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}

        try:
            stats = self.performance_optimizer.get_stats()
            return stats.get("caching", {})
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "message": str(e)}
