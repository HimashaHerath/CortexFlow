# Performance Optimization in CortexFlow

This document explains the performance optimization features in CortexFlow that improve the efficiency of knowledge graph operations, particularly for large-scale graphs.

## Overview

CortexFlow's performance optimization module provides several mechanisms to enhance the performance and scalability of knowledge graph operations:

1. **Graph Partitioning**: Divides large graphs into more manageable subgraphs to improve query performance
2. **Multi-hop Indexing**: Creates indexes for multi-hop paths to speed up complex traversal operations
3. **Query Planning**: Optimizes reasoning paths by selecting the most efficient algorithms and strategies
4. **Result Caching**: Stores common reasoning patterns to avoid redundant computation

These features are particularly valuable when working with large knowledge graphs or when performing complex reasoning operations.

## Configuration

To enable performance optimization features, update your CortexFlow configuration:

```python
from cortexflow.config import CortexFlowConfig

config = CortexFlowConfig(
    # Enable performance optimization features
    use_performance_optimization=True,
    
    # Graph partitioning settings
    use_graph_partitioning=True,
    graph_partition_method="louvain",  # Options: louvain, spectral, modularity
    target_partition_count=5,  # Target number of partitions (if applicable)
    
    # Multi-hop indexing settings
    use_multihop_indexing=True,
    max_indexed_hops=2,  # Maximum number of hops to index
    
    # Query planning and caching settings
    use_query_planning=True,
    use_reasoning_cache=True,
    reasoning_cache_max_size=1000,
    query_cache_max_size=500,
    cache_ttl=3600  # Cache time-to-live in seconds (0 for no expiration)
)
```

## Features in Detail

### Graph Partitioning

Graph partitioning divides a large graph into smaller, more manageable subgraphs based on the connectivity patterns. This improves performance by allowing operations to focus only on relevant subgraphs.

CortexFlow offers multiple partitioning algorithms:

- **Louvain Method** (default): Community detection based on modularity optimization
- **Spectral Clustering**: Uses the graph's eigenvalues to identify natural divisions
- **Modularity-based**: Finds partitions that maximize the density of connections within partitions

Example usage:

```python
from cortexflow import CortexFlowManager

manager = CortexFlowManager(config)

# Partition the knowledge graph
result = manager.partition_graph(
    method="louvain",  # Optional, defaults to config setting
    partition_count=5  # Optional, defaults to config setting
)

print(f"Created {result['partitions']} partitions")
```

### Multi-hop Indexing

Multi-hop indexing creates database indexes that store pre-computed paths between entities, significantly speeding up multi-hop queries. This is especially useful for complex reasoning operations that require path traversal.

Example usage:

```python
# Create indexes for paths up to 2 hops
result = manager.create_hop_indexes(max_hops=2)

print(f"Created {result['indexes_created']} indexes")
```

### Query Planning

The query planning system analyzes queries and dynamically generates execution plans that optimize performance. It considers factors such as:

- Available indexes and partitions
- Graph connectivity patterns
- Query complexity and constraints
- Historical query patterns

Example usage:

```python
# Directly generate a query plan
query = {
    "type": "path",
    "start_entity": "Albert Einstein",
    "end_entity": "Nobel Prize",
    "max_hops": 3,
    "relation_constraints": ["awarded", "known_for"]
}
plan = manager.optimize_query(query)

# Or use the convenience method for path queries
plan = manager.optimize_path_query(
    start_entity="Albert Einstein",
    end_entity="Nobel Prize",
    max_hops=3,
    relation_constraints=["awarded", "known_for"]
)

print(f"Query plan using {plan['strategy']} strategy with estimated cost {plan['estimated_cost']}")
```

### Reasoning Pattern Caching

CortexFlow caches the results of common reasoning patterns to avoid redundant computation. This is particularly useful for frequently repeated query patterns.

Example usage:

```python
# Cache a reasoning pattern result
manager.cache_reasoning_pattern(
    pattern_key="relatedness|Albert Einstein|Physics",
    pattern_result={"relatedness_score": 0.95, "common_properties": ["field", "expertise"]}
)

# View cache statistics
stats = manager.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}%")

# Clear caches if needed
manager.clear_performance_caches()
```

## Performance Statistics

You can monitor the performance of the optimization features:

```python
# Get all performance statistics
stats = manager.get_performance_stats()

# Or get system-wide statistics including performance metrics
system_stats = manager.get_stats()
performance_stats = system_stats.get("performance", {})

print(f"Partitioning: {performance_stats.get('partitioning', {})}")
print(f"Caching: {performance_stats.get('caching', {})}")
```

## Requirements

The performance optimization features have the following dependencies:

- **NetworkX**: Required for graph algorithms and partitioning
- **python-louvain**: Optional, enables Louvain community detection for partitioning
- **scikit-learn**: Optional, enables spectral clustering for partitioning

You can install these dependencies using pip:

```bash
pip install networkx python-louvain scikit-learn
```

## Best Practices

1. **Start with default settings**: The default configuration works well for many use cases. Adjust settings based on performance monitoring.

2. **Monitor cache hit rates**: Low hit rates might indicate that your queries are too diverse for caching to be effective.

3. **Choose the right partitioning method**:
   - **Louvain**: Best for graphs with natural community structure
   - **Spectral**: Good for graphs with clear clusters
   - **Modularity**: Good for general-purpose partitioning

4. **Adjust indexed hop count based on query patterns**: If most of your queries span 3 hops, set `max_indexed_hops=3`.

5. **Consider memory usage**: Indexing too many hops or caching too many patterns can consume significant memory.

## Limitations

- Graph partitioning requires the full graph to be loaded in memory during partitioning
- Multi-hop indexing can require significant storage for large graphs
- Some optimization features require additional libraries (NetworkX, python-louvain, scikit-learn)
- Performance benefits vary based on graph structure and query patterns 