#!/usr/bin/env python3
"""
CortexFlow Performance Optimization Demo

This script demonstrates the performance optimization features of CortexFlow,
including graph partitioning, multi-hop indexing, query planning, and caching.
"""

import time
import json
import random
from pprint import pprint

from cortexflow import CortexFlowManager
from cortexflow.config import CortexFlowConfig

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def main():
    """Main function demonstrating performance optimization features."""
    print_section("CortexFlow Performance Optimization Demo")
    
    # Initialize with performance optimization enabled
    config = CortexFlowConfig(
        # Basic settings
        knowledge_store_path=":memory:",  # Use in-memory DB for this demo
        
        # Enable Graph RAG features
        use_graph_rag=True,
        enable_multi_hop_queries=True,
        max_graph_hops=4,
        
        # Enable performance optimization
        use_performance_optimization=True,
        use_graph_partitioning=True,
        graph_partition_method="louvain",
        use_multihop_indexing=True,
        max_indexed_hops=2,
        use_reasoning_cache=True,
        
        # Enable verbose logging
        verbose_logging=True
    )
    
    print("Initializing CortexFlow with performance optimization enabled...\n")
    manager = CortexFlowManager(config)
    
    # Create a sample knowledge graph
    print_section("Creating Sample Knowledge Graph")
    create_sample_graph(manager)
    
    # Demonstrate graph partitioning
    print_section("Graph Partitioning")
    print("Partitioning the graph to improve query performance...")
    result, execution_time = measure_time(
        manager.partition_graph,
        method="louvain"
    )
    print(f"Graph partitioning completed in {execution_time:.4f} seconds")
    print(f"Created {result.get('partitions', 0)} partitions")
    print("\nPartition statistics:")
    pprint(result.get('partition_stats', {}))
    
    # Demonstrate multi-hop indexing
    print_section("Multi-hop Indexing")
    print("Creating multi-hop indexes to speed up path queries...")
    result, execution_time = measure_time(
        manager.create_hop_indexes,
        max_hops=2
    )
    print(f"Index creation completed in {execution_time:.4f} seconds")
    print(f"Created {result.get('indexes_created', 0)} indexes")
    
    # Demonstrate query planning
    print_section("Query Planning")
    print("Generating an optimized query plan for a path query...\n")
    
    query = {
        "type": "path",
        "start_entity": "Albert Einstein",
        "end_entity": "Physics",
        "max_hops": 2,
        "relation_constraints": ["field_of_work", "known_for"]
    }
    
    print("Query parameters:")
    pprint(query)
    print()
    
    plan, execution_time = measure_time(manager.optimize_query, query)
    print(f"Query plan generated in {execution_time:.6f} seconds")
    print("\nOptimized Query Plan:")
    pprint(plan)
    
    # Run the same query multiple times to demonstrate caching
    print_section("Query Caching")
    print("Running the same query multiple times to demonstrate caching benefits...\n")
    
    # First query (should be a cache miss)
    result1, time1 = measure_time(
        manager.optimize_path_query,
        start_entity="Albert Einstein",
        end_entity="Physics",
        max_hops=2,
        relation_constraints=["field_of_work", "known_for"]
    )
    
    # Second query (should be a cache hit)
    result2, time2 = measure_time(
        manager.optimize_path_query,
        start_entity="Albert Einstein",
        end_entity="Physics",
        max_hops=2,
        relation_constraints=["field_of_work", "known_for"]
    )
    
    print(f"First query execution time: {time1:.6f} seconds")
    print(f"Second query execution time: {time2:.6f} seconds")
    print(f"Speed improvement: {time1 / time2:.2f}x faster\n")
    
    # Check cache statistics
    cache_stats = manager.get_cache_stats()
    print("Cache Statistics:")
    pprint(cache_stats)
    
    # Show overall performance statistics
    print_section("Performance Statistics")
    print("Getting overall performance statistics...\n")
    stats = manager.get_performance_stats()
    pprint(stats)
    
    # Clean up
    manager.close()
    print("\nDemo completed successfully!")

def create_sample_graph(manager):
    """Create a sample knowledge graph for the demo."""
    # Scientists
    scientists = [
        "Albert Einstein",
        "Isaac Newton",
        "Marie Curie",
        "Niels Bohr",
        "Richard Feynman",
        "Stephen Hawking",
        "Charles Darwin",
        "Nikola Tesla",
        "Rosalind Franklin",
        "Alan Turing"
    ]
    
    # Fields
    fields = [
        "Physics",
        "Chemistry",
        "Biology",
        "Mathematics",
        "Computer Science",
        "Astronomy",
        "Genetics",
        "Quantum Mechanics",
        "Relativity",
        "Evolution"
    ]
    
    # Institutions
    institutions = [
        "Princeton University",
        "University of Cambridge",
        "University of Paris",
        "University of Copenhagen",
        "Caltech",
        "University of Oxford",
        "University of Chicago",
        "ETH Zurich",
        "MIT",
        "Imperial College London"
    ]
    
    # Awards
    awards = [
        "Nobel Prize",
        "Fields Medal",
        "Copley Medal",
        "Turing Award",
        "Wolf Prize",
        "Breakthrough Prize",
        "Crafoord Prize",
        "Max Planck Medal",
        "Royal Medal",
        "Lorentz Medal"
    ]
    
    # Add entities
    print("Adding entities...")
    for entity in scientists + fields + institutions + awards:
        entity_type = None
        if entity in scientists:
            entity_type = "Scientist"
        elif entity in fields:
            entity_type = "Field"
        elif entity in institutions:
            entity_type = "Institution"
        elif entity in awards:
            entity_type = "Award"
        
        manager.knowledge_store.add_entity(entity, entity_type)
    
    # Add relationships
    print("Adding relationships...")
    
    # Scientists to Fields (field_of_work)
    for scientist in scientists:
        # Each scientist works in 1-3 fields
        scientist_fields = random.sample(fields, random.randint(1, 3))
        for field in scientist_fields:
            manager.knowledge_store.add_relationship(
                source=scientist,
                relation="field_of_work",
                target=field,
                metadata={"confidence": random.uniform(0.7, 0.99)}
            )
    
    # Scientists to Institutions (affiliated_with)
    for scientist in scientists:
        # Each scientist is affiliated with 1-2 institutions
        scientist_institutions = random.sample(institutions, random.randint(1, 2))
        for institution in scientist_institutions:
            manager.knowledge_store.add_relationship(
                source=scientist,
                relation="affiliated_with",
                target=institution,
                metadata={"confidence": random.uniform(0.8, 0.99)}
            )
    
    # Scientists to Awards (received)
    for scientist in scientists:
        # Some scientists receive awards
        if random.random() > 0.3:  # 70% chance of receiving an award
            scientist_awards = random.sample(awards, random.randint(0, 2))
            for award in scientist_awards:
                manager.knowledge_store.add_relationship(
                    source=scientist,
                    relation="received",
                    target=award,
                    metadata={"confidence": random.uniform(0.9, 0.99)}
                )
    
    # Scientists to Scientists (collaborated_with)
    for i, scientist1 in enumerate(scientists):
        for scientist2 in scientists[i+1:]:
            # Some scientists collaborate
            if random.random() > 0.7:  # 30% chance of collaboration
                manager.knowledge_store.add_relationship(
                    source=scientist1,
                    relation="collaborated_with",
                    target=scientist2,
                    metadata={"confidence": random.uniform(0.7, 0.95)}
                )
    
    # Scientists known for specific contributions
    contributions = [
        ("Albert Einstein", "Relativity"),
        ("Isaac Newton", "Physics"),
        ("Marie Curie", "Chemistry"),
        ("Niels Bohr", "Quantum Mechanics"),
        ("Richard Feynman", "Quantum Mechanics"),
        ("Stephen Hawking", "Physics"),
        ("Charles Darwin", "Evolution"),
        ("Nikola Tesla", "Physics"),
        ("Rosalind Franklin", "Genetics"),
        ("Alan Turing", "Computer Science")
    ]
    
    for scientist, field in contributions:
        manager.knowledge_store.add_relationship(
            source=scientist,
            relation="known_for",
            target=field,
            metadata={"confidence": random.uniform(0.9, 0.99)}
        )
    
    # Fields related to other fields
    field_relations = [
        ("Physics", "Quantum Mechanics"),
        ("Physics", "Relativity"),
        ("Physics", "Astronomy"),
        ("Chemistry", "Biology"),
        ("Biology", "Genetics"),
        ("Biology", "Evolution"),
        ("Mathematics", "Physics"),
        ("Mathematics", "Computer Science"),
        ("Computer Science", "Mathematics")
    ]
    
    for field1, field2 in field_relations:
        manager.knowledge_store.add_relationship(
            source=field1,
            relation="related_to",
            target=field2,
            metadata={"confidence": random.uniform(0.7, 0.9)}
        )
    
    # Count entities and relationships
    print("Sample knowledge graph created successfully!")
    
    # Get graph stats 
    entity_count = len(scientists) + len(fields) + len(institutions) + len(awards)
    relationship_count = (
        # Scientists to Fields
        sum(random.randint(1, 3) for _ in scientists) +
        # Scientists to Institutions
        sum(random.randint(1, 2) for _ in scientists) +
        # Scientists to Awards (approximation)
        len(scientists) * 0.7 * 1.5 +
        # Scientists to Scientists (approximation)
        len(scientists) * (len(scientists) - 1) / 2 * 0.3 +
        # Known for relationships
        len(contributions) +
        # Field relationships
        len(field_relations)
    )
    
    print(f"  - Entities: {entity_count}")
    print(f"  - Relationships: ~{int(relationship_count)}")

if __name__ == "__main__":
    main() 