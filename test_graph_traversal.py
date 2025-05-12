#!/usr/bin/env python3
"""
Test script for the advanced graph traversal features.
"""

import os
import logging
import sqlite3
import networkx as nx
import time

from cortexflow.graph_store import GraphStore
from cortexflow.config import CortexFlowConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

def format_path(path):
    """Format a path for pretty printing"""
    if not path:
        return "No path found"
    
    path_str = ""
    for i, node in enumerate(path):
        entity = node.get("entity", "Unknown")
        path_str += entity
        
        if i < len(path) - 1:
            relation = node.get("next_relation", {}).get("type", "related to")
            path_str += f" → [{relation}] → "
    
    return path_str

def main():
    # Create a test database path
    db_path = "test_graph_traversal.db"
    
    # Clean up any existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Set up a minimal configuration
    config = CortexFlowConfig(knowledge_store_path=db_path)
    
    # Initialize the graph store
    graph_store = GraphStore(config)
    
    print("=== POPULATING GRAPH ===")
    
    # Add some entities and relations for testing
    graph_store.add_entity("Albert Einstein", entity_type="Person", confidence=0.9)
    graph_store.add_entity("Theory of Relativity", entity_type="Theory", confidence=0.95)
    graph_store.add_entity("Nobel Prize in Physics", entity_type="Award", confidence=0.95)
    graph_store.add_entity("Princeton University", entity_type="Organization", confidence=0.9)
    graph_store.add_entity("Marie Curie", entity_type="Person", confidence=0.9)
    graph_store.add_entity("Radium", entity_type="Element", confidence=0.95)
    graph_store.add_entity("University of Paris", entity_type="Organization", confidence=0.9)
    
    # Add relations with weights and confidence scores
    graph_store.add_relation(
        source_entity="Albert Einstein",
        relation_type="developed",
        target_entity="Theory of Relativity",
        weight=0.9,
        confidence=0.95
    )
    
    graph_store.add_relation(
        source_entity="Albert Einstein",
        relation_type="received",
        target_entity="Nobel Prize in Physics",
        weight=0.8,
        confidence=0.9
    )
    
    graph_store.add_relation(
        source_entity="Albert Einstein",
        relation_type="worked_at",
        target_entity="Princeton University",
        weight=0.7,
        confidence=0.85
    )
    
    graph_store.add_relation(
        source_entity="Marie Curie",
        relation_type="discovered",
        target_entity="Radium",
        weight=0.9,
        confidence=0.95
    )
    
    graph_store.add_relation(
        source_entity="Marie Curie",
        relation_type="received",
        target_entity="Nobel Prize in Physics",
        weight=0.8,
        confidence=0.9
    )
    
    graph_store.add_relation(
        source_entity="Marie Curie",
        relation_type="worked_at",
        target_entity="University of Paris",
        weight=0.7,
        confidence=0.85
    )
    
    # Allow the graph to be loaded
    time.sleep(1)
    
    print("\n=== BASIC PATH QUERY ===")
    paths = graph_store.path_query(
        start_entity="Albert Einstein",
        end_entity="Nobel Prize in Physics",
        max_hops=2
    )
    
    print(f"Found {len(paths)} paths from Einstein to Nobel Prize:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    print("\n=== WEIGHTED PATH QUERY ===")
    paths = graph_store.weighted_path_query(
        start_entity="Albert Einstein",
        end_entity="Nobel Prize in Physics",
        importance_weight=0.7,
        confidence_weight=0.3
    )
    
    print(f"Found {len(paths)} weighted paths from Einstein to Nobel Prize:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    print("\n=== BIDIRECTIONAL SEARCH ===")
    paths = graph_store.bidirectional_search(
        start_entity="Marie Curie",
        end_entity="Albert Einstein",
        max_hops=3
    )
    
    print(f"Found {len(paths)} paths between Curie and Einstein:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    print("\n=== CONSTRAINED PATH SEARCH ===")
    paths = graph_store.constrained_path_search(
        start_entity="Albert Einstein",
        end_entity="Nobel Prize in Physics",
        allowed_relations=["developed", "received"],
        max_hops=2
    )
    
    print(f"Found {len(paths)} constrained paths from Einstein to Nobel Prize:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    # Clean up
    graph_store.close()
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main() 