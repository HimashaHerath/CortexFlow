#!/usr/bin/env python3
"""
CortexFlow Advanced Graph Traversal Example

This example demonstrates how to use the advanced graph traversal features
in CortexFlow to efficiently query and navigate knowledge graphs.
"""

import logging
import os
import time
from pprint import pprint

from cortexflow import CortexFlowManager, CortexFlowConfig

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
    db_path = "graph_traversal_example.db"
    
    # Clean up any existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Configure CortexFlow with graph features enabled
    config = CortexFlowConfig(
        use_graph_rag=True,
        knowledge_store_path=db_path
    )
    
    # Create the manager
    manager = CortexFlowManager(config)
    
    print("=== POPULATING KNOWLEDGE GRAPH ===")
    
    # Add some knowledge to the graph
    documents = [
        """
        Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.
        He was born in Ulm, Germany, in 1879. Einstein received the Nobel Prize in Physics in 1921 for 
        his discovery of the law of the photoelectric effect.
        """,
        
        """
        Einstein worked at the Swiss Patent Office in Bern from 1902 to 1909. Later, he became a professor
        at the University of Berlin. After World War II, he moved to the United States and worked at 
        Princeton University until his death in 1955.
        """,
        
        """
        The theory of relativity was developed by Einstein in the early 20th century. It consists of 
        two theories: special relativity and general relativity. The famous equation E=mc² comes from
        the special theory of relativity.
        """,
        
        """
        Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
        She discovered the elements polonium and radium. Curie was the first woman to win a Nobel Prize
        and the only person to win Nobel Prizes in multiple scientific fields (Physics and Chemistry).
        """,
        
        """
        Marie Curie worked at the University of Paris where she conducted her research. She collaborated
        with her husband Pierre Curie on much of her early work. Their daughter Irène Joliot-Curie also
        became a renowned physicist and won a Nobel Prize.
        """,
        
        """
        Richard Feynman was an American theoretical physicist known for his work in quantum mechanics and
        particle physics. He developed the path integral formulation of quantum mechanics and Feynman diagrams.
        Feynman received the Nobel Prize in Physics in 1965.
        """,
        
        """
        Feynman worked on the Manhattan Project during World War II at Los Alamos Laboratory. Later, he became
        a professor at Caltech. He also served on the Rogers Commission investigating the Space Shuttle
        Challenger disaster in 1986.
        """
    ]
    
    # Process documents to build the knowledge graph
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}...")
        manager.knowledge_store.remember_knowledge(doc)
    
    # Add some additional specific relationships
    print("Adding specific relationships...")
    graph_store = manager.knowledge_store.graph_store
    
    # Add relationships with confidence scores and weights
    graph_store.add_relation(
        source_entity="Albert Einstein",
        relation_type="developed",
        target_entity="Theory of Relativity",
        confidence=0.95,
        weight=0.9
    )
    
    graph_store.add_relation(
        source_entity="Albert Einstein",
        relation_type="wrote",
        target_entity="Annus Mirabilis papers",
        confidence=0.9,
        weight=0.8
    )
    
    graph_store.add_relation(
        source_entity="Annus Mirabilis papers",
        relation_type="discusses",
        target_entity="Photoelectric effect",
        confidence=0.85,
        weight=0.7
    )
    
    graph_store.add_relation(
        source_entity="Photoelectric effect",
        relation_type="led to",
        target_entity="Nobel Prize in Physics",
        confidence=0.8,
        weight=0.6
    )
    
    graph_store.add_relation(
        source_entity="Marie Curie",
        relation_type="won",
        target_entity="Nobel Prize in Physics",
        confidence=0.95,
        weight=0.9
    )
    
    graph_store.add_relation(
        source_entity="Marie Curie",
        relation_type="won",
        target_entity="Nobel Prize in Chemistry",
        confidence=0.95,
        weight=0.9
    )
    
    graph_store.add_relation(
        source_entity="Richard Feynman",
        relation_type="won",
        target_entity="Nobel Prize in Physics",
        confidence=0.95,
        weight=0.9
    )
    
    # Allow the graph to be loaded
    time.sleep(1)
    
    print("\n=== BASIC PATH QUERY ===")
    paths = graph_store.path_query(
        start_entity="Albert Einstein",
        end_entity="Nobel Prize in Physics",
        max_hops=3
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
        max_hops=4
    )
    
    print(f"Found {len(paths)} paths between Curie and Einstein:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    print("\n=== CONSTRAINED PATH SEARCH ===")
    # Find paths that only go through certain relation types
    paths = graph_store.constrained_path_search(
        start_entity="Albert Einstein",
        end_entity="Nobel Prize in Physics",
        allowed_relations=["developed", "discovered", "led to", "won"],
        max_hops=4
    )
    
    print(f"Found {len(paths)} constrained paths from Einstein to Nobel Prize:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    print("\n=== GRAPH CONTRACTION ===")
    # Contract the graph for efficiency
    contraction_stats = graph_store.contract_graph(
        min_edge_weight=0.2,
        min_confidence=0.3,
        combine_parallel_edges=True
    )
    
    print("Graph contraction results:")
    pprint(contraction_stats)
    
    print("\n=== GRAPH ABSTRACTION ===")
    # Create a hierarchical abstraction
    abstraction_stats = graph_store.create_graph_abstraction(
        community_resolution=1.0,
        min_community_size=3
    )
    
    print("Graph abstraction results:")
    pprint(abstraction_stats)
    
    print("\n=== TRAVERSAL WITH ABSTRACTION ===")
    # Use abstraction for efficient traversal
    paths = graph_store.path_query_with_abstraction(
        start_entity="Albert Einstein",
        end_entity="Richard Feynman",
        max_hops=5
    )
    
    print(f"Found {len(paths)} paths using abstraction:")
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {format_path(path)}")
    
    # Clean up
    manager.close()
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main() 