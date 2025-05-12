"""
Demonstration of CortexFlow's reasoning capabilities.

This script shows how to use the reasoning engine and path inference features.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the necessary modules
from cortexflow.config import CortexFlowConfig
from cortexflow.knowledge import KnowledgeStore
from cortexflow.graph_store import GraphStore
from cortexflow.agent_chain import AgentChainManager
from cortexflow.reasoning_engine import register_reasoning_engine
from cortexflow.path_inference import register_path_inference

def setup_knowledge_store():
    """Set up and populate the knowledge store."""
    # Create configuration
    config = CortexFlowConfig()
    config.knowledge_store_path = "reasoning_demo.db"
    config.use_graph_rag = True
    config.use_inference_engine = True
    
    # Initialize knowledge store
    knowledge_store = KnowledgeStore(config)
    
    # Register reasoning engine
    register_reasoning_engine(knowledge_store, config)
    
    # Register path inference
    register_path_inference(knowledge_store.graph_store)
    
    # Populate with test data
    populate_test_data(knowledge_store)
    
    return knowledge_store, config

def populate_test_data(knowledge_store):
    """Populate the knowledge store with test data."""
    # Add some facts
    knowledge_store.remember_explicit("Python is a programming language.", "demo")
    knowledge_store.remember_explicit("Python was created by Guido van Rossum.", "demo")
    knowledge_store.remember_explicit("Guido van Rossum is a software engineer.", "demo")
    knowledge_store.remember_explicit("Python is widely used in data science.", "demo")
    knowledge_store.remember_explicit("Data science involves statistical analysis.", "demo")
    knowledge_store.remember_explicit("Statistical analysis is a form of mathematics.", "demo")
    
    # Add more complex knowledge
    knowledge_store.remember_explicit("""
        Machine learning is a subfield of artificial intelligence. It uses statistical techniques
        to enable computers to learn from data without being explicitly programmed.
        Deep learning is a type of machine learning that uses neural networks with multiple layers.
        Python has libraries like TensorFlow and PyTorch that support deep learning.
    """, "demo")
    
    knowledge_store.remember_explicit("""
        Graph theory is a field of mathematics that studies graphs, which are structures
        used to model pairwise relations between objects. A graph is made up of vertices
        (nodes) and edges that connect them. Graph theory has applications in computer science,
        particularly in algorithms and data structures.
    """, "demo")
    
    # Add entities and relations directly to graph store
    graph_store = knowledge_store.graph_store
    
    # Add entities
    python_id = graph_store.add_entity("Python", "programming_language")
    guido_id = graph_store.add_entity("Guido van Rossum", "person")
    data_science_id = graph_store.add_entity("Data Science", "field")
    stats_id = graph_store.add_entity("Statistical Analysis", "method")
    math_id = graph_store.add_entity("Mathematics", "discipline")
    ml_id = graph_store.add_entity("Machine Learning", "field")
    ai_id = graph_store.add_entity("Artificial Intelligence", "field")
    dl_id = graph_store.add_entity("Deep Learning", "field")
    graph_theory_id = graph_store.add_entity("Graph Theory", "field")
    tensor_id = graph_store.add_entity("TensorFlow", "library")
    pytorch_id = graph_store.add_entity("PyTorch", "library")
    
    # Add basic relations
    graph_store.add_relation("Python", "created_by", "Guido van Rossum")
    graph_store.add_relation("Python", "used_in", "Data Science")
    graph_store.add_relation("Data Science", "involves", "Statistical Analysis")
    graph_store.add_relation("Statistical Analysis", "is_a", "Mathematics")
    
    # Add more complex relations
    graph_store.add_relation("Machine Learning", "subfield_of", "Artificial Intelligence")
    graph_store.add_relation("Deep Learning", "type_of", "Machine Learning")
    graph_store.add_relation("TensorFlow", "used_for", "Deep Learning")
    graph_store.add_relation("PyTorch", "used_for", "Deep Learning")
    graph_store.add_relation("Python", "has_library", "TensorFlow")
    graph_store.add_relation("Python", "has_library", "PyTorch")
    graph_store.add_relation("Graph Theory", "is_a", "Mathematics")
    graph_store.add_relation("Graph Theory", "applied_in", "Computer Science")
    
    # Add weighted relations
    graph_store.add_relation("Python", "popular_for", "Machine Learning", confidence=0.9, weight=0.8)
    graph_store.add_relation("Data Science", "uses", "Machine Learning", confidence=0.9, weight=0.7)
    
    logger.info("Knowledge store populated with test data")

def demonstrate_reasoning_engine(knowledge_store):
    """Demonstrate the reasoning engine capabilities."""
    logger.info("\n====== REASONING ENGINE DEMONSTRATION ======")
    
    # Get the reasoning engine
    reasoning_engine = knowledge_store.reasoning_engine
    
    # Example 1: Simple reasoning
    logger.info("\n----- Example 1: Simple Reasoning -----")
    query1 = "How is Python connected to mathematics?"
    logger.info(f"Query: {query1}")
    
    result1 = reasoning_engine.reason(query1)
    logger.info(f"Answer: {result1['answer']}")
    logger.info(f"Confidence: {result1['confidence']}")
    
    # Example 2: Multi-step reasoning
    logger.info("\n----- Example 2: Multi-step Reasoning -----")
    query2 = "What path connects Python to Artificial Intelligence?"
    logger.info(f"Query: {query2}")
    
    result2 = reasoning_engine.reason(query2)
    logger.info(f"Answer: {result2['answer']}")
    
    # Print reasoning steps
    if "reasoning_steps" in result2:
        logger.info("Reasoning Steps:")
        for i, step in enumerate(result2["reasoning_steps"]):
            logger.info(f"  Step {i+1}: {step['description']}")
            if "explanation" in step:
                logger.info(f"    {step['explanation']}")
    
    # Example 3: Weighted path search
    logger.info("\n----- Example 3: Weighted Path Search -----")
    
    graph_store = knowledge_store.graph_store
    paths = graph_store.weighted_path_query(
        start_entity="Python",
        end_entity="Artificial Intelligence",
        max_hops=4,
        importance_weight=0.7,
        confidence_weight=0.3
    )
    
    logger.info(f"Found {len(paths)} weighted paths")
    
    if paths:
        # Get the first path
        first_path = paths[0]
        
        # Generate explanation
        explanation = graph_store.explain_path(first_path)
        logger.info(f"Path explanation: {explanation}")
        
        # Show path details
        logger.info("Path details:")
        for i, step in enumerate(first_path):
            logger.info(f"  Step {i+1}: {step['source']} {step['relation']} {step['target']} (confidence: {step['confidence']:.2f})")

def demonstrate_agent_chain(knowledge_store, config):
    """Demonstrate the agent chain with reasoning integration."""
    logger.info("\n====== AGENT CHAIN DEMONSTRATION ======")
    
    # Create an agent chain manager
    agent_chain = AgentChainManager(config, knowledge_store)
    
    # Example query
    query = "How does Python relate to artificial intelligence and what path connects them?"
    logger.info(f"Query: {query}")
    
    # Process the query
    result = agent_chain.process_query(query)
    
    # Extract and display the synthesizer's response
    if "synthesis_results" in result:
        synthesis = result["synthesis_results"]
        if "response" in synthesis:
            logger.info(f"Response: {synthesis['response']}")
        
        # Show reasoning steps if available
        if "reasoning_steps" in synthesis:
            logger.info("Reasoning Steps:")
            steps = synthesis["reasoning_steps"]
            for i, step in enumerate(steps):
                logger.info(f"  Step {i+1}: {step['description']}")

def main():
    """Main demonstration function."""
    logger.info("Starting CortexFlow reasoning demonstration")
    
    # Set up knowledge store
    knowledge_store, config = setup_knowledge_store()
    
    # Demonstrate reasoning engine
    demonstrate_reasoning_engine(knowledge_store)
    
    # Demonstrate agent chain integration
    demonstrate_agent_chain(knowledge_store, config)
    
    # Clean up
    knowledge_store.close()
    logger.info("Demonstration completed")

if __name__ == "__main__":
    main() 