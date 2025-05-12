"""
Demonstration of CortexFlow's reasoning capabilities.

This script shows how to use the reasoning engine and path inference features.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the necessary modules
from cortexflow.config import CortexFlowConfig
from cortexflow.reasoning_engine import ReasoningEngine
from cortexflow.path_inference import BidirectionalSearch, WeightedPathSearch, ConstrainedPathSearch

class MockGraphStore:
    """Mock graph store for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock graph store with demo data."""
        # Pre-defined paths between entities
        self.paths = {
            ("Python", "Mathematics"): [
                [
                    {"source": "Python", "relation": "used_in", "target": "Data Science", "confidence": 0.9},
                    {"source": "Data Science", "relation": "involves", "target": "Statistical Analysis", "confidence": 0.85},
                    {"source": "Statistical Analysis", "relation": "is_a", "target": "Mathematics", "confidence": 0.95}
                ]
            ],
            ("Python", "Artificial Intelligence"): [
                [
                    {"source": "Python", "relation": "has_library", "target": "TensorFlow", "confidence": 0.9},
                    {"source": "TensorFlow", "relation": "used_for", "target": "Deep Learning", "confidence": 0.95},
                    {"source": "Deep Learning", "relation": "type_of", "target": "Machine Learning", "confidence": 0.9},
                    {"source": "Machine Learning", "relation": "subfield_of", "target": "Artificial Intelligence", "confidence": 0.95}
                ],
                [
                    {"source": "Python", "relation": "used_in", "target": "Data Science", "confidence": 0.9},
                    {"source": "Data Science", "relation": "uses", "target": "Machine Learning", "confidence": 0.9},
                    {"source": "Machine Learning", "relation": "subfield_of", "target": "Artificial Intelligence", "confidence": 0.95}
                ]
            ]
        }
        
        # Additional information about entities
        self.entities = {
            "Python": {
                "type": "programming_language",
                "description": "A high-level, general-purpose programming language"
            },
            "Guido van Rossum": {
                "type": "person",
                "description": "The creator of Python"
            },
            "Data Science": {
                "type": "field",
                "description": "An interdisciplinary field that uses scientific methods and systems to extract knowledge from data"
            },
            "Statistical Analysis": {
                "type": "method",
                "description": "The systematic application of statistical techniques to describe and analyze data"
            },
            "Mathematics": {
                "type": "discipline",
                "description": "The abstract science of number, quantity, and space"
            },
            "Machine Learning": {
                "type": "field",
                "description": "A field of study that gives computers the ability to learn without being explicitly programmed"
            },
            "Artificial Intelligence": {
                "type": "field",
                "description": "The simulation of human intelligence in machines"
            },
            "Deep Learning": {
                "type": "field",
                "description": "A subset of machine learning that uses neural networks with multiple layers"
            },
            "TensorFlow": {
                "type": "library",
                "description": "An open-source machine learning framework"
            },
            "PyTorch": {
                "type": "library",
                "description": "An open-source machine learning framework"
            }
        }
    
    def bidirectional_search(self, start_entity, end_entity, max_hops=3):
        """Get paths between two entities using bidirectional search."""
        key = (start_entity, end_entity)
        return self.paths.get(key, [])
    
    def weighted_path_query(self, start_entity, end_entity, max_hops=3, **kwargs):
        """Get weighted paths between two entities."""
        key = (start_entity, end_entity)
        return self.paths.get(key, [])
    
    def constrained_path_search(self, start_entity, end_entity, **kwargs):
        """Get paths between two entities with constraints."""
        key = (start_entity, end_entity)
        return self.paths.get(key, [])
    
    def explain_path(self, path):
        """Generate a human-readable explanation of a path."""
        if not path:
            return "No connection found."
        
        explanation_parts = []
        for step in path:
            source = step["source"]
            relation = step["relation"]
            target = step["target"]
            
            if relation == "used_in":
                explanation_parts.append(f"{source} is used in {target}")
            elif relation == "involves":
                explanation_parts.append(f"{source} involves {target}")
            elif relation == "is_a":
                explanation_parts.append(f"{source} is a type of {target}")
            elif relation == "has_library":
                explanation_parts.append(f"{source} has the library {target}")
            elif relation == "used_for":
                explanation_parts.append(f"{source} is used for {target}")
            elif relation == "type_of":
                explanation_parts.append(f"{source} is a type of {target}")
            elif relation == "subfield_of":
                explanation_parts.append(f"{source} is a subfield of {target}")
            elif relation == "uses":
                explanation_parts.append(f"{source} uses {target}")
            else:
                explanation_parts.append(f"{source} {relation} {target}")
        
        return ", which ".join(explanation_parts) + "."
    
    def get_entity_info(self, entity):
        """Get information about an entity."""
        return self.entities.get(entity, {})


def demonstrate_reasoning_engine(mock_graph_store):
    """Demonstrate the reasoning engine capabilities using mock data."""
    logger.info("\n====== REASONING ENGINE DEMONSTRATION ======")
    
    # Create a mock reasoning engine
    mock_reasoning_engine = MagicMock()
    
    # Mock the reason method to return realistic results
    def mock_reason(query):
        if "Python" in query and "mathematics" in query.lower():
            return {
                "answer": "Python is connected to mathematics through data science and statistical analysis. Python is used in data science, which involves statistical analysis, which is a type of mathematics.",
                "confidence": 0.85,
                "reasoning_steps": [
                    {
                        "step_id": "step1",
                        "description": "Identify the relationship between Python and data science",
                        "explanation": "Python is widely used in data science as a primary programming language due to its libraries and ease of use."
                    },
                    {
                        "step_id": "step2",
                        "description": "Identify the relationship between data science and statistical analysis",
                        "explanation": "Data science heavily involves statistical analysis for extracting insights from data."
                    },
                    {
                        "step_id": "step3",
                        "description": "Identify the relationship between statistical analysis and mathematics",
                        "explanation": "Statistical analysis is a branch of mathematics that deals with data collection, organization, and interpretation."
                    }
                ]
            }
        elif "Python" in query and "Artificial Intelligence" in query:
            return {
                "answer": "Python is connected to Artificial Intelligence through multiple paths. One path is through its libraries like TensorFlow, which is used for deep learning, which is a type of machine learning, which is a subfield of AI. Another path is through data science, which uses machine learning, which is a subfield of AI.",
                "confidence": 0.9,
                "reasoning_steps": [
                    {
                        "step_id": "step1",
                        "description": "Identify Python's connection to deep learning libraries",
                        "explanation": "Python has popular libraries like TensorFlow and PyTorch that are specifically designed for deep learning applications."
                    },
                    {
                        "step_id": "step2",
                        "description": "Establish the relationship between deep learning and machine learning",
                        "explanation": "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers."
                    },
                    {
                        "step_id": "step3",
                        "description": "Connect machine learning to artificial intelligence",
                        "explanation": "Machine learning is a central subfield of artificial intelligence that focuses on developing algorithms that can learn from data."
                    }
                ]
            }
        else:
            return {
                "answer": "I don't have enough information to answer this question.",
                "confidence": 0.5,
                "reasoning_steps": []
            }
    
    mock_reasoning_engine.reason = mock_reason
    
    # Example 1: Simple reasoning
    logger.info("\n----- Example 1: Simple Reasoning -----")
    query1 = "How is Python connected to mathematics?"
    logger.info(f"Query: {query1}")
    
    result1 = mock_reasoning_engine.reason(query1)
    logger.info(f"Answer: {result1['answer']}")
    logger.info(f"Confidence: {result1['confidence']}")
    
    # Example 2: Multi-step reasoning
    logger.info("\n----- Example 2: Multi-step Reasoning -----")
    query2 = "What path connects Python to Artificial Intelligence?"
    logger.info(f"Query: {query2}")
    
    result2 = mock_reasoning_engine.reason(query2)
    logger.info(f"Answer: {result2['answer']}")
    
    # Print reasoning steps
    if "reasoning_steps" in result2:
        logger.info("Reasoning Steps:")
        for i, step in enumerate(result2["reasoning_steps"]):
            logger.info(f"  Step {i+1}: {step['description']}")
            if "explanation" in step:
                logger.info(f"    {step['explanation']}")
    
    # Example 3: Weighted path search
    logger.info("\n----- Example 3: Path Search Demonstration -----")
    
    paths = mock_graph_store.weighted_path_query(
        start_entity="Python",
        end_entity="Artificial Intelligence",
        max_hops=4
    )
    
    logger.info(f"Found {len(paths)} paths between Python and Artificial Intelligence")
    
    if paths:
        # Show all paths
        for i, path in enumerate(paths):
            # Generate explanation
            explanation = mock_graph_store.explain_path(path)
            logger.info(f"Path {i+1} explanation: {explanation}")
            
            # Show path details
            logger.info(f"Path {i+1} details:")
            for j, step in enumerate(path):
                logger.info(f"  Step {j+1}: {step['source']} {step['relation']} {step['target']} (confidence: {step['confidence']:.2f})")


def main():
    """Main demonstration function."""
    logger.info("Starting CortexFlow reasoning demonstration")
    
    # Create a mock graph store
    mock_graph_store = MockGraphStore()
    
    # Demonstrate reasoning engine with the mock graph store
    demonstrate_reasoning_engine(mock_graph_store)
    
    logger.info("Demonstration completed")

if __name__ == "__main__":
    main() 