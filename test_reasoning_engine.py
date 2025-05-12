"""
Test the reasoning engine and path inference functionality.
"""

import unittest
import os
import sys
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the necessary modules
from cortexflow.config import CortexFlowConfig
from cortexflow.knowledge import KnowledgeStore
from cortexflow.graph_store import GraphStore
from cortexflow.reasoning_engine import ReasoningEngine, register_reasoning_engine
from cortexflow.path_inference import register_path_inference

class TestReasoningEngine(unittest.TestCase):
    """Test the reasoning engine functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create an in-memory database for testing
        self.config = CortexFlowConfig()
        self.config.knowledge_store_path = ":memory:"
        self.config.use_graph_rag = True
        
        # Initialize knowledge store
        self.knowledge_store = KnowledgeStore(self.config)
        
        # Populate with test data
        self._populate_test_data()
        
        # Register reasoning engine
        register_reasoning_engine(self.knowledge_store, self.config)
        
        # Register path inference
        register_path_inference(self.knowledge_store.graph_store)
        
    def _populate_test_data(self):
        """Populate the knowledge store with test data."""
        # Add some facts
        self.knowledge_store.remember_explicit("Python is a programming language.", "test")
        self.knowledge_store.remember_explicit("Python was created by Guido van Rossum.", "test")
        self.knowledge_store.remember_explicit("Guido van Rossum is a software engineer.", "test")
        self.knowledge_store.remember_explicit("Python is widely used in data science.", "test")
        self.knowledge_store.remember_explicit("Data science involves statistical analysis.", "test")
        self.knowledge_store.remember_explicit("Statistical analysis is a form of mathematics.", "test")
        
        # Add entities and relations directly to graph store
        graph_store = self.knowledge_store.graph_store
        
        # Add entities
        python_id = graph_store.add_entity("Python", "programming_language")
        guido_id = graph_store.add_entity("Guido van Rossum", "person")
        data_science_id = graph_store.add_entity("Data Science", "field")
        stats_id = graph_store.add_entity("Statistical Analysis", "method")
        math_id = graph_store.add_entity("Mathematics", "discipline")
        ml_id = graph_store.add_entity("Machine Learning", "field")
        ai_id = graph_store.add_entity("Artificial Intelligence", "field")
        
        # Add relations
        graph_store.add_relation("Python", "created_by", "Guido van Rossum")
        graph_store.add_relation("Python", "used_in", "Data Science")
        graph_store.add_relation("Data Science", "involves", "Statistical Analysis")
        graph_store.add_relation("Statistical Analysis", "is_a", "Mathematics")
        graph_store.add_relation("Data Science", "includes", "Machine Learning")
        graph_store.add_relation("Machine Learning", "is_a", "Artificial Intelligence")
        
    def test_reasoning_engine_initialization(self):
        """Test that the reasoning engine is properly initialized."""
        self.assertIsNotNone(self.knowledge_store.reasoning_engine)
        
    def test_query_planner(self):
        """Test the query planner component."""
        engine = self.knowledge_store.reasoning_engine
        planner = engine.query_planner
        
        # Test a causal query
        causal_query = "What causes Python to be popular in data science?"
        causal_plan = planner.plan_query(causal_query)
        self.assertTrue(len(causal_plan) > 0)
        
        # Test a comparison query
        comparison_query = "Compare Python and Java for data science applications"
        comparison_plan = planner.plan_query(comparison_query)
        self.assertTrue(len(comparison_plan) > 0)
        
    def test_bidirectional_search(self):
        """Test bidirectional search functionality."""
        graph_store = self.knowledge_store.graph_store
        
        # Test bidirectional search
        paths = graph_store.bidirectional_search(
            start_entity="Python",
            end_entity="Mathematics",
            max_hops=3
        )
        
        # Should find at least one path
        self.assertTrue(len(paths) > 0)
        
        # Verify path structure
        first_path = paths[0]
        self.assertTrue(len(first_path) > 0)
        
        # Check path endpoints
        self.assertEqual(first_path[0]["source"], "Python")
        self.assertEqual(first_path[-1]["target"], "Mathematics")
        
    def test_weighted_path_query(self):
        """Test weighted path query functionality."""
        graph_store = self.knowledge_store.graph_store
        
        # Test weighted path query
        paths = graph_store.weighted_path_query(
            start_entity="Python",
            end_entity="Artificial Intelligence",
            max_hops=4
        )
        
        # Should find at least one path
        self.assertTrue(len(paths) > 0)
        
    def test_constrained_path_search(self):
        """Test constrained path search functionality."""
        graph_store = self.knowledge_store.graph_store
        
        # Test constrained path search with allowed relations
        paths = graph_store.constrained_path_search(
            start_entity="Python",
            end_entity="Artificial Intelligence",
            allowed_relations=["used_in", "includes", "is_a"],
            max_hops=4
        )
        
        # Should find at least one path
        self.assertTrue(len(paths) > 0)
        
    def test_path_explanation(self):
        """Test path explanation generation."""
        graph_store = self.knowledge_store.graph_store
        
        # Get a path
        paths = graph_store.bidirectional_search(
            start_entity="Python",
            end_entity="Mathematics",
            max_hops=3
        )
        
        if paths:
            # Generate explanation
            explanation = graph_store.explain_path(paths[0])
            
            # Should have a non-empty explanation
            self.assertTrue(len(explanation) > 0)
            
    def test_reasoning_process(self):
        """Test the full reasoning process."""
        engine = self.knowledge_store.reasoning_engine
        
        # Perform reasoning on a query
        result = engine.reason("How is Python connected to mathematics?")
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertIn("answer", result)
        self.assertIn("reasoning_steps", result)
        
    def test_agent_integration(self):
        """Test integration with agent chain."""
        # This would typically require more complex setup with agent chain
        # For now, just verify that the reasoning engine is accessible
        self.assertIsNotNone(self.knowledge_store.reasoning_engine)
        

if __name__ == "__main__":
    unittest.main() 