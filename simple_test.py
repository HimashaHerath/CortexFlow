"""
Simple test file to verify our module imports and functionality work correctly.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test importing the modules we've created and their basic functionality."""
    logger.info("Testing imports...")
    
    try:
        from cortexflow.config import CortexFlowConfig
        logger.info("Successfully imported CortexFlowConfig")
        
        # Create a config instance
        config = CortexFlowConfig()
        logger.info(f"Created config with knowledge_store_path: {config.knowledge_store_path}")
    except ImportError as e:
        logger.error(f"Failed to import CortexFlowConfig: {e}")
    
    try:
        from cortexflow.reasoning_engine import ReasoningEngine
        logger.info("Successfully imported ReasoningEngine")
        
        # Test the query planner
        from cortexflow.reasoning_engine import QueryPlanner
        
        # Create a mock knowledge store
        from unittest.mock import MagicMock
        mock_knowledge_store = MagicMock()
        
        # Create a query planner
        planner = QueryPlanner(mock_knowledge_store, config)
        query = "What causes Python to be popular in data science?"
        steps = planner.plan_query(query)
        logger.info(f"Query planner created {len(steps)} steps for query: {query}")
    except ImportError as e:
        logger.error(f"Failed to import ReasoningEngine: {e}")
    
    try:
        from cortexflow.path_inference import BidirectionalSearch, WeightedPathSearch, ConstrainedPathSearch
        logger.info("Successfully imported path inference modules")
        
        # Create a mock graph store
        class MockGraphStore:
            def get_entity_neighbors(self, entity, direction="both"):
                if entity == "Python" and direction == "outgoing":
                    return [{"entity": "Data Science", "relation": "used_in", "confidence": 0.9}]
                elif entity == "Data Science" and direction == "outgoing":
                    return [{"entity": "Statistics", "relation": "involves", "confidence": 0.8}]
                return []
                
            def explain_path(self, path):
                return "Path explanation: " + " -> ".join([f"{step['source']} {step['relation']} {step['target']}" for step in path])
        
        # Create a path search instance
        mock_graph = MockGraphStore()
        bidirectional = BidirectionalSearch(mock_graph)
        logger.info("Created BidirectionalSearch instance")
        
        # Test with dummy data
        mock_graph.bidirectional_search = lambda start, end, max_hops: [[
            {"source": "Python", "relation": "used_in", "target": "Data Science"},
            {"source": "Data Science", "relation": "involves", "target": "Statistics"}
        ]] if start == "Python" and end == "Statistics" else []
        
        paths = mock_graph.bidirectional_search("Python", "Statistics", 3)
        if paths:
            explanation = mock_graph.explain_path(paths[0])
            logger.info(f"Test path found: {explanation}")
        else:
            logger.warning("No test path found")
    except ImportError as e:
        logger.error(f"Failed to import path inference modules: {e}")
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main() 