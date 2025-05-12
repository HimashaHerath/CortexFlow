"""
Simple test file to verify our module imports work correctly.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test importing the modules we've created."""
    logger.info("Testing imports...")
    
    try:
        from cortexflow.config import CortexFlowConfig
        logger.info("Successfully imported CortexFlowConfig")
    except ImportError as e:
        logger.error(f"Failed to import CortexFlowConfig: {e}")
    
    try:
        from cortexflow.reasoning_engine import ReasoningEngine
        logger.info("Successfully imported ReasoningEngine")
    except ImportError as e:
        logger.error(f"Failed to import ReasoningEngine: {e}")
    
    try:
        from cortexflow.path_inference import BidirectionalSearch
        logger.info("Successfully imported BidirectionalSearch")
    except ImportError as e:
        logger.error(f"Failed to import BidirectionalSearch: {e}")
    
    logger.info("Import tests completed")

if __name__ == "__main__":
    main() 