#!/usr/bin/env python3
"""
Demonstration script for using the CortexFlow Evaluation Framework.

This script shows how to use the evaluation framework to assess
the reasoning capabilities of the CortexFlow knowledge graph system.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cortexflow import CortexFlowManager, CortexFlowConfig
from benchmark.evaluation_framework import EvaluationFramework
from benchmark.test_generation import ReasoningTestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("evaluate_reasoning")

def load_sample_knowledge(manager, sample_file=None):
    """
    Load sample knowledge into the system for demonstration.
    
    Args:
        manager: CortexFlowManager instance
        sample_file: Optional path to a JSON file with sample knowledge
    """
    # If a sample file is provided, load it
    if sample_file and os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            knowledge_items = json.load(f)
        
        logger.info(f"Loading {len(knowledge_items)} knowledge items from {sample_file}")
        
        for item in knowledge_items:
            manager.knowledge_store.remember_explicit(
                text=item.get("text", ""),
                source=item.get("source", "sample_data"),
                confidence=item.get("confidence", 0.9)
            )
        
        return
    
    # Otherwise use default sample knowledge
    logger.info("Loading default sample knowledge")
    
    # Technology domain
    tech_knowledge = [
        "Python is a programming language created by Guido van Rossum.",
        "Guido van Rossum worked at Google from 2005 to 2012.",
        "Google's headquarters are located in Mountain View, California.",
        "Python is widely used for artificial intelligence and machine learning.",
        "TensorFlow is a machine learning framework developed by Google.",
        "PyTorch is a machine learning framework developed by Facebook.",
        "Silicon Valley is a region in California known for technology companies.",
        "JavaScript is a programming language commonly used for web development.",
        "React is a JavaScript library developed by Facebook.",
        "Facebook was founded by Mark Zuckerberg in 2004."
    ]
    
    # Science domain
    science_knowledge = [
        "Albert Einstein developed the theory of relativity.",
        "Quantum mechanics is a fundamental theory in physics.",
        "Isaac Newton formulated the laws of motion and universal gravitation.",
        "DNA is the molecule that carries genetic information in living organisms.",
        "CERN operates the Large Hadron Collider, the world's largest particle accelerator.",
        "Stephen Hawking made significant contributions to the study of black holes.",
        "Marie Curie discovered the elements polonium and radium.",
        "The periodic table organizes chemical elements based on their properties.",
        "Charles Darwin proposed the theory of evolution by natural selection.",
        "The human genome contains approximately 3 billion base pairs."
    ]
    
    # Business domain
    business_knowledge = [
        "Elon Musk is the CEO of Tesla and SpaceX.",
        "SpaceX is a company that designs and manufactures aerospace technology.",
        "Tesla produces electric vehicles and clean energy products.",
        "Jeff Bezos founded Amazon in 1994.",
        "Amazon started as an online bookstore and expanded to other products.",
        "Satya Nadella is the CEO of Microsoft.",
        "Microsoft was founded by Bill Gates and Paul Allen.",
        "Apple was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
        "Warren Buffett is the CEO of Berkshire Hathaway.",
        "The New York Stock Exchange is located on Wall Street in Manhattan."
    ]
    
    # Load all knowledge
    all_knowledge = tech_knowledge + science_knowledge + business_knowledge
    
    for text in all_knowledge:
        manager.knowledge_store.remember_explicit(
            text=text,
            source="sample_data",
            confidence=0.9
        )
    
    logger.info(f"Loaded {len(all_knowledge)} knowledge items")

def run_simple_benchmark(manager, num_tests=5):
    """
    Run a simple multi-hop reasoning benchmark.
    
    Args:
        manager: CortexFlowManager instance
        num_tests: Number of tests to run
    """
    logger.info(f"Running simple benchmark with {num_tests} tests")
    
    # Create test generator
    test_generator = ReasoningTestGenerator(
        graph_store=manager.knowledge_store.graph_store,
        knowledge_store=manager.knowledge_store
    )
    
    # Generate a small test suite
    test_suite = test_generator.generate_test_suite(
        num_multi_hop=num_tests,
        num_counterfactual=2
    )
    
    logger.info(f"Generated {len(test_suite['tests'].get('multi_hop', []))} multi-hop tests")
    logger.info(f"Generated {len(test_suite['tests'].get('counterfactual', []))} counterfactual tests")
    
    # Display a few sample queries
    logger.info("Sample test queries:")
    
    for test_type, tests in test_suite["tests"].items():
        if not tests:
            continue
            
        for i, test in enumerate(tests[:3]):  # Show up to 3 of each type
            logger.info(f"  [{test_type}] {test['query']}")
            if "expected_path" in test and test["expected_path"]:
                path_str = " → ".join(test["expected_path"])
                logger.info(f"    Expected path: {path_str}")
    
    # Run a few queries directly
    if test_suite["tests"].get("multi_hop"):
        test = test_suite["tests"]["multi_hop"][0]
        query = test["query"]
        
        logger.info(f"\nRunning query: {query}")
        result = manager.query(query)
        
        if "path" in result and result["path"]:
            path_str = " → ".join(result["path"])
            logger.info(f"Path found: {path_str}")
        
        if "answer" in result:
            logger.info(f"Answer: {result['answer']}")

def run_comprehensive_evaluation(manager, results_dir="evaluation_results"):
    """
    Run a comprehensive evaluation using the evaluation framework.
    
    Args:
        manager: CortexFlowManager instance
        results_dir: Directory to store results
    """
    logger.info("Starting comprehensive evaluation")
    
    # Initialize evaluation framework
    framework = EvaluationFramework(
        manager=manager,
        results_dir=results_dir,
        reasoning_log_db=os.path.join(results_dir, "reasoning_logs.db")
    )
    
    # Run multi-hop benchmarks with a small number of tests
    logger.info("Running multi-hop reasoning benchmarks")
    benchmark_results = None
    try:
        benchmark_results = framework.run_multi_hop_benchmarks(num_generated_tests=10)
        
        # Display summary results
        if benchmark_results and "aggregated" in benchmark_results:
            logger.info("Benchmark results summary:")
            
            for query_type, metrics in benchmark_results["aggregated"].items():
                if query_type == "overall":
                    logger.info(f"  Overall composite score: {metrics.get('composite_score', 0):.3f}")
                else:
                    logger.info(f"  {query_type} composite score: {metrics.get('composite_score', 0):.3f}")
    except Exception as e:
        logger.error(f"Error running multi-hop benchmarks: {e}")
    
    # Evaluate knowledge consistency
    logger.info("\nEvaluating knowledge consistency")
    consistency_results = None
    try:
        consistency_results = framework.evaluate_knowledge_consistency(time_window_days=7)
        
        if consistency_results and "consistency_metrics" in consistency_results:
            metrics = consistency_results["consistency_metrics"]
            logger.info("Consistency metrics summary:")
            logger.info(f"  Consistency score: {metrics.get('consistency_score', 0):.3f}")
            logger.info(f"  Stability score: {metrics.get('stability_score', 0):.3f}")
            logger.info(f"  Contradiction rate: {metrics.get('contradiction_rate', 0):.3f}")
    except Exception as e:
        logger.error(f"Error evaluating knowledge consistency: {e}")
    
    # Analyze reasoning paths
    logger.info("\nAnalyzing reasoning paths")
    reasoning_analysis = None
    try:
        reasoning_analysis = framework.analyze_reasoning_paths(n_sessions=5)
        
        if reasoning_analysis and "summary" in reasoning_analysis:
            summary = reasoning_analysis["summary"]
            logger.info("Reasoning path analysis summary:")
            logger.info(f"  Average path length: {summary.get('avg_path_length', 0):.2f}")
            logger.info(f"  Success rate: {summary.get('success_rate', 0):.2f}")
            
            if "common_entities" in summary:
                top_entities = list(summary["common_entities"].keys())[:3]
                logger.info(f"  Top entities: {', '.join(top_entities)}")
    except Exception as e:
        logger.error(f"Error analyzing reasoning paths: {e}")
    
    logger.info(f"\nEvaluation complete. Results stored in: {results_dir}")

def main():
    """Main function to demonstrate the evaluation framework."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demonstrate CortexFlow Evaluation Framework')
    parser.add_argument('--db_path', type=str, default='cortexflow_eval.db', help='Path to knowledge database')
    parser.add_argument('--sample_file', type=str, help='Path to sample knowledge file')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory for results')
    parser.add_argument('--simple', action='store_true', help='Run simple benchmark only')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a fresh database for demonstration
    if os.path.exists(args.db_path):
        logger.info(f"Using existing database: {args.db_path}")
    else:
        logger.info(f"Creating new database: {args.db_path}")
    
    # Initialize CortexFlow manager
    config = CortexFlowConfig(
        knowledge_store_path=args.db_path,
        use_graph_rag=True,
        enable_multi_hop_queries=True,
        max_graph_hops=3,
        use_uncertainty_handling=True
    )
    
    manager = CortexFlowManager(config)
    
    # Add query method to manager for the evaluation framework
    def multi_hop_query(self, query):
        """
        Perform multi-hop reasoning on a query.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with reasoning results
        """
        # Use the knowledge store's graph search to find paths
        results = self.knowledge_store._graph_search(query, max_results=5)
        
        # Extract path and entities from results
        path = []
        entities = []
        score = 0.0
        
        for result in results:
            if result.get('type') == 'graph_path':
                path_text = result.get('text', '')
                if path_text:
                    # Parse the path from text (assuming format like "A → B → C")
                    path = [p.strip() for p in path_text.replace('→', '|').split('|')]
                    score = result.get('score', 0.0)
            
            # Collect entities
            if 'entities' in result:
                entities.extend(result.get('entities', []))
        
        # Return formatted result
        return {
            "path": path,
            "entities": list(set(entities)),
            "score": score
        }
    
    # Add query method as a fallback
    def query(self, query):
        """Fallback query method that calls multi_hop_query"""
        return self.multi_hop_query(query)
    
    # Monkey patch the methods into the manager class
    CortexFlowManager.multi_hop_query = multi_hop_query
    CortexFlowManager.query = query
    
    # Add snapshot support to KnowledgeStore for consistency evaluation
    def get_snapshots(self):
        """
        Get snapshots of the knowledge store for consistency evaluation.
        
        Returns:
            List of knowledge snapshots, each with a timestamp
        """
        # Create a single snapshot of the current state
        snapshot = {
            "timestamp": datetime.now().timestamp(),
            "entities": [],
            "relations": []
        }
        
        # Try to get entities and relations from database
        try:
            if hasattr(self, 'conn') and self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
                
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get entities
            cursor.execute("SELECT * FROM graph_entities")
            entities = [dict(row) for row in cursor.fetchall()]
            snapshot["entities"] = entities
            
            # Get relations
            cursor.execute("SELECT * FROM graph_relationships")
            relations = [dict(row) for row in cursor.fetchall()]
            snapshot["relations"] = relations
            
            if not hasattr(self, 'conn') or self.conn is None:
                conn.close()
        except Exception as e:
            print(f"Error getting snapshots: {e}")
        
        # Return a list with just one snapshot for demonstration
        return [snapshot]
    
    # Add take_snapshot method
    def take_snapshot(self):
        """
        Take a snapshot of the current knowledge state.
        
        Returns:
            Dictionary representing the current knowledge state
        """
        return self.get_snapshots()[0]
    
    # Monkey patch the methods into the KnowledgeStore class
    from cortexflow.knowledge import KnowledgeStore
    KnowledgeStore.get_snapshots = get_snapshots
    KnowledgeStore.take_snapshot = take_snapshot
    
    # Add belief revision history method if uncertainty handler exists
    try:
        from cortexflow.uncertainty import UncertaintyHandler
        
        def get_belief_revision_history(self):
            """
            Get the history of belief revisions.
            
            Returns:
                List of belief revision events
            """
            # Return an empty list for the demo
            return []
        
        # Add the method to the UncertaintyHandler class
        UncertaintyHandler.get_belief_revision_history = get_belief_revision_history
    except ImportError:
        # UncertaintyHandler doesn't exist or cannot be imported
        logger.warning("UncertaintyHandler not found, belief revision history will be empty")
    
    # Load sample knowledge
    load_sample_knowledge(manager, args.sample_file)
    
    # Run evaluation
    if args.simple:
        run_simple_benchmark(manager)
    else:
        run_comprehensive_evaluation(manager, args.results_dir)

if __name__ == "__main__":
    main() 