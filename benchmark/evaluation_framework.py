"""
Evaluation Framework for CortexFlow Knowledge Graph.

This module provides a comprehensive evaluation framework for assessing
the performance and capabilities of the CortexFlow knowledge graph system.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to sys.path to import cortexflow modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cortexflow import CortexFlowManager, CortexFlowConfig
from cortexflow.reasoning_logger import ReasoningLogger
from benchmark.metrics.multi_hop_metrics import (
    multi_hop_reasoning_score, 
    benchmark_multi_hop_reasoning
)
from benchmark.metrics.consistency_metrics import (
    temporal_consistency_score,
    track_knowledge_growth,
    evaluate_knowledge_consistency
)
from benchmark.test_generation import ReasoningTestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("evaluation_framework")

class EvaluationFramework:
    """
    Evaluation framework for CortexFlow knowledge graph system.
    
    This framework provides:
    1. Multi-hop reasoning benchmarks
    2. Knowledge consistency evaluation
    3. Automatic test generation
    4. Reasoning path logging and analysis
    """
    
    def __init__(
        self,
        manager: Optional[CortexFlowManager] = None,
        config_path: Optional[str] = None,
        db_path: Optional[str] = None,
        results_dir: str = "evaluation_results",
        reasoning_log_db: Optional[str] = None
    ):
        """
        Initialize the evaluation framework.
        
        Args:
            manager: Optional CortexFlowManager instance
            config_path: Path to CortexFlow configuration file
            db_path: Path to knowledge database
            results_dir: Directory for evaluation results
            reasoning_log_db: Path to reasoning log database
        """
        self.manager = manager
        self.config_path = config_path
        self.db_path = db_path
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize manager if not provided
        if not manager and (config_path or db_path):
            logger.info(f"Initializing CortexFlow manager with database: {db_path}")
            config = None
            
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                config = CortexFlowConfig(**config_data)
            elif db_path:
                config = CortexFlowConfig(knowledge_store_path=db_path)
            
            if config:
                self.manager = CortexFlowManager(config)
        
        # Initialize reasoning logger
        self.reasoning_logger = ReasoningLogger(
            db_path=reasoning_log_db or os.path.join(results_dir, "reasoning_logs.db"),
            log_dir=os.path.join(results_dir, "logs")
        )
        
        # Initialize test generator
        self.test_generator = None
        if self.manager:
            self.test_generator = ReasoningTestGenerator(
                graph_store=self.manager.knowledge_store.graph_store if hasattr(self.manager, "knowledge_store") else None,
                knowledge_store=self.manager.knowledge_store if hasattr(self.manager, "knowledge_store") else None
            )
        
        # Initialize results storage
        self.benchmark_results = {}
        self.consistency_results = {}
        self.test_suites = {}
        
        logger.info("Evaluation framework initialized")
    
    def run_multi_hop_benchmarks(
        self,
        test_queries: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        num_generated_tests: int = 20,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run multi-hop reasoning benchmarks.
        
        Args:
            test_queries: Dictionary of test queries by type
            num_generated_tests: Number of tests to generate if test_queries not provided
            save_results: Whether to save results to a file
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.manager:
            logger.error("CortexFlow manager not initialized")
            return {}
        
        # Generate tests if not provided
        if not test_queries and self.test_generator:
            logger.info(f"Generating {num_generated_tests} test queries")
            test_suite = self.test_generator.generate_test_suite(
                suite_name=f"multi_hop_suite_{datetime.now().strftime('%Y%m%d_%H%M')}",
                num_multi_hop=int(num_generated_tests * 0.7),  # 70% multi-hop
                num_counterfactual=int(num_generated_tests * 0.3)  # 30% counterfactual
            )
            
            test_queries = test_suite["tests"]
            self.test_suites[test_suite["name"]] = test_suite
            
            # Save test suite
            if save_results:
                test_suite_path = os.path.join(self.results_dir, f"{test_suite['name']}.json")
                with open(test_suite_path, 'w') as f:
                    json.dump(test_suite, f, indent=2)
                logger.info(f"Saved test suite to {test_suite_path}")
        
        if not test_queries:
            logger.warning("No test queries available for benchmarking")
            return {}
        
        # Define reasoning function that uses manager and logger
        def reasoning_function(query):
            # Start reasoning session
            with self.reasoning_logger.create_context_manager(query) as ctx:
                # Log query processing
                self.reasoning_logger.log_reasoning_step(
                    "query_processing", 
                    f"Processing query: {query}"
                )
                
                # Run query through manager
                if hasattr(self.manager, "multi_hop_query"):
                    result = self.manager.multi_hop_query(query)
                else:
                    # Fall back to standard query if multi_hop_query not available
                    result = self.manager.query(query)
                
                # Extract path and entities from result
                path = result.get("path", [])
                entities = result.get("entities", [])
                
                # Log the path
                if path:
                    self.reasoning_logger.log_path(path, result.get("score"))
                
                # Return result in expected format
                return {
                    "path": path,
                    "entities": entities,
                    "score": result.get("score"),
                    "reasoning_id": ctx.session_id
                }
        
        # Run benchmarks
        logger.info("Running multi-hop reasoning benchmarks")
        start_time = time.time()
        
        results = benchmark_multi_hop_reasoning(
            test_queries,
            reasoning_function,
            logger=logger
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Benchmarks completed in {execution_time:.2f} seconds")
        
        # Add overall execution time
        results["execution_time"] = execution_time
        
        # Store benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"multi_hop_benchmark_{timestamp}"
        self.benchmark_results[result_id] = results
        
        # Save results if requested
        if save_results:
            results_path = os.path.join(self.results_dir, f"{result_id}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved benchmark results to {results_path}")
            
            # Generate visualizations
            self._visualize_multi_hop_results(results, result_id)
        
        return results
    
    def evaluate_knowledge_consistency(
        self,
        time_window_days: Optional[int] = 30,
        take_snapshot: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate knowledge consistency over time.
        
        Args:
            time_window_days: Optional time window in days for evaluation
            take_snapshot: Whether to take a snapshot of the current state
            save_results: Whether to save results to a file
            
        Returns:
            Dictionary with consistency metrics
        """
        if not self.manager or not hasattr(self.manager, "knowledge_store"):
            logger.error("CortexFlow manager or knowledge store not initialized")
            return {}
        
        # Convert time window to timedelta
        time_window = timedelta(days=time_window_days) if time_window_days else None
        
        # Evaluate knowledge consistency
        logger.info(f"Evaluating knowledge consistency over {time_window_days} days")
        start_time = time.time()
        
        results = evaluate_knowledge_consistency(
            self.manager.knowledge_store,
            time_window=time_window,
            take_snapshot=take_snapshot
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Consistency evaluation completed in {execution_time:.2f} seconds")
        
        # Add execution time
        results["execution_time"] = execution_time
        
        # Store consistency results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"consistency_evaluation_{timestamp}"
        self.consistency_results[result_id] = results
        
        # Save results if requested
        if save_results:
            results_path = os.path.join(self.results_dir, f"{result_id}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved consistency results to {results_path}")
            
            # Generate visualizations
            self._visualize_consistency_results(results, result_id)
        
        return results
    
    def generate_and_run_tests(
        self,
        num_tests: int = 30,
        save_results: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate and run tests for reasoning capability assessment.
        
        Args:
            num_tests: Number of tests to generate
            save_results: Whether to save results to a file
            
        Returns:
            Tuple of (test_suite, benchmark_results)
        """
        if not self.test_generator:
            logger.error("Test generator not initialized")
            return {}, {}
        
        # Generate test suite
        logger.info(f"Generating {num_tests} tests")
        suite_name = f"generated_suite_{datetime.now().strftime('%Y%m%d_%H%M')}"
        test_suite = self.test_generator.generate_test_suite(
            suite_name=suite_name,
            num_multi_hop=int(num_tests * 0.7),  # 70% multi-hop
            num_counterfactual=int(num_tests * 0.3)  # 30% counterfactual
        )
        
        # Save test suite
        if save_results:
            test_suite_path = os.path.join(self.results_dir, f"{suite_name}.json")
            with open(test_suite_path, 'w') as f:
                json.dump(test_suite, f, indent=2)
            logger.info(f"Saved test suite to {test_suite_path}")
        
        # Run benchmarks with generated tests
        benchmark_results = self.run_multi_hop_benchmarks(
            test_queries=test_suite["tests"],
            save_results=save_results
        )
        
        return test_suite, benchmark_results
    
    def analyze_reasoning_paths(
        self,
        n_sessions: int = 10,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze reasoning paths from recent sessions.
        
        Args:
            n_sessions: Number of recent sessions to analyze
            save_results: Whether to save results to a file
            
        Returns:
            Dictionary with analysis results
        """
        # Get recent sessions
        recent_sessions = self.reasoning_logger.get_recent_sessions(n_sessions)
        
        if not recent_sessions:
            logger.warning("No reasoning sessions found for analysis")
            return {}
        
        # Analyze each session
        logger.info(f"Analyzing {len(recent_sessions)} reasoning sessions")
        
        analysis_results = {
            "sessions": {},
            "summary": {
                "avg_path_length": 0,
                "avg_confidence": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "common_entities": {},
                "common_relations": {},
                "step_type_distribution": {}
            }
        }
        
        # Analyze individual sessions
        for session in recent_sessions:
            session_id = session["id"]
            session_analysis = self.reasoning_logger.analyze_session(session_id)
            
            if session_analysis:
                analysis_results["sessions"][session_id] = session_analysis
        
        # Calculate summary metrics
        if analysis_results["sessions"]:
            # Extract values for averaging
            path_lengths = [s.get("avg_path_length", 0) for s in analysis_results["sessions"].values()]
            confidences = [s.get("avg_confidence", 0) for s in analysis_results["sessions"].values()]
            success_rates = [1 if s.get("success", False) else 0 for s in analysis_results["sessions"].values()]
            durations = [s.get("duration", 0) for s in analysis_results["sessions"].values()]
            
            # Average metrics
            analysis_results["summary"]["avg_path_length"] = sum(path_lengths) / len(path_lengths) if path_lengths else 0
            analysis_results["summary"]["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
            analysis_results["summary"]["success_rate"] = sum(success_rates) / len(success_rates) if success_rates else 0
            analysis_results["summary"]["avg_duration"] = sum(durations) / len(durations) if durations else 0
            
            # Aggregate entity and relation frequencies
            entity_freq = {}
            relation_freq = {}
            step_type_dist = {}
            
            for session_analysis in analysis_results["sessions"].values():
                # Entities
                for entity, count in session_analysis.get("entity_frequencies", {}).items():
                    if entity not in entity_freq:
                        entity_freq[entity] = 0
                    entity_freq[entity] += count
                
                # Relations
                for relation, count in session_analysis.get("relation_frequencies", {}).items():
                    if relation not in relation_freq:
                        relation_freq[relation] = 0
                    relation_freq[relation] += count
                
                # Step types
                for step_type, count in session_analysis.get("step_type_distribution", {}).items():
                    if step_type not in step_type_dist:
                        step_type_dist[step_type] = 0
                    step_type_dist[step_type] += count
            
            # Sort and limit to top 10
            analysis_results["summary"]["common_entities"] = dict(
                sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            analysis_results["summary"]["common_relations"] = dict(
                sorted(relation_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            analysis_results["summary"]["step_type_distribution"] = step_type_dist
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"reasoning_analysis_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            logger.info(f"Saved reasoning analysis to {results_path}")
            
            # Generate visualizations
            self._visualize_reasoning_analysis(analysis_results, f"reasoning_analysis_{timestamp}")
        
        return analysis_results
    
    def _visualize_multi_hop_results(self, results: Dict[str, Any], result_id: str):
        """
        Generate visualizations for multi-hop benchmark results.
        
        Args:
            results: Dictionary with benchmark results
            result_id: Identifier for the results
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract benchmark categories
            categories = [cat for cat in results["aggregated"] if cat != "overall"]
            
            # Prepare data
            metrics = ["path_overlap", "path_order", "entity_coverage", "hop_accuracy", "composite_score"]
            
            for i, metric in enumerate(metrics):
                values = [results["aggregated"][cat][metric] for cat in categories]
                plt.bar([x + i*0.15 for x in range(len(categories))], values, width=0.15, label=metric)
            
            plt.xlabel("Query Types")
            plt.ylabel("Score")
            plt.title("Multi-hop Reasoning Benchmark Results")
            plt.xticks(range(len(categories)), categories)
            plt.legend()
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{result_id}_metrics.png"))
            
            # Create execution time visualization
            plt.figure(figsize=(8, 5))
            exec_times = [np.mean([r["scores"]["execution_time"] for r in results[cat]]) for cat in categories]
            
            plt.bar(range(len(categories)), exec_times)
            plt.xlabel("Query Types")
            plt.ylabel("Execution Time (s)")
            plt.title("Average Execution Time by Query Type")
            plt.xticks(range(len(categories)), categories)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{result_id}_exec_time.png"))
            
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _visualize_consistency_results(self, results: Dict[str, Any], result_id: str):
        """
        Generate visualizations for consistency evaluation results.
        
        Args:
            results: Dictionary with consistency results
            result_id: Identifier for the results
        """
        try:
            # Consistency metrics visualization
            plt.figure(figsize=(10, 6))
            
            # Extract metrics
            consistency_metrics = results.get("consistency_metrics", {})
            metrics = [
                "consistency_score", 
                "stability_score", 
                "contradiction_rate",
                "entity_consistency",
                "relation_consistency"
            ]
            
            values = [consistency_metrics.get(metric, 0) for metric in metrics]
            
            plt.bar(range(len(metrics)), values)
            plt.xlabel("Metrics")
            plt.ylabel("Score")
            plt.title("Knowledge Consistency Metrics")
            plt.xticks(range(len(metrics)), metrics, rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{result_id}_consistency.png"))
            
            # Knowledge growth visualization
            growth_metrics = results.get("growth_metrics", {})
            
            if growth_metrics and growth_metrics.get("timestamps"):
                plt.figure(figsize=(12, 8))
                
                timestamps = growth_metrics.get("timestamps", [])
                entity_counts = growth_metrics.get("entity_count", [])
                relation_counts = growth_metrics.get("relation_count", [])
                
                plt.subplot(2, 1, 1)
                plt.plot(timestamps, entity_counts, 'b-', label="Entities")
                plt.plot(timestamps, relation_counts, 'r-', label="Relations")
                plt.xlabel("Time")
                plt.ylabel("Count")
                plt.title("Knowledge Growth Over Time")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Growth rates
                plt.subplot(2, 1, 2)
                entity_growth = growth_metrics.get("cumulative_entity_growth", [])[1:] if len(growth_metrics.get("cumulative_entity_growth", [])) > 1 else []
                relation_growth = growth_metrics.get("cumulative_relation_growth", [])[1:] if len(growth_metrics.get("cumulative_relation_growth", [])) > 1 else []
                
                if entity_growth and relation_growth:
                    plt.plot(timestamps[1:], entity_growth, 'b--', label="Entity Growth Rate")
                    plt.plot(timestamps[1:], relation_growth, 'r--', label="Relation Growth Rate")
                    plt.xlabel("Time")
                    plt.ylabel("Growth Rate")
                    plt.title("Knowledge Growth Rates")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{result_id}_growth.png"))
            
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _visualize_reasoning_analysis(self, results: Dict[str, Any], result_id: str):
        """
        Generate visualizations for reasoning path analysis.
        
        Args:
            results: Dictionary with analysis results
            result_id: Identifier for the results
        """
        try:
            # Summary metrics visualization
            plt.figure(figsize=(8, 5))
            
            # Extract metrics
            summary = results.get("summary", {})
            metrics = ["avg_path_length", "avg_confidence", "success_rate", "avg_duration"]
            values = [summary.get(metric, 0) for metric in metrics]
            
            plt.bar(range(len(metrics)), values)
            plt.xlabel("Metrics")
            plt.ylabel("Value")
            plt.title("Reasoning Path Analysis Summary")
            plt.xticks(range(len(metrics)), metrics, rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{result_id}_summary.png"))
            
            # Entity frequencies visualization
            common_entities = summary.get("common_entities", {})
            if common_entities:
                plt.figure(figsize=(10, 6))
                
                entities = list(common_entities.keys())[:10]  # Top 10
                frequencies = [common_entities[e] for e in entities]
                
                plt.barh(range(len(entities)), frequencies, color='skyblue')
                plt.xlabel("Frequency")
                plt.ylabel("Entity")
                plt.title("Most Common Entities in Reasoning Paths")
                plt.yticks(range(len(entities)), entities)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{result_id}_entities.png"))
            
            # Step type distribution visualization
            step_types = summary.get("step_type_distribution", {})
            if step_types:
                plt.figure(figsize=(9, 6))
                
                types = list(step_types.keys())
                counts = [step_types[t] for t in types]
                
                plt.pie(counts, labels=types, autopct='%1.1f%%', startangle=90, shadow=True)
                plt.axis('equal')
                plt.title("Reasoning Step Type Distribution")
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{result_id}_step_types.png"))
            
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def run_full_evaluation(
        self,
        n_tests: int = 30,
        consistency_days: int = 30,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run a full evaluation suite.
        
        Args:
            n_tests: Number of tests to generate and run
            consistency_days: Time window in days for consistency evaluation
            save_results: Whether to save results to a file
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting full evaluation suite")
        start_time = time.time()
        
        # Generate and run tests
        test_suite, benchmark_results = self.generate_and_run_tests(n_tests, save_results)
        
        # Evaluate knowledge consistency
        consistency_results = self.evaluate_knowledge_consistency(consistency_days, save_results=save_results)
        
        # Analyze reasoning paths
        reasoning_analysis = self.analyze_reasoning_paths(save_results=save_results)
        
        # Combine all results
        full_results = {
            "benchmark_results": benchmark_results,
            "consistency_results": consistency_results,
            "reasoning_analysis": reasoning_analysis,
            "execution_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_tests": n_tests,
                "consistency_days": consistency_days
            }
        }
        
        # Save full results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"full_evaluation_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(full_results, f, indent=2)
            logger.info(f"Saved full evaluation results to {results_path}")
        
        logger.info(f"Full evaluation completed in {full_results['execution_time']:.2f} seconds")
        
        return full_results

def main():
    """Main function for running the evaluation framework."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CortexFlow Evaluation Framework')
    parser.add_argument('--db_path', type=str, help='Path to knowledge database')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory for results')
    parser.add_argument('--mode', type=str, choices=['full', 'multi_hop', 'consistency', 'test_gen', 'reasoning'], 
                        default='full', help='Evaluation mode')
    parser.add_argument('--n_tests', type=int, default=30, help='Number of tests to generate')
    parser.add_argument('--days', type=int, default=30, help='Time window in days for consistency evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluation framework
    framework = EvaluationFramework(
        db_path=args.db_path,
        config_path=args.config,
        results_dir=args.results_dir
    )
    
    # Run evaluation based on mode
    if args.mode == 'full':
        framework.run_full_evaluation(args.n_tests, args.days)
    elif args.mode == 'multi_hop':
        framework.run_multi_hop_benchmarks(num_generated_tests=args.n_tests)
    elif args.mode == 'consistency':
        framework.evaluate_knowledge_consistency(time_window_days=args.days)
    elif args.mode == 'test_gen':
        framework.generate_and_run_tests(num_tests=args.n_tests)
    elif args.mode == 'reasoning':
        framework.analyze_reasoning_paths()

if __name__ == "__main__":
    main() 