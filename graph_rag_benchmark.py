#!/usr/bin/env python3
"""
Benchmark script for GraphRAG functionality in AdaptiveContext.
This script evaluates retrieval quality, accuracy, and performance metrics.
"""

import os
import sys
import time
import logging
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not available. Ollama integration will be disabled.")

from adaptive_context import AdaptiveContextManager
from adaptive_context.config import AdaptiveContextConfig
from metrics_utils import (
    calculate_precision, calculate_recall, calculate_f1, 
    calculate_mrr, calculate_path_accuracy, evaluate_hop_accuracy,
    evaluate_llm_answer, calculate_benchmark_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test data with connected facts for knowledge graph
from graph_rag_test import TEST_KNOWLEDGE

# Benchmark queries with expected entities or answers
BENCHMARK_QUERIES = {
    "single_hop": [
        {"query": "What is Python?", 
         "expected_entities": ["Python", "programming language"],
         "hop_count": 1},
        {"query": "Who created Python?", 
         "expected_entities": ["Guido van Rossum"],
         "hop_count": 1},
        {"query": "Where is Google's headquarters?", 
         "expected_entities": ["Mountain View", "California", "Silicon Valley"],
         "hop_count": 1},
        {"query": "What is in California?", 
         "expected_entities": ["San Francisco", "Los Angeles", "Silicon Valley"],
         "hop_count": 1},
    ],
    "multi_hop": [
        {"query": "What is the connection between Python and Google?",
         "expected_entities": ["Guido van Rossum", "Google", "Python", "Silicon Valley"],
         "expected_paths": [["Guido van Rossum", "worked", "Google"]],
         "hop_count": 2},
        {"query": "How are Silicon Valley and Python related?",
         "expected_entities": ["Silicon Valley", "Python", "companies", "development"],
         "hop_count": 2},
        {"query": "What connects Facebook and machine learning?",
         "expected_entities": ["Facebook", "PyTorch", "machine learning framework"],
         "hop_count": 2},
        {"query": "Is there a connection between California and Guido van Rossum?",
         "expected_entities": ["California", "Guido van Rossum", "Google", "Silicon Valley"],
         "expected_paths": [["Guido van Rossum", "worked", "Google", "headquarters", "Mountain View", "city in", "Silicon Valley", "region in", "California"]],
         "hop_count": 3},
    ],
    "counterfactual": [
        {"query": "What is the connection between Python and Japan?",
         "expected_entities": ["Python"],
         "hop_count": 0},  # There should be no connection
        {"query": "Did Larry Page create Python?",
         "expected_entities": ["Larry Page", "Python"],
         "hop_count": 0},  # False connection
    ]
}

class GraphRAGBenchmark:
    """Benchmark for evaluating GraphRAG functionality."""
    
    def __init__(self, args):
        """Initialize benchmark."""
        self.args = args
        self.results = {
            "retrieval_precision": {},
            "retrieval_recall": {},
            "path_accuracy": {},
            "timing": {},
            "hop_accuracy": {},
            "llm_answer_accuracy": {},
        }
        
        # Create a test DB path
        self.test_db_path = args.db_path
        if os.path.exists(self.test_db_path) and not args.keep_db:
            os.remove(self.test_db_path)
        
        # Initialize config with GraphRAG enabled
        self.config = AdaptiveContextConfig(
            knowledge_store_path=self.test_db_path,
            use_graph_rag=True,
            graph_weight=args.graph_weight,
            enable_multi_hop_queries=True,
            max_graph_hops=args.max_hops,
            default_model=args.model
        )
        
        # Initialize manager
        self.manager = AdaptiveContextManager(self.config)
        
        # Load or initialize knowledge base
        if not os.path.exists(self.test_db_path) or args.reload_knowledge:
            self._load_knowledge()
    
    def _load_knowledge(self):
        """Load knowledge into the system."""
        print("\nLoading knowledge base...")
        start_time = time.time()
        
        for i, knowledge in enumerate(TEST_KNOWLEDGE):
            if self.args.verbose:
                print(f"[{i+1}/{len(TEST_KNOWLEDGE)}] Adding: {knowledge}")
            else:
                if i % 10 == 0:
                    print(f"Added {i}/{len(TEST_KNOWLEDGE)} facts...", end='\r')
                    
            self.manager.knowledge_store.remember_explicit(
                text=knowledge,
                source="benchmark_data",
                confidence=0.9
            )
        
        load_time = time.time() - start_time
        print(f"\nKnowledge base loaded in {load_time:.2f} seconds")
        
        # Print graph statistics
        graph_store = self.manager.knowledge_store.graph_store
        if graph_store.graph:
            print(f"Graph Statistics:")
            print(f"  - Entities (nodes): {graph_store.graph.number_of_nodes()}")
            print(f"  - Relationships (edges): {graph_store.graph.number_of_edges()}")
    
    def evaluate_retrieval_precision(self, query_type: str, query_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate precision of retrieved results for a query.
        
        Args:
            query_type: Type of query (single_hop, multi_hop, etc.)
            query_data: Query data with expected entities
            
        Returns:
            Precision score and detailed results
        """
        query = query_data["query"]
        expected_entities = query_data.get("expected_entities", [])
        
        start_time = time.time()
        results = self.manager.knowledge_store.get_relevant_knowledge(query, max_results=5)
        query_time = time.time() - start_time
        
        # Get texts from results
        retrieved_texts = [result.get('text', '') for result in results]
        
        # Use metrics utility to calculate precision
        precision = calculate_precision(expected_entities, retrieved_texts)
        
        # Calculate additional metrics
        recall = calculate_recall(expected_entities, retrieved_texts)
        f1 = calculate_f1(precision, recall)
        mrr = calculate_mrr(expected_entities, retrieved_texts)
        
        return precision, {
            "query": query,
            "query_time": query_time,
            "expected_entities": expected_entities,
            "results": retrieved_texts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
    
    def evaluate_retrieval_recall(self, query_type: str, query_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate recall of retrieved results for a query.
        
        Args:
            query_type: Type of query (single_hop, multi_hop, etc.)
            query_data: Query data with expected entities
            
        Returns:
            Recall score and detailed results
        """
        query = query_data["query"]
        expected_entities = query_data.get("expected_entities", [])
        
        results = self.manager.knowledge_store.get_relevant_knowledge(query, max_results=5)
        retrieved_texts = [result.get('text', '') for result in results]
        
        # Use metrics utility to calculate recall
        recall = calculate_recall(expected_entities, retrieved_texts)
        
        return recall, {
            "query": query,
            "expected_entities": expected_entities,
            "retrieved_texts": retrieved_texts,
            "recall": recall
        }
    
    def evaluate_path_accuracy(self, query_type: str, query_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate accuracy of paths found for a query.
        
        Args:
            query_type: Type of query (single_hop, multi_hop, etc.)
            query_data: Query data with expected paths
            
        Returns:
            Path accuracy score and detailed results
        """
        query = query_data["query"]
        expected_paths = query_data.get("expected_paths", [])
        expected_hop_count = query_data.get("hop_count", 0)
        
        if not expected_paths:
            return 1.0, {"query": query, "message": "No expected paths defined"}
        
        # Get direct graph search results
        graph_results = self.manager.knowledge_store._graph_search(query, max_results=5)
        
        # Find path results
        path_found = False
        path_text = ""
        hop_count = 0
        
        for result in graph_results:
            result_type = result.get('type', '')
            if result_type == 'graph_path':
                path_found = True
                path_text = result.get('text', '')
                # Count hops in the path
                hop_count = path_text.count('→')
                break
        
        # Use metrics utility to calculate path accuracy
        path_accuracy = 0.0
        if path_found and expected_paths:
            path_accuracy = calculate_path_accuracy(expected_paths[0], path_text)
            
        # Use metrics utility to calculate hop accuracy
        hop_accuracy = evaluate_hop_accuracy(expected_hop_count, hop_count)
        
        return path_accuracy, {
            "query": query,
            "expected_paths": expected_paths,
            "path_found": path_found,
            "path_text": path_text,
            "expected_hop_count": expected_hop_count,
            "actual_hop_count": hop_count,
            "path_accuracy": path_accuracy,
            "hop_accuracy": hop_accuracy
        }
    
    def evaluate_llm_answers(self, query_type: str, query_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate LLM answer accuracy using retrieved context.
        
        Args:
            query_type: Type of query (single_hop, multi_hop, etc.)
            query_data: Query data
            
        Returns:
            Answer accuracy score and detailed results
        """
        if not REQUESTS_AVAILABLE:
            return 0.0, {"error": "Requests library not available"}
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.config.ollama_host}/api/tags", timeout=2)
            if response.status_code != 200:
                return 0.0, {"error": "Ollama not available"}
        except:
            return 0.0, {"error": "Error connecting to Ollama"}
        
        query = query_data["query"]
        expected_entities = query_data.get("expected_entities", [])
        
        # Get relevant knowledge using GraphRAG
        context = self.manager.knowledge_store.get_relevant_knowledge(query, max_results=5)
        context_text = "\n".join([item["text"] for item in context])
        
        # Build prompt for Ollama
        prompt = f"""
        Based on the following context, please answer the question. 
        If you cannot answer from the provided context, say "I don't have enough information."
        
        Context:
        {context_text}
        
        Question: {query}
        Answer:
        """
        
        try:
            # Call Ollama
            response = requests.post(
                f"{self.config.ollama_host}/api/generate",
                json={"model": self.config.default_model, "prompt": prompt, "stream": False},
                timeout=30
            )
            
            if response.status_code != 200:
                return 0.0, {"error": f"Ollama error: {response.status_code}"}
                
            answer = response.json().get("response", "")
            
            # Use metrics utility to evaluate LLM answer
            accuracy = evaluate_llm_answer(answer, expected_entities)
            
            return accuracy, {
                "query": query,
                "expected_entities": expected_entities,
                "answer": answer,
                "context": context_text,
                "accuracy": accuracy
            }
            
        except Exception as e:
            return 0.0, {"error": f"Error: {str(e)}"}
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "="*80)
        print("Running GraphRAG Benchmarks")
        print("="*80)
        
        for query_type, queries in BENCHMARK_QUERIES.items():
            print(f"\nBenchmarking {query_type} queries...")
            
            precision_scores = []
            recall_scores = []
            path_accuracy_scores = []
            timing_data = []
            hop_accuracy_scores = []
            llm_answer_scores = []
            
            for i, query_data in enumerate(queries):
                query = query_data["query"]
                if self.args.verbose:
                    print(f"\nQUERY: {query}")
                
                # Measure retrieval time
                start_time = time.time()
                results = self.manager.knowledge_store.get_relevant_knowledge(query, max_results=5)
                query_time = time.time() - start_time
                timing_data.append(query_time)
                
                # Evaluate precision
                precision, precision_details = self.evaluate_retrieval_precision(query_type, query_data)
                precision_scores.append(precision)
                
                # Evaluate recall
                recall, recall_details = self.evaluate_retrieval_recall(query_type, query_data)
                recall_scores.append(recall)
                
                # Evaluate path accuracy
                path_accuracy, path_details = self.evaluate_path_accuracy(query_type, query_data)
                path_accuracy_scores.append(path_accuracy)
                hop_accuracy_scores.append(path_details.get("hop_accuracy", 0))
                
                # Evaluate LLM answers if enabled
                if self.args.evaluate_llm:
                    llm_accuracy, llm_details = self.evaluate_llm_answers(query_type, query_data)
                    llm_answer_scores.append(llm_accuracy)
                    
                    if self.args.verbose and "error" not in llm_details:
                        print(f"LLM Answer: {llm_details.get('answer', '')}")
                        print(f"LLM Accuracy: {llm_accuracy:.2f}")
                
                if self.args.verbose:
                    print(f"Retrieval Time: {query_time:.4f}s")
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"Path Accuracy: {path_accuracy:.2f}")
                    
                    print("Results:")
                    for j, result in enumerate(results):
                        print(f"  {j+1}. {result['text']}")
            
            # Calculate average scores
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            avg_path_accuracy = np.mean(path_accuracy_scores) if path_accuracy_scores else 0
            avg_timing = np.mean(timing_data) if timing_data else 0
            avg_hop_accuracy = np.mean(hop_accuracy_scores) if hop_accuracy_scores else 0
            avg_llm_accuracy = np.mean(llm_answer_scores) if llm_answer_scores and self.args.evaluate_llm else 0
            
            # Store results
            self.results["retrieval_precision"][query_type] = avg_precision
            self.results["retrieval_recall"][query_type] = avg_recall
            self.results["path_accuracy"][query_type] = avg_path_accuracy
            self.results["timing"][query_type] = avg_timing
            self.results["hop_accuracy"][query_type] = avg_hop_accuracy
            
            if self.args.evaluate_llm:
                self.results["llm_answer_accuracy"][query_type] = avg_llm_accuracy
            
            print(f"\n{query_type.upper()} Query Results:")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")
            print(f"  Average Path Accuracy: {avg_path_accuracy:.4f}")
            print(f"  Average Hop Accuracy: {avg_hop_accuracy:.4f}")
            print(f"  Average Query Time: {avg_timing:.4f}s")
            
            if self.args.evaluate_llm:
                print(f"  Average LLM Answer Accuracy: {avg_llm_accuracy:.4f}")
        
        # Print overall results
        print("\n" + "="*80)
        print("Overall Benchmark Results")
        print("="*80)
        
        # Calculate overall metrics
        overall_precision = np.mean(list(self.results["retrieval_precision"].values()))
        overall_recall = np.mean(list(self.results["retrieval_recall"].values()))
        overall_path_accuracy = np.mean(list(self.results["path_accuracy"].values()))
        overall_hop_accuracy = np.mean(list(self.results["hop_accuracy"].values()))
        overall_timing = np.mean(list(self.results["timing"].values()))
        
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall Path Accuracy: {overall_path_accuracy:.4f}")
        print(f"Overall Hop Accuracy: {overall_hop_accuracy:.4f}")
        print(f"Overall Average Query Time: {overall_timing:.4f}s")
        
        if self.args.evaluate_llm:
            overall_llm_accuracy = np.mean(list(self.results["llm_answer_accuracy"].values()))
            print(f"Overall LLM Answer Accuracy: {overall_llm_accuracy:.4f}")
        
        if self.args.output:
            self._save_results()
        
        if self.args.plot:
            self._generate_plots()
    
    def _save_results(self):
        """Save benchmark results to a file."""
        output_file = self.args.output
        
        # Add timestamp to the results
        self.results["timestamp"] = time.time()
        self.results["config"] = {
            "model": self.args.model,
            "graph_weight": self.args.graph_weight,
            "max_hops": self.args.max_hops
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def _generate_plots(self):
        """Generate plots of benchmark results."""
        try:
            # Create figure for comparison plot
            plt.figure(figsize=(12, 8))
            
            # Query types
            query_types = list(BENCHMARK_QUERIES.keys())
            
            # Plot precision and recall
            x = range(len(query_types))
            width = 0.15
            
            plt.bar([i - width*2 for i in x], 
                   [self.results["retrieval_precision"].get(qt, 0) for qt in query_types],
                   width=width, label="Precision")
            
            plt.bar([i - width for i in x], 
                   [self.results["retrieval_recall"].get(qt, 0) for qt in query_types],
                   width=width, label="Recall")
            
            plt.bar([i for i in x], 
                   [self.results["path_accuracy"].get(qt, 0) for qt in query_types],
                   width=width, label="Path Accuracy")
            
            plt.bar([i + width for i in x], 
                   [self.results["hop_accuracy"].get(qt, 0) for qt in query_types],
                   width=width, label="Hop Accuracy")
            
            if self.args.evaluate_llm:
                plt.bar([i + width*2 for i in x], 
                       [self.results["llm_answer_accuracy"].get(qt, 0) for qt in query_types],
                       width=width, label="LLM Accuracy")
            
            plt.xlabel("Query Type")
            plt.ylabel("Score")
            plt.title("GraphRAG Benchmark Results by Query Type")
            plt.xticks(x, query_types)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save plot
            plot_file = "graph_rag_benchmark_results.png"
            plt.savefig(plot_file)
            print(f"\nPlot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.manager.close()
        print("\nBenchmark completed.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphRAG functionality")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model to use")
    parser.add_argument("--graph-weight", type=float, default=0.3, help="Weight for graph-based results")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum path length for graph traversal")
    parser.add_argument("--db-path", default="benchmark_graph_rag.db", help="Database path")
    parser.add_argument("--keep-db", action="store_true", help="Keep existing database")
    parser.add_argument("--reload-knowledge", action="store_true", help="Force reload knowledge even if DB exists")
    parser.add_argument("--output", help="Output file for benchmark results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--evaluate-llm", action="store_true", help="Evaluate LLM answer accuracy")
    
    args = parser.parse_args()
    
    benchmark = GraphRAGBenchmark(args)
    benchmark.run_benchmarks()
    benchmark.cleanup()


if __name__ == "__main__":
    main() 