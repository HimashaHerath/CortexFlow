#!/usr/bin/env python3
"""
Benchmark comparison runner for all implemented systems.
"""
import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark")

# Ensure path includes parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import benchmark modules
from benchmark.registry import get_available_systems, get_system_adapter
from benchmark.datasets import available_datasets, load_dataset
from benchmark.metrics.retrieval import calculate_precision_recall
from benchmark.metrics.memory import calculate_memory_efficiency, calculate_knowledge_retention
from benchmark.metrics.complexity import evaluate_multi_hop, evaluate_relationship_queries, calculate_complex_query_score
from benchmark.visualize import visualize_results


def run_retrieval_benchmarks(systems: List[str], dataset_name: str, model: str, verbose: bool) -> Dict[str, Any]:
    """
    Run retrieval benchmarks for the specified systems.
    
    Args:
        systems: List of system names to benchmark
        dataset_name: Dataset name to use
        model: LLM model to use
        verbose: Whether to output verbose logs
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running retrieval benchmarks on dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Results dictionary
    results = {}
    
    # Run benchmark for each system
    for system_name in systems:
        logger.info(f"Benchmarking system: {system_name}")
        
        try:
            # Initialize system adapter
            adapter_class = get_system_adapter(system_name)
            adapter = adapter_class(model=model, verbose=verbose)
            
            # Initialize conversation
            adapter.initialize_conversation()
            
            # Results for this system
            system_results = {
                "precision": [],
                "recall": [],
                "f1": [],
                "query_times": []
            }
            
            # Run queries
            for i, query_data in enumerate(tqdm(dataset["queries"], desc=f"Queries for {system_name}")):
                query = query_data["query"]
                expected_answers = query_data["answers"]
                
                # Record start time
                start_time = time.time()
                
                # Get context if available
                context = dataset.get("context", [])
                
                # Submit query
                response = adapter.query(query, context=context)
                
                # Record query time
                query_time = time.time() - start_time
                system_results["query_times"].append(query_time)
                
                # Calculate precision and recall
                precision, recall, f1 = calculate_precision_recall(response, expected_answers)
                system_results["precision"].append(precision)
                system_results["recall"].append(recall)
                system_results["f1"].append(f1)
                
                if verbose:
                    logger.info(f"Query {i+1}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            # Calculate aggregate metrics
            avg_precision = sum(system_results["precision"]) / len(system_results["precision"]) if system_results["precision"] else 0
            avg_recall = sum(system_results["recall"]) / len(system_results["recall"]) if system_results["recall"] else 0
            avg_f1 = sum(system_results["f1"]) / len(system_results["f1"]) if system_results["f1"] else 0
            avg_time = sum(system_results["query_times"]) / len(system_results["query_times"]) if system_results["query_times"] else 0
            
            # Add aggregate metrics
            system_results["aggregate"] = {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "avg_time": avg_time
            }
            
            results[system_name] = system_results
            
            logger.info(f"{system_name} results: Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, F1={avg_f1:.3f}, Time={avg_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error benchmarking {system_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with next system instead of failing
            results[system_name] = {
                "error": str(e),
                "aggregate": {
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "avg_f1": 0.0,
                    "avg_time": 0.0
                }
            }
    
    return results


def run_memory_benchmarks(systems: List[str], dataset_name: str, model: str, verbose: bool) -> Dict[str, Any]:
    """
    Run memory benchmarks for the specified systems.
    
    Args:
        systems: List of system names to benchmark
        dataset_name: Dataset name to use
        model: LLM model to use
        verbose: Whether to output verbose logs
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running memory benchmarks on dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Results dictionary
    results = {}
    
    # Run benchmark for each system
    for system_name in systems:
        logger.info(f"Benchmarking system: {system_name}")
        
        try:
            # Initialize system adapter
            adapter_class = get_system_adapter(system_name)
            adapter = adapter_class(model=model, verbose=verbose)
            
            # Initialize conversation
            adapter.initialize_conversation()
            
            # Results for this system
            system_results = {
                "retention_scores": [],
                "memory_stats": []
            }
            
            # Load conversations (for long_context dataset) or tests (for memory_test dataset)
            conversations = []
            
            if "conversations" in dataset:
                conversations = dataset["conversations"]
            elif "tests" in dataset:
                # Convert memory tests to conversations
                for test in dataset["tests"]:
                    conversations.append({
                        "id": test["id"],
                        "messages": test["setup_messages"],
                        "queries": [test["pre_flush_query"]]
                    })
            
            # Run through conversations
            for conv_idx, conversation in enumerate(conversations):
                logger.info(f"Running conversation {conv_idx+1}/{len(conversations)}")
                
                # Initialize new conversation
                adapter.initialize_conversation()
                
                # Add all conversation messages
                for msg in conversation["messages"]:
                    adapter.add_message(msg["role"], msg["content"])
                
                # Capture memory stats during conversation
                memory_stats = adapter.get_memory_stats()
                system_results["memory_stats"].append(memory_stats)
                
                # Run retention tests
                for i, query_data in enumerate(conversation["queries"]):
                    query = query_data["query"]
                    expected_answer = query_data["expected_answer"] if "expected_answer" in query_data else query_data.get("answers", [""])[0]
                    
                    # Submit query
                    response = adapter.query(query)
                    
                    # Calculate retention score
                    retention_score = calculate_knowledge_retention(response, query)
                    system_results[f"retention_topic_{i}"] = {
                        "query": query,
                        "response": response,
                        "expected_answer": expected_answer,
                        "retention_score": retention_score
                    }
                    
                    system_results["retention_scores"].append(retention_score)
                    
                    if verbose:
                        logger.info(f"Topic {i+1}: Retention={retention_score:.3f}")
            
            # Get final memory stats
            final_stats = adapter.get_memory_stats()
            system_results["final_token_usage"] = final_stats.get("total_tokens", 0)
            
            # Calculate memory efficiency
            memory_efficiency = calculate_memory_efficiency({
                "final_token_usage": system_results["final_token_usage"],
                **{f"memory_{i}": stats for i, stats in enumerate(system_results["memory_stats"])}
            })
            
            # Calculate aggregate metrics
            avg_retention = sum(system_results["retention_scores"]) / len(system_results["retention_scores"]) if system_results["retention_scores"] else 0
            
            # Add aggregate metrics
            system_results["aggregate"] = {
                "avg_retention": avg_retention,
                "memory_efficiency": memory_efficiency
            }
            
            results[system_name] = system_results
            
            logger.info(f"{system_name} results: Retention={avg_retention:.3f}, Efficiency={memory_efficiency:.3f}")
            
        except Exception as e:
            logger.error(f"Error benchmarking {system_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results


def run_complexity_benchmarks(systems: List[str], dataset_name: str, model: str, verbose: bool) -> Dict[str, Any]:
    """
    Run query complexity benchmarks for the specified systems.
    
    Args:
        systems: List of system names to benchmark
        dataset_name: Dataset name to use
        model: LLM model to use
        verbose: Whether to output verbose logs
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running complexity benchmarks on dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Results dictionary
    results = {}
    
    # Run benchmark for each system
    for system_name in systems:
        logger.info(f"Benchmarking system: {system_name}")
        
        try:
            # Initialize system adapter
            adapter_class = get_system_adapter(system_name)
            adapter = adapter_class(model=model, verbose=verbose)
            
            # Initialize conversation
            adapter.initialize_conversation()
            
            # Results for this system
            system_results = {
                "multi_hop_scores": [],
                "relationship_scores": [],
                "query_times": []
            }
            
            # Run queries
            for i, query_data in enumerate(tqdm(dataset["queries"], desc=f"Queries for {system_name}")):
                query = query_data["query"]
                expected_answers = query_data["answers"]
                hop_count = query_data.get("hop_count", 1)
                query_type = query_data.get("type", "standard")
                
                # Record start time
                start_time = time.time()
                
                # Get context if available
                context = dataset.get("context", [])
                
                # Submit query
                response = adapter.query(query, context=context)
                
                # Record query time
                query_time = time.time() - start_time
                system_results["query_times"].append(query_time)
                
                # Calculate multi-hop score
                multi_hop_score = evaluate_multi_hop(response, expected_answers, hop_count)
                system_results["multi_hop_scores"].append(multi_hop_score)
                
                # Calculate relationship score for relationship queries
                if query_type == "relationship":
                    relationship_score = evaluate_relationship_queries(response, expected_answers)
                    system_results["relationship_scores"].append(relationship_score)
                
                if verbose:
                    logger.info(f"Query {i+1}: Multi-hop={multi_hop_score:.3f}, Time={query_time:.3f}s")
            
            # Calculate aggregate metrics
            avg_multi_hop = sum(system_results["multi_hop_scores"]) / len(system_results["multi_hop_scores"]) if system_results["multi_hop_scores"] else 0
            avg_relationship = sum(system_results["relationship_scores"]) / len(system_results["relationship_scores"]) if system_results["relationship_scores"] else 0
            avg_time = sum(system_results["query_times"]) / len(system_results["query_times"]) if system_results["query_times"] else 0
            
            # Calculate complex query score
            complex_score = calculate_complex_query_score(
                avg_multi_hop, 
                avg_relationship if system_results["relationship_scores"] else avg_multi_hop,
                0.8  # Default factuality score (could be calculated separately)
            )
            
            # Add aggregate metrics
            system_results["aggregate"] = {
                "avg_multi_hop": avg_multi_hop,
                "avg_relationship": avg_relationship,
                "avg_time": avg_time,
                "complex_score": complex_score
            }
            
            results[system_name] = system_results
            
            logger.info(f"{system_name} results: Multi-hop={avg_multi_hop:.3f}, Relationship={avg_relationship:.3f}, Complex={complex_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error benchmarking {system_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results


def run_all_benchmarks(systems: List[str], datasets: List[str], model: str, verbose: bool, output_dir: str) -> Dict[str, Any]:
    """
    Run all benchmarks for the specified systems.
    
    Args:
        systems: List of system names to benchmark
        datasets: List of dataset names to use
        model: LLM model to use
        verbose: Whether to output verbose logs
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall results
    all_results = {}
    
    # Run retrieval benchmarks on HotPotQA or complex dataset
    retrieval_dataset = "hotpotqa" if "hotpotqa" in datasets else "complex"
    retrieval_results = run_retrieval_benchmarks(systems, retrieval_dataset, model, verbose)
    all_results["retrieval"] = retrieval_results
    
    # Run memory benchmarks on long_context or memory_test dataset
    memory_dataset = "memory_test" if "memory_test" in datasets else "long_context"
    memory_results = run_memory_benchmarks(systems, memory_dataset, model, verbose)
    all_results["memory"] = memory_results
    
    # Run complexity benchmarks on complex or HotPotQA dataset
    complexity_dataset = "complex" if "complex" in datasets else "hotpotqa"
    complexity_results = run_complexity_benchmarks(systems, complexity_dataset, model, verbose)
    all_results["complexity"] = complexity_results
    
    # Save results to file
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualizations
    visualize_results(results_file, output_dir)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="AdaptiveContext Benchmarking Framework")
    parser.add_argument("--systems", type=str, default="adaptivecontext", 
                         help="Comma-separated list of systems to benchmark")
    parser.add_argument("--all-systems", action="store_true", 
                         help="Benchmark all available systems")
    parser.add_argument("--datasets", type=str, default="hotpotqa,memory_test", 
                         help="Comma-separated list of datasets to use")
    parser.add_argument("--model", type=str, default="llama3", 
                         help="LLM model to use")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                         help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", 
                         help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Get systems to benchmark
    if args.all_systems:
        systems = get_available_systems()
    else:
        systems = [s.strip() for s in args.systems.split(",")]
    
    # Get datasets to use
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    logger.info(f"Starting benchmarks for systems: {', '.join(systems)}")
    logger.info(f"Using datasets: {', '.join(datasets)}")
    
    # Run benchmarks
    run_all_benchmarks(systems, datasets, args.model, args.verbose, args.output_dir)
    
    logger.info(f"Benchmarks complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 