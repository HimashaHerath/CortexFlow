#!/usr/bin/env python3
"""
AdaptiveContext Benchmark Framework

This script runs benchmarks comparing AdaptiveContext with other RAG and context management systems.
"""

import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import adapters
from benchmark.adapters.base import BenchmarkSystemAdapter
from benchmark.adapters.adaptivecontext_adapter import AdaptiveContextAdapter
from benchmark.registry import get_available_systems, get_system_adapter

# Import evaluation metrics
from benchmark.metrics.retrieval import calculate_precision_recall, calculate_map, calculate_ndcg
from benchmark.metrics.memory import calculate_memory_efficiency, calculate_knowledge_retention
from benchmark.metrics.complexity import evaluate_multi_hop, evaluate_relationship_queries

# Import datasets
from benchmark.datasets import load_dataset, available_datasets

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark AdaptiveContext against other systems")
    
    # System selection
    systems_group = parser.add_mutually_exclusive_group(required=True)
    systems_group.add_argument("--all-systems", action="store_true", help="Benchmark against all available systems")
    systems_group.add_argument("--systems", type=str, help="Comma-separated list of systems to benchmark (e.g., adaptivecontext,llamaindex,langchain)")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="all", help=f"Dataset to use for benchmarking. Available: {', '.join(available_datasets())}")
    
    # Test parameters
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to run per test")
    parser.add_argument("--conversation-length", type=int, default=20, help="Length of conversations for memory tests")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops for multi-hop query tests")
    
    # Model selection
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model to use for all systems")
    
    # Output options
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for benchmark results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def initialize_systems(args):
    """Initialize all systems for benchmarking."""
    if args.all_systems:
        system_names = get_available_systems()
    else:
        system_names = args.systems.split(",")
    
    systems = {}
    for name in system_names:
        if args.verbose:
            print(f"Initializing {name}...")
        
        try:
            adapter = get_system_adapter(name)
            systems[name] = adapter(model=args.model, verbose=args.verbose)
        except Exception as e:
            print(f"Error initializing {name}: {e}")
            continue
            
    return systems

def run_retrieval_benchmark(systems, dataset, args):
    """Run retrieval quality benchmark."""
    results = {name: {} for name in systems}
    
    print(f"Running retrieval benchmark on {dataset['name']}...")
    
    # Process each query in the dataset
    for i, query_item in enumerate(dataset["queries"][:args.num_queries]):
        query = query_item["query"]
        expected_answers = query_item["answers"]
        
        if args.verbose:
            print(f"Query {i+1}/{min(args.num_queries, len(dataset['queries']))}: {query}")
        
        # Test each system
        for name, system in systems.items():
            if args.verbose:
                print(f"  Testing {name}...")
                
            start_time = time.time()
            response = system.query(query, dataset["context"])
            end_time = time.time()
            
            # Calculate metrics
            precision, recall, f1 = calculate_precision_recall(response, expected_answers)
            
            # Store results
            results[name][f"query_{i}"] = {
                "query": query,
                "response": response,
                "time": end_time - start_time,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
    
    # Calculate aggregate metrics
    for name in systems:
        results[name]["aggregate"] = {
            "avg_precision": np.mean([results[name][f"query_{i}"]["precision"] for i in range(min(args.num_queries, len(dataset["queries"])))]),
            "avg_recall": np.mean([results[name][f"query_{i}"]["recall"] for i in range(min(args.num_queries, len(dataset["queries"])))]),
            "avg_f1": np.mean([results[name][f"query_{i}"]["f1"] for i in range(min(args.num_queries, len(dataset["queries"])))]),
            "avg_time": np.mean([results[name][f"query_{i}"]["time"] for i in range(min(args.num_queries, len(dataset["queries"])))])
        }
    
    return results

def run_memory_benchmark(systems, args):
    """Run memory efficiency benchmark."""
    results = {name: {} for name in systems}
    
    print("Running memory efficiency benchmark...")
    
    # Generate a conversation of specified length
    conversation = []
    for i in range(args.conversation_length):
        conversation.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"This is message {i+1} in the conversation. It contains important information about topic {(i % 5) + 1}."
        })
    
    # Test each system
    for name, system in systems.items():
        if args.verbose:
            print(f"Testing {name}...")
        
        # Initialize the conversation
        system.initialize_conversation()
        
        # Add messages
        for i, message in enumerate(conversation):
            system.add_message(message["role"], message["content"])
            
            # Periodically check memory usage
            if i % 5 == 0 or i == len(conversation) - 1:
                memory_stats = system.get_memory_stats()
                results[name][f"memory_{i}"] = memory_stats
                
                if args.verbose:
                    print(f"  Message {i+1}: {memory_stats['total_tokens']} tokens used")
        
        # Test memory retention by asking questions about earlier messages
        retention_scores = []
        for topic in range(1, 6):
            query = f"What did we discuss about topic {topic}?"
            response = system.query(query)
            
            # Score retention (1.0 = perfect retention, 0.0 = no retention)
            retention = calculate_knowledge_retention(response, f"topic {topic}")
            retention_scores.append(retention)
            
            results[name][f"retention_topic_{topic}"] = {
                "query": query,
                "response": response,
                "retention_score": retention
            }
        
        # Calculate aggregate metrics
        results[name]["aggregate"] = {
            "avg_retention": np.mean(retention_scores),
            "final_token_usage": results[name][f"memory_{len(conversation)-1}"]["total_tokens"],
            "memory_efficiency": calculate_memory_efficiency(results[name])
        }
    
    return results

def run_complexity_benchmark(systems, dataset, args):
    """Run query complexity handling benchmark."""
    results = {name: {} for name in systems}
    
    print(f"Running complexity benchmark on {dataset['name']}...")
    
    # Get multi-hop queries
    multi_hop_queries = [q for q in dataset["queries"] if q.get("hop_count", 1) > 1][:args.num_queries]
    
    # Test each system
    for name, system in systems.items():
        if args.verbose:
            print(f"Testing {name}...")
        
        # Test multi-hop queries
        hop_scores = []
        for i, query_item in enumerate(multi_hop_queries):
            query = query_item["query"]
            hop_count = query_item.get("hop_count", 2)
            expected_answers = query_item["answers"]
            
            if args.verbose:
                print(f"  Multi-hop query {i+1}/{len(multi_hop_queries)}: {query} ({hop_count} hops)")
            
            start_time = time.time()
            response = system.query(query, dataset["context"])
            end_time = time.time()
            
            # Evaluate multi-hop reasoning
            hop_score = evaluate_multi_hop(response, expected_answers, hop_count)
            hop_scores.append(hop_score)
            
            results[name][f"multi_hop_{i}"] = {
                "query": query,
                "hop_count": hop_count,
                "response": response,
                "time": end_time - start_time,
                "hop_score": hop_score
            }
        
        # Test relationship queries
        relationship_queries = [q for q in dataset["queries"] if "relationship" in q.get("type", "")][:args.num_queries]
        relationship_scores = []
        
        for i, query_item in enumerate(relationship_queries):
            query = query_item["query"]
            expected_answers = query_item["answers"]
            
            if args.verbose:
                print(f"  Relationship query {i+1}/{len(relationship_queries)}: {query}")
            
            start_time = time.time()
            response = system.query(query, dataset["context"])
            end_time = time.time()
            
            # Evaluate relationship understanding
            rel_score = evaluate_relationship_queries(response, expected_answers)
            relationship_scores.append(rel_score)
            
            results[name][f"relationship_{i}"] = {
                "query": query,
                "response": response,
                "time": end_time - start_time,
                "relationship_score": rel_score
            }
        
        # Calculate aggregate metrics
        results[name]["aggregate"] = {
            "avg_hop_score": np.mean(hop_scores) if hop_scores else 0,
            "avg_relationship_score": np.mean(relationship_scores) if relationship_scores else 0,
            "avg_complex_query_time": np.mean([results[name][f"multi_hop_{i}"]["time"] for i in range(len(multi_hop_queries))])
        }
    
    return results

def generate_visualizations(all_results, output_prefix):
    """Generate visualization charts from benchmark results."""
    print("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Extract system names
    systems = list(all_results["retrieval"].keys())
    
    # Retrieval metrics
    plt.figure(figsize=(12, 6))
    metrics = ["avg_precision", "avg_recall", "avg_f1"]
    x = np.arange(len(systems))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [all_results["retrieval"][system]["aggregate"][metric] for system in systems]
        plt.bar(x + (i - 1) * width, values, width, label=metric)
    
    plt.xlabel("System")
    plt.ylabel("Score")
    plt.title("Retrieval Quality Metrics")
    plt.xticks(x, systems)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_retrieval.png")
    
    # Memory efficiency
    plt.figure(figsize=(12, 6))
    metrics = ["avg_retention", "memory_efficiency"]
    
    for i, metric in enumerate(metrics):
        values = [all_results["memory"][system]["aggregate"][metric] for system in systems]
        plt.bar(x + (i - 0.5) * width, values, width, label=metric)
    
    plt.xlabel("System")
    plt.ylabel("Score")
    plt.title("Memory Efficiency Metrics")
    plt.xticks(x, systems)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_memory.png")
    
    # Complexity handling
    plt.figure(figsize=(12, 6))
    metrics = ["avg_hop_score", "avg_relationship_score"]
    
    for i, metric in enumerate(metrics):
        values = [all_results["complexity"][system]["aggregate"][metric] for system in systems]
        plt.bar(x + (i - 0.5) * width, values, width, label=metric)
    
    plt.xlabel("System")
    plt.ylabel("Score")
    plt.title("Query Complexity Handling Metrics")
    plt.xticks(x, systems)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_complexity.png")
    
    print(f"Visualizations saved with prefix: {output_prefix}")

def main():
    args = parse_args()
    
    print(f"Starting AdaptiveContext benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize all systems
    systems = initialize_systems(args)
    if not systems:
        print("No systems to benchmark. Exiting.")
        return
    
    print(f"Benchmarking systems: {', '.join(systems.keys())}")
    
    all_results = {}
    
    # Run retrieval benchmark
    if args.dataset == "all" or args.dataset == "hotpotqa":
        dataset = load_dataset("hotpotqa")
        all_results["retrieval"] = run_retrieval_benchmark(systems, dataset, args)
    
    # Run memory benchmark
    all_results["memory"] = run_memory_benchmark(systems, args)
    
    # Run complexity benchmark
    if args.dataset == "all" or args.dataset == "complex":
        dataset = load_dataset("complex")
        all_results["complexity"] = run_complexity_benchmark(systems, dataset, args)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Benchmark results saved to {args.output}")
    
    # Generate visualizations
    if args.visualize:
        output_prefix = os.path.splitext(args.output)[0]
        generate_visualizations(all_results, output_prefix)
    
    print(f"Benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 