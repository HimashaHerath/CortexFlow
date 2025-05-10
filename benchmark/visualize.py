#!/usr/bin/env python3
"""
Visualize benchmark results.
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file.
    
    Args:
        filename: Path to results file
        
    Returns:
        Dictionary with results
    """
    with open(filename, 'r') as f:
        return json.load(f)


def visualize_retrieval_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualize retrieval benchmark results.
    
    Args:
        results: Retrieval benchmark results
        output_dir: Directory to save visualizations
    """
    if not results:
        print("No retrieval results to visualize")
        return
    
    # Extract systems and metrics
    systems = list(results.keys())
    metrics = ["avg_precision", "avg_recall", "avg_f1", "avg_time"]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot precision, recall, and F1
    x = np.arange(len(systems))
    width = 0.25
    
    for i, metric in enumerate(metrics[:3]):  # Precision, recall, F1
        values = [results[system]["aggregate"][metric] for system in systems]
        ax1.bar(x + (i - 1) * width, values, width, label=metric)
    
    ax1.set_xlabel("System")
    ax1.set_ylabel("Score")
    ax1.set_title("Retrieval Quality Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend()
    
    # Plot query time
    times = [results[system]["aggregate"]["avg_time"] for system in systems]
    ax2.bar(systems, times, color='skyblue')
    ax2.set_xlabel("System")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Average Query Time")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retrieval_metrics.png"))
    plt.close()


def visualize_memory_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualize memory benchmark results.
    
    Args:
        results: Memory benchmark results
        output_dir: Directory to save visualizations
    """
    if not results:
        print("No memory results to visualize")
        return
    
    # Extract systems
    systems = list(results.keys())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot retention scores
    retention_scores = [results[system]["aggregate"]["avg_retention"] for system in systems]
    memory_efficiency = [results[system]["aggregate"]["memory_efficiency"] for system in systems]
    
    x = np.arange(len(systems))
    width = 0.35
    
    ax1.bar(x - width/2, retention_scores, width, label='Knowledge Retention')
    ax1.bar(x + width/2, memory_efficiency, width, label='Memory Efficiency')
    
    ax1.set_xlabel("System")
    ax1.set_ylabel("Score")
    ax1.set_title("Memory Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend()
    
    # Plot topic retention for first system (if multiple topics)
    if systems:
        system = systems[0]
        topics = []
        scores = []
        
        for key, value in results[system].items():
            if key.startswith("retention_topic_") and isinstance(value, dict):
                if "retention_score" in value:
                    topics.append(key.replace("retention_topic_", "Topic "))
                    scores.append(value["retention_score"])
        
        if topics:
            ax2.bar(topics, scores, color='lightgreen')
            ax2.set_xlabel("Topic")
            ax2.set_ylabel("Retention Score")
            ax2.set_title(f"{system} Topic Retention")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_metrics.png"))
    plt.close()


def visualize_results(results_file: str, output_dir: str) -> None:
    """
    Visualize benchmark results.
    
    Args:
        results_file: Path to results file
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_file)
    
    # Visualize retrieval results
    if "retrieval" in results:
        visualize_retrieval_results(results["retrieval"], output_dir)
    
    # Visualize memory results
    if "memory" in results:
        visualize_memory_results(results["memory"], output_dir)
    
    # Generate summary report
    generate_summary_report(results, os.path.join(output_dir, "benchmark_summary.md"))


def generate_summary_report(results: Dict[str, Any], output_file: str) -> None:
    """
    Generate a summary report of benchmark results.
    
    Args:
        results: Benchmark results
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("# AdaptiveContext Benchmark Results\n\n")
        
        # Retrieval results
        if "retrieval" in results:
            f.write("## Retrieval Quality\n\n")
            f.write("| System | Precision | Recall | F1 Score | Query Time (s) |\n")
            f.write("|--------|-----------|--------|----------|---------------|\n")
            
            for system, data in results["retrieval"].items():
                if "aggregate" in data:
                    agg = data["aggregate"]
                    f.write(f"| {system} | {agg['avg_precision']:.3f} | {agg['avg_recall']:.3f} | {agg['avg_f1']:.3f} | {agg['avg_time']:.3f} |\n")
            
            f.write("\n")
        
        # Memory results
        if "memory" in results:
            f.write("## Memory Efficiency\n\n")
            f.write("| System | Knowledge Retention | Memory Efficiency |\n")
            f.write("|--------|---------------------|-------------------|\n")
            
            for system, data in results["memory"].items():
                if "aggregate" in data:
                    agg = data["aggregate"]
                    f.write(f"| {system} | {agg.get('avg_retention', 0):.3f} | {agg.get('memory_efficiency', 0):.3f} |\n")
            
            f.write("\n")
            
            # Topic retention for first system
            systems = list(results["memory"].keys())
            if systems:
                system = systems[0]
                f.write(f"### {system} Topic Retention\n\n")
                f.write("| Topic | Retention Score |\n")
                f.write("|-------|----------------|\n")
                
                for key, value in results["memory"][system].items():
                    if key.startswith("retention_topic_") and isinstance(value, dict):
                        if "retention_score" in value:
                            topic = key.replace("retention_topic_", "")
                            f.write(f"| Topic {topic} | {value['retention_score']:.3f} |\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", type=str, default="benchmark_results.json", help="Input results file")
    parser.add_argument("--output", type=str, default="benchmark_visualizations", help="Output directory")
    args = parser.parse_args()
    
    visualize_results(args.input, args.output)
    print(f"Visualizations saved to {args.output}")


if __name__ == "__main__":
    main() 