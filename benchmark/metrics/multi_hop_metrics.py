"""
Multi-hop Reasoning Accuracy Metrics for CortexFlow Evaluation.

This module provides metrics specifically designed to evaluate multi-hop reasoning 
accuracy in knowledge graphs.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import json
from datetime import datetime

def normalize_path(path: str) -> List[str]:
    """
    Normalize a reasoning path for comparison.
    
    Args:
        path: A string representation of a reasoning path
        
    Returns:
        List of normalized path components
    """
    if not path:
        return []
    
    # Normalize formatting and splitting
    normalized = path.lower().replace(" → ", "→").replace("->", "→")
    components = [c.strip() for c in normalized.split("→")]
    return [c for c in components if c]  # Filter out empty strings

def path_overlap_score(expected_path: List[str], actual_path: List[str]) -> float:
    """
    Calculate the overlap between expected and actual paths.
    
    Args:
        expected_path: List of expected path components
        actual_path: List of actual path components
        
    Returns:
        A score between 0 and 1 representing the path overlap
    """
    if not expected_path or not actual_path:
        return 0.0
    
    # Convert to sets for overlap calculation
    expected_set = set(expected_path)
    actual_set = set(actual_path)
    
    # Calculate Jaccard similarity
    intersection = len(expected_set.intersection(actual_set))
    union = len(expected_set.union(actual_set))
    
    return intersection / union if union > 0 else 0.0

def path_order_accuracy(expected_path: List[str], actual_path: List[str]) -> float:
    """
    Evaluate how well the ordering of elements in the path matches the expected ordering.
    
    Args:
        expected_path: List of expected path components in correct order
        actual_path: List of actual path components
        
    Returns:
        A score between 0 and 1 representing order accuracy
    """
    if not expected_path or not actual_path:
        return 0.0
    
    # Find the longest common subsequence (LCS)
    m, n = len(expected_path), len(actual_path)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if expected_path[i-1] == actual_path[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    return lcs_length / max(m, n)

def multi_hop_reasoning_score(
    expected_path: List[str], 
    actual_path: List[str], 
    expected_entities: List[str],
    actual_entities: List[str],
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculate a comprehensive multi-hop reasoning score.
    
    Args:
        expected_path: List of expected path components
        actual_path: List of actual path components
        expected_entities: List of expected entities
        actual_entities: List of actual entities
        weights: Dictionary of weights for different components of the score
        
    Returns:
        Dictionary with various scores
    """
    if weights is None:
        weights = {
            "path_overlap": 0.3,
            "path_order": 0.3,
            "entity_coverage": 0.2,
            "hop_count": 0.2
        }
    
    # Calculate path overlap
    overlap = path_overlap_score(expected_path, actual_path)
    
    # Calculate path order accuracy
    order = path_order_accuracy(expected_path, actual_path)
    
    # Calculate entity coverage
    expected_entity_set = set(expected_entities)
    actual_entity_set = set(actual_entities)
    entity_coverage = 0.0
    if expected_entity_set:
        entity_coverage = len(expected_entity_set.intersection(actual_entity_set)) / len(expected_entity_set)
    
    # Calculate hop count accuracy
    expected_hops = max(0, len(expected_path) - 1)
    actual_hops = max(0, len(actual_path) - 1)
    hop_accuracy = 0.0
    if expected_hops == actual_hops:
        hop_accuracy = 1.0
    elif expected_hops > 0 and actual_hops > 0:
        hop_accuracy = min(expected_hops / actual_hops, actual_hops / expected_hops)
    
    # Calculate composite score
    composite_score = (
        weights["path_overlap"] * overlap +
        weights["path_order"] * order +
        weights["entity_coverage"] * entity_coverage +
        weights["hop_count"] * hop_accuracy
    )
    
    return {
        "path_overlap": overlap,
        "path_order": order,
        "entity_coverage": entity_coverage,
        "hop_accuracy": hop_accuracy,
        "composite_score": composite_score
    }

def evaluate_reasoning_chain(
    expected_chain: List[Dict[str, Any]],
    actual_chain: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate a reasoning chain with multiple steps.
    
    Args:
        expected_chain: List of expected reasoning steps
        actual_chain: List of actual reasoning steps
        
    Returns:
        Dictionary with evaluation scores
    """
    if not expected_chain or not actual_chain:
        return {"chain_accuracy": 0.0}
    
    # Initialize scores
    step_scores = []
    
    # Match steps and evaluate each step
    min_length = min(len(expected_chain), len(actual_chain))
    for i in range(min_length):
        expected_step = expected_chain[i]
        actual_step = actual_chain[i]
        
        # Extract paths and entities
        expected_path = expected_step.get("path", [])
        actual_path = actual_step.get("path", [])
        expected_entities = expected_step.get("entities", [])
        actual_entities = actual_step.get("entities", [])
        
        # Evaluate step
        step_score = multi_hop_reasoning_score(
            expected_path,
            actual_path,
            expected_entities,
            actual_entities
        )
        
        step_scores.append(step_score)
    
    # Calculate step coverage
    step_coverage = min_length / len(expected_chain)
    
    # Calculate chain accuracy as weighted average of step scores
    chain_accuracy = 0.0
    if step_scores:
        chain_accuracy = sum(score["composite_score"] for score in step_scores) / len(step_scores)
    
    return {
        "step_scores": step_scores,
        "step_coverage": step_coverage,
        "chain_accuracy": chain_accuracy * step_coverage  # Penalize for missing steps
    }

def benchmark_multi_hop_reasoning(
    benchmark_queries: Dict[str, List[Dict[str, Any]]], 
    reasoning_function,
    logger=None
) -> Dict[str, Any]:
    """
    Run benchmarks for multi-hop reasoning.
    
    Args:
        benchmark_queries: Dictionary of benchmark queries
        reasoning_function: Function to call for each query
        logger: Optional logger for logging results
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "single_hop": [],
        "multi_hop": [],
        "counterfactual": [],
        "aggregated": {}
    }
    
    # Run benchmarks for each query type
    for query_type, queries in benchmark_queries.items():
        for query_data in queries:
            query = query_data["query"]
            
            # Log the current benchmark
            if logger:
                logger.info(f"Benchmarking {query_type} query: {query}")
            
            # Run reasoning
            start_time = datetime.now()
            actual_result = reasoning_function(query)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract expected values
            expected_path = query_data.get("expected_path", [])
            expected_entities = query_data.get("expected_entities", [])
            
            # Extract actual values
            actual_path = actual_result.get("path", [])
            actual_entities = actual_result.get("entities", [])
            
            # Calculate scores
            scores = multi_hop_reasoning_score(
                expected_path,
                actual_path,
                expected_entities,
                actual_entities
            )
            
            # Add execution time
            scores["execution_time"] = execution_time
            
            # Add to results
            result_entry = {
                "query": query,
                "expected": {
                    "path": expected_path,
                    "entities": expected_entities
                },
                "actual": {
                    "path": actual_path,
                    "entities": actual_entities
                },
                "scores": scores
            }
            
            if query_type in results:
                results[query_type].append(result_entry)
            else:
                results[query_type] = [result_entry]
    
    # Calculate aggregated scores
    for query_type, query_results in results.items():
        if query_type == "aggregated":
            continue
            
        if not query_results:
            continue
            
        # Calculate average scores
        avg_scores = {
            "path_overlap": np.mean([r["scores"]["path_overlap"] for r in query_results]),
            "path_order": np.mean([r["scores"]["path_order"] for r in query_results]),
            "entity_coverage": np.mean([r["scores"]["entity_coverage"] for r in query_results]),
            "hop_accuracy": np.mean([r["scores"]["hop_accuracy"] for r in query_results]),
            "composite_score": np.mean([r["scores"]["composite_score"] for r in query_results]),
            "execution_time": np.mean([r["scores"]["execution_time"] for r in query_results])
        }
        
        results["aggregated"][query_type] = avg_scores
    
    # Calculate overall score
    if results["aggregated"]:
        overall_score = np.mean([
            scores["composite_score"] 
            for query_type, scores in results["aggregated"].items()
        ])
        
        results["aggregated"]["overall"] = {
            "composite_score": overall_score
        }
    
    return results 