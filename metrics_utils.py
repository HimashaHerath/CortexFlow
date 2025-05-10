"""
Utility functions for evaluating RAG systems.
This module provides standardized metrics for retrieval quality evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple

def normalize_text(text: str) -> str:
    """
    Normalize text for better comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    for char in ",.:;!?()[]{}<>\"'`~/\\|@#$%^&*_+-=":
        text = text.replace(char, ' ')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_entities(text: str) -> Set[str]:
    """
    Extract potential entities from text.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Set of normalized entity texts
    """
    # This is a simple approximation - in production you'd use NER
    normalized = normalize_text(text)
    
    # Extract tokens as potential entities (removing stop words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
    tokens = [token for token in normalized.split() if token not in stop_words and len(token) > 1]
    
    # Extract bigrams too for multi-word entities
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    
    return set(tokens + bigrams)

def calculate_precision(expected_entities: List[str], retrieved_texts: List[str]) -> float:
    """
    Calculate precision of retrieved results.
    
    Args:
        expected_entities: List of expected entities
        retrieved_texts: List of retrieved text fragments
        
    Returns:
        Precision score (0-1)
    """
    if not retrieved_texts:
        return 0.0
    
    normalized_expected = {normalize_text(entity) for entity in expected_entities}
    
    relevant_count = 0
    for text in retrieved_texts:
        entities = extract_entities(text)
        if any(expected in entities or any(expected in entity for entity in entities) for expected in normalized_expected):
            relevant_count += 1
    
    return relevant_count / len(retrieved_texts)

def calculate_recall(expected_entities: List[str], retrieved_texts: List[str]) -> float:
    """
    Calculate recall of retrieved results.
    
    Args:
        expected_entities: List of expected entities
        retrieved_texts: List of retrieved text fragments
        
    Returns:
        Recall score (0-1)
    """
    if not expected_entities:
        return 1.0
    
    normalized_expected = {normalize_text(entity) for entity in expected_entities}
    all_retrieved_entities = set()
    
    for text in retrieved_texts:
        all_retrieved_entities.update(extract_entities(text))
    
    found_count = 0
    for expected in normalized_expected:
        if expected in all_retrieved_entities or any(expected in entity for entity in all_retrieved_entities):
            found_count += 1
    
    return found_count / len(normalized_expected)

def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        
    Returns:
        F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(expected_entities: List[str], ranked_results: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for ranking quality.
    
    Args:
        expected_entities: List of expected entities
        ranked_results: List of retrieved texts in rank order
        
    Returns:
        MRR score (0-1)
    """
    if not ranked_results or not expected_entities:
        return 0.0
    
    normalized_expected = {normalize_text(entity) for entity in expected_entities}
    
    for i, text in enumerate(ranked_results):
        entities = extract_entities(text)
        if any(expected in entities or any(expected in entity for entity in entities) for expected in normalized_expected):
            # Return reciprocal of first relevant result position (1-indexed)
            return 1.0 / (i + 1)
    
    return 0.0  # No relevant results found

def calculate_path_accuracy(expected_path: List[str], actual_path: str) -> float:
    """
    Calculate accuracy of a graph path.
    
    Args:
        expected_path: List of expected entities and relations in the path
        actual_path: Actual path as a string
        
    Returns:
        Path accuracy score (0-1)
    """
    if not expected_path or not actual_path:
        return 0.0
    
    # Normalize and convert expected path to string for comparison
    expected_path_str = " → ".join(expected_path)
    expected_path_norm = normalize_text(expected_path_str)
    actual_path_norm = normalize_text(actual_path)
    
    # Count matching segments
    expected_segments = expected_path_norm.split(" → ")
    actual_segments = actual_path_norm.split(" → ")
    
    matches = 0
    for exp_seg in expected_segments:
        if any(exp_seg in act_seg for act_seg in actual_segments):
            matches += 1
    
    return matches / len(expected_segments) if expected_segments else 0.0

def evaluate_hop_accuracy(expected_hops: int, actual_hops: int) -> float:
    """
    Calculate accuracy of hop count.
    
    Args:
        expected_hops: Expected number of hops
        actual_hops: Actual number of hops
        
    Returns:
        Hop accuracy score (0-1)
    """
    if expected_hops == 0 and actual_hops == 0:
        return 1.0  # Perfect match for no-hop expectation
    
    if expected_hops == 0:
        return 0.0  # Expected no hops but got some
    
    # Return ratio based on difference (1 for exact match, decreases as difference increases)
    hop_ratio = min(actual_hops / expected_hops, expected_hops / actual_hops) if actual_hops > 0 else 0
    return hop_ratio

def evaluate_llm_answer(answer: str, expected_entities: List[str]) -> float:
    """
    Evaluate accuracy of LLM answer against expected entities.
    
    Args:
        answer: LLM generated answer
        expected_entities: List of expected entities
        
    Returns:
        Answer accuracy score (0-1)
    """
    if not answer or not expected_entities:
        return 0.0
    
    normalized_answer = normalize_text(answer)
    answer_entities = extract_entities(normalized_answer)
    
    found_count = 0
    for entity in expected_entities:
        normalized_entity = normalize_text(entity)
        if (normalized_entity in normalized_answer or 
            normalized_entity in answer_entities or
            any(normalized_entity in e for e in answer_entities)):
            found_count += 1
    
    return found_count / len(expected_entities)

def calculate_benchmark_metrics(query_results: List[Dict[str, Any]], expected_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for benchmark evaluation.
    
    Args:
        query_results: List of retrieval results 
        expected_data: Expected data with entities, paths, etc.
        
    Returns:
        Dictionary with computed metrics
    """
    retrieved_texts = [r.get('text', '') for r in query_results]
    expected_entities = expected_data.get('expected_entities', [])
    
    # Calculate retrieval quality metrics
    precision = calculate_precision(expected_entities, retrieved_texts)
    recall = calculate_recall(expected_entities, retrieved_texts)
    f1 = calculate_f1(precision, recall)
    mrr = calculate_mrr(expected_entities, retrieved_texts)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mrr': mrr
    } 