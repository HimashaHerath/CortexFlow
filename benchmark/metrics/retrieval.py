"""
Retrieval quality evaluation metrics.

This module contains functions for evaluating the quality of retrieved content
based on relevance to expected answers.
"""
from typing import List, Dict, Tuple, Set, Any
import numpy as np
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for better comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    for char in ",.:;!?()[]{}<>\"'`~/\\|@#$%^&*_+-=":
        text = text.replace(char, ' ')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def extract_key_phrases(text: str) -> Set[str]:
    """
    Extract key phrases from text.
    
    Args:
        text: Text to extract phrases from
        
    Returns:
        Set of extracted key phrases
    """
    normalized = normalize_text(text)
    
    # Extract tokens (excluding stopwords)
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
        'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
        'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'to', 'from', 'by', 'with', 'in',
        'on', 'at', 'of'
    }
    
    tokens = normalized.split()
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Extract phrases (consecutive tokens)
    phrases = set()
    phrases.update(filtered_tokens)  # Add single tokens
    
    # Add bigrams
    for i in range(len(filtered_tokens) - 1):
        phrases.add(f"{filtered_tokens[i]} {filtered_tokens[i+1]}")
    
    # Add trigrams
    for i in range(len(filtered_tokens) - 2):
        phrases.add(f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}")
    
    return phrases


def calculate_precision_recall(response: str, expected_answers: List[str]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall and F1 score for a response.
    
    Args:
        response: System response text
        expected_answers: List of expected answer strings
        
    Returns:
        Tuple of (precision, recall, F1 score)
    """
    if not response or not expected_answers:
        return 0.0, 0.0, 0.0
        
    # Extract key phrases from response and expected answers
    response_phrases = extract_key_phrases(response)
    
    all_expected_phrases = set()
    for answer in expected_answers:
        all_expected_phrases.update(extract_key_phrases(answer))
    
    # Find intersection (relevant retrieved phrases)
    relevant_retrieved = response_phrases.intersection(all_expected_phrases)
    
    # Calculate precision and recall
    precision = len(relevant_retrieved) / len(response_phrases) if response_phrases else 0.0
    recall = len(relevant_retrieved) / len(all_expected_phrases) if all_expected_phrases else 1.0
    
    # Calculate F1 score
    f1 = 0.0
    if precision > 0 or recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def calculate_map(ranked_results: List[str], expected_answers: List[str]) -> float:
    """
    Calculate Mean Average Precision for ranked retrieval results.
    
    Args:
        ranked_results: List of retrieved documents in rank order
        expected_answers: List of expected answer strings
        
    Returns:
        MAP score (0-1)
    """
    if not ranked_results or not expected_answers:
        return 0.0
    
    # Extract phrases from each result and expected answer
    result_phrases_list = [extract_key_phrases(result) for result in ranked_results]
    
    all_expected_phrases = set()
    for answer in expected_answers:
        all_expected_phrases.update(extract_key_phrases(answer))
        
    # Calculate precision at each position with a relevant result
    precisions = []
    relevant_count = 0
    
    for i, result_phrases in enumerate(result_phrases_list):
        # Check if this result contains any expected phrases
        is_relevant = bool(result_phrases.intersection(all_expected_phrases))
        
        if is_relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
    
    # Calculate MAP
    if not precisions:
        return 0.0
        
    return sum(precisions) / len(precisions)


def calculate_ndcg(ranked_results: List[str], expected_answers: List[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain for ranked retrieval results.
    
    Args:
        ranked_results: List of retrieved documents in rank order
        expected_answers: List of expected answer strings
        k: Number of results to consider
        
    Returns:
        NDCG score (0-1)
    """
    if not ranked_results or not expected_answers:
        return 0.0
    
    # Limit to top k results
    ranked_results = ranked_results[:k]
    
    # Extract phrases from each result and expected answer
    result_phrases_list = [extract_key_phrases(result) for result in ranked_results]
    
    all_expected_phrases = set()
    for answer in expected_answers:
        all_expected_phrases.update(extract_key_phrases(answer))
    
    # Calculate relevance scores (how many expected phrases are in each result)
    relevance_scores = []
    for result_phrases in result_phrases_list:
        overlap = result_phrases.intersection(all_expected_phrases)
        relevance = len(overlap) / len(all_expected_phrases) if all_expected_phrases else 0
        relevance_scores.append(relevance)
    
    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed and log2(1) = 0
    
    # Calculate ideal DCG (best possible ranking)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance_scores):
        idcg += rel / np.log2(i + 2)
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
        
    return dcg / idcg 