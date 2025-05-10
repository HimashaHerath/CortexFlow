"""
Query complexity handling evaluation metrics.

This module contains functions for evaluating how well a system handles
complex queries like multi-hop questions and relationship queries.
"""
from typing import List, Dict, Set, Any
import re
import numpy as np

from benchmark.metrics.retrieval import normalize_text, extract_key_phrases


def evaluate_multi_hop(response: str, expected_answers: List[str], hop_count: int) -> float:
    """
    Evaluate multi-hop reasoning capabilities.
    
    Args:
        response: System response text
        expected_answers: List of expected answer strings
        hop_count: Expected number of reasoning hops
        
    Returns:
        Multi-hop score (0-1)
    """
    if not response or not expected_answers:
        return 0.0
    
    # Extract key phrases from response and expected answers
    response_phrases = extract_key_phrases(response)
    
    all_expected_phrases = set()
    for answer in expected_answers:
        all_expected_phrases.update(extract_key_phrases(answer))
    
    # Find overlap (relevant retrieved phrases)
    relevant_retrieved = response_phrases.intersection(all_expected_phrases)
    
    # Base content relevance score
    content_relevance = len(relevant_retrieved) / len(all_expected_phrases) if all_expected_phrases else 0.0
    
    # Look for reasoning steps in the response
    reasoning_indicators = [
        "first", "firstly", "second", "secondly", "third", "thirdly",
        "next", "then", "finally", "because", "therefore", "thus", "so",
        "since", "as a result", "consequently", "this means that"
    ]
    
    # Count reasoning indicators
    reasoning_count = 0
    response_lower = response.lower()
    for indicator in reasoning_indicators:
        reasoning_count += response_lower.count(f" {indicator} ")
    
    # Reasoning structure score (higher for responses with multiple reasoning steps)
    expected_indicators = max(1, hop_count - 1)  # Number of expected reasoning transitions
    reasoning_structure = min(1.0, reasoning_count / expected_indicators)
    
    # Evaluate inter-fact connections
    connection_phrases = [
        "related to", "connects to", "linked to", "associated with",
        "connection between", "relationship between", "correlation between"
    ]
    
    connection_count = 0
    for phrase in connection_phrases:
        connection_count += response_lower.count(phrase)
    
    # Connection score (higher for responses with explicit connections)
    connection_score = min(1.0, connection_count / hop_count)
    
    # Combined score with weights
    return (0.5 * content_relevance) + (0.3 * reasoning_structure) + (0.2 * connection_score)


def evaluate_relationship_queries(response: str, expected_answers: List[str]) -> float:
    """
    Evaluate relationship query handling capabilities.
    
    Args:
        response: System response text
        expected_answers: List of expected answer strings
        
    Returns:
        Relationship handling score (0-1)
    """
    if not response or not expected_answers:
        return 0.0
    
    # Extract key phrases from response and expected answers
    response_phrases = extract_key_phrases(response)
    
    all_expected_phrases = set()
    for answer in expected_answers:
        all_expected_phrases.update(extract_key_phrases(answer))
    
    # Find overlap (relevant retrieved phrases)
    relevant_retrieved = response_phrases.intersection(all_expected_phrases)
    
    # Base content relevance score
    content_relevance = len(relevant_retrieved) / len(all_expected_phrases) if all_expected_phrases else 0.0
    
    # Check for relationship terminology
    relationship_terms = [
        "relationship", "connection", "link", "association", "correlation",
        "connected to", "related to", "linked to", "associated with",
        "impacts", "affects", "influences", "causes", "leads to", "results in"
    ]
    
    # Count relationship terms
    term_count = 0
    response_lower = response.lower()
    for term in relationship_terms:
        term_count += response_lower.count(term)
    
    # Relationship terminology score
    terminology_score = min(1.0, term_count / 3)  # Cap at 3 terms
    
    # Look for explicit relationship description patterns
    relationship_patterns = [
        r"([A-Za-z\s]+)\s+is\s+([A-Za-z\s]+)\s+to\s+([A-Za-z\s]+)",
        r"([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)\s+are\s+([A-Za-z\s]+)",
        r"relationship\s+between\s+([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)",
        r"connection\s+between\s+([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)",
        r"([A-Za-z\s]+)\s+([A-Za-z\s]+)\s+([A-Za-z\s]+)"  # Subject-verb-object
    ]
    
    # Count relationship pattern matches
    pattern_count = 0
    for pattern in relationship_patterns:
        matches = re.findall(pattern, response_lower)
        pattern_count += len(matches)
    
    # Relationship pattern score
    pattern_score = min(1.0, pattern_count / 2)  # Cap at 2 patterns
    
    # Combined score with weights
    return (0.4 * content_relevance) + (0.3 * terminology_score) + (0.3 * pattern_score)


def evaluate_response_factuality(response: str, facts: List[str]) -> float:
    """
    Evaluate factual accuracy of a response.
    
    Args:
        response: System response text
        facts: List of factual statements to check for
        
    Returns:
        Factuality score (0-1)
    """
    if not response or not facts:
        return 0.0
    
    # Normalize response
    response_lower = normalize_text(response)
    
    # Count facts present in the response
    fact_present_count = 0
    fact_phrases = []
    
    for fact in facts:
        fact_lower = normalize_text(fact)
        fact_phrases.extend(extract_key_phrases(fact))
        
        # Check if key parts of the fact are present
        key_phrases = list(extract_key_phrases(fact))
        if len(key_phrases) >= 3:
            present_phrases = 0
            for phrase in key_phrases:
                if phrase in response_lower:
                    present_phrases += 1
            
            # If most key phrases are present, count the fact as present
            if present_phrases / len(key_phrases) >= 0.7:
                fact_present_count += 1
    
    # Calculate basic factuality score
    basic_factuality = fact_present_count / len(facts)
    
    # Check for contradictions
    contradiction_markers = [
        "however,", "but,", "although", "though", "nevertheless",
        "on the contrary", "in contrast", "instead", "rather than",
        "not true", "incorrect", "false", "mistake", "error", "wrong"
    ]
    
    contradiction_count = 0
    for marker in contradiction_markers:
        contradiction_count += response_lower.count(" " + marker + " ")
    
    # Adjust score based on contradictions
    contradiction_penalty = min(0.5, contradiction_count * 0.1)
    
    # Final score
    return max(0.0, basic_factuality - contradiction_penalty)


def calculate_complex_query_score(multi_hop_score: float, relationship_score: float, 
                                 factuality_score: float) -> float:
    """
    Calculate an overall complex query handling score.
    
    Args:
        multi_hop_score: Score for multi-hop query handling
        relationship_score: Score for relationship query handling
        factuality_score: Score for factual accuracy
        
    Returns:
        Complex query handling score (0-1)
    """
    return (0.4 * multi_hop_score) + (0.4 * relationship_score) + (0.2 * factuality_score) 