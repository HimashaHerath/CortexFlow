"""
Memory efficiency evaluation metrics.

This module contains functions for evaluating memory usage efficiency and
knowledge retention over time.
"""
from typing import Dict, List, Any
import re
import numpy as np


def calculate_memory_efficiency(memory_stats: Dict[str, Any]) -> float:
    """
    Calculate memory efficiency score based on token usage and performance.
    
    Args:
        memory_stats: Dictionary containing memory usage statistics
        
    Returns:
        Efficiency score (0-1)
    """
    # Extract relevant metrics
    final_tokens = memory_stats.get("final_token_usage", 0)
    
    # If we have historical stats, we can do a more nuanced calculation
    token_history = []
    for key in memory_stats:
        if key.startswith("memory_") and isinstance(memory_stats[key], dict):
            if "total_tokens" in memory_stats[key]:
                token_history.append(memory_stats[key]["total_tokens"])
    
    # If no token history, return 0
    if not token_history:
        return 0.0
        
    # Calculate growth rate of token usage (lower is better)
    if len(token_history) > 1:
        growth_rates = []
        for i in range(1, len(token_history)):
            if token_history[i-1] > 0:
                growth_rate = (token_history[i] - token_history[i-1]) / token_history[i-1]
                growth_rates.append(growth_rate)
        
        # Average growth rate (lower is better)
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        
        # Efficiency score based on growth rate (inversely related)
        growth_efficiency = max(0, 1.0 - (avg_growth_rate * 2))
    else:
        growth_efficiency = 0.5  # Neutral if we don't have enough data
    
    # Retention scores
    retention_scores = []
    for key in memory_stats:
        if key.startswith("retention_topic_") and isinstance(memory_stats[key], dict):
            if "retention_score" in memory_stats[key]:
                retention_scores.append(memory_stats[key]["retention_score"])
    
    # Retention efficiency
    retention_efficiency = np.mean(retention_scores) if retention_scores else 0.5
    
    # Combined score (equal weights)
    return 0.5 * growth_efficiency + 0.5 * retention_efficiency


def calculate_knowledge_retention(response: str, topic: str) -> float:
    """
    Calculate knowledge retention score based on response relevance to a topic.
    
    Args:
        response: System response text
        topic: Topic that should be remembered
        
    Returns:
        Retention score (0-1)
    """
    if not response or not topic:
        return 0.0
    
    # Normalize text
    response_lower = response.lower()
    topic_lower = topic.lower()
    
    # Check for direct mention of the topic
    topic_mentioned = topic_lower in response_lower
    if not topic_mentioned:
        return 0.0
    
    # Count mentions of the topic
    topic_count = response_lower.count(topic_lower)
    
    # Look for detailed information about the topic
    detail_indicators = [
        "discussed",
        "talked about",
        "mentioned",
        "regarding",
        "related to",
        "about",
        "concerning",
        "information on",
        "details on",
    ]
    
    detail_score = 0.0
    for indicator in detail_indicators:
        pattern = f"{indicator}\\s+[^.]*{topic_lower}|{topic_lower}\\s+[^.]*{indicator}"
        matches = re.findall(pattern, response_lower)
        if matches:
            detail_score += 0.1 * len(matches)
    
    detail_score = min(detail_score, 0.5)  # Cap at 0.5
    
    # Base retention score from topic mentions
    mention_score = min(0.3, topic_count * 0.1)
    
    # Length-based component (longer responses about the topic are likely better)
    length_score = min(0.2, (len(response) / 500) * 0.2)
    
    # Combined score
    retention_score = mention_score + detail_score + length_score
    
    return min(1.0, retention_score)


def calculate_tier_utilization(tier_stats: Dict[str, int], total_tokens: int) -> Dict[str, float]:
    """
    Calculate tier utilization efficiency.
    
    Args:
        tier_stats: Dictionary with token counts per tier
        total_tokens: Total tokens across all tiers
        
    Returns:
        Dictionary with tier utilization scores
    """
    if total_tokens == 0:
        return {tier: 0.0 for tier in tier_stats}
    
    # Calculate utilization percentage per tier
    utilization = {tier: count / total_tokens for tier, count in tier_stats.items()}
    
    # Ideal distribution (example)
    ideal = {
        "active": 0.25,
        "working": 0.35,
        "archive": 0.40
    }
    
    # Score based on closeness to ideal distribution
    scores = {}
    for tier in tier_stats:
        if tier in ideal:
            # Closer to ideal is better (1.0 is perfect match)
            difference = abs(utilization.get(tier, 0) - ideal.get(tier, 0))
            scores[tier] = max(0.0, 1.0 - (difference * 2))
        else:
            scores[tier] = 0.5  # Neutral for unknown tiers
    
    return scores


def calculate_knowledge_preservation_rate(before_flush: List[str], after_flush: List[str], 
                                         query: str) -> float:
    """
    Calculate how well knowledge is preserved after a memory flush.
    
    Args:
        before_flush: List of responses before memory flush
        after_flush: List of responses after memory flush
        query: Query used to test knowledge preservation
        
    Returns:
        Knowledge preservation rate (0-1)
    """
    if not before_flush or not after_flush:
        return 0.0
    
    # Combine all before and after responses
    before_text = " ".join(before_flush)
    after_text = " ".join(after_flush)
    
    # Extract key entities from before responses
    before_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', before_text))
    before_entities.update(re.findall(r'\b[A-Z]{2,}\b', before_text))  # Acronyms
    
    # Count how many entities are preserved after flush
    preserved_count = 0
    for entity in before_entities:
        if entity.lower() in after_text.lower():
            preserved_count += 1
    
    # Calculate preservation rate
    preservation_rate = preserved_count / len(before_entities) if before_entities else 0.0
    
    return preservation_rate 