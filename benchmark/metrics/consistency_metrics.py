"""
Knowledge Consistency Metrics for CortexFlow Evaluation.

This module provides metrics to evaluate knowledge consistency over time,
detecting inconsistencies and measuring knowledge stability.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import json
import math
import numpy as np
from datetime import datetime, timedelta
import copy

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def temporal_consistency_score(snapshots, time_window=None):
    """
    Calculate temporal consistency score based on knowledge snapshots.
    
    Args:
        snapshots: List of knowledge snapshots
        time_window: Optional time window for evaluation
        
    Returns:
        Dictionary with consistency metrics
    """
    # Need at least 2 snapshots to evaluate consistency
    if len(snapshots) < 2:
        return {
            "consistency_score": 1.0,  # Perfect consistency for single snapshot
            "change_rate": 0.0,
            "stability_score": 1.0,
            "contradiction_rate": 0.0,
            "entity_count": len(snapshots[0]["entities"]) if snapshots else 0,
            "relation_count": len(snapshots[0]["relations"]) if snapshots else 0
        }
    
    # Sort snapshots by timestamp
    sorted_snapshots = sorted(snapshots, key=lambda x: x["timestamp"])
    
    # Filter by time window if provided
    if time_window:
        current_time = datetime.now().timestamp()
        cutoff_time = 0
        
        # Handle different time_window types
        if isinstance(time_window, timedelta):
            cutoff_time = (datetime.now() - time_window).timestamp()
        elif isinstance(time_window, (int, float)):
            # Assume time_window is in days
            cutoff_time = current_time - (time_window * 86400)
        
        sorted_snapshots = [s for s in sorted_snapshots if s["timestamp"] >= cutoff_time]
    
    # Need at least 2 snapshots to evaluate consistency
    if len(sorted_snapshots) < 2:
        return {
            "consistency_score": 1.0,
            "change_rate": 0.0,
            "stability_score": 1.0,
            "contradiction_rate": 0.0,
            "entity_count": len(sorted_snapshots[0]["entities"]) if sorted_snapshots else 0,
            "relation_count": len(sorted_snapshots[0]["relations"]) if sorted_snapshots else 0
        }
    
    # Calculate metrics
    entity_changes = []
    relation_changes = []
    contradictions = []
    
    for i in range(1, len(sorted_snapshots)):
        prev = sorted_snapshots[i-1]
        curr = sorted_snapshots[i]
        
        # Compare entities
        prev_entities = {e.get("entity", e.get("id", "")): e for e in prev.get("entities", [])}
        curr_entities = {e.get("entity", e.get("id", "")): e for e in curr.get("entities", [])}
        
        # Entity changes (added, removed, modified)
        prev_entity_keys = set(prev_entities.keys())
        curr_entity_keys = set(curr_entities.keys())
        
        added_entities = curr_entity_keys - prev_entity_keys
        removed_entities = prev_entity_keys - curr_entity_keys
        common_entities = prev_entity_keys.intersection(curr_entity_keys)
        
        # Check for entity modifications
        modified_entities = set()
        for entity in common_entities:
            if prev_entities[entity].get("entity_type") != curr_entities[entity].get("entity_type"):
                modified_entities.add(entity)
        
        entity_change = {
            "timestamp": curr["timestamp"],
            "added": len(added_entities),
            "removed": len(removed_entities),
            "modified": len(modified_entities),
            "total_change": len(added_entities) + len(removed_entities) + len(modified_entities)
        }
        entity_changes.append(entity_change)
        
        # Compare relations
        prev_relations = {}
        curr_relations = {}
        
        # Format relations for comparison
        for rel in prev.get("relations", []):
            rel_key = f"{rel.get('source_id')}_{rel.get('relation_type')}_{rel.get('target_id')}"
            prev_relations[rel_key] = rel
            
            # Also add formatted version if available
            if "formatted" in rel:
                prev_relations[rel["formatted"]] = rel
                
        for rel in curr.get("relations", []):
            rel_key = f"{rel.get('source_id')}_{rel.get('relation_type')}_{rel.get('target_id')}"
            curr_relations[rel_key] = rel
            
            # Also add formatted version if available
            if "formatted" in rel:
                curr_relations[rel["formatted"]] = rel
        
        # Relation changes
        prev_relation_keys = set(prev_relations.keys())
        curr_relation_keys = set(curr_relations.keys())
        
        added_relations = curr_relation_keys - prev_relation_keys
        removed_relations = prev_relation_keys - curr_relation_keys
        common_relations = prev_relation_keys.intersection(curr_relation_keys)
        
        # Check for relation modifications (confidence change)
        modified_relations = set()
        contradictory_relations = set()
        
        for rel in common_relations:
            if abs(prev_relations[rel].get("confidence", 0.5) - curr_relations[rel].get("confidence", 0.5)) > 0.2:
                modified_relations.add(rel)
            
            # Check for direction inversions as potential contradictions
            if "source_id" in prev_relations[rel] and "target_id" in prev_relations[rel]:
                prev_source = prev_relations[rel]["source_id"]
                prev_target = prev_relations[rel]["target_id"]
                
                if "source_id" in curr_relations[rel] and "target_id" in curr_relations[rel]:
                    curr_source = curr_relations[rel]["source_id"]
                    curr_target = curr_relations[rel]["target_id"]
                    
                    if prev_source == curr_target and prev_target == curr_source:
                        contradictory_relations.add(rel)
        
        relation_change = {
            "timestamp": curr["timestamp"],
            "added": len(added_relations),
            "removed": len(removed_relations),
            "modified": len(modified_relations),
            "contradictions": len(contradictory_relations),
            "total_change": len(added_relations) + len(removed_relations) + len(modified_relations)
        }
        relation_changes.append(relation_change)
        
        # Track contradictions
        contradictions.append({
            "timestamp": curr["timestamp"],
            "count": len(contradictory_relations),
            "relations": list(contradictory_relations)
        })
    
    # Calculate aggregate scores
    total_entities = len(sorted_snapshots[-1].get("entities", []))
    total_relations = len(sorted_snapshots[-1].get("relations", []))
    
    avg_entity_change_rate = sum(c["total_change"] for c in entity_changes) / len(entity_changes) / max(total_entities, 1)
    avg_relation_change_rate = sum(c["total_change"] for c in relation_changes) / len(relation_changes) / max(total_relations, 1)
    
    avg_contradiction_rate = sum(c["count"] for c in contradictions) / len(contradictions) / max(total_relations, 1)
    
    # Higher score means more consistent (less change)
    entity_stability = 1.0 - min(avg_entity_change_rate, 1.0)
    relation_stability = 1.0 - min(avg_relation_change_rate, 1.0)
    
    # Overall consistency score (weighted average)
    stability_score = (entity_stability * 0.4) + (relation_stability * 0.6)
    consistency_score = stability_score * (1.0 - min(avg_contradiction_rate * 3, 0.5))
    
    return {
        "consistency_score": consistency_score,
        "change_rate": (avg_entity_change_rate + avg_relation_change_rate) / 2,
        "stability_score": stability_score,
        "contradiction_rate": avg_contradiction_rate,
        "entity_count": total_entities,
        "relation_count": total_relations
    }

def track_knowledge_growth(
    knowledge_snapshots: List[Dict[str, Any]],
    time_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Track knowledge growth over time, divided into time bins.
    
    Args:
        knowledge_snapshots: List of knowledge snapshots with timestamps
        time_bins: Number of time bins to divide the data into
        
    Returns:
        Dictionary with growth metrics over time
    """
    if not knowledge_snapshots:
        return {
            "timestamps": [],
            "entity_count": [],
            "relation_count": [],
            "cumulative_entity_growth": [],
            "cumulative_relation_growth": []
        }
    
    # Sort by timestamp
    sorted_snapshots = sorted(knowledge_snapshots, key=lambda x: x["timestamp"])
    
    # Get time range
    start_time = sorted_snapshots[0]["timestamp"]
    end_time = sorted_snapshots[-1]["timestamp"]
    
    # Create time bins
    time_range = end_time - start_time
    bin_size = time_range / time_bins
    bin_edges = [start_time + i * bin_size for i in range(time_bins + 1)]
    
    # Initialize result arrays
    timestamps = []
    entity_counts = []
    relation_counts = []
    cumulative_entity_growth = []
    cumulative_relation_growth = []
    
    # Process each bin
    for i in range(time_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_center = bin_start + (bin_end - bin_start) / 2
        
        # Filter snapshots in this bin
        bin_snapshots = [s for s in sorted_snapshots if bin_start <= s["timestamp"] < bin_end]
        
        if not bin_snapshots:
            continue
        
        # Use the latest snapshot in the bin
        latest_snapshot = max(bin_snapshots, key=lambda x: x["timestamp"])
        
        # Count entities and relations
        entity_count = len(latest_snapshot.get("entities", []))
        relation_count = len(latest_snapshot.get("relations", []))
        
        # Add to result arrays
        timestamps.append(bin_center)
        entity_counts.append(entity_count)
        relation_counts.append(relation_count)
        
        # Calculate cumulative growth
        if i == 0:
            cumulative_entity_growth.append(0.0)
            cumulative_relation_growth.append(0.0)
        else:
            prev_entity_count = entity_counts[-2]
            prev_relation_count = relation_counts[-2]
            
            entity_growth = ((entity_count - prev_entity_count) / prev_entity_count) if prev_entity_count > 0 else 1.0
            relation_growth = ((relation_count - prev_relation_count) / prev_relation_count) if prev_relation_count > 0 else 1.0
            
            cumulative_entity_growth.append(entity_growth)
            cumulative_relation_growth.append(relation_growth)
    
    return {
        "timestamps": timestamps,
        "entity_count": entity_counts,
        "relation_count": relation_counts,
        "cumulative_entity_growth": cumulative_entity_growth,
        "cumulative_relation_growth": cumulative_relation_growth
    }

def belief_revision_impact(
    knowledge_snapshots: List[Dict[str, Any]],
    revision_events: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Measure the impact of belief revision events on knowledge consistency.
    
    Args:
        knowledge_snapshots: List of knowledge snapshots
        revision_events: List of belief revision events
        
    Returns:
        Dictionary with impact metrics
    """
    if not knowledge_snapshots or not revision_events:
        return {"revision_impact": 0.0, "recovery_time": 0.0}
    
    # Sort snapshots and events by timestamp
    sorted_snapshots = sorted(knowledge_snapshots, key=lambda x: x["timestamp"])
    sorted_events = sorted(revision_events, key=lambda x: x["timestamp"])
    
    # Initialize metrics
    consistency_before_revisions = []
    consistency_after_revisions = []
    recovery_times = []
    
    # For each revision event, measure consistency before and after
    for event in sorted_events:
        event_time = event["timestamp"]
        
        # Find snapshots before and after the event
        snapshots_before = [s for s in sorted_snapshots if s["timestamp"] < event_time]
        snapshots_after = [s for s in sorted_snapshots if s["timestamp"] >= event_time]
        
        if not snapshots_before or not snapshots_after:
            continue
        
        # Calculate consistency before the event
        before_snapshots = snapshots_before[-min(3, len(snapshots_before)):]  # Last 3 snapshots before
        consistency_before = temporal_consistency_score(before_snapshots)["consistency_score"]
        
        # Calculate consistency immediately after the event
        after_snapshots = [snapshots_before[-1], snapshots_after[0]]  # Last before + first after
        consistency_after = temporal_consistency_score(after_snapshots)["consistency_score"]
        
        # Calculate recovery time
        recovery_time = 0.0
        recovered = False
        
        for i, snapshot in enumerate(snapshots_after):
            if i == 0:
                continue
                
            recovery_snapshots = [snapshots_after[0], snapshot]
            recovery_consistency = temporal_consistency_score(recovery_snapshots)["consistency_score"]
            
            if recovery_consistency >= consistency_before * 0.95:  # 95% of original consistency
                recovery_time = snapshot["timestamp"] - event_time
                recovered = True
                break
        
        if not recovered and len(snapshots_after) > 1:
            recovery_time = snapshots_after[-1]["timestamp"] - event_time
        
        # Add to metrics
        consistency_before_revisions.append(consistency_before)
        consistency_after_revisions.append(consistency_after)
        recovery_times.append(recovery_time)
    
    # Calculate average impact and recovery time
    avg_consistency_before = np.mean(consistency_before_revisions) if consistency_before_revisions else 1.0
    avg_consistency_after = np.mean(consistency_after_revisions) if consistency_after_revisions else 1.0
    revision_impact = 1.0 - (avg_consistency_after / avg_consistency_before) if avg_consistency_before > 0 else 0.0
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
    
    return {
        "revision_impact": revision_impact,
        "recovery_time": avg_recovery_time,
        "consistency_before": avg_consistency_before,
        "consistency_after": avg_consistency_after
    }

def evaluate_knowledge_consistency(
    knowledge_store, 
    time_window: Optional[timedelta] = None,
    take_snapshot: bool = True
) -> Dict[str, Any]:
    """
    Evaluate knowledge consistency of a knowledge store.
    
    Args:
        knowledge_store: The knowledge store to evaluate
        time_window: Optional time window for evaluation
        take_snapshot: Whether to take a snapshot of the current state
        
    Returns:
        Dictionary with consistency metrics
    """
    # Get snapshots from the knowledge store
    snapshots = knowledge_store.get_snapshots()
    
    if take_snapshot:
        # Take a snapshot of the current state
        current_snapshot = knowledge_store.take_snapshot()
        snapshots.append(current_snapshot)
    
    # Get belief revision events
    revision_events = []
    if hasattr(knowledge_store, "graph_store") and hasattr(knowledge_store.graph_store, "uncertainty_handler"):
        revision_events = knowledge_store.graph_store.uncertainty_handler.get_belief_revision_history()
    
    # Calculate consistency metrics
    consistency_metrics = temporal_consistency_score(snapshots, time_window)
    
    # Calculate growth metrics
    growth_metrics = track_knowledge_growth(snapshots)
    
    # Calculate revision impact metrics
    revision_metrics = belief_revision_impact(snapshots, revision_events)
    
    # Combine metrics
    result = {
        "consistency_metrics": consistency_metrics,
        "growth_metrics": growth_metrics,
        "revision_metrics": revision_metrics,
        "snapshot_count": len(snapshots),
        "revision_count": len(revision_events)
    }
    
    return result 