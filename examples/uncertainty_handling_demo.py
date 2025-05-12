#!/usr/bin/env python3
"""
Uncertainty and Contradictions Handling Demo

This script demonstrates the uncertainty handling features in CortexFlow:
1. Belief revision when new contradictory information arrives
2. Explicit uncertainty representation using confidence scores and probability distributions
3. Conflict resolution strategies based on source reliability and recency
4. Reasoning with incomplete information
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to sys.path to import the cortexflow module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cortexflow import CortexFlowManager, CortexFlowConfig

def print_separator(title: str):
    """Print a section separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_result(title: str, data: Any):
    """Print results in a formatted way."""
    print(f"\n-- {title} --")
    if isinstance(data, list):
        for i, item in enumerate(data, 1):
            print(f"  {i}. {item}")
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {data}")
    print()

def demonstrate_belief_revision(manager: CortexFlowManager):
    """Demonstrate belief revision when contradictory information arrives."""
    print_separator("1. Belief Revision with Contradictory Information")
    
    # Add initial knowledge
    print("Adding initial knowledge...")
    manager.remember_knowledge(
        "Mount Everest is 8,848 meters tall.", 
        source="geography_textbook",
        confidence=0.9
    )
    
    print("Waiting 2 seconds before adding contradictory information...")
    time.sleep(2)
    
    # Add contradictory information with a different source
    print("Adding contradictory information...")
    manager.remember_knowledge(
        "Mount Everest is 8,849 meters tall.", 
        source="recent_survey",
        confidence=0.85
    )
    
    # Auto-detection and resolution should happen after adding the knowledge
    
    # Check the current state
    print("Checking the current state after auto-resolution...")
    contradictions = manager.detect_contradictions()
    print_result("Remaining Contradictions", contradictions)
    
    # Get the revision history
    revisions = manager.get_belief_revision_history()
    print_result("Belief Revision History", revisions)
    
    # Test different resolution strategies
    print("\nTesting different resolution strategies manually...")
    
    # Add another set of contradictory information
    print("Adding new contradictory pair...")
    manager.remember_knowledge(
        "The Nile River is 6,650 kilometers long.", 
        source="old_encyclopedia",
        confidence=0.7
    )
    
    time.sleep(1)
    
    manager.remember_knowledge(
        "The Nile River is 6,695 kilometers long.", 
        source="geographic_society",
        confidence=0.8
    )
    
    # Find contradictions 
    contradictions = manager.detect_contradictions()
    
    # If we have a contradiction, demonstrate resolving it with different strategies
    if contradictions:
        contradiction = contradictions[0]
        
        # Resolve with recency
        recency_result = manager.resolve_contradiction(contradiction, "recency")
        print_result("Resolution with Recency", recency_result)
        
        # Resolve with confidence
        confidence_result = manager.resolve_contradiction(contradiction, "confidence")
        print_result("Resolution with Confidence", confidence_result)
        
        # Resolve with weighted approach
        weighted_result = manager.resolve_contradiction(contradiction, "weighted")
        print_result("Resolution with Weighted Approach", weighted_result)
    else:
        print("No contradictions found to demonstrate different resolution strategies.")

def demonstrate_uncertainty_representation(manager: CortexFlowManager):
    """Demonstrate explicit uncertainty representation."""
    print_separator("2. Explicit Uncertainty Representation")
    
    # Add knowledge with different confidence levels
    print("Adding knowledge with different confidence levels...")
    manager.remember_knowledge(
        "Jupiter has 79 known moons.", 
        source="astronomy_website",
        confidence=0.9
    )
    
    manager.remember_knowledge(
        "The average surface temperature of Venus is around 462°C.", 
        source="nasa_data",
        confidence=0.95
    )
    
    manager.remember_knowledge(
        "There might be water on Mars.", 
        source="research_paper",
        confidence=0.6
    )
    
    # Add a probability distribution for uncertain values
    print("\nAdding probability distribution for an uncertain value...")
    
    # First add an entity to get its ID
    item_ids = manager.remember_knowledge(
        "The global average temperature will rise by 1.5-4.5°C by 2100.",
        source="climate_model",
        confidence=0.7
    )
    
    # Get the entity and relation IDs (this is simplified and depends on implementation)
    entity_id = 1  # This would normally be retrieved from the knowledge store
    relation_id = 1  # This would normally be retrieved from the knowledge store
    
    # Add a probability distribution for the temperature rise
    manager.add_probability_distribution(
        entity_id,
        relation_id,
        distribution_type="discrete",
        distribution_data={
            "values": ["1.5°C", "2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C", "4.5°C"],
            "probabilities": [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]
        }
    )
    
    # Retrieve and display the probability distribution
    distribution = manager.get_probability_distribution(entity_id, relation_id)
    print_result("Probability Distribution", distribution)

def demonstrate_source_reliability(manager: CortexFlowManager):
    """Demonstrate conflict resolution based on source reliability."""
    print_separator("3. Conflict Resolution based on Source Reliability")
    
    # Set up source reliability scores
    print("Setting up source reliability scores...")
    manager.update_source_reliability("scientific_journal", 0.95)
    manager.update_source_reliability("news_article", 0.6)
    manager.update_source_reliability("social_media", 0.3)
    manager.update_source_reliability("expert_opinion", 0.85)
    
    # Display reliability scores
    print("\nSource reliability scores:")
    print(f"  scientific_journal: {manager.get_source_reliability('scientific_journal')}")
    print(f"  news_article: {manager.get_source_reliability('news_article')}")
    print(f"  social_media: {manager.get_source_reliability('social_media')}")
    print(f"  expert_opinion: {manager.get_source_reliability('expert_opinion')}")
    
    # Add contradictory information from sources with different reliability
    print("\nAdding contradictory information from different sources...")
    manager.remember_knowledge(
        "Coffee consumption reduces the risk of heart disease.", 
        source="scientific_journal",
        confidence=0.8
    )
    
    time.sleep(1)
    
    manager.remember_knowledge(
        "Coffee consumption increases the risk of heart disease.", 
        source="social_media",
        confidence=0.7
    )
    
    # The system should automatically prefer the scientific journal's information
    
    # Check if any contradictions remain
    contradictions = manager.detect_contradictions()
    print_result("Remaining Contradictions", contradictions)
    
    # Get the revision history
    revisions = manager.get_belief_revision_history()
    print_result("Belief Revision History (most recent first)", revisions[:3])

def demonstrate_incomplete_information(manager: CortexFlowManager):
    """Demonstrate reasoning with incomplete information."""
    print_separator("4. Reasoning with Incomplete Information")
    
    # Set up a query that requires multiple pieces of information
    query = {
        "question": "What is the capital of France and its population?",
        "required_fields": ["capital", "population"]
    }
    
    # Case 1: Complete information available
    print("Case 1: Complete information available")
    complete_knowledge = [
        {
            "capital": "Paris",
            "population": "2.2 million",
            "answer": "The capital of France is Paris with a population of 2.2 million in the city proper.",
            "confidence": 0.9,
            "source": "geographic_database"
        }
    ]
    
    result = manager.reason_with_incomplete_information(query, complete_knowledge)
    print_result("Reasoning Result (Complete Information)", result)
    
    # Case 2: Partial information available
    print("\nCase 2: Partial information available")
    partial_knowledge = [
        {
            "capital": "Paris",
            "answer": "The capital of France is Paris.",
            "confidence": 0.95,
            "source": "geographic_database"
        },
        {
            "country": "France",
            "continent": "Europe",
            "answer": "France is a country in Western Europe.",
            "confidence": 0.9,
            "source": "geographic_database"
        }
    ]
    
    result = manager.reason_with_incomplete_information(query, partial_knowledge)
    print_result("Reasoning Result (Partial Information)", result)
    
    # Case 3: Very limited information
    print("\nCase 3: Very limited information")
    limited_knowledge = [
        {
            "continent": "Europe",
            "answer": "France is in Europe.",
            "confidence": 0.9,
            "source": "geographic_database"
        }
    ]
    
    result = manager.reason_with_incomplete_information(query, limited_knowledge)
    print_result("Reasoning Result (Limited Information)", result)

def main():
    """Main function to run the demonstration."""
    # Initialize with uncertainty handling enabled
    config = CortexFlowConfig(
        use_uncertainty_handling=True,
        auto_detect_contradictions=True,
        default_contradiction_strategy="weighted",
        recency_weight=0.6,
        reliability_weight=0.4,
        confidence_threshold=0.7,
        uncertainty_representation="confidence",
        reason_with_incomplete_info=True,
        knowledge_store_path=":memory:"  # Use in-memory database for the demo
    )
    
    manager = CortexFlowManager(config)
    
    try:
        # Demonstrate belief revision with contradictory information
        demonstrate_belief_revision(manager)
        
        # Demonstrate explicit uncertainty representation
        demonstrate_uncertainty_representation(manager)
        
        # Demonstrate source reliability for conflict resolution
        demonstrate_source_reliability(manager)
        
        # Demonstrate reasoning with incomplete information
        demonstrate_incomplete_information(manager)
        
    finally:
        # Clean up
        manager.close()
    
    print_separator("End of Demonstration")
    print("The CortexFlow system now includes robust mechanisms for:")
    print("1. Revising beliefs when contradictory information arrives")
    print("2. Representing uncertainty with confidence scores and probability distributions")  
    print("3. Resolving conflicts based on source reliability and recency")
    print("4. Reasoning with incomplete information")

if __name__ == "__main__":
    main() 