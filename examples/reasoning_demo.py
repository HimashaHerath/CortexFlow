#!/usr/bin/env python
"""
CortexFlow Reasoning Demo

This script demonstrates the reasoning capabilities of the CortexFlow inference engine.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexflow.config import CortexFlowConfig
from cortexflow.manager import CortexFlowManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_knowledge_graph():
    """Set up a sample knowledge graph with facts for demonstration."""
    # Create configuration with inference engine enabled
    config = CortexFlowConfig(
        use_graph_rag=True,
        use_inference_engine=True,
        max_inference_depth=5,
        inference_confidence_threshold=0.6,
        abductive_reasoning_enabled=True
    )
    
    # Initialize CortexFlow manager
    cf = CortexFlowManager(config)
    
    # Add sample entities and relations
    graph_store = cf.knowledge_store.graph_store
    
    # Animal taxonomy example
    graph_store.add_entity("animal", "category", confidence=1.0)
    graph_store.add_entity("mammal", "category", confidence=1.0)
    graph_store.add_entity("bird", "category", confidence=1.0)
    graph_store.add_entity("reptile", "category", confidence=1.0)
    
    graph_store.add_entity("dog", "animal_species", confidence=1.0)
    graph_store.add_entity("cat", "animal_species", confidence=1.0)
    graph_store.add_entity("eagle", "animal_species", confidence=1.0)
    graph_store.add_entity("snake", "animal_species", confidence=1.0)
    graph_store.add_entity("crocodile", "animal_species", confidence=1.0)
    
    graph_store.add_entity("fur", "feature", confidence=1.0)
    graph_store.add_entity("feathers", "feature", confidence=1.0)
    graph_store.add_entity("scales", "feature", confidence=1.0)
    graph_store.add_entity("warm_blooded", "trait", confidence=1.0)
    graph_store.add_entity("cold_blooded", "trait", confidence=1.0)
    
    # Add taxonomy relations
    graph_store.add_relation("mammal", "is_a", "animal", confidence=1.0)
    graph_store.add_relation("bird", "is_a", "animal", confidence=1.0)
    graph_store.add_relation("reptile", "is_a", "animal", confidence=1.0)
    
    graph_store.add_relation("dog", "is_a", "mammal", confidence=1.0)
    graph_store.add_relation("cat", "is_a", "mammal", confidence=1.0)
    graph_store.add_relation("eagle", "is_a", "bird", confidence=1.0)
    graph_store.add_relation("snake", "is_a", "reptile", confidence=1.0)
    graph_store.add_relation("crocodile", "is_a", "reptile", confidence=1.0)
    
    # Add feature relations
    graph_store.add_relation("mammal", "has_property", "fur", confidence=0.9)
    graph_store.add_relation("bird", "has_property", "feathers", confidence=0.95)
    graph_store.add_relation("reptile", "has_property", "scales", confidence=0.9)
    
    graph_store.add_relation("mammal", "has_property", "warm_blooded", confidence=1.0)
    graph_store.add_relation("bird", "has_property", "warm_blooded", confidence=1.0)
    graph_store.add_relation("reptile", "has_property", "cold_blooded", confidence=1.0)
    
    # Add more custom logical rules
    inference_engine = cf.knowledge_store.inference_engine
    
    # Rule for "can_fly" property
    inference_engine.add_rule(
        name="birds_can_fly",
        premise=[
            {"source": "?X", "relation": "is_a", "target": "bird"}
        ],
        conclusion={"source": "?X", "relation": "can_fly", "target": "true"},
        confidence=0.8,
        metadata={"category": "ability"}
    )
    
    # Rule for "vertebrate" classification
    inference_engine.add_rule(
        name="animals_are_vertebrates",
        premise=[
            {"source": "?X", "relation": "is_a", "target": "animal"}
        ],
        conclusion={"source": "?X", "relation": "is_a", "target": "vertebrate"},
        confidence=0.85,
        metadata={"category": "classification"}
    )
    
    return cf

def demonstrate_backward_chaining(cf: CortexFlowManager):
    """Demonstrate backward chaining for "why" questions."""
    print("\n" + "="*50)
    print("BACKWARD CHAINING DEMO: Answering 'Why' Questions")
    print("="*50)
    
    why_questions = [
        "Why is a dog an animal?",
        "Why does a dog have fur?",
        "Why can an eagle fly?",
        "Why is a snake cold blooded?"
    ]
    
    for question in why_questions:
        print(f"\nQuestion: {question}")
        explanation = cf.answer_why_question(question)
        
        if explanation:
            print("Explanation:")
            for step in explanation:
                if "error" in step:
                    print(f"  Error: {step['error']}")
                else:
                    step_type = step.get("type", "")
                    message = step.get("message", "")
                    confidence = step.get("confidence", 0)
                    
                    print(f"  [{step_type}] {message} (Confidence: {confidence:.2f})")
        else:
            print("  No explanation found.")

def demonstrate_forward_chaining(cf: CortexFlowManager):
    """Demonstrate forward chaining to discover novel implications."""
    print("\n" + "="*50)
    print("FORWARD CHAINING DEMO: Discovering Novel Implications")
    print("="*50)
    
    print("\nRunning forward chaining to derive new facts...")
    inferred_facts = cf.generate_novel_implications(iterations=2)
    
    if inferred_facts:
        print(f"Discovered {len(inferred_facts)} new facts:")
        for i, fact in enumerate(inferred_facts, 1):
            source = fact.get("source", "")
            relation = fact.get("relation", "")
            target = fact.get("target", "")
            rule = fact.get("rule", "")
            confidence = fact.get("confidence", 0)
            
            print(f"  {i}. {source} {relation} {target} (Rule: {rule}, Confidence: {confidence:.2f})")
    else:
        print("  No new facts were inferred.")

def demonstrate_abductive_reasoning(cf: CortexFlowManager):
    """Demonstrate abductive reasoning for hypothesis generation."""
    print("\n" + "="*50)
    print("ABDUCTIVE REASONING DEMO: Generating Hypotheses")
    print("="*50)
    
    observations = [
        "Eagles have wings",
        "Mammals give birth to live young",
        "Reptiles lay eggs",
        "Cats have retractable claws"
    ]
    
    for observation in observations:
        print(f"\nObservation: {observation}")
        hypotheses = cf.generate_hypotheses(observation)
        
        if hypotheses:
            print("Possible explanations:")
            for i, hypothesis in enumerate(hypotheses, 1):
                text = hypothesis.get("text", "")
                confidence = hypothesis.get("confidence", 0)
                is_known = hypothesis.get("is_known", False)
                reasoning = hypothesis.get("reasoning_path", "")
                
                known_status = "Known fact" if is_known else "Novel hypothesis"
                print(f"  {i}. {text} ({known_status}, Confidence: {confidence:.2f})")
                print(f"     Reasoning: {reasoning}")
        else:
            print("  No hypotheses generated.")

def main():
    """Run the reasoning demo."""
    print("Initializing CortexFlow with sample knowledge graph...")
    cf = setup_knowledge_graph()
    
    # Run the demonstrations
    demonstrate_backward_chaining(cf)
    demonstrate_forward_chaining(cf)
    demonstrate_abductive_reasoning(cf)
    
    print("\nReasoning demo completed.")

if __name__ == "__main__":
    main() 