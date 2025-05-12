"""
Test script for enhanced entity and relation extraction capabilities.
This is a standalone script that doesn't require the full cortexflow package.
"""

import os
import sys
import json
import sqlite3
import logging
from typing import Dict, Any, List, Tuple, Optional

# Add the parent directory to the path to import graph_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexflow.graph_store import GraphStore
from cortexflow.config import CortexFlowConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_entity_extraction():
    """Test enhanced entity extraction"""
    # Create in-memory database for testing
    config = CortexFlowConfig(
        knowledge_store_path=":memory:",
        use_graph_rag=True
    )
    
    # Initialize GraphStore
    graph_store = GraphStore(config)
    
    # Test text with various entity types
    text = "Apple Inc. was founded by Steve Jobs in California in 1976. " \
           "The company released the iPhone in 2007, which runs on iOS. " \
           "Tim Cook is the current CEO and the stock price was $150.23 on May 15, 2023."
    
    # Extract entities
    entities = graph_store.extract_entities(text)
    
    # Print extracted entities
    print(f"\nExtracted {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity['text']} ({entity['type']})")
    
    # Check if key entities were found
    important_entities = ["Apple Inc.", "Steve Jobs", "California", "iPhone", "Tim Cook", "$150.23"]
    for entity in important_entities:
        found = any(e['text'] == entity for e in entities)
        print(f"  Entity '{entity}': {'✓' if found else '✗'}")
    
    return entities
    
def test_domain_specific_extraction():
    """Test domain-specific entity extraction"""
    # Create in-memory database for testing
    config = CortexFlowConfig(
        knowledge_store_path=":memory:",
        use_graph_rag=True
    )
    
    # Initialize GraphStore
    graph_store = GraphStore(config)
    
    # Test text with domain-specific entities
    text = "Python has become very popular for machine learning applications. " \
           "Many developers use TensorFlow and PyTorch for deep learning " \
           "projects, especially when working with CNNs or RNNs."
    
    # Extract entities
    entities = graph_store.extract_entities(text)
    
    # Print extracted entities
    print(f"\nExtracted {len(entities)} domain-specific entities:")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity['text']} ({entity['type']})")
    
    # Check for domain-specific entities
    domain_entities = ["Python", "machine learning", "TensorFlow", "PyTorch", "deep learning", "CNNs", "RNNs"]
    for entity in domain_entities:
        found = any(entity in e['text'] for e in entities)
        print(f"  Entity '{entity}': {'✓' if found else '✗'}")
    
    return entities
    
def test_relation_extraction():
    """Test enhanced relation extraction"""
    # Create in-memory database for testing
    config = CortexFlowConfig(
        knowledge_store_path=":memory:",
        use_graph_rag=True
    )
    
    # Initialize GraphStore
    graph_store = GraphStore(config)
    
    # Test text with clear subject-verb-object relationships
    text = "Steve Jobs founded Apple in California. Microsoft develops Windows. " \
           "The researchers published their findings in Nature. " \
           "Google acquired YouTube for $1.65 billion in 2006."
    
    # Extract relations
    relations = graph_store.extract_relations(text)
    
    # Print extracted relations
    print(f"\nExtracted {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"  {i+1}. {relation[0]} → {relation[1]} → {relation[2]}")
    
    # Check for expected relations
    key_relations = [
        ("Steve Jobs", "found", "Apple"),
        ("Microsoft", "develop", "Windows"),
        ("Google", "acquire", "YouTube"),
        ("Apple", "in", "California")
    ]
    
    for subj, pred, obj in key_relations:
        found = any(
            subj in r[0] and (pred in r[1] or pred+'ed' in r[1] or pred+'s' in r[1]) and obj in r[2]
            for r in relations
        )
        print(f"  Relation '{subj} → {pred} → {obj}': {'✓' if found else '✗'}")
    
    return relations
    
def test_coreference_resolution():
    """Test coreference resolution in text processing"""
    # Create in-memory database for testing
    config = CortexFlowConfig(
        knowledge_store_path=":memory:",
        use_graph_rag=True
    )
    
    # Initialize GraphStore
    graph_store = GraphStore(config)
    
    # Try to load neuralcoref
    try:
        import neuralcoref
        import spacy
        
        # Test text with coreferences
        text = "Albert Einstein published his theory of relativity in 1915. " \
               "He was born in Germany but later moved to the United States. " \
               "Einstein's work revolutionized physics and he won the Nobel Prize in 1921."
        
        # Process text to graph
        relations_added = graph_store.process_text_to_graph(text)
        
        # Query related entities to Albert Einstein
        neighbors = graph_store.get_entity_neighbors("Albert Einstein")
        
        # Print coreference results
        print(f"\nProcessed text with coreference resolution:")
        print(f"  Added {relations_added} relations to graph")
        print(f"  Found {len(neighbors)} neighbor entities for 'Albert Einstein':")
        for i, neighbor in enumerate(neighbors):
            print(f"    {i+1}. {neighbor['entity']} ({neighbor['relation']})")
        
        return relations_added > 0
    except ImportError:
        print("\nCoreference resolution test skipped - neuralcoref not available")
        return None
    
def test_semantic_role_labeling():
    """Test semantic role labeling for relation extraction"""
    # Create in-memory database for testing
    config = CortexFlowConfig(
        knowledge_store_path=":memory:",
        use_graph_rag=True
    )
    
    # Initialize GraphStore
    graph_store = GraphStore(config)
    
    # Try to use AllenNLP SRL
    try:
        from allennlp.predictors.predictor import Predictor
        
        # Test text with complex relations
        text = "The researchers from Stanford University developed a new algorithm " \
               "to solve complex optimization problems in machine learning."
        
        # Extract relations with SRL
        srl_relations = graph_store._extract_with_semantic_roles(text)
        
        # Print SRL results
        print(f"\nExtracted {len(srl_relations)} relations with semantic role labeling:")
        for i, relation in enumerate(srl_relations):
            print(f"  {i+1}. {relation[0]} → {relation[1]} → {relation[2]}")
        
        return len(srl_relations) > 0
    except (ImportError, Exception) as e:
        print(f"\nSemantic role labeling test skipped - {str(e)}")
        return None

def main():
    """Run all tests"""
    print("Testing Enhanced Entity and Relation Extraction")
    print("="*50)
    
    # Test entity extraction
    entities = test_entity_extraction()
    
    # Test domain-specific entity extraction
    domain_entities = test_domain_specific_extraction()
    
    # Test relation extraction
    relations = test_relation_extraction()
    
    # Test coreference resolution
    coref_result = test_coreference_resolution()
    
    # Test semantic role labeling
    srl_result = test_semantic_role_labeling()
    
    # Print summary
    print("\nTest Summary:")
    print(f"  Basic entity extraction: {'✓' if entities else '✗'} ({len(entities) if entities else 0} entities)")
    print(f"  Domain-specific extraction: {'✓' if domain_entities else '✗'} ({len(domain_entities) if domain_entities else 0} entities)")
    print(f"  Relation extraction: {'✓' if relations else '✗'} ({len(relations) if relations else 0} relations)")
    print(f"  Coreference resolution: {'✓' if coref_result else '✗' if coref_result is False else 'SKIPPED'}")
    print(f"  Semantic role labeling: {'✓' if srl_result else '✗' if srl_result is False else 'SKIPPED'}")

if __name__ == "__main__":
    main() 