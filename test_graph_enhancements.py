#!/usr/bin/env python
"""
Test script for CortexFlow enhanced entity recognition and relation extraction.
"""

import logging
from cortexflow.config import CortexFlowConfig
from cortexflow.graph_store import GraphStore, RelationExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_recognition():
    """Test enhanced entity recognition capabilities."""
    config = CortexFlowConfig()
    # Use in-memory database for testing
    config.knowledge_store_path = ":memory:"
    
    graph = GraphStore(config)
    
    # Test text with various entity types
    text = """
    Albert Einstein developed the theory of relativity while working in Berlin.
    Apple Inc. is headquartered in Cupertino and was founded by Steve Jobs and Steve Wozniak.
    Python is a popular programming language used in machine learning and data science.
    The film Inception, directed by Christopher Nolan, was released on July 16, 2010.
    """
    
    # Extract entities
    entities = graph.extract_entities(text)
    
    # Log results
    logger.info(f"Found {len(entities)} entities:")
    
    # Group entities by source
    sources = {}
    for entity in entities:
        source = entity.get('source', 'unknown')
        sources.setdefault(source, []).append(entity)
    
    # Log counts by source
    for source, source_entities in sources.items():
        logger.info(f"  - {source}: {len(source_entities)} entities")
        for entity in source_entities[:3]:  # Show first 3 examples
            logger.info(f"    - {entity['text']} ({entity['type']})")
    
    return len(entities) > 0

def test_entity_linking():
    """Test entity linking and fuzzy matching."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"
    
    graph = GraphStore(config)
    
    # Add canonical entities
    einstein_id = graph.add_entity(
        entity="Albert Einstein",
        entity_type="PERSON",
        metadata={
            "aliases": ["Einstein", "A. Einstein"],
            "birth_year": 1879,
            "death_year": 1955
        }
    )
    
    python_id = graph.add_entity(
        entity="Python",
        entity_type="PROGRAMMING_LANGUAGE",
        metadata={
            "creator": "Guido van Rossum",
            "year": 1991
        }
    )
    
    # Add an alias
    graph.add_entity_alias(einstein_id, "Father of Relativity")
    
    # Test text with variations of the entities
    text = """
    Einstein's work on relativity revolutionized physics.
    The Father of Relativity received the Nobel Prize in 1921.
    Many data scientists prefer Python programming for AI development.
    """
    
    # Extract entities with linking
    entities = graph.extract_entities(text)
    
    # Check for linked entities
    linked_entities = [e for e in entities if e.get('linked', False)]
    
    logger.info(f"Found {len(linked_entities)} linked entities:")
    for entity in linked_entities:
        canonical = entity.get('canonical', entity['text'])
        logger.info(f"  - '{entity['text']}' linked to canonical entity '{canonical}'")
    
    return len(linked_entities) > 0

def test_relation_extraction():
    """Test enhanced relation extraction capabilities."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"
    
    graph = GraphStore(config)
    
    # Test text with various relation types
    text = """
    Albert Einstein developed the theory of relativity.
    Marie Curie discovered radium and polonium.
    Alan Turing invented the Turing machine.
    Amazon is headquartered in Seattle.
    Bill Gates founded Microsoft with Paul Allen.
    """
    
    # Extract relations
    relations = graph.extract_relations(text)
    
    logger.info(f"Found {len(relations)} relations:")
    for i, (subj, pred, obj) in enumerate(relations):
        logger.info(f"  {i+1}. {subj} - {pred} - {obj}")
    
    # Test the standalone relation extractor
    extractor = RelationExtractor()
    more_relations = extractor.extract_relations(text)
    
    logger.info(f"RelationExtractor found {len(more_relations)} relations")
    
    return len(relations) > 0

def test_process_text_to_graph():
    """Test end-to-end processing of text to graph."""
    config = CortexFlowConfig()
    config.knowledge_store_path = ":memory:"
    
    graph = GraphStore(config)
    
    # Test text
    text = """
    Albert Einstein developed the theory of relativity while working in Berlin.
    He was born in Ulm, Germany and won the Nobel Prize in Physics in 1921.
    """
    
    # Process text to graph
    relations_added = graph.process_text_to_graph(text, source="test")
    
    logger.info(f"Added {relations_added} relations to the graph")
    
    # Query the graph to verify
    query = "Albert Einstein"
    subgraph = graph.build_knowledge_subgraph(query)
    
    logger.info(f"Subgraph for '{query}' has {len(subgraph['nodes'])} nodes and {len(subgraph['edges'])} edges")
    
    return relations_added > 0

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Entity Recognition", test_entity_recognition),
        ("Entity Linking", test_entity_linking),
        ("Relation Extraction", test_relation_extraction),
        ("Process Text to Graph", test_process_text_to_graph)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n{'='*50}\nRunning test: {name}\n{'='*50}")
        try:
            success = test_func()
            results.append((name, success))
            logger.info(f"Test '{name}' {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test '{name}' ERROR: {e}")
            results.append((name, False))
    
    # Print summary with very clear markers
    logger.info("\n\n")
    logger.info("*" * 80)
    logger.info("*" + " " * 30 + "TEST SUMMARY" + " " * 30 + "*")
    logger.info("*" * 80)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"* {name:30} {status:10} *")
    logger.info("*" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"* OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%) {'SUCCESS' if passed == total else 'FAILURE'} *")
    logger.info("*" * 80)
    logger.info("\n\n")
    
    # Also print to stdout to ensure visibility
    print("\n\n")
    print("*" * 80)
    print("*" + " " * 30 + "TEST SUMMARY" + " " * 30 + "*")
    print("*" * 80)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"* {name:30} {status:10} *")
    print("*" * 80)
    print(f"* OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%) {'SUCCESS' if passed == total else 'FAILURE'} *")
    print("*" * 80)
    print("\n\n")

if __name__ == "__main__":
    run_all_tests() 