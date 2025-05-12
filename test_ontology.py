"""
Test script for the enhanced knowledge graph structure with ontology and n-ary relationships.

This script tests the core functionality of:
- Flexible ontology system
- Relation typing with inheritance
- Metadata framework for tracking provenance, confidence, and temporal information
"""

import os
import logging
import json
import sys
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import CortexFlow modules
from cortexflow.config import CortexFlowConfig
from cortexflow.ontology import Ontology, OntologyClass, RelationType

def test_ontology_system():
    """Test the flexible ontology system."""
    logging.info("Testing ontology system...")
    
    try:
        # Create a temporary database path for ontology
        test_db_path = f"test_onto_{int(datetime.now().timestamp())}.db"
        
        # Initialize ontology
        ontology = Ontology(test_db_path)
        
        # Test class creation and inheritance
        person_class = ontology.get_class("Person")
        if not person_class:
            logging.error("Basic ontology initialization failed - Person class not found")
            return False
        
        # Add a more specific class
        scientist = OntologyClass(
            name="Scientist",
            parent_classes=["Person"],
            properties={
                "field": "string",
                "publications": "integer"
            },
            metadata={"domain": "academic"}
        )
        
        ontology.add_class(scientist)
        
        # Verify subclass relationship
        if not ontology.is_subclass_of("Scientist", "Person"):
            logging.error("Subclass relationship failed - Scientist should be a subclass of Person")
            return False
        
        # Test relation type inheritance
        works_with = RelationType(
            name="collaborates_with",
            parent_types=["knows"],
            source_classes=["Scientist"],
            target_classes=["Scientist"],
            cardinality="many-to-many",
            properties={"project": "string", "start_date": "date"}
        )
        
        ontology.add_relation_type(works_with)
        
        # Verify relation type inheritance
        if not ontology.is_subtype_of("collaborates_with", "knows"):
            logging.error("Relation subtyping failed - collaborates_with should be a subtype of knows")
            return False
        
        # Test ontology evolution
        physicist = OntologyClass(
            name="Physicist",
            parent_classes=["Scientist"],
            properties={"specialization": "string"}
        )
        
        ontology.add_class(physicist)
        
        # Test getting all subclasses
        all_person_subclasses = ontology.get_all_subclasses("Person")
        if "Scientist" not in all_person_subclasses or "Physicist" not in all_person_subclasses:
            logging.error("Getting all subclasses failed")
            return False
        
        # Clean up the test database
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except:
            pass
            
        logging.info("Ontology system test passed!")
        return True
    except Exception as e:
        logging.error(f"Error in ontology system test: {e}")
        return False

def test_metadata_structure():
    """Test the metadata structure implementation."""
    logging.info("Testing metadata structure...")
    
    try:
        # Check metadata framework fields in OntologyClass
        test_metadata = {"source": "test", "confidence": 0.9, "timestamp": datetime.now().isoformat()}
        test_class = OntologyClass(
            name="TestClass",
            metadata=test_metadata
        )
        
        # Verify metadata persistence
        if test_class.metadata.get("source") != "test" or test_class.metadata.get("confidence") != 0.9:
            logging.error("Metadata persistence failed in OntologyClass")
            return False
            
        # Check creation time and last modified time
        if not hasattr(test_class, "creation_time") or not hasattr(test_class, "last_modified_time"):
            logging.error("Time tracking failed in OntologyClass")
            return False
            
        # Test RelationType metadata
        test_relation = RelationType(
            name="test_relation",
            metadata={"provenance": "manual", "confidence": 0.85}
        )
        
        if test_relation.metadata.get("provenance") != "manual" or test_relation.metadata.get("confidence") != 0.85:
            logging.error("Metadata persistence failed in RelationType")
            return False
            
        logging.info("Metadata structure test passed!")
        return True
    except Exception as e:
        logging.error(f"Error in metadata structure test: {e}")
        return False

def test_config_structure():
    """Test the configuration structure for the enhanced knowledge graph."""
    logging.info("Testing configuration structure...")
    
    try:
        # Create a config with all the new settings
        config = CortexFlowConfig()
        
        # Check ontology settings
        if not hasattr(config, "use_ontology"):
            logging.error("Missing use_ontology in config")
            return False
            
        if not hasattr(config, "enable_ontology_evolution"):
            logging.error("Missing enable_ontology_evolution in config")
            return False
            
        # Check metadata framework settings
        if not hasattr(config, "track_provenance"):
            logging.error("Missing track_provenance in config")
            return False
            
        if not hasattr(config, "track_confidence"):
            logging.error("Missing track_confidence in config")
            return False
            
        if not hasattr(config, "track_temporal"):
            logging.error("Missing track_temporal in config")
            return False
            
        # Test config serialization
        config_dict = config.to_dict()
        if "use_ontology" not in config_dict or "track_provenance" not in config_dict:
            logging.error("Missing fields in config serialization")
            return False
            
        logging.info("Configuration structure test passed!")
        return True
    except Exception as e:
        logging.error(f"Error in config structure test: {e}")
        return False

def main():
    """Run all tests."""
    logging.info("Starting enhanced knowledge graph structure tests...")
    
    # Run tests
    ontology_success = test_ontology_system()
    metadata_success = test_metadata_structure()
    config_success = test_config_structure()
    
    # Report results
    logging.info("\n----- Test Results -----")
    logging.info(f"Ontology System: {'PASS' if ontology_success else 'FAIL'}")
    logging.info(f"Metadata Structure: {'PASS' if metadata_success else 'FAIL'}")
    logging.info(f"Configuration Structure: {'PASS' if config_success else 'FAIL'}")
    
    # Return overall success
    return all([ontology_success, metadata_success, config_success])

if __name__ == "__main__":
    main() 