"""
Tests for the enhanced knowledge graph structure with ontology and n-ary relationships.

Tests the core functionality of:
- Flexible ontology system
- Relation typing with inheritance
- Metadata framework for tracking provenance, confidence, and temporal information
"""

import os
import pytest
import time
from datetime import datetime

from cortexflow.config import CortexFlowConfig
from cortexflow.ontology import Ontology, OntologyClass, RelationType


@pytest.fixture
def ontology_db_path(tmp_path):
    """Provide a temporary database path for ontology tests."""
    return str(tmp_path / "test_ontology.db")


@pytest.fixture
def ontology(ontology_db_path):
    """Create an Ontology instance with a temporary database."""
    onto = Ontology(ontology_db_path)
    yield onto
    onto.close()


class TestOntologySystem:
    """Test the flexible ontology system."""

    def test_basic_initialization(self, ontology):
        """Test that ontology initializes with basic classes."""
        person_class = ontology.get_class("Person")
        assert person_class is not None, "Person class should exist after initialization"

    def test_add_subclass(self, ontology):
        """Test adding a subclass with properties and metadata."""
        scientist = OntologyClass(
            name="Scientist",
            parent_classes=["Person"],
            properties={
                "field": "string",
                "publications": "integer"
            },
            metadata={"domain": "academic"}
        )
        result = ontology.add_class(scientist)
        assert result is True, "Adding Scientist class should succeed"

    def test_subclass_relationship(self, ontology):
        """Test that subclass relationships are correctly established."""
        scientist = OntologyClass(
            name="Scientist",
            parent_classes=["Person"],
            properties={"field": "string", "publications": "integer"},
            metadata={"domain": "academic"}
        )
        ontology.add_class(scientist)

        assert ontology.is_subclass_of("Scientist", "Person"), \
            "Scientist should be a subclass of Person"

    def test_relation_type_inheritance(self, ontology):
        """Test relation type inheritance."""
        # First add Scientist class
        scientist = OntologyClass(
            name="Scientist",
            parent_classes=["Person"],
            properties={"field": "string", "publications": "integer"},
        )
        ontology.add_class(scientist)

        works_with = RelationType(
            name="collaborates_with",
            parent_types=["knows"],
            source_classes=["Scientist"],
            target_classes=["Scientist"],
            cardinality="many-to-many",
            properties={"project": "string", "start_date": "date"}
        )
        ontology.add_relation_type(works_with)

        assert ontology.is_subtype_of("collaborates_with", "knows"), \
            "collaborates_with should be a subtype of knows"

    def test_multi_level_inheritance(self, ontology):
        """Test multi-level class inheritance (Person -> Scientist -> Physicist)."""
        scientist = OntologyClass(
            name="Scientist",
            parent_classes=["Person"],
            properties={"field": "string"}
        )
        ontology.add_class(scientist)

        physicist = OntologyClass(
            name="Physicist",
            parent_classes=["Scientist"],
            properties={"specialization": "string"}
        )
        ontology.add_class(physicist)

        all_person_subclasses = ontology.get_all_subclasses("Person")
        assert "Scientist" in all_person_subclasses, \
            "Scientist should be in Person's subclasses"
        assert "Physicist" in all_person_subclasses, \
            "Physicist should be in Person's subclasses"

    def test_is_subclass_of_self(self, ontology):
        """Test that a class is considered a subclass of itself."""
        assert ontology.is_subclass_of("Person", "Person"), \
            "Person should be a subclass of itself"

    def test_is_subclass_of_nonexistent(self, ontology):
        """Test subclass check with a nonexistent class."""
        assert not ontology.is_subclass_of("NonExistent", "Person"), \
            "NonExistent should not be a subclass of Person"

    def test_get_class_returns_none_for_unknown(self, ontology):
        """Test that get_class returns None for unknown class names."""
        result = ontology.get_class("CompletelyUnknownClass")
        assert result is None

    def test_delete_class(self, ontology):
        """Test deleting a class from the ontology."""
        custom = OntologyClass(name="CustomClass", parent_classes=["Thing"])
        ontology.add_class(custom)
        assert ontology.get_class("CustomClass") is not None

        result = ontology.delete_class("CustomClass")
        assert result is True
        assert ontology.get_class("CustomClass") is None

    def test_delete_nonexistent_class(self, ontology):
        """Test deleting a class that does not exist."""
        result = ontology.delete_class("DoesNotExist")
        assert result is False


class TestMetadataStructure:
    """Test the metadata structure implementation."""

    def test_ontology_class_metadata_persistence(self):
        """Test metadata is correctly stored in OntologyClass."""
        test_metadata = {
            "source": "test",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat()
        }
        test_class = OntologyClass(name="TestClass", metadata=test_metadata)

        assert test_class.metadata.get("source") == "test", \
            "Metadata 'source' should persist"
        assert test_class.metadata.get("confidence") == 0.9, \
            "Metadata 'confidence' should persist"

    def test_ontology_class_time_tracking(self):
        """Test that OntologyClass tracks creation and modification times."""
        test_class = OntologyClass(name="TestClass")
        assert hasattr(test_class, "creation_time"), \
            "OntologyClass should have creation_time"
        assert hasattr(test_class, "last_modified_time"), \
            "OntologyClass should have last_modified_time"
        assert test_class.creation_time <= time.time()
        assert test_class.last_modified_time <= time.time()

    def test_relation_type_metadata_persistence(self):
        """Test metadata is correctly stored in RelationType."""
        test_relation = RelationType(
            name="test_relation",
            metadata={"provenance": "manual", "confidence": 0.85}
        )

        assert test_relation.metadata.get("provenance") == "manual", \
            "RelationType metadata 'provenance' should persist"
        assert test_relation.metadata.get("confidence") == 0.85, \
            "RelationType metadata 'confidence' should persist"

    def test_ontology_class_to_dict(self):
        """Test OntologyClass serialization to dict."""
        cls = OntologyClass(
            name="Foo",
            parent_classes=["Thing"],
            properties={"bar": "string"},
            metadata={"info": "test"}
        )
        d = cls.to_dict()
        assert d["name"] == "Foo"
        assert d["parent_classes"] == ["Thing"]
        assert d["properties"] == {"bar": "string"}
        assert d["metadata"] == {"info": "test"}
        assert "creation_time" in d
        assert "last_modified_time" in d

    def test_ontology_class_from_dict(self):
        """Test OntologyClass deserialization from dict."""
        original = OntologyClass(
            name="Bar",
            parent_classes=["Thing"],
            properties={"baz": "integer"},
        )
        data = original.to_dict()
        restored = OntologyClass.from_dict(data)

        assert restored.name == "Bar"
        assert restored.parent_classes == ["Thing"]
        assert restored.properties == {"baz": "integer"}

    def test_relation_type_to_dict(self):
        """Test RelationType serialization to dict."""
        rt = RelationType(
            name="works_at",
            parent_types=["related_to"],
            source_classes=["Person"],
            target_classes=["Organization"],
            cardinality="many-to-one",
        )
        d = rt.to_dict()
        assert d["name"] == "works_at"
        assert d["cardinality"] == "many-to-one"
        assert d["source_classes"] == ["Person"]

    def test_relation_type_from_dict(self):
        """Test RelationType deserialization from dict."""
        original = RelationType(
            name="lives_in",
            parent_types=["located_in"],
            cardinality="many-to-one",
        )
        data = original.to_dict()
        restored = RelationType.from_dict(data)

        assert restored.name == "lives_in"
        assert restored.parent_types == ["located_in"]
        assert restored.cardinality == "many-to-one"


class TestConfigStructure:
    """Test the configuration structure for the enhanced knowledge graph."""

    def test_config_has_ontology_settings(self):
        """Test that CortexFlowConfig has ontology settings via proxy."""
        config = CortexFlowConfig()
        assert hasattr(config, "use_ontology"), "Config should have use_ontology"
        assert hasattr(config, "enable_ontology_evolution"), \
            "Config should have enable_ontology_evolution"

    def test_config_has_metadata_framework_settings(self):
        """Test that CortexFlowConfig has metadata framework settings."""
        config = CortexFlowConfig()
        assert hasattr(config, "track_provenance"), \
            "Config should have track_provenance"
        assert hasattr(config, "track_confidence"), \
            "Config should have track_confidence"
        assert hasattr(config, "track_temporal"), \
            "Config should have track_temporal"

    def test_config_serialization_includes_ontology_fields(self):
        """Test that config serialization includes ontology and metadata fields."""
        config = CortexFlowConfig()
        config_dict = config.to_dict()
        assert "use_ontology" in config_dict, \
            "Serialized config should include use_ontology"
        assert "track_provenance" in config_dict, \
            "Serialized config should include track_provenance"

    def test_config_default_values(self):
        """Test that ontology and metadata config have expected defaults."""
        config = CortexFlowConfig()
        assert config.use_ontology is False
        assert config.enable_ontology_evolution is True
        assert config.track_provenance is True
        assert config.track_confidence is True
        assert config.track_temporal is True
