"""
CortexFlow Ontology Module.

This module provides a flexible ontology system for the knowledge graph that can evolve
based on new information. It supports:
- Concept hierarchy with subclass/superclass relationships
- Relationship types with inheritance
- Properties and constraints for entity types
- Dynamic schema evolution
"""

import logging
import json
import time
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple, Union

class OntologyClass:
    """Represents a class in the ontology with inheritance capabilities."""
    
    def __init__(self, name: str, parent_classes: List[str] = None, properties: Dict[str, Any] = None,
                 constraints: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """
        Initialize an ontology class.
        
        Args:
            name: The class name
            parent_classes: List of parent class names (for inheritance)
            properties: Dictionary of class properties and their types
            constraints: Dictionary of constraints on properties
            metadata: Additional metadata about the class
        """
        self.name = name
        self.parent_classes = parent_classes or []
        self.properties = properties or {}
        self.constraints = constraints or {}
        self.metadata = metadata or {}
        self.subclasses = []  # Will be populated when ontology is built
        self.creation_time = time.time()
        self.last_modified_time = self.creation_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the class to a dictionary for storage."""
        return {
            "name": self.name,
            "parent_classes": self.parent_classes,
            "properties": self.properties,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "last_modified_time": self.last_modified_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OntologyClass':
        """Create an OntologyClass instance from a dictionary."""
        instance = cls(
            name=data["name"],
            parent_classes=data.get("parent_classes", []),
            properties=data.get("properties", {}),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {})
        )
        instance.creation_time = data.get("creation_time", time.time())
        instance.last_modified_time = data.get("last_modified_time", instance.creation_time)
        return instance

class RelationType:
    """Represents a type of relationship in the ontology with inheritance capabilities."""
    
    def __init__(self, name: str, parent_types: List[str] = None, source_classes: List[str] = None,
                 target_classes: List[str] = None, cardinality: str = "many-to-many",
                 properties: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """
        Initialize a relationship type.
        
        Args:
            name: The relation type name
            parent_types: List of parent relation types (for inheritance)
            source_classes: List of valid source class names
            target_classes: List of valid target class names
            cardinality: Cardinality constraint ("one-to-one", "one-to-many", "many-to-one", "many-to-many")
            properties: Dictionary of relation properties and their types
            metadata: Additional metadata about the relation type
        """
        self.name = name
        self.parent_types = parent_types or []
        self.source_classes = source_classes or []
        self.target_classes = target_classes or []
        self.cardinality = cardinality
        self.properties = properties or {}
        self.metadata = metadata or {}
        self.subtypes = []  # Will be populated when ontology is built
        self.creation_time = time.time()
        self.last_modified_time = self.creation_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship type to a dictionary for storage."""
        return {
            "name": self.name,
            "parent_types": self.parent_types,
            "source_classes": self.source_classes,
            "target_classes": self.target_classes,
            "cardinality": self.cardinality,
            "properties": self.properties,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "last_modified_time": self.last_modified_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationType':
        """Create a RelationType instance from a dictionary."""
        instance = cls(
            name=data["name"],
            parent_types=data.get("parent_types", []),
            source_classes=data.get("source_classes", []),
            target_classes=data.get("target_classes", []),
            cardinality=data.get("cardinality", "many-to-many"),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {})
        )
        instance.creation_time = data.get("creation_time", time.time())
        instance.last_modified_time = data.get("last_modified_time", instance.creation_time)
        return instance

class Ontology:
    """Manages the knowledge graph ontology with schema evolution capabilities."""
    
    def __init__(self, db_path: str):
        """
        Initialize the ontology.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.classes: Dict[str, OntologyClass] = {}
        self.relation_types: Dict[str, RelationType] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}  # Maps class name to subclass names
        self.relation_hierarchy: Dict[str, List[str]] = {}  # Maps relation type to subtype names
        
        # Initialize database tables for ontology
        self._init_db()
        
        # Load existing ontology data
        self._load_ontology()
        
        # Build hierarchy maps
        self._build_hierarchy_maps()
    
    def _init_db(self):
        """Initialize the database tables for ontology."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create ontology_classes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ontology_classes (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    creation_time REAL,
                    last_modified_time REAL
                )
            ''')
            
            # Create ontology_relation_types table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ontology_relation_types (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    creation_time REAL,
                    last_modified_time REAL
                )
            ''')
            
            conn.commit()
            
        except sqlite3.Error as e:
            logging.error(f"Error initializing ontology database: {e}")
            
        finally:
            conn.close()
    
    def _load_ontology(self):
        """Load the existing ontology data from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load classes
            cursor.execute('SELECT name, data FROM ontology_classes')
            for row in cursor.fetchall():
                class_data = json.loads(row[1])
                self.classes[row[0]] = OntologyClass.from_dict(class_data)
            
            # Load relation types
            cursor.execute('SELECT name, data FROM ontology_relation_types')
            for row in cursor.fetchall():
                relation_data = json.loads(row[1])
                self.relation_types[row[0]] = RelationType.from_dict(relation_data)
                
            # If no ontology data exists, initialize with basic types
            if not self.classes:
                self._initialize_basic_ontology()
                
        except sqlite3.Error as e:
            logging.error(f"Error loading ontology data: {e}")
            
        finally:
            conn.close()
    
    def _initialize_basic_ontology(self):
        """Initialize the ontology with basic classes and relation types."""
        # Add basic entity classes
        basic_classes = [
            OntologyClass(name="Thing", properties={"name": "string"}),
            OntologyClass(name="Person", parent_classes=["Thing"], 
                        properties={"age": "integer", "occupation": "string"}),
            OntologyClass(name="Organization", parent_classes=["Thing"],
                        properties={"founded": "date", "industry": "string"}),
            OntologyClass(name="Location", parent_classes=["Thing"],
                        properties={"latitude": "float", "longitude": "float"}),
            OntologyClass(name="Concept", parent_classes=["Thing"],
                        properties={"definition": "string"}),
            OntologyClass(name="Event", parent_classes=["Thing"],
                        properties={"start_time": "datetime", "end_time": "datetime"})
        ]
        
        # Add basic relation types
        basic_relation_types = [
            RelationType(name="related_to", source_classes=["Thing"], target_classes=["Thing"]),
            RelationType(name="located_in", parent_types=["related_to"], 
                        source_classes=["Thing"], target_classes=["Location"]),
            RelationType(name="works_for", parent_types=["related_to"],
                        source_classes=["Person"], target_classes=["Organization"]),
            RelationType(name="knows", parent_types=["related_to"],
                        source_classes=["Person"], target_classes=["Person"]),
            RelationType(name="part_of", parent_types=["related_to"],
                        source_classes=["Thing"], target_classes=["Thing"])
        ]
        
        # Add all basic classes
        for cls in basic_classes:
            self.add_class(cls)
            
        # Add all basic relation types
        for rel_type in basic_relation_types:
            self.add_relation_type(rel_type)
    
    def _build_hierarchy_maps(self):
        """Build the hierarchy maps for classes and relation types."""
        # Reset hierarchy maps
        self.class_hierarchy = {cls_name: [] for cls_name in self.classes.keys()}
        self.relation_hierarchy = {rel_name: [] for rel_name in self.relation_types.keys()}
        
        # Build class hierarchy
        for cls_name, cls in self.classes.items():
            for parent_name in cls.parent_classes:
                if parent_name in self.classes:
                    self.class_hierarchy[parent_name].append(cls_name)
                    self.classes[parent_name].subclasses.append(cls_name)
        
        # Build relation type hierarchy
        for rel_name, rel_type in self.relation_types.items():
            for parent_name in rel_type.parent_types:
                if parent_name in self.relation_types:
                    self.relation_hierarchy[parent_name].append(rel_name)
                    self.relation_types[parent_name].subtypes.append(rel_name)
    
    def add_class(self, cls: OntologyClass) -> bool:
        """
        Add a new class to the ontology.
        
        Args:
            cls: The class to add
            
        Returns:
            True if class was added successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add class to database
            cursor.execute('''
                INSERT OR REPLACE INTO ontology_classes (name, data, creation_time, last_modified_time)
                VALUES (?, ?, ?, ?)
            ''', (
                cls.name,
                json.dumps(cls.to_dict()),
                cls.creation_time,
                cls.last_modified_time
            ))
            
            conn.commit()
            
            # Add to in-memory storage
            self.classes[cls.name] = cls
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error adding class to ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def add_relation_type(self, rel_type: RelationType) -> bool:
        """
        Add a new relation type to the ontology.
        
        Args:
            rel_type: The relation type to add
            
        Returns:
            True if relation type was added successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add relation type to database
            cursor.execute('''
                INSERT OR REPLACE INTO ontology_relation_types (name, data, creation_time, last_modified_time)
                VALUES (?, ?, ?, ?)
            ''', (
                rel_type.name,
                json.dumps(rel_type.to_dict()),
                rel_type.creation_time,
                rel_type.last_modified_time
            ))
            
            conn.commit()
            
            # Add to in-memory storage
            self.relation_types[rel_type.name] = rel_type
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error adding relation type to ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def get_class(self, name: str) -> Optional[OntologyClass]:
        """Get a class by name."""
        return self.classes.get(name)
    
    def get_relation_type(self, name: str) -> Optional[RelationType]:
        """Get a relation type by name."""
        return self.relation_types.get(name)
    
    def get_all_subclasses(self, class_name: str, include_indirect: bool = True) -> List[str]:
        """
        Get all subclasses of a given class.
        
        Args:
            class_name: The parent class name
            include_indirect: Whether to include indirect subclasses
            
        Returns:
            List of subclass names
        """
        if class_name not in self.class_hierarchy:
            return []
            
        direct_subclasses = self.class_hierarchy[class_name]
        
        if not include_indirect:
            return direct_subclasses
            
        all_subclasses = direct_subclasses.copy()
        for subclass in direct_subclasses:
            all_subclasses.extend(self.get_all_subclasses(subclass))
            
        return list(set(all_subclasses))  # Remove duplicates
    
    def get_all_subtypes(self, relation_type: str, include_indirect: bool = True) -> List[str]:
        """
        Get all subtypes of a given relation type.
        
        Args:
            relation_type: The parent relation type name
            include_indirect: Whether to include indirect subtypes
            
        Returns:
            List of subtype names
        """
        if relation_type not in self.relation_hierarchy:
            return []
            
        direct_subtypes = self.relation_hierarchy[relation_type]
        
        if not include_indirect:
            return direct_subtypes
            
        all_subtypes = direct_subtypes.copy()
        for subtype in direct_subtypes:
            all_subtypes.extend(self.get_all_subtypes(subtype))
            
        return list(set(all_subtypes))  # Remove duplicates
    
    def is_subclass_of(self, class_name: str, potential_parent: str) -> bool:
        """
        Check if a class is a subclass of another class.
        
        Args:
            class_name: The class to check
            potential_parent: The potential parent class
            
        Returns:
            True if class_name is a subclass of potential_parent
        """
        if class_name == potential_parent:
            return True
            
        if class_name not in self.classes:
            return False
            
        # Check direct parent classes
        for parent in self.classes[class_name].parent_classes:
            if parent == potential_parent or self.is_subclass_of(parent, potential_parent):
                return True
                
        return False
    
    def is_subtype_of(self, relation_type: str, potential_parent: str) -> bool:
        """
        Check if a relation type is a subtype of another relation type.
        
        Args:
            relation_type: The relation type to check
            potential_parent: The potential parent relation type
            
        Returns:
            True if relation_type is a subtype of potential_parent
        """
        if relation_type == potential_parent:
            return True
            
        if relation_type not in self.relation_types:
            return False
            
        # Check direct parent types
        for parent in self.relation_types[relation_type].parent_types:
            if parent == potential_parent or self.is_subtype_of(parent, potential_parent):
                return True
                
        return False
    
    def update_class(self, name: str, properties: Dict[str, Any] = None, 
                   parent_classes: List[str] = None, constraints: Dict[str, Any] = None,
                   metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing class in the ontology.
        
        Args:
            name: The class name to update
            properties: Dictionary of class properties to update
            parent_classes: List of parent class names to update
            constraints: Dictionary of constraints to update
            metadata: Additional metadata about the class to update
            
        Returns:
            True if class was updated successfully
        """
        if name not in self.classes:
            return False
            
        cls = self.classes[name]
        
        # Update fields if provided
        if properties is not None:
            cls.properties.update(properties)
            
        if parent_classes is not None:
            cls.parent_classes = parent_classes
            
        if constraints is not None:
            cls.constraints.update(constraints)
            
        if metadata is not None:
            cls.metadata.update(metadata)
            
        cls.last_modified_time = time.time()
        
        # Save updates
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE ontology_classes 
                SET data = ?, last_modified_time = ?
                WHERE name = ?
            ''', (
                json.dumps(cls.to_dict()),
                cls.last_modified_time,
                name
            ))
            
            conn.commit()
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error updating class in ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def update_relation_type(self, name: str, parent_types: List[str] = None,
                           source_classes: List[str] = None, target_classes: List[str] = None,
                           cardinality: str = None, properties: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing relation type in the ontology.
        
        Args:
            name: The relation type name to update
            parent_types: List of parent relation types to update
            source_classes: List of valid source class names to update
            target_classes: List of valid target class names to update
            cardinality: Cardinality constraint to update
            properties: Dictionary of relation properties to update
            metadata: Additional metadata about the relation type to update
            
        Returns:
            True if relation type was updated successfully
        """
        if name not in self.relation_types:
            return False
            
        rel_type = self.relation_types[name]
        
        # Update fields if provided
        if parent_types is not None:
            rel_type.parent_types = parent_types
            
        if source_classes is not None:
            rel_type.source_classes = source_classes
            
        if target_classes is not None:
            rel_type.target_classes = target_classes
            
        if cardinality is not None:
            rel_type.cardinality = cardinality
            
        if properties is not None:
            rel_type.properties.update(properties)
            
        if metadata is not None:
            rel_type.metadata.update(metadata)
            
        rel_type.last_modified_time = time.time()
        
        # Save updates
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE ontology_relation_types 
                SET data = ?, last_modified_time = ?
                WHERE name = ?
            ''', (
                json.dumps(rel_type.to_dict()),
                rel_type.last_modified_time,
                name
            ))
            
            conn.commit()
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error updating relation type in ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def suggest_new_class(self, entity_name: str, entity_type: str, 
                        entity_properties: Dict[str, Any]) -> Optional[OntologyClass]:
        """
        Suggest a new class in the ontology based on entity patterns.
        This enables automatic ontology evolution.
        
        Args:
            entity_name: Example entity name
            entity_type: Current entity type
            entity_properties: Properties of the entity
            
        Returns:
            Suggested new ontology class or None
        """
        # Start with current entity type
        if entity_type in self.classes:
            base_class = entity_type
        else:
            # Default to Thing if no match
            base_class = "Thing"
        
        # Create a new class name based on pattern
        new_class_name = f"{entity_type.capitalize()}"
        
        # Don't suggest if class already exists
        if new_class_name in self.classes:
            return None
        
        # Create suggested class
        suggested_class = OntologyClass(
            name=new_class_name,
            parent_classes=[base_class],
            properties=entity_properties,
            metadata={
                "suggested_from": entity_name,
                "automatic": True,
                "confidence": 0.7
            }
        )
        
        return suggested_class
    
    def suggest_new_relation_type(self, source_entity: str, relation: str, 
                                target_entity: str) -> Optional[RelationType]:
        """
        Suggest a new relation type in the ontology based on observed patterns.
        This enables automatic ontology evolution.
        
        Args:
            source_entity: Source entity
            relation: Observed relation
            target_entity: Target entity
            
        Returns:
            Suggested new relation type or None
        """
        # Don't suggest if relation already exists
        if relation in self.relation_types:
            return None
        
        # Default parent relation
        parent_relation = "related_to"
        
        # Try to find source and target types
        source_type = "Thing"
        target_type = "Thing"
        
        # Create suggested relation type
        suggested_relation = RelationType(
            name=relation,
            parent_types=[parent_relation],
            source_classes=[source_type],
            target_classes=[target_type],
            metadata={
                "suggested_from": f"{source_entity} {relation} {target_entity}",
                "automatic": True,
                "confidence": 0.7
            }
        )
        
        return suggested_relation
    
    def delete_class(self, name: str) -> bool:
        """
        Delete a class from the ontology.
        
        Args:
            name: The class name to delete
            
        Returns:
            True if class was deleted successfully
        """
        if name not in self.classes:
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM ontology_classes WHERE name = ?', (name,))
            conn.commit()
            
            # Remove from in-memory storage
            del self.classes[name]
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error deleting class from ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def delete_relation_type(self, name: str) -> bool:
        """
        Delete a relation type from the ontology.
        
        Args:
            name: The relation type name to delete
            
        Returns:
            True if relation type was deleted successfully
        """
        if name not in self.relation_types:
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM ontology_relation_types WHERE name = ?', (name,))
            conn.commit()
            
            # Remove from in-memory storage
            del self.relation_types[name]
            
            # Update hierarchy maps
            self._build_hierarchy_maps()
            
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error deleting relation type from ontology: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close() 