"""
CortexFlow Graph Store module.

This module provides graph-based knowledge representation for CortexFlow.
"""

import os
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import json
import time
import re

# Try importing graph libraries
try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    NETWORKX_ENABLED = False
    logging.warning("networkx not found. Knowledge graph functionality will be limited.")

# Try importing NER for entity extraction
try:
    import spacy
    SPACY_ENABLED = True
except ImportError:
    SPACY_ENABLED = False
    logging.warning("spacy not found. Automatic entity extraction will be limited.")

# Try importing Flair for advanced NER
try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    FLAIR_ENABLED = True
except ImportError:
    FLAIR_ENABLED = False
    logging.warning("flair not found. Advanced entity recognition will be limited.")

# Try importing SpanBERT for entity recognition
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    SPANBERT_ENABLED = True
except ImportError:
    SPANBERT_ENABLED = False
    logging.warning("transformers/torch not found. SpanBERT entity recognition will be disabled.")

# Try importing libraries for fuzzy matching
try:
    from thefuzz import fuzz, process
    FUZZY_MATCHING_ENABLED = True
except ImportError:
    FUZZY_MATCHING_ENABLED = False
    logging.warning("thefuzz not found. Fuzzy entity matching will be disabled.")

from cortexflow.config import CortexFlowConfig
try:
    from cortexflow.ontology import Ontology
    ONTOLOGY_ENABLED = True
except ImportError:
    ONTOLOGY_ENABLED = False
    logging.warning("Ontology module not found. Advanced knowledge graph capabilities will be limited.")

class RelationExtractor:
    """
    Dedicated relation extraction class for CortexFlow knowledge graph.
    Provides advanced relation extraction capabilities using dependency parsing,
    semantic role labeling, and relation classification.
    """
    
    def __init__(self, nlp=None):
        """
        Initialize relation extractor.
        
        Args:
            nlp: spaCy language model, or None to create a new one
        """
        # Initialize spaCy model if needed
        self.nlp = nlp
        if SPACY_ENABLED and not self.nlp:
            try:
                # Use a model with dependency parsing for relation extraction
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Relation Extractor: spaCy model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading spaCy model for relation extraction: {e}")
                self.nlp = None
        
        # Initialize SRL (Semantic Role Labeling) components
        self.srl_predictor = None
        try:
            from allennlp.predictors.predictor import Predictor
            self.srl_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
            logging.info("SRL model loaded successfully")
        except ImportError:
            logging.debug("AllenNLP SRL not available. Semantic role extraction will be limited.")
        except Exception as e:
            logging.error(f"Error loading SRL model: {e}")
        
        # Initialize relation classification model
        self.relation_classifier = None
        try:
            self.relation_classifier = AutoModelForTokenClassification.from_pretrained("Babelscape/rebel-large")
            self.relation_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
            logging.info("Relation classification model loaded successfully")
        except ImportError:
            logging.debug("Relation classification model not available")
        except Exception as e:
            logging.error(f"Error loading relation classification model: {e}")
        
        # Define relation patterns and templates
        self.relation_patterns = self._init_relation_patterns()
    
    def _init_relation_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize common relation patterns for rule-based extraction.
        
        Returns:
            Dictionary of relation patterns
        """
        patterns = {
            "is_a": [
                {"pattern": r"([^\s]+) is (?:a|an) ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) are ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) classified as ([^\s]+)", "groups": (1, 2)},
            ],
            "part_of": [
                {"pattern": r"([^\s]+) is part of ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) belongs to ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) contains ([^\s]+)", "groups": (2, 1)},
            ],
            "located_in": [
                {"pattern": r"([^\s]+) is (?:in|at|on) ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) is located (?:in|at|on) ([^\s]+)", "groups": (1, 2)},
            ],
            "has_property": [
                {"pattern": r"([^\s]+) has (?:a|an)? ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) with (?:a|an)? ([^\s]+)", "groups": (1, 2)},
            ],
            "causes": [
                {"pattern": r"([^\s]+) causes ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) leads to ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) results in ([^\s]+)", "groups": (1, 2)},
            ],
        }
        return patterns
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text using multiple techniques.
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        relations = []
        
        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # 1. Extract relations using dependency parsing
                dep_relations = self.extract_svo_from_dependency(doc)
                relations.extend(dep_relations)
                
                # 2. Extract prepositional relations
                prep_relations = self.extract_prep_relations(doc)
                relations.extend(prep_relations)
                
                # 3. Extract relations using semantic role labeling
                if self.srl_predictor:
                    srl_relations = self.extract_with_semantic_roles(text)
                    relations.extend(srl_relations)
                
                # 4. Apply relation classification if available
                if self.relation_classifier:
                    classified_relations = self.classify_relations(text)
                    relations.extend(classified_relations)
                
                # 5. Extract relations using pattern matching
                pattern_relations = self.extract_with_patterns(text)
                relations.extend(pattern_relations)
                
            except Exception as e:
                logging.error(f"Error in relation extraction: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Deduplicate relations
        unique_relations = []
        relation_strings = set()
        
        for subj, pred, obj in relations:
            rel_str = f"{subj.lower()}|{pred.lower()}|{obj.lower()}"
            if rel_str not in relation_strings:
                relation_strings.add(rel_str)
                unique_relations.append((subj, pred, obj))
        
        return unique_relations
    
    def extract_svo_from_dependency(self, doc) -> List[Tuple[str, str, str]]:
        """
        Extract Subject-Verb-Object triples from a document using dependency parsing.
        
        Args:
            doc: spaCy document
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        
        for sent in doc.sents:
            # Find root verbs in the sentence
            root_verbs = [token for token in sent if token.dep_ == "ROOT" and token.pos_ == "VERB"]
            if not root_verbs:
                # If no root verb, find any verb
                root_verbs = [token for token in sent if token.pos_ == "VERB"]
            
            for verb in root_verbs:
                # Find potential subjects
                subjects = []
                for token in sent:
                    # Check if token is a subject dependent on our verb
                    if token.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent"] and token.head == verb:
                        # Get full noun phrase
                        subject_span = self._get_span_text(token)
                        subjects.append(subject_span)
                
                # Find potential objects
                objects = []
                for token in sent:
                    # Check if token is an object dependent on our verb
                    if token.dep_ in ["dobj", "pobj", "iobj", "obj"] and token.head == verb:
                        object_span = self._get_span_text(token)
                        objects.append(object_span)
                    # Handle prep phrases like "to the store" connected to our verb
                    elif token.dep_ == "prep" and token.head == verb:
                        for child in token.children:
                            if child.dep_ == "pobj":
                                # Include the preposition in the relation
                                pred = f"{verb.lemma_} {token.text}"
                                object_span = self._get_span_text(child)
                                objects.append((pred, object_span))
                
                # Create triples for all subject-object pairs
                for subject in subjects:
                    for obj in objects:
                        if isinstance(obj, tuple):
                            # Handle special case of prep phrases
                            pred, obj_text = obj
                            triples.append((subject, pred, obj_text))
                        else:
                            triples.append((subject, verb.lemma_, obj))
        
        return triples
    
    def extract_prep_relations(self, doc) -> List[Tuple[str, str, str]]:
        """
        Extract relations based on prepositional phrases like "X in Y".
        
        Args:
            doc: spaCy document
            
        Returns:
            List of (entity1, preposition, entity2) tuples
        """
        prep_relations = []
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                    # Get the head (the first entity)
                    head_span = self._get_span_text(token.head)
                    
                    # Get the object of the preposition (the second entity)
                    for child in token.children:
                        if child.dep_ == "pobj":
                            object_span = self._get_span_text(child)
                            # Create relation with the preposition as predicate
                            prep_relations.append((head_span, token.text, object_span))
        
        return prep_relations
    
    def extract_with_semantic_roles(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relations using semantic role labeling (SRL).
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        if not self.srl_predictor:
            return []
            
        try:
            srl_output = self.srl_predictor.predict(sentence=text)
            
            # Extract relations from SRL output
            relations = []
            for verb_data in srl_output.get('verbs', []):
                predicate = verb_data['verb']
                
                # Process tagged spans to extract arguments
                arg0 = None
                arg1 = None
                arg2 = None
                loc = None
                tmp = None
                
                # Extract arguments from tags
                tagged_string = verb_data['description']
                current_arg = None
                current_text = ""
                
                for part in tagged_string.split():
                    if part.startswith('['):
                        # Start of new argument
                        if current_arg and current_text:
                            if current_arg == 'ARG0':
                                arg0 = current_text.strip()
                            elif current_arg == 'ARG1':
                                arg1 = current_text.strip()
                            elif current_arg == 'ARG2':
                                arg2 = current_text.strip()
                            elif current_arg.startswith('ARGM-LOC'):
                                loc = current_text.strip()
                            elif current_arg.startswith('ARGM-TMP'):
                                tmp = current_text.strip()
                                
                        # Set new current argument
                        if '*' in part:
                            label_end = part.find('*')
                            current_arg = part[1:label_end]
                            current_text = part[label_end+1:]
                            if part.endswith(']'):
                                current_text = current_text[:-1]
                    elif part.endswith(']'):
                        # End of current argument
                        current_text += " " + part[:-1]
                        
                        if current_arg == 'ARG0':
                            arg0 = current_text.strip()
                        elif current_arg == 'ARG1':
                            arg1 = current_text.strip()
                        elif current_arg == 'ARG2':
                            arg2 = current_text.strip()
                        elif current_arg.startswith('ARGM-LOC'):
                            loc = current_text.strip()
                        elif current_arg.startswith('ARGM-TMP'):
                            tmp = current_text.strip()
                            
                        current_arg = None
                        current_text = ""
                    elif current_arg:
                        # Continue current argument
                        current_text += " " + part
                
                # Create relations from arguments
                if arg0 and arg1:
                    relations.append((arg0, predicate, arg1))
                if arg0 and arg2:
                    relations.append((arg0, predicate + " to", arg2))
                if arg1 and loc:
                    relations.append((arg1, "located in", loc))
                if arg0 and loc:
                    relations.append((arg0, "located in", loc))
                if arg0 and tmp:
                    relations.append((arg0, "at time", tmp))
            
            return relations
            
        except Exception as e:
            logging.error(f"Error in semantic role labeling: {e}")
            return []
    
    def classify_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Use a pretrained model to classify relationship types between entities.
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, relation, object) tuples
        """
        if not self.relation_classifier or not SPANBERT_ENABLED:
            return []
            
        try:
            # Tokenize the text
            inputs = self.relation_tokenizer(text, return_tensors="pt")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.relation_classifier(**inputs)
            
            # Process the outputs to extract relations
            # Note: This is a simplified implementation and would need to be
            # adapted based on the specific relation classification model used
            relations = []
            
            # Process model outputs to extract structured relations
            # Actual implementation depends on the specific model's output format
            
            return relations
            
        except Exception as e:
            logging.error(f"Error in relation classification: {e}")
            return []
    
    def extract_with_patterns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relations using pattern-based rules.
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, relation, object) tuples
        """
        relations = []
        
        # Apply each relation pattern
        for relation_type, patterns in self.relation_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                group_indices = pattern_info["groups"]
                
                # Find matches in text
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        # Extract subject and object from specified groups
                        subject = match.group(group_indices[0]).strip()
                        obj = match.group(group_indices[1]).strip()
                        
                        # Add to relations
                        relations.append((subject, relation_type, obj))
                    except IndexError:
                        continue
        
        return relations
    
    def _get_span_text(self, token) -> str:
        """
        Get the text of the full noun phrase for a token.
        
        Args:
            token: spaCy token
            
        Returns:
            Text of the full noun phrase
        """
        # If token is part of a compound, get the full compound
        if token.pos_ in ["NOUN", "PROPN"]:
            # Start with the token itself
            start = token
            # Traverse left children to find compound parts
            lefts = list(token.lefts)
            for left in lefts:
                if left.dep_ in ["compound", "amod", "det", "nummod"]:
                    if left.i < start.i:
                        start = left
            
            # Traverse right children to find the end of the noun phrase
            end = token
            rights = list(token.rights)
            for right in rights:
                if right.dep_ in ["compound", "amod"]:
                    if right.i > end.i:
                        end = right
            
            # Get the full span text
            span_tokens = [t for t in start.doc if start.i <= t.i <= end.i]
            return " ".join([t.text for t in span_tokens])
        
        return token.text

class GraphStore:
    """Knowledge graph storage and query functionality for GraphRAG."""
    
    def __init__(self, config: CortexFlowConfig):
        """
        Initialize graph store.
        
        Args:
            config: CortexFlow configuration
        """
        self.config = config
        self.db_path = config.knowledge_store_path
        
        # For in-memory databases, we need to maintain a persistent connection
        self.conn = None
        if self.db_path == ':memory:':
            # For Windows compatibility, use empty string instead of ":memory:"
            self.conn = sqlite3.connect("")
        
        # Initialize NetworkX graph if available
        self.graph = nx.DiGraph() if NETWORKX_ENABLED else None
        
        # Initialize NER model for entity extraction if available
        self.nlp = None
        if SPACY_ENABLED:
            try:
                # Use a small efficient model by default
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Spacy NER model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading Spacy model: {e}")
        
        # Initialize advanced NER models if available
        self.flair_ner = None
        if FLAIR_ENABLED:
            try:
                # Load Flair NER model
                self.flair_ner = SequenceTagger.load("flair/ner-english-ontonotes-large")
                logging.info("Flair NER model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading Flair model: {e}")
                
        # Initialize SpanBERT model if available
        self.spanbert_tokenizer = None
        self.spanbert_model = None
        if SPANBERT_ENABLED:
            try:
                # Load SpanBERT model for entity recognition
                self.spanbert_tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
                self.spanbert_model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased")
                logging.info("SpanBERT model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading SpanBERT model: {e}")
        
        # Initialize entity linking system
        self.entity_db = {}  # Map of canonical entities
        self.entity_embeddings = {}  # For semantic similarity between entities
        
        # Initialize relation extractor
        self.relation_extractor = RelationExtractor(self.nlp)
        
        # Initialize ontology if available
        self.ontology = None
        if ONTOLOGY_ENABLED:
            try:
                self.ontology = Ontology(self.db_path)
                logging.info("Ontology system initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing ontology: {e}")
        
        # Initialize graph database tables
        self._init_db()
        
        # Load existing graph from database
        self._load_graph_from_db()
        
        # Load entity database for entity linking
        self._load_entity_db()
        
        logging.info(f"Graph store initialized with NetworkX: {NETWORKX_ENABLED}, Spacy: {SPACY_ENABLED}, Flair: {FLAIR_ENABLED}, SpanBERT: {SPANBERT_ENABLED}, Ontology: {ONTOLOGY_ENABLED}")
    
    def _load_entity_db(self):
        """Load existing entity database for entity linking."""
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Load all entities for linking
            cursor.execute('SELECT id, entity, entity_type, metadata FROM graph_entities')
            entities = cursor.fetchall()
            
            for entity in entities:
                canonical_name = entity['entity']
                entity_id = entity['id']
                entity_type = entity['entity_type']
                metadata = json.loads(entity['metadata']) if entity['metadata'] else {}
                
                # Store entity in the linking database
                self.entity_db[canonical_name] = {
                    'id': entity_id,
                    'type': entity_type,
                    'metadata': metadata,
                    'aliases': metadata.get('aliases', [])
                }
                
                # Add aliases to the lookup
                for alias in metadata.get('aliases', []):
                    self.entity_db[alias] = {
                        'canonical': canonical_name,
                        'id': entity_id
                    }
                    
        except Exception as e:
            logging.error(f"Error loading entity database: {e}")
            
        if self.conn is None:
            conn.close()
    
    def _init_db(self):
        """Initialize the SQLite database with required tables for graph storage."""
        if self.conn is not None:
            # We're using an in-memory database
            cursor = self.conn.cursor()
        else:
            # Using a file-based database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Create entities table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT NOT NULL,
            entity_type TEXT,
            metadata TEXT,
            embedding BLOB,
            timestamp REAL,
            provenance TEXT,
            confidence REAL DEFAULT 0.8,
            temporal_start TEXT,
            temporal_end TEXT,
            extraction_method TEXT,
            version INTEGER DEFAULT 1,
            last_updated REAL,
            UNIQUE(entity)
        )
        ''')
        
        # Create relation_types table for proper relation type ontology
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relation_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            parent_type TEXT,
            description TEXT,
            symmetric BOOLEAN DEFAULT 0,
            transitive BOOLEAN DEFAULT 0,
            inverse_relation TEXT,
            taxonomy_level INTEGER DEFAULT 0,
            metadata TEXT,
            UNIQUE(name)
        )
        ''')
        
        # Create relationships table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            timestamp REAL,
            provenance TEXT,
            confidence REAL DEFAULT 0.5,
            temporal_start TEXT,
            temporal_end TEXT,
            extraction_method TEXT,
            version INTEGER DEFAULT 1,
            last_updated REAL,
            FOREIGN KEY (source_id) REFERENCES graph_entities (id),
            FOREIGN KEY (target_id) REFERENCES graph_entities (id),
            FOREIGN KEY (relation_type) REFERENCES relation_types (name),
            UNIQUE(source_id, target_id, relation_type)
        )
        ''')
        
        # Create entity_versions table for tracking entity changes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            entity TEXT NOT NULL,
            entity_type TEXT,
            metadata TEXT,
            provenance TEXT,
            confidence REAL,
            temporal_start TEXT,
            temporal_end TEXT,
            extraction_method TEXT,
            version INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            change_type TEXT NOT NULL,
            changed_by TEXT,
            FOREIGN KEY (entity_id) REFERENCES graph_entities (id)
        )
        ''')
        
        # Create relationship_versions table for tracking relationship changes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationship_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relationship_id INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            provenance TEXT,
            confidence REAL,
            temporal_start TEXT,
            temporal_end TEXT,
            extraction_method TEXT,
            version INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            change_type TEXT NOT NULL,
            changed_by TEXT,
            FOREIGN KEY (relationship_id) REFERENCES graph_relationships (id),
            FOREIGN KEY (source_id) REFERENCES graph_entities (id),
            FOREIGN KEY (target_id) REFERENCES graph_entities (id)
        )
        ''')
        
        # Create table for n-ary relationships
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nary_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation_type TEXT NOT NULL,
            metadata TEXT,
            provenance TEXT,
            confidence REAL,
            extraction_method TEXT,
            version INTEGER DEFAULT 1,
            timestamp REAL,
            last_updated REAL,
            FOREIGN KEY (relation_type) REFERENCES relation_types (name)
        )
        ''')
        
        # Create table for n-ary relationship participants
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nary_participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relationship_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            metadata TEXT,
            timestamp REAL,
            FOREIGN KEY (relationship_id) REFERENCES nary_relationships (id),
            FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
            UNIQUE(relationship_id, entity_id, role)
        )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON graph_entities(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON graph_entities(entity_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version ON graph_entities(version)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_provenance ON graph_entities(provenance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_extraction ON graph_entities(extraction_method)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_type_name ON relation_types(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_type_parent ON relation_types(parent_type)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON graph_relationships(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON graph_relationships(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON graph_relationships(relation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_version ON graph_relationships(version)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_provenance ON graph_relationships(provenance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_extraction ON graph_relationships(extraction_method)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_entity ON entity_versions(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_number ON entity_versions(version)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_type ON entity_versions(change_type)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_rel ON relationship_versions(relationship_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_number ON relationship_versions(version)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_type ON relationship_versions(change_type)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_type ON nary_relationships(relation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_extraction ON nary_relationships(extraction_method)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_version ON nary_relationships(version)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_participant ON nary_participants(relationship_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_entity ON nary_participants(entity_id)')
        
        # Insert basic relation types if not exist
        basic_relation_types = [
            ('is_a', None, 'Taxonomic relationship', 0, 0, None, 1),
            ('part_of', None, 'Meronymic relationship', 0, 1, 'contains', 1),
            ('located_in', None, 'Spatial relationship', 0, 1, 'contains', 1),
            ('has_property', None, 'Attributional relationship', 0, 0, 'is_property_of', 1),
            ('causes', None, 'Causal relationship', 0, 0, 'caused_by', 1),
            ('related_to', None, 'Generic relationship', 1, 0, 'related_to', 0),
            ('same_as', None, 'Identity relationship', 1, 1, 'same_as', 1),
            ('temporal_before', None, 'Temporal relationship', 0, 1, 'temporal_after', 1),
            ('temporal_after', None, 'Temporal relationship', 0, 1, 'temporal_before', 1),
            ('contains', None, 'Containment relationship', 0, 0, 'part_of', 1),
            ('created_by', None, 'Creative relationship', 0, 0, 'created', 1),
            ('instance_of', 'is_a', 'Instance relationship', 0, 0, 'has_instance', 2),
            ('subclass_of', 'is_a', 'Subclass relationship', 0, 1, 'has_subclass', 2)
        ]
        
        for relation in basic_relation_types:
            cursor.execute('''
                INSERT OR IGNORE INTO relation_types 
                (name, parent_type, description, symmetric, transitive, inverse_relation, taxonomy_level) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', relation)
        
        # Alter existing tables to add new metadata columns if they don't exist
        try:
            # Check if extraction_method column exists in graph_relationships
            cursor.execute("PRAGMA table_info(graph_relationships)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add extraction_method column if it doesn't exist
            if "extraction_method" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN extraction_method TEXT")
                
            # Add version column if it doesn't exist
            if "version" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN version INTEGER DEFAULT 1")
                
            # Add last_updated column if it doesn't exist
            if "last_updated" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN last_updated REAL")
                
            # Check if similar columns need to be added to graph_entities
            cursor.execute("PRAGMA table_info(graph_entities)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add extraction_method column if it doesn't exist
            if "extraction_method" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN extraction_method TEXT")
                
            # Add version column if it doesn't exist
            if "version" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN version INTEGER DEFAULT 1")
                
            # Add last_updated column if it doesn't exist
            if "last_updated" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN last_updated REAL")
                
        except sqlite3.OperationalError as e:
            logging.error(f"Error adding metadata columns: {e}")
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
    
    def _load_graph_from_db(self):
        """Load existing graph data from the database."""
        if not NETWORKX_ENABLED:
            return
            
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Load all entities
            cursor.execute('SELECT id, entity, entity_type, metadata FROM graph_entities')
            entities = cursor.fetchall()
            
            for entity in entities:
                metadata = json.loads(entity['metadata']) if entity['metadata'] else {}
                self.graph.add_node(
                    entity['id'], 
                    name=entity['entity'],
                    entity_type=entity['entity_type'],
                    **metadata
                )
            
            # Load all relationships
            cursor.execute('''
                SELECT source_id, target_id, relation_type, weight, metadata 
                FROM graph_relationships
            ''')
            relationships = cursor.fetchall()
            
            for rel in relationships:
                metadata = json.loads(rel['metadata']) if rel['metadata'] else {}
                weight = rel['weight'] if rel['weight'] is not None else 1.0
                self.graph.add_edge(
                    rel['source_id'], 
                    rel['target_id'],
                    relation=rel['relation_type'],
                    weight=weight,
                    **metadata
                )
                
            logging.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except sqlite3.OperationalError as e:
            logging.error(f"Error loading graph from database: {e}")
            
        finally:
            if self.conn is None:
                conn.close()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using multiple techniques including
        advanced NER models, entity linking, and fuzzy matching.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of entity dictionaries with text, type, start, end
        """
        entities = []
        
        # 1. First try SpaCy NER if available
        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Get named entities from spaCy's NER
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'source': 'spacy'
                    })
                    
                # 2. Attempt coreference resolution if neuralcoref is available
                try:
                    import neuralcoref
                    if not hasattr(self, 'coref_nlp'):
                        # Initialize neuralcoref if not already done
                        self.coref_nlp = spacy.load('en_core_web_sm')
                        neuralcoref.add_to_pipe(self.coref_nlp)
                        logging.info("Coreference resolution model loaded successfully")
                        
                    # Process the text for coreference resolution
                    coref_doc = self.coref_nlp(text)
                    
                    # Add coreferenced entities
                    coref_clusters = coref_doc._.coref_clusters
                    if coref_clusters:
                        for cluster in coref_clusters:
                            main_mention = cluster.main
                            # Add main mention as entity if it's not already in our list
                            main_mention_text = main_mention.text
                            main_start = main_mention.start_char
                            main_end = main_mention.end_char
                            
                            # Check if this mention overlaps with existing entities
                            is_new_entity = True
                            for entity in entities:
                                if (main_start >= entity['start'] and main_start < entity['end']) or \
                                   (main_end > entity['start'] and main_end <= entity['end']):
                                    is_new_entity = False
                                    break
                                    
                            if is_new_entity:
                                entities.append({
                                    'text': main_mention_text,
                                    'type': 'COREF',
                                    'start': main_start,
                                    'end': main_end,
                                    'mentions': [m.text for m in cluster.mentions],
                                    'source': 'coref'
                                })
                except ImportError:
                    logging.debug("neuralcoref not available, skipping coreference resolution")
                except Exception as e:
                    logging.error(f"Error in coreference resolution: {e}")
            except Exception as e:
                logging.error(f"Error in SpaCy NER: {e}")
        
        # 3. Use Flair for NER if available
        if FLAIR_ENABLED and self.flair_ner is not None:
            try:
                # Create Flair sentence
                flair_sentence = Sentence(text)
                
                # Run NER
                self.flair_ner.predict(flair_sentence)
                
                # Extract entities
                for entity in flair_sentence.get_spans('ner'):
                    # Calculate character offsets
                    start_pos = text.find(entity.text)
                    if start_pos >= 0:
                        end_pos = start_pos + len(entity.text)
                        
                        # Check for overlap with existing entities
                        is_new_entity = True
                        for existing_entity in entities:
                            if (start_pos >= existing_entity['start'] and start_pos < existing_entity['end']) or \
                               (end_pos > existing_entity['start'] and end_pos <= existing_entity['end']):
                                is_new_entity = False
                                break
                                
                        if is_new_entity:
                            entities.append({
                                'text': entity.text,
                                'type': entity.tag,
                                'start': start_pos,
                                'end': end_pos,
                                'score': entity.score,
                                'source': 'flair'
                            })
            except Exception as e:
                logging.error(f"Error in Flair NER: {e}")
        
        # 4. Use SpanBERT for NER if available
        if SPANBERT_ENABLED and self.spanbert_model is not None and self.spanbert_tokenizer is not None:
            try:
                # Tokenize input
                inputs = self.spanbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.spanbert_model(**inputs)
                    
                # Process predictions to extract entities
                # This is a simplified implementation and would need to be adapted for the specific model
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                tokens = self.spanbert_tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze().tolist())
                
                # Map predictions to entity spans (simplified)
                current_entity = None
                current_type = None
                current_start = 0
                
                # Skip special tokens like [CLS]
                offset = 1
                char_offset = 0
                
                for i, (token, prediction) in enumerate(zip(tokens[offset:], predictions[offset:])):
                    # Skip special tokens
                    if token.startswith("##") or token in ["[SEP]", "[PAD]"]:
                        continue
                        
                    # Basic BIO scheme processing (simplification)
                    if prediction > 0:  # Non-O tag
                        # Get entity type (simplified mapping)
                        entity_type = f"TYPE_{prediction}"
                        
                        if current_entity is None:
                            # Start of new entity
                            current_entity = token.replace("##", "")
                            current_type = entity_type
                            current_start = char_offset
                        else:
                            # Continue current entity
                            current_entity += " " + token.replace("##", "")
                    else:
                        # End of entity
                        if current_entity is not None:
                            # Add entity if not already overlapping
                            entity_end = current_start + len(current_entity)
                            
                            # Check for overlap
                            is_new_entity = True
                            for entity in entities:
                                if (current_start >= entity['start'] and current_start < entity['end']) or \
                                   (entity_end > entity['start'] and entity_end <= entity['end']):
                                    is_new_entity = False
                                    break
                                    
                            if is_new_entity:
                                entities.append({
                                    'text': current_entity,
                                    'type': current_type,
                                    'start': current_start,
                                    'end': entity_end,
                                    'source': 'spanbert'
                                })
                                
                            current_entity = None
                            current_type = None
                            
                    # Update character offset (simplified)
                    char_offset += len(token) + 1
                    
                # Add final entity if there is one
                if current_entity is not None:
                    entity_end = current_start + len(current_entity)
                    entities.append({
                        'text': current_entity,
                        'type': current_type,
                        'start': current_start,
                        'end': entity_end,
                        'source': 'spanbert'
                    })
                    
            except Exception as e:
                logging.error(f"Error in SpanBERT NER: {e}")
        
        # 5. Add pattern-based entity extraction
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'URL': r'https?://\S+',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'TIME': r'\b\d{1,2}:\d{2}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'NUMBER': r'\b\d+(?:\.\d+)?\b',
            'PERCENTAGE': r'\b\d+(?:\.\d+)?%\b',
            'MONEY': r'\$\d+(?:\.\d+)?\b',
            'HASHTAG': r'#[A-Za-z][A-Za-z0-9_]*',
            'MENTION': r'@[A-Za-z][A-Za-z0-9_]*'
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                # Check if this match overlaps with existing entities
                overlap = False
                for entity in entities:
                    if (match.start() >= entity['start'] and match.start() < entity['end']) or \
                       (match.end() > entity['start'] and match.end() <= entity['end']):
                        overlap = True
                        break
                
                if not overlap:
                    entities.append({
                        'text': match.group(0),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'source': 'pattern'
                    })
        
        # 6. Add noun phrase extraction if no entities found yet or to supplement
        if SPACY_ENABLED and self.nlp is not None:
            try:
                if not 'doc' in locals():  # Only parse if we haven't already
                    doc = self.nlp(text)
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:  # Skip very short chunks
                        # Check for overlap with existing entities
                        overlap = False
                        for entity in entities:
                            if (chunk.start_char >= entity['start'] and chunk.start_char < entity['end']) or \
                               (chunk.end_char > entity['start'] and chunk.end_char <= entity['end']):
                                overlap = True
                                break
                                
                        if not overlap:
                            entities.append({
                                'text': chunk.text,
                                'type': 'NOUN_PHRASE',
                                'start': chunk.start_char,
                                'end': chunk.end_char,
                                'source': 'noun_chunk'
                            })
                        
                # Add proper nouns not already captured
                for token in doc:
                    if token.pos_ == "PROPN" and len(token.text) > 2:
                        # Check if already part of an entity
                        is_part_of_entity = False
                        for entity in entities:
                            if token.idx >= entity['start'] and token.idx + len(token.text) <= entity['end']:
                                is_part_of_entity = True
                                break
                                
                        if not is_part_of_entity:
                            entities.append({
                                'text': token.text,
                                'type': 'PROPER_NOUN',
                                'start': token.idx,
                                'end': token.idx + len(token.text),
                                'source': 'pos_tag'
                            })
            except Exception as e:
                logging.error(f"Error extracting noun phrases: {e}")

        # 7. Add domain-specific entity extraction
        try:
            # Check for domain-specific patterns based on config
            domain_entities = self._extract_domain_specific_entities(text)
            for entity in domain_entities:
                # Check for overlap
                overlap = False
                for existing_entity in entities:
                    if (entity['start'] >= existing_entity['start'] and entity['start'] < existing_entity['end']) or \
                       (entity['end'] > existing_entity['start'] and entity['end'] <= existing_entity['end']):
                        overlap = True
                        break
                
                if not overlap:
                    entity['source'] = 'domain'
                    entities.append(entity)
        except Exception as e:
            logging.error(f"Error in domain-specific entity extraction: {e}")
            
        # 8. Perform entity linking to connect mentions to canonical entities
        linked_entities = []
        for entity in entities:
            entity_text = entity['text']
            linked_entity = self._link_entity(entity_text)
            
            if linked_entity:
                # Copy original entity and add linking information
                linked_entity_data = entity.copy()
                linked_entity_data['canonical'] = linked_entity['canonical'] if 'canonical' in linked_entity else entity_text
                linked_entity_data['entity_id'] = linked_entity['id']
                linked_entity_data['linked'] = True
                
                # Use canonical entity type if available
                if 'type' in linked_entity and linked_entity['type']:
                    linked_entity_data['canonical_type'] = linked_entity['type']
                
                linked_entities.append(linked_entity_data)
            else:
                # No linking found, keep original entity
                entity['linked'] = False
                linked_entities.append(entity)
                
        return linked_entities
    
    def _link_entity(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Link entity mention to canonical entity using exact, fuzzy, and embedding-based matching.
        
        Args:
            entity_text: Text of the entity mention to link
            
        Returns:
            Dictionary with linking information or None if no match found
        """
        # First try exact match
        if entity_text in self.entity_db:
            return self.entity_db[entity_text]
            
        # Next try case-insensitive match
        entity_lower = entity_text.lower()
        for key in self.entity_db:
            if key.lower() == entity_lower:
                return self.entity_db[key]
                
        # Try fuzzy matching if available
        if FUZZY_MATCHING_ENABLED:
            try:
                # Get only canonical entities (not aliases)
                canonical_entities = [key for key in self.entity_db 
                                     if not 'canonical' in self.entity_db[key]]
                
                # Find closest match with threshold
                matches = process.extractBests(entity_text, canonical_entities, 
                                               scorer=fuzz.token_sort_ratio, 
                                               score_cutoff=85,
                                               limit=1)
                                               
                if matches and len(matches) > 0:
                    match, score = matches[0]
                    result = self.entity_db[match].copy()
                    result['match_score'] = score
                    result['match_type'] = 'fuzzy'
                    return result
            except Exception as e:
                logging.error(f"Error in fuzzy entity matching: {e}")
                
        # No match found
        return None
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text using the RelationExtractor.
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        # Use the dedicated RelationExtractor for relation extraction
        return self.relation_extractor.extract_relations(text)
    
    def _extract_domain_specific_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract domain-specific entities based on patterns.
        This can be extended for specialized knowledge domains.
        
        Args:
            text: Input text to extract domain entities from
            
        Returns:
            List of entity dictionaries
        """
        domain_entities = []
        
        # Example: Extract programming language entities
        programming_langs = [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
            "Swift", "Kotlin", "Ruby", "PHP", "SQL", "R", "MATLAB", "Scala", "Perl",
            "Haskell", "Clojure", "Erlang", "Elixir", "Julia"
        ]
        
        # Look for programming languages
        for lang in programming_langs:
            for match in re.finditer(r'\b' + re.escape(lang) + r'\b', text):
                domain_entities.append({
                    'text': match.group(0),
                    'type': 'PROGRAMMING_LANGUAGE',
                    'start': match.start(),
                    'end': match.end()
                })
                
        # Example: Extract ML/AI techniques
        ml_techniques = [
            "Neural Network", "Deep Learning", "Machine Learning", "Natural Language Processing",
            "Computer Vision", "Reinforcement Learning", "Transformer", "BERT", "GPT",
            "CNN", "RNN", "LSTM", "GAN", "Decision Tree", "Random Forest", "SVM",
            "K-means", "PCA", "t-SNE", "XGBoost"
        ]
        
        # Look for ML/AI terms
        for technique in ml_techniques:
            pattern = r'\b' + re.escape(technique) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                domain_entities.append({
                    'text': match.group(0),
                    'type': 'AI_ML_TERM',
                    'start': match.start(),
                    'end': match.end()
                })
                
        return domain_entities
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text using semantic role labeling
        and enhanced dependency parsing techniques.
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        relations = []
        
        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # 1. First attempt with semantic role labeling if available
                try:
                    srl_relations = self._extract_with_semantic_roles(text)
                    if srl_relations:
                        relations.extend(srl_relations)
                except Exception as e:
                    logging.debug(f"SRL extraction not available: {e}")
                
                # 2. Process each sentence separately with enhanced dependency parsing
                for sent in doc.sents:
                    # Extract all subject-verb-object patterns
                    extracted_svo = self._extract_svo_from_dependency(sent)
                    relations.extend(extracted_svo)
                    
                    # Extract prep relations (X in Y, X on Y, etc.)
                    extracted_preps = self._extract_prep_relations(sent)
                    relations.extend(extracted_preps)
                
                # 3. Add coreference-based relations if available
                try:
                    coref_relations = self._extract_with_coreference(text)
                    if coref_relations:
                        relations.extend(coref_relations)
                except Exception as e:
                    logging.debug(f"Coreference relation extraction not available: {e}")
                
                # 4. If no relations found with advanced methods, fall back to simpler approaches
                if not relations:
                    entities = self.extract_entities(text)
                    if len(entities) >= 2:
                        # Create relations between consecutive entities with heuristics
                        for i in range(len(entities) - 1):
                            entity1 = entities[i]
                            entity2 = entities[i + 1]
                            
                            # Get text between entities
                            between_start = entity1['end']
                            between_end = entity2['start']
                            
                            if between_end > between_start:
                                between_text = text[between_start:between_end].strip()
                                
                                # If there's text between, use it as predicate
                                if between_text:
                                    # Clean up the predicate
                                    predicate = between_text.strip()
                                    # Remove common stopwords
                                    for stopword in [" the ", " a ", " an ", " and ", " or ", " but ", " of "]:
                                        predicate = predicate.replace(stopword, " ")
                                    predicate = predicate.strip()
                                    
                                    if predicate:
                                        relations.append((entity1['text'], predicate, entity2['text']))
                                else:
                                    # If no text between, use generic relation
                                    relations.append((entity1['text'], "related_to", entity2['text']))
                
            except Exception as e:
                logging.error(f"Error extracting relations: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                
        return relations
    
    def _extract_svo_from_dependency(self, sent) -> List[Tuple[str, str, str]]:
        """
        Extract Subject-Verb-Object triples from a sentence using dependency parsing.
        Uses more advanced patterns than the basic implementation.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        
        # Find all verbs in the sentence
        verbs = [token for token in sent if token.pos_ == "VERB"]
        
        for verb in verbs:
            # Find potential subjects
            subjects = []
            for token in verb.lefts:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent"]:
                    # Get full noun phrase
                    subject_span = self._get_span_text(token)
                    subjects.append(subject_span)
            
            # Find potential objects
            objects = []
            for token in verb.rights:
                # Direct object or prepositional object
                if token.dep_ in ["dobj", "pobj", "iobj", "obj"]:
                    object_span = self._get_span_text(token)
                    objects.append(object_span)
                # Handle prep phrases like "to the store"
                elif token.dep_ == "prep":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            # Include the preposition in the relation
                            pred = f"{verb.lemma_} {token.text}"
                            object_span = self._get_span_text(child)
                            objects.append((pred, object_span))
            
            # Create triples for all subject-object pairs
            for subject in subjects:
                for obj in objects:
                    if isinstance(obj, tuple):
                        # Handle special case of prep phrases
                        pred, obj_text = obj
                        triples.append((subject, pred, obj_text))
                    else:
                        triples.append((subject, verb.lemma_, obj))
        
        return triples
    
    def _extract_prep_relations(self, sent) -> List[Tuple[str, str, str]]:
        """
        Extract relations based on prepositional phrases like "X in Y".
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            List of (entity1, preposition, entity2) tuples
        """
        prep_relations = []
        
        for token in sent:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Get the head (the first entity)
                head_span = self._get_span_text(token.head)
                
                # Get the object of the preposition (the second entity)
                for child in token.children:
                    if child.dep_ == "pobj":
                        object_span = self._get_span_text(child)
                        # Create relation with the preposition as predicate
                        prep_relations.append((head_span, token.text, object_span))
        
        return prep_relations
    
    def _extract_with_semantic_roles(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relations using semantic role labeling (SRL) if available.
        Requires AllenNLP SRL model.
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        try:
            from allennlp.predictors.predictor import Predictor
            
            # Initialize SRL predictor if not already done
            if not hasattr(self, 'srl_predictor'):
                self.srl_predictor = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
                logging.info("SRL model loaded successfully")
            
            srl_output = self.srl_predictor.predict(sentence=text)
            
            # Extract relations from SRL output
            relations = []
            for verb_data in srl_output.get('verbs', []):
                predicate = verb_data['verb']
                args = {}
                
                # Extract arguments from tags
                for tag, words in zip(verb_data['tags'], text.split()):
                    if tag.startswith('B-ARG0'):
                        args['ARG0'] = words
                    elif tag.startswith('B-ARG1'):
                        args['ARG1'] = words
                    elif tag.startswith('B-ARG2'):
                        args['ARG2'] = words
                    elif tag.startswith('I-ARG0') and 'ARG0' in args:
                        args['ARG0'] += ' ' + words
                    elif tag.startswith('I-ARG1') and 'ARG1' in args:
                        args['ARG1'] += ' ' + words
                    elif tag.startswith('I-ARG2') and 'ARG2' in args:
                        args['ARG2'] += ' ' + words
                
                # Create relations from arguments
                if 'ARG0' in args and 'ARG1' in args:
                    relations.append((args['ARG0'], predicate, args['ARG1']))
                if 'ARG0' in args and 'ARG2' in args:
                    relations.append((args['ARG0'], predicate + ' to', args['ARG2']))
            
            return relations
            
        except ImportError:
            logging.debug("AllenNLP SRL not available. Skipping semantic role extraction.")
            return []
        
    def _extract_with_coreference(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relations with coreference resolution to connect entities.
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, predicate, object) tuples with resolved references
        """
        try:
            import neuralcoref
            
            # Initialize neuralcoref if not already done
            if not hasattr(self, 'coref_nlp'):
                self.coref_nlp = spacy.load('en_core_web_sm')
                neuralcoref.add_to_pipe(self.coref_nlp)
                logging.info("Coreference resolution model loaded successfully")
            
            # Process the text
            doc = self.coref_nlp(text)
            
            # Get resolved text with pronouns replaced by their referents
            resolved_text = doc._.coref_resolved
            
            # Extract relations from resolved text using standard method
            # First, parse the resolved text with the standard spaCy model
            resolved_doc = self.nlp(resolved_text)
            
            relations = []
            
            # Extract SVO triples from each sentence in the resolved text
            for sent in resolved_doc.sents:
                relations.extend(self._extract_svo_from_dependency(sent))
            
            return relations
            
        except ImportError:
            logging.debug("neuralcoref not available. Skipping coreference resolution.")
            return []
        
    def _get_span_text(self, token) -> str:
        """
        Get the full text span for a token, including its children.
        
        Args:
            token: The token to get span text for
            
        Returns:
            Text of the span
        """
        # If token has no children, just return its text
        if not list(token.children):
            return token.text
            
        # Otherwise, get the leftmost and rightmost token in the subtree
        leftmost = token
        rightmost = token
        
        # Find leftmost token
        for descendant in token.subtree:
            if descendant.i < leftmost.i:
                leftmost = descendant
                
        # Find rightmost token
        for descendant in token.subtree:
            if descendant.i > rightmost.i:
                rightmost = descendant
                
        # Return the span text
        return token.doc[leftmost.i:rightmost.i + 1].text
    
    def add_entity(self, entity: str, entity_type: str = None, metadata: Dict[str, Any] = None,
                  provenance: str = None, confidence: float = 0.8,
                  temporal_start: str = None, temporal_end: str = None,
                  extraction_method: str = None, changed_by: str = None) -> int:
        """
        Add an entity to the graph with enhanced metadata and versioning.
        
        Args:
            entity: Entity text
            entity_type: Type of entity (e.g., PERSON, LOCATION)
            metadata: Additional entity metadata
            provenance: Source of the entity information
            confidence: Confidence score for this entity (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this entity
            changed_by: Identifier of who/what made this change
            
        Returns:
            ID of the created entity
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        timestamp = time.time()
        
        try:
            # Check if entity already exists
            cursor.execute('SELECT id, version FROM graph_entities WHERE entity = ?', (entity,))
            existing = cursor.fetchone()
            
            if existing:
                # Get the current entity data for version history
                cursor.execute('''
                    SELECT entity, entity_type, metadata, provenance, confidence, 
                    temporal_start, temporal_end, extraction_method
                    FROM graph_entities WHERE id = ?
                ''', (existing[0],))
                current_data = cursor.fetchone()
                
                # Store the current version in version history
                new_version = existing[1] + 1
                cursor.execute('''
                    INSERT INTO entity_versions 
                    (entity_id, entity, entity_type, metadata, provenance, confidence, 
                     temporal_start, temporal_end, extraction_method, version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    existing[0],
                    current_data[0],
                    current_data[1],
                    current_data[2],
                    current_data[3],
                    current_data[4],
                    current_data[5],
                    current_data[6],
                    current_data[7],
                    existing[1],
                    timestamp,
                    "UPDATE",
                    changed_by
                ))
                
                # Update the entity with new data
                cursor.execute('''
                    UPDATE graph_entities 
                    SET entity_type = COALESCE(?, entity_type),
                        metadata = COALESCE(?, metadata),
                        provenance = COALESCE(?, provenance),
                        confidence = COALESCE(?, confidence),
                        temporal_start = COALESCE(?, temporal_start),
                        temporal_end = COALESCE(?, temporal_end),
                        extraction_method = COALESCE(?, extraction_method),
                        version = ?,
                        last_updated = ?
                    WHERE id = ?
                ''', (
                    entity_type,
                    json.dumps(metadata) if metadata else None,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    new_version,
                    timestamp,
                    existing[0]
                ))
                
                entity_id = existing[0]
            else:
                # Insert new entity
                cursor.execute('''
                    INSERT INTO graph_entities 
                    (entity, entity_type, metadata, timestamp, provenance, confidence, 
                     temporal_start, temporal_end, extraction_method, version, last_updated) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity,
                    entity_type,
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    1,  # Initial version
                    timestamp
                ))
                
                entity_id = cursor.lastrowid
                
                # Add to version history
                cursor.execute('''
                    INSERT INTO entity_versions 
                    (entity_id, entity, entity_type, metadata, provenance, confidence,
                     temporal_start, temporal_end, extraction_method, version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity_id,
                    entity,
                    entity_type,
                    json.dumps(metadata) if metadata else None,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    1,  # Initial version
                    timestamp,
                    "CREATE",
                    changed_by
                ))
            
            conn.commit()
            
            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                node_attrs = {
                    'name': entity,
                    'entity_type': entity_type,
                    'provenance': provenance,
                    'confidence': confidence,
                    'temporal_start': temporal_start,
                    'temporal_end': temporal_end,
                    'extraction_method': extraction_method,
                    'version': 1 if existing is None else existing[1] + 1,
                    'timestamp': timestamp
                }
                
                if metadata:
                    node_attrs.update(metadata)
                    
                self.graph.add_node(entity_id, **node_attrs)
            
            return entity_id
            
        except Exception as e:
            logging.error(f"Error adding entity: {e}")
            conn.rollback()
            return -1
            
        finally:
            if self.conn is None and conn:
                conn.close()
    
    def add_entity_alias(self, entity_id: int, alias: str, confidence: float = 0.8) -> bool:
        """
        Add an alias to an existing entity for entity linking.
        
        Args:
            entity_id: ID of the existing entity
            alias: Alternative name or reference to the entity
            confidence: Confidence score for this alias (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Verify entity exists
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get entity information
            cursor.execute('SELECT entity, metadata FROM graph_entities WHERE id = ?', (entity_id,))
            result = cursor.fetchone()
            
            if not result:
                logging.error(f"Entity with ID {entity_id} not found")
                return False
                
            canonical_name = result['entity']
            metadata_str = result['metadata']
            
            # Update metadata to include the new alias
            if metadata_str:
                metadata = json.loads(metadata_str)
            else:
                metadata = {}
                
            if 'aliases' not in metadata:
                metadata['aliases'] = []
                
            # Add alias if not already present
            if alias not in metadata['aliases']:
                metadata['aliases'].append(alias)
                metadata['alias_confidence'] = metadata.get('alias_confidence', {})
                metadata['alias_confidence'][alias] = confidence
                
                # Update database
                cursor.execute(
                    'UPDATE graph_entities SET metadata = ? WHERE id = ?',
                    (json.dumps(metadata), entity_id)
                )
                
                # Update entity linking database
                self.entity_db[canonical_name] = {
                    'id': entity_id,
                    'metadata': metadata,
                    'aliases': metadata['aliases']
                }
                
                self.entity_db[alias] = {
                    'canonical': canonical_name,
                    'id': entity_id
                }
                
                conn.commit()
                return True
            else:
                # Alias already exists
                return True
                
        except Exception as e:
            logging.error(f"Error adding entity alias: {e}")
            conn.rollback()
            return False
        finally:
            if self.conn is None:
                conn.close()
    
    def add_relation(self, source_entity: str, relation_type: str, target_entity: str, 
                     weight: float = 1.0, metadata: Dict[str, Any] = None,
                     provenance: str = None, confidence: float = 0.5,
                     temporal_start: str = None, temporal_end: str = None,
                     extraction_method: str = None, changed_by: str = None) -> bool:
        """
        Add a relation between two entities with enhanced metadata and versioning.
        
        Args:
            source_entity: Source entity text
            relation_type: Type of relation
            target_entity: Target entity text
            weight: Weight/importance of the relation
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this relation
            changed_by: Identifier of who/what made this change
            
        Returns:
            True if relation was added successfully
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        timestamp = time.time()
        
        try:
            # Verify relation type exists
            cursor.execute('SELECT id, symmetric, transitive, inverse_relation FROM relation_types WHERE name = ?', (relation_type,))
            relation_type_data = cursor.fetchone()
            
            if not relation_type_data:
                # Add the relation type dynamically
                logging.info(f"Adding new relation type: {relation_type}")
                cursor.execute('''
                    INSERT INTO relation_types (name, parent_type, description, metadata) 
                    VALUES (?, 'related_to', ?, ?)
                ''', (
                    relation_type,
                    f"Dynamically added relation type: {relation_type}",
                    json.dumps({"automatic": True, "added_at": timestamp})
                ))
                
                # Get symmetry and transitivity for new relation type
                is_symmetric = False
                is_transitive = False
                inverse_relation = None
            else:
                is_symmetric = bool(relation_type_data[1])
                is_transitive = bool(relation_type_data[2])
                inverse_relation = relation_type_data[3]
                
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (source_entity,))
            source = cursor.fetchone()
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (target_entity,))
            target = cursor.fetchone()
            
            # If either entity doesn't exist, create them
            if not source:
                source_id = self.add_entity(
                    source_entity, 
                    provenance=provenance, 
                    confidence=confidence, 
                    extraction_method=extraction_method,
                    changed_by=changed_by
                )
            else:
                source_id = source[0]
                
            if not target:
                target_id = self.add_entity(
                    target_entity, 
                    provenance=provenance, 
                    confidence=confidence, 
                    extraction_method=extraction_method,
                    changed_by=changed_by
                )
            else:
                target_id = target[0]
            
            # Check if relation already exists
            cursor.execute('''
                SELECT id, version FROM graph_relationships 
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
            ''', (source_id, target_id, relation_type))
            existing = cursor.fetchone()
            
            if existing:
                # Get the current relation data for version history
                cursor.execute('''
                    SELECT source_id, target_id, relation_type, weight, metadata, 
                    provenance, confidence, temporal_start, temporal_end, extraction_method
                    FROM graph_relationships WHERE id = ?
                ''', (existing[0],))
                current_data = cursor.fetchone()
                
                # Store the current version in version history
                new_version = existing[1] + 1
                cursor.execute('''
                    INSERT INTO relationship_versions 
                    (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                     provenance, confidence, temporal_start, temporal_end, extraction_method, 
                     version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    existing[0],
                    current_data[0],
                    current_data[1],
                    current_data[2],
                    current_data[3],
                    current_data[4],
                    current_data[5],
                    current_data[6],
                    current_data[7],
                    current_data[8],
                    current_data[9],
                    existing[1],
                    timestamp,
                    "UPDATE",
                    changed_by
                ))
                
                # Update existing relation
                cursor.execute('''
                    UPDATE graph_relationships 
                    SET weight = COALESCE(?, weight),
                        metadata = COALESCE(?, metadata),
                        provenance = COALESCE(?, provenance),
                        confidence = COALESCE(?, confidence),
                        temporal_start = COALESCE(?, temporal_start),
                        temporal_end = COALESCE(?, temporal_end),
                        extraction_method = COALESCE(?, extraction_method),
                        version = ?,
                        last_updated = ?
                    WHERE id = ?
                ''', (
                    weight,
                    json.dumps(metadata) if metadata else None,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    new_version,
                    timestamp,
                    existing[0]
                ))
                
                relation_id = existing[0]
            else:
                # Insert new relation
                cursor.execute('''
                    INSERT INTO graph_relationships 
                    (source_id, target_id, relation_type, weight, metadata, timestamp,
                     provenance, confidence, temporal_start, temporal_end, extraction_method, 
                     version, last_updated) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    source_id, 
                    target_id, 
                    relation_type,
                    weight,
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    1,  # Initial version
                    timestamp
                ))
                
                relation_id = cursor.lastrowid
                
                # Add to version history
                cursor.execute('''
                    INSERT INTO relationship_versions 
                    (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                     provenance, confidence, temporal_start, temporal_end, extraction_method, 
                     version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    relation_id,
                    source_id,
                    target_id,
                    relation_type,
                    weight,
                    json.dumps(metadata) if metadata else None,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    extraction_method,
                    1,  # Initial version
                    timestamp,
                    "CREATE",
                    changed_by
                ))
                
                # If relation is symmetric, add the reverse relation
                if is_symmetric and source_id != target_id:
                    cursor.execute('''
                        INSERT INTO graph_relationships 
                        (source_id, target_id, relation_type, weight, metadata, timestamp,
                         provenance, confidence, temporal_start, temporal_end, extraction_method, 
                         version, last_updated) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        target_id, 
                        source_id, 
                        relation_type,
                        weight,
                        json.dumps({**(metadata or {}), "symmetric_of": relation_id}),
                        timestamp,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp
                    ))
                    
                    symmetric_id = cursor.lastrowid
                    
                    # Add symmetric relation to version history
                    cursor.execute('''
                        INSERT INTO relationship_versions 
                        (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                         provenance, confidence, temporal_start, temporal_end, extraction_method, 
                         version, timestamp, change_type, changed_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symmetric_id,
                        target_id,
                        source_id,
                        relation_type,
                        weight,
                        json.dumps({**(metadata or {}), "symmetric_of": relation_id}),
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                        "CREATE_SYMMETRIC",
                        changed_by
                    ))
                
                # If inverse relation is defined, add the inverse relation
                if inverse_relation and source_id != target_id:
                    cursor.execute('''
                        INSERT INTO graph_relationships 
                        (source_id, target_id, relation_type, weight, metadata, timestamp,
                         provenance, confidence, temporal_start, temporal_end, extraction_method, 
                         version, last_updated) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        target_id, 
                        source_id, 
                        inverse_relation,
                        weight,
                        json.dumps({**(metadata or {}), "inverse_of": relation_id}),
                        timestamp,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp
                    ))
                    
                    inverse_id = cursor.lastrowid
                    
                    # Add inverse relation to version history
                    cursor.execute('''
                        INSERT INTO relationship_versions 
                        (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                         provenance, confidence, temporal_start, temporal_end, extraction_method, 
                         version, timestamp, change_type, changed_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        inverse_id,
                        target_id,
                        source_id,
                        inverse_relation,
                        weight,
                        json.dumps({**(metadata or {}), "inverse_of": relation_id}),
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                        "CREATE_INVERSE",
                        changed_by
                    ))
                
                # If relation is transitive, check for implied transitive relations
                if is_transitive:
                    self._add_transitive_relations(
                        source_id, target_id, relation_type, 
                        weight, provenance, confidence, extraction_method, changed_by
                    )
            
            conn.commit()
            
            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                edge_attrs = {
                    'relation': relation_type,
                    'weight': weight,
                    'provenance': provenance,
                    'confidence': confidence,
                    'temporal_start': temporal_start,
                    'temporal_end': temporal_end,
                    'extraction_method': extraction_method,
                    'timestamp': timestamp,
                    'version': 1 if existing is None else existing[1] + 1
                }
                
                if metadata:
                    edge_attrs.update(metadata)
                    
                self.graph.add_edge(source_id, target_id, **edge_attrs)
                
                # Add symmetric and inverse edges to NetworkX if applicable
                if is_symmetric and source_id != target_id:
                    self.graph.add_edge(target_id, source_id, **{
                        **edge_attrs,
                        'symmetric_of': relation_id
                    })
                    
                if inverse_relation and source_id != target_id:
                    self.graph.add_edge(target_id, source_id, **{
                        **edge_attrs,
                        'relation': inverse_relation,
                        'inverse_of': relation_id
                    })
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding relation: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            conn.rollback()
            return False
            
        finally:
            if self.conn is None:
                conn.close()
    
    def _add_transitive_relations(self, source_id: int, target_id: int, relation_type: str,
                                weight: float, provenance: str, confidence: float,
                                extraction_method: str, changed_by: str) -> None:
        """
        Add implied transitive relations.
        
        For example, if A related_to B and B related_to C, then A related_to C 
        for transitive relations like "is_a" or "part_of".
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relation
            weight: Weight of the relation
            provenance: Source of the relation information
            confidence: Confidence score for this relation
            extraction_method: Method used to extract this relation
            changed_by: Identifier of who/what made this change
        """
        if self.conn is None:
            return
            
        cursor = self.conn.cursor()
        timestamp = time.time()
        
        try:
            # Find all entities that the source is related to with this relation
            cursor.execute('''
                SELECT target_id FROM graph_relationships
                WHERE source_id = ? AND relation_type = ?
            ''', (source_id, relation_type))
            sources_targets = cursor.fetchall()
            
            # Find all entities that have this relation to the target
            cursor.execute('''
                SELECT source_id FROM graph_relationships
                WHERE target_id = ? AND relation_type = ?
            ''', (target_id, relation_type))
            targets_sources = cursor.fetchall()
            
            # Add transitive relations: source -> target's sources
            for row in targets_sources:
                third_id = row[0]
                if third_id != source_id:  # Avoid self-relations
                    # Check if this relation already exists
                    cursor.execute('''
                        SELECT id FROM graph_relationships
                        WHERE source_id = ? AND target_id = ? AND relation_type = ?
                    ''', (source_id, third_id, relation_type))
                    if not cursor.fetchone():
                        # Add the transitive relation with reduced confidence
                        new_confidence = confidence * 0.9  # Reduce confidence for transitive inference
                        
                        # Add the relation
                        transitive_metadata = {
                            "transitive": True,
                            "via_entity": target_id,
                            "inference_type": "transitive_relation"
                        }
                        
                        cursor.execute('''
                            INSERT INTO graph_relationships
                            (source_id, target_id, relation_type, weight, metadata, timestamp,
                             provenance, confidence, extraction_method, version, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            source_id,
                            third_id,
                            relation_type,
                            weight * 0.8,  # Reduce weight for transitive relations
                            json.dumps(transitive_metadata),
                            timestamp,
                            provenance,
                            new_confidence,
                            "transitive_inference",
                            1,
                            timestamp
                        ))
                        
                        transitive_id = cursor.lastrowid
                        
                        # Add to version history
                        cursor.execute('''
                            INSERT INTO relationship_versions
                            (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                             provenance, confidence, extraction_method, version, timestamp, change_type, changed_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            transitive_id,
                            source_id,
                            third_id,
                            relation_type,
                            weight * 0.8,
                            json.dumps(transitive_metadata),
                            provenance,
                            new_confidence,
                            "transitive_inference",
                            1,
                            timestamp,
                            "CREATE_TRANSITIVE",
                            changed_by
                        ))
            
            # Add transitive relations: source's sources -> target
            for row in sources_targets:
                third_id = row[0]
                if third_id != target_id:  # Avoid self-relations
                    # Check if this relation already exists
                    cursor.execute('''
                        SELECT id FROM graph_relationships
                        WHERE source_id = ? AND target_id = ? AND relation_type = ?
                    ''', (third_id, target_id, relation_type))
                    if not cursor.fetchone():
                        # Add the transitive relation with reduced confidence
                        new_confidence = confidence * 0.9  # Reduce confidence for transitive inference
                        
                        # Add the relation
                        transitive_metadata = {
                            "transitive": True,
                            "via_entity": source_id,
                            "inference_type": "transitive_relation"
                        }
                        
                        cursor.execute('''
                            INSERT INTO graph_relationships
                            (source_id, target_id, relation_type, weight, metadata, timestamp,
                             provenance, confidence, extraction_method, version, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            third_id,
                            target_id,
                            relation_type,
                            weight * 0.8,  # Reduce weight for transitive relations
                            json.dumps(transitive_metadata),
                            timestamp,
                            provenance,
                            new_confidence,
                            "transitive_inference",
                            1,
                            timestamp
                        ))
                        
                        transitive_id = cursor.lastrowid
                        
                        # Add to version history
                        cursor.execute('''
                            INSERT INTO relationship_versions
                            (relationship_id, source_id, target_id, relation_type, weight, metadata, 
                             provenance, confidence, extraction_method, version, timestamp, change_type, changed_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            transitive_id,
                            third_id,
                            target_id,
                            relation_type,
                            weight * 0.8,
                            json.dumps(transitive_metadata),
                            provenance,
                            new_confidence,
                            "transitive_inference",
                            1,
                            timestamp,
                            "CREATE_TRANSITIVE",
                            changed_by
                        ))
                        
        except Exception as e:
            logging.error(f"Error adding transitive relations: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - this is a best-effort operation
    
    def add_nary_relation(self, relation_type: str, participants: Dict[str, str], 
                        metadata: Dict[str, Any] = None, provenance: str = None, 
                        confidence: float = 0.5) -> int:
        """
        Add an n-ary relation involving multiple entities with different roles.
        
        Args:
            relation_type: Type of the n-ary relation
            participants: Dictionary mapping role names to entity names
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)
            
        Returns:
            ID of the created n-ary relation or -1 if failed
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        timestamp = time.time()
        
        try:
            # If ontology is enabled, check the relation type in the ontology
            if ONTOLOGY_ENABLED and self.ontology:
                if not self.ontology.get_relation_type(relation_type):
                    # Add a basic relation type if needed
                    from cortexflow.ontology import RelationType
                    self.ontology.add_relation_type(
                        RelationType(
                            name=relation_type,
                            parent_types=["related_to"],
                            metadata={
                                "n_ary": True,
                                "automatic": True,
                                "confidence": 0.7
                            }
                        )
                    )
                    logging.info(f"Added n-ary relation type to ontology: {relation_type}")
            
            # Insert the n-ary relation
            cursor.execute('''
                INSERT INTO nary_relationships 
                (relation_type, metadata, provenance, confidence, timestamp) 
                VALUES (?, ?, ?, ?, ?)
            ''', (
                relation_type,
                json.dumps(metadata) if metadata else None,
                provenance,
                confidence,
                timestamp
            ))
            
            relation_id = cursor.lastrowid
            
            # Add all participants
            for role, entity_name in participants.items():
                # Get or create the entity
                cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity_name,))
                entity_row = cursor.fetchone()
                
                if not entity_row:
                    entity_id = self.add_entity(
                        entity=entity_name, 
                        provenance=provenance,
                        confidence=confidence
                    )
                else:
                    entity_id = entity_row[0]
                
                # Add the participant
                cursor.execute('''
                    INSERT INTO nary_participants
                    (relationship_id, entity_id, role, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    relation_id,
                    entity_id,
                    role,
                    timestamp
                ))
            
            conn.commit()
            
            # Add to NetworkX graph if enabled (using a hypergraph-like representation)
            if NETWORKX_ENABLED and self.graph is not None:
                # Create a special node for the n-ary relation
                relation_node_id = f"nary_{relation_id}"
                self.graph.add_node(
                    relation_node_id,
                    relation_type=relation_type,
                    n_ary=True,
                    provenance=provenance,
                    confidence=confidence,
                    timestamp=timestamp,
                    **(metadata or {})
                )
                
                # Connect all participants to the relation node
                for role, entity_name in participants.items():
                    cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity_name,))
                    entity_row = cursor.fetchone()
                    if entity_row:
                        entity_id = entity_row[0]
                        self.graph.add_edge(
                            entity_id,
                            relation_node_id,
                            role=role,
                            weight=1.0,
                            timestamp=timestamp
                        )
            
            return relation_id
            
        except Exception as e:
            logging.error(f"Error adding n-ary relation: {e}")
            conn.rollback()
            return -1
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_nary_relation(self, relation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get details of an n-ary relation by ID.
        
        Args:
            relation_id: ID of the n-ary relation
            
        Returns:
            Dictionary with relation details or None if not found
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get the relation details
            cursor.execute('''
                SELECT relation_type, metadata, provenance, confidence, timestamp
                FROM nary_relationships
                WHERE id = ?
            ''', (relation_id,))
            
            relation_row = cursor.fetchone()
            
            if not relation_row:
                return None
                
            # Convert row to dictionary
            relation = dict(relation_row)
            
            # Parse metadata JSON
            if relation['metadata']:
                relation['metadata'] = json.loads(relation['metadata'])
            else:
                relation['metadata'] = {}
                
            # Get all participants
            cursor.execute('''
                SELECT np.role, ge.entity, ge.entity_type, ge.id as entity_id
                FROM nary_participants np
                JOIN graph_entities ge ON np.entity_id = ge.id
                WHERE np.relationship_id = ?
            ''', (relation_id,))
            
            participants = {}
            for row in cursor.fetchall():
                participants[row['role']] = {
                    'entity': row['entity'],
                    'entity_type': row['entity_type'],
                    'entity_id': row['entity_id']
                }
                
            relation['participants'] = participants
            relation['id'] = relation_id
            
            return relation
            
        except Exception as e:
            logging.error(f"Error getting n-ary relation: {e}")
            return None
            
        finally:
            if self.conn is None:
                conn.close()
    
    def query_nary_relations(self, relation_type: str = None, 
                          participant_entity: str = None, 
                          participant_role: str = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query n-ary relations with optional filters.
        
        Args:
            relation_type: Type of relation to filter by (optional)
            participant_entity: Filter by entity name participating in the relation (optional)
            participant_role: Filter by role in the relation (optional)
            limit: Maximum number of results
            
        Returns:
            List of n-ary relation dictionaries
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            params = []
            
            # Start with base query
            query = '''
                SELECT DISTINCT nr.id as relation_id
                FROM nary_relationships nr
            '''
            
            # Add join conditions if needed
            if participant_entity or participant_role:
                query += '''
                    JOIN nary_participants np ON nr.id = np.relationship_id
                    JOIN graph_entities ge ON np.entity_id = ge.id
                '''
            
            # Add WHERE clause
            where_clauses = []
            
            if relation_type:
                where_clauses.append("nr.relation_type = ?")
                params.append(relation_type)
                
            if participant_entity:
                where_clauses.append("ge.entity = ?")
                params.append(participant_entity)
                
            if participant_role:
                where_clauses.append("np.role = ?")
                params.append(participant_role)
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            # Add order and limit
            query += " ORDER BY nr.timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query to get relation IDs
            cursor.execute(query, params)
            relation_ids = [row['relation_id'] for row in cursor.fetchall()]
            
            # Get full details for each relation
            results = []
            for relation_id in relation_ids:
                relation = self.get_nary_relation(relation_id)
                if relation:
                    results.append(relation)
                    
            return results
            
        except Exception as e:
            logging.error(f"Error querying n-ary relations: {e}")
            return []
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_entity_metadata(self, entity_id: int) -> Dict[str, Any]:
        """
        Get full metadata for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary with entity metadata
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT entity, entity_type, metadata, provenance, confidence, 
                       temporal_start, temporal_end, timestamp
                FROM graph_entities
                WHERE id = ?
            ''', (entity_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
                
            result = dict(row)
            
            # Parse metadata JSON
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
            else:
                result['metadata'] = {}
                
            result['id'] = entity_id
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting entity metadata: {e}")
            return {}
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_relation_metadata(self, relation_id: int) -> Dict[str, Any]:
        """
        Get full metadata for a relation.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Dictionary with relation metadata
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT source_id, target_id, relation_type, weight, metadata, 
                       provenance, confidence, temporal_start, temporal_end, timestamp
                FROM graph_relationships
                WHERE id = ?
            ''', (relation_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
                
            result = dict(row)
            
            # Parse metadata JSON
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
            else:
                result['metadata'] = {}
                
            # Get source and target entity details
            source = self.get_entity_metadata(result['source_id'])
            target = self.get_entity_metadata(result['target_id'])
            
            result['source'] = source
            result['target'] = target
            result['id'] = relation_id
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting relation metadata: {e}")
            return {}
            
        finally:
            if self.conn is None:
                conn.close()
    
    def process_text_to_graph(self, text: str, source: str = None) -> int:
        """
        Process text to extract entities and relations and add them to the graph.
        Enhanced to support n-ary relationships and metadata tracking.
        
        Args:
            text: The text to process
            source: The source of the text (for provenance)
            
        Returns:
            Number of relations added to the graph
        """
        relations_added = 0
        
        try:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Extract binary relations using existing methods
            binary_relations = self.extract_relations(text)
            
            # Add binary relations to graph
            for subj, pred, obj in binary_relations:
                success = self.add_relation(
                    source_entity=subj,
                    relation_type=pred,
                    target_entity=obj,
                    provenance=source,
                    confidence=0.7  # Default confidence for extracted relations
                )
                
                if success:
                    relations_added += 1
            
            # Try to extract n-ary relations with roles
            nary_relations = self._extract_complex_events(text)
            
            # Add n-ary relations to graph
            for relation in nary_relations:
                relation_type = relation.get('type', 'event')
                participants = relation.get('participants', {})
                
                if participants:
                    nary_id = self.add_nary_relation(
                        relation_type=relation_type,
                        participants=participants,
                        provenance=source,
                        confidence=0.6  # Default confidence for n-ary relations
                    )
                    
                    if nary_id > 0:
                        relations_added += 1
            
            return relations_added
            
        except Exception as e:
            logging.error(f"Error processing text to graph: {e}")
            return 0
    
    def _extract_complex_events(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract complex events and scenarios as n-ary relations with roles.
        
        Args:
            text: Text to process
            
        Returns:
            List of n-ary relation dictionaries
        """
        complex_events = []
        
        # Basic implementation using existing NLP capabilities
        if not SPACY_ENABLED or not self.nlp:
            return complex_events
            
        try:
            doc = self.nlp(text)
            
            # Look for sentences with multiple entities or complex structure
            for sent in doc.sents:
                # Check if this is a complex sentence (multiple verbs or entities)
                entities = [e for e in sent.ents]
                verbs = [token for token in sent if token.pos_ == "VERB"]
                
                # Skip simple sentences
                if len(entities) < 2 or len(verbs) < 1:
                    continue
                    
                # For each main verb, try to identify an event
                for verb in verbs:
                    # Skip auxiliary verbs
                    if verb.dep_ in ("aux", "auxpass"):
                        continue
                        
                    # Get the verb lemma as event type
                    event_type = verb.lemma_
                    
                    # Collect participants by role
                    participants = {}
                    
                    # Check for subject
                    subjects = [token for token in verb.children if token.dep_ in ("nsubj", "nsubjpass")]
                    for subject in subjects:
                        # Extend to noun phrases
                        subj_span = self._get_span_text(subject)
                        participants["agent"] = subj_span
                    
                    # Check for object
                    objects = [token for token in verb.children if token.dep_ in ("dobj", "pobj")]
                    for obj in objects:
                        # Extend to noun phrases
                        obj_span = self._get_span_text(obj)
                        participants["theme"] = obj_span
                    
                    # Check for indirect object
                    ind_objects = [token for token in verb.children if token.dep_ == "iobj"]
                    for ind_obj in ind_objects:
                        # Extend to noun phrases
                        ind_obj_span = self._get_span_text(ind_obj)
                        participants["recipient"] = ind_obj_span
                    
                    # Check for time expressions
                    time_preps = [token for token in verb.children if token.dep_ == "prep" and token.text in ("at", "on", "in")]
                    for prep in time_preps:
                        for child in prep.children:
                            if child.dep_ == "pobj":
                                time_span = self._get_span_text(child)
                                participants["time"] = time_span
                    
                    # Check for location
                    loc_preps = [token for token in verb.children if token.dep_ == "prep" and token.text in ("at", "in", "on", "near", "by")]
                    for prep in loc_preps:
                        for child in prep.children:
                            if child.dep_ == "pobj":
                                # Check if this is actually a location
                                if child.ent_type_ in ("LOC", "GPE"):
                                    loc_span = self._get_span_text(child)
                                    participants["location"] = loc_span
                    
                    # If we have at least two participants, add the event
                    if len(participants) >= 2:
                        complex_events.append({
                            "type": event_type,
                            "participants": participants,
                            "sentence": sent.text
                        })
            
        except Exception as e:
            logging.error(f"Error extracting complex events: {e}")
            
        return complex_events
    
    def get_entity_neighbors(self, entity: str, direction: str = "both", 
                           relation_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get neighbors of an entity in the graph.
        
        Args:
            entity: Entity name to find neighbors for
            direction: Direction of relations: "outgoing", "incoming", or "both"
            relation_type: Filter by specific relation type (optional)
            limit: Maximum number of neighbors to return
            
        Returns:
            List of neighbor dictionaries with relation details
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        neighbors = []
        
        try:
            # Get entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity,))
            entity_row = cursor.fetchone()
            
            if not entity_row:
                return []
                
            entity_id = entity_row['id']
            
            # Get outgoing relations
            if direction in ["outgoing", "both"]:
                query = '''
                    SELECT r.id, r.relation_type, r.weight, r.provenance, r.confidence,
                           e.id as target_id, e.entity, e.entity_type
                    FROM graph_relationships r
                    JOIN graph_entities e ON r.target_id = e.id
                    WHERE r.source_id = ?
                '''
                
                params = [entity_id]
                
                if relation_type:
                    query += " AND r.relation_type = ?"
                    params.append(relation_type)
                    
                query += " ORDER BY r.confidence DESC, r.weight DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    neighbor = dict(row)
                    neighbor['direction'] = 'outgoing'
                    neighbor['relation'] = neighbor['relation_type']
                    neighbors.append(neighbor)
            
            # Get incoming relations
            if direction in ["incoming", "both"]:
                remaining_limit = limit - len(neighbors)
                
                if remaining_limit > 0:
                    query = '''
                        SELECT r.id, r.relation_type, r.weight, r.provenance, r.confidence,
                               e.id as source_id, e.entity, e.entity_type
                        FROM graph_relationships r
                        JOIN graph_entities e ON r.source_id = e.id
                        WHERE r.target_id = ?
                    '''
                    
                    params = [entity_id]
                    
                    if relation_type:
                        query += " AND r.relation_type = ?"
                        params.append(relation_type)
                        
                    query += " ORDER BY r.confidence DESC, r.weight DESC LIMIT ?"
                    params.append(remaining_limit)
                    
                    cursor.execute(query, params)
                    
                    for row in cursor.fetchall():
                        neighbor = dict(row)
                        neighbor['direction'] = 'incoming'
                        neighbor['relation'] = neighbor['relation_type']
                        neighbors.append(neighbor)
            
            # Sort by weight and confidence
            neighbors = sorted(neighbors, key=lambda x: (x.get('confidence', 0.0), x.get('weight', 0.0)), reverse=True)
            
            return neighbors[:limit]
            
        except Exception as e:
            logging.error(f"Error getting entity neighbors: {e}")
            return []
            
        finally:
            if self.conn is None:
                conn.close()
    
    def build_knowledge_subgraph(self, query: str, max_nodes: int = 20) -> Dict[str, Any]:
        """
        Build a knowledge subgraph relevant to a query.
        
        Args:
            query: Query text to extract entities from
            max_nodes: Maximum number of nodes to include in the subgraph
            
        Returns:
            Dictionary with nodes and edges of the subgraph
        """
        subgraph = {"nodes": [], "edges": []}
        
        try:
            # Extract potential entities from the query
            query_entities = []
            
            # Use NLP to extract entities if available
            if SPACY_ENABLED and self.nlp:
                doc = self.nlp(query)
                for ent in doc.ents:
                    query_entities.append(ent.text)
                    
                # Also add noun chunks as potential entities
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2 and chunk.text not in query_entities:
                        query_entities.append(chunk.text)
            
            # If no entities found with NLP, use simple word extraction
            if not query_entities:
                for word in query.split():
                    if len(word) > 3 and (word[0].isupper() or word in query.lower()):
                        query_entities.append(word)
            
            # Check for common entity patterns in queries
            connection_pattern = r"(?:connection|relationship|relation)(?:\s+between\s+)([^,]+?)(?:\s+and\s+)([^?\.]+)"
            match = re.search(connection_pattern, query, re.IGNORECASE)
            if match:
                start_entity = match.group(1).strip()
                end_entity = match.group(2).strip()
                if start_entity not in query_entities:
                    query_entities.append(start_entity)
                if end_entity not in query_entities:
                    query_entities.append(end_entity)
            
            # Track nodes and edges to avoid duplicates
            node_ids = set()
            edge_ids = set()
            
            # For each entity in the query, get neighbors
            for entity_text in query_entities:
                neighbors = self.get_entity_neighbors(
                    entity=entity_text,
                    direction="both",
                    limit=5
                )
                
                if not neighbors:
                    continue
                
                # Safely get source entity ID
                source_id = None
                for neighbor in neighbors:
                    if 'source_id' in neighbor and neighbor['direction'] == 'outgoing':
                        source_id = neighbor['source_id']
                        break
                    elif 'target_id' in neighbor and neighbor['direction'] == 'incoming':
                        source_id = neighbor['target_id']
                        break
                
                # If we can't find source ID, try a direct lookup
                if source_id is None:
                    entity_info = self.get_entity_id(entity_text)
                    if entity_info:
                        source_id = entity_info.get('id')
                
                # Add the entity to nodes if we have an ID and it's not already added
                if source_id and source_id not in node_ids:
                    try:
                        # Get entity details
                        entity_details = self.get_entity_metadata(source_id) or {}
                        
                        # Create node
                        node = {
                            "id": source_id,
                            "label": entity_text,
                            "type": entity_details.get('entity_type', 'unknown'),
                            "confidence": entity_details.get('confidence', 0.5)
                        }
                        
                        subgraph["nodes"].append(node)
                        node_ids.add(source_id)
                    except Exception as e:
                        logging.error(f"Error adding source node {source_id}: {e}")
                
                # Process each neighbor
                for neighbor in neighbors:
                    try:
                        # Skip neighbors without required fields
                        if 'entity' not in neighbor:
                            logging.warning(f"Skipping neighbor without entity field: {neighbor}")
                            continue
                            
                        neighbor_entity = neighbor['entity']
                        relation = neighbor.get('relation', 'related_to')
                        
                        # Safely get neighbor ID
                        neighbor_id = None
                        if neighbor['direction'] == 'outgoing' and 'target_id' in neighbor:
                            neighbor_id = neighbor['target_id']
                        elif neighbor['direction'] == 'incoming' and 'source_id' in neighbor:
                            neighbor_id = neighbor['source_id']
                        
                        # If we can't get neighbor ID, try direct lookup
                        if neighbor_id is None:
                            nb_info = self.get_entity_id(neighbor_entity)
                            if nb_info:
                                neighbor_id = nb_info.get('id')
                        
                        # Skip if we still don't have required IDs
                        if not source_id or not neighbor_id:
                            logging.warning(f"Missing ID for {entity_text} or {neighbor_entity}")
                            continue
                        
                        # Add neighbor node if new
                        if neighbor_id not in node_ids:
                            try:
                                # Get entity details
                                entity_details = self.get_entity_metadata(neighbor_id) or {}
                                
                                # Create node
                                node = {
                                    "id": neighbor_id,
                                    "label": neighbor_entity,
                                    "type": entity_details.get('entity_type', 'unknown'),
                                    "confidence": entity_details.get('confidence', 0.5)
                                }
                                
                                subgraph["nodes"].append(node)
                                node_ids.add(neighbor_id)
                            except Exception as e:
                                logging.error(f"Error adding neighbor node {neighbor_id}: {e}")
                                continue
                        
                        # Add edge
                        edge_id = f"{source_id}_{neighbor_id}_{relation}" if neighbor['direction'] == 'outgoing' else f"{neighbor_id}_{source_id}_{relation}"
                        
                        if edge_id not in edge_ids:
                            edge = {
                                "source": source_id if neighbor['direction'] == 'outgoing' else neighbor_id,
                                "target": neighbor_id if neighbor['direction'] == 'outgoing' else source_id,
                                "label": relation,
                                "weight": neighbor.get('weight', 1.0),
                                "confidence": neighbor.get('confidence', 0.5)
                            }
                            
                            subgraph["edges"].append(edge)
                            edge_ids.add(edge_id)
                    
                    except Exception as e:
                        logging.error(f"Error processing neighbor: {e}")
                
                # Check if we reached max nodes
                if len(node_ids) >= max_nodes:
                    break
            
            return subgraph
            
        except Exception as e:
            logging.error(f"Error building knowledge subgraph: {e}")
            return {"nodes": [], "edges": []}
            
    def get_entity_id(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Get entity ID from text.
        
        Args:
            entity_text: Entity text to look up
            
        Returns:
            Dictionary with entity information or None if not found
        """
        try:
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute('SELECT id, entity, entity_type FROM graph_entities WHERE entity = ?', (entity_text,))
            entity_row = cursor.fetchone()
            
            if not entity_row:
                # Try case-insensitive matching
                cursor.execute('SELECT id, entity, entity_type FROM graph_entities WHERE LOWER(entity) = LOWER(?)', (entity_text,))
                entity_row = cursor.fetchone()
            
            if not entity_row:
                # Try fuzzy matching as last resort
                cursor.execute('SELECT id, entity, entity_type FROM graph_entities WHERE entity LIKE ? LIMIT 1', (f"%{entity_text}%",))
                entity_row = cursor.fetchone()
            
            if self.conn is None:
                conn.close()
                
            if entity_row:
                return dict(entity_row)
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting entity ID: {e}")
            return None
    
    def path_query(self, start_entity: str, end_entity: str, max_hops: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities in the graph.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length
            
        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for path queries")
            return []
        
        paths = []
        
        try:
            # Get entity IDs
            source_id = None
            target_id = None
            
            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()
            
            if source_row:
                source_id = source_row[0]
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_id = target_row[0]
            
            if self.conn is None:
                conn.close()
            
            # Return empty if entities not found
            if not source_id or not target_id:
                return []
            
            # Use NetworkX for path finding
            try:
                # Find all simple paths (can be slow for large graphs)
                simple_paths = nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops)
                
                # Convert paths to our format
                for path in list(simple_paths)[:5]:  # Limit to top 5 paths
                    formatted_path = []
                    
                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)
                        
                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }
                        
                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)
                            
                            if edge_data:
                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_data.get('weight', 1.0),
                                    "confidence": edge_data.get('confidence', 0.5)
                                }
                                node_info["next_relation"] = relation_info
                        
                        formatted_path.append(node_info)
                    
                    paths.append(formatted_path)
                
            except nx.NetworkXNoPath:
                # No path exists
                pass
            
            return paths
            
        except Exception as e:
            logging.error(f"Error in path query: {e}")
            return []
    
    def weighted_path_query(self, start_entity: str, end_entity: str, 
                          max_hops: int = 3, importance_weight: float = 0.6, 
                          confidence_weight: float = 0.4) -> List[List[Dict[str, Any]]]:
        """
        Find weighted paths between entities considering relation importance and confidence.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length
            importance_weight: Weight factor for relation importance (0-1)
            confidence_weight: Weight factor for relation confidence (0-1)
            
        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for weighted path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for weighted path queries")
            return []
        
        weighted_paths = []
        
        try:
            # Get entity IDs
            source_id = None
            target_id = None
            
            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()
            
            if source_row:
                source_id = source_row[0]
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_id = target_row[0]
            
            if self.conn is None:
                conn.close()
            
            # Return empty if entities not found
            if not source_id or not target_id:
                return []
            
            # Create a copy of the graph with calculated weights
            weighted_graph = nx.DiGraph()
            
            # Copy nodes 
            for node in self.graph.nodes():
                weighted_graph.add_node(node)
            
            # Copy edges with inverted weights
            for u, v, data in self.graph.edges(data=True):
                # Calculate combined weight based on importance and confidence
                edge_weight = data.get('weight', 0.5)
                edge_confidence = data.get('confidence', 0.5)
                
                # Normalize weights to 0-1 range
                norm_weight = min(max(edge_weight, 0.1), 1.0)
                norm_confidence = min(max(edge_confidence, 0.1), 1.0)
                
                # Calculate combined weight
                combined_weight = (importance_weight * norm_weight) + (confidence_weight * norm_confidence)
                
                # Invert weight for shortest path algorithm (higher weight/confidence = shorter path)
                inverted_weight = 1.0 / combined_weight if combined_weight > 0 else float('inf')
                
                # Create a copy of the data without the weight to avoid conflict
                edge_data = data.copy()
                if 'weight' in edge_data:
                    del edge_data['weight']
                
                # Add edge with inverted weight
                weighted_graph.add_edge(u, v, weight=inverted_weight, **edge_data)
            
            # Find k shortest paths
            try:
                # Get k-shortest paths using Dijkstra
                for path in nx.shortest_simple_paths(weighted_graph, source_id, target_id, weight='weight'):
                    # Check max hops
                    if len(path) > max_hops + 1:
                        break
                        
                    formatted_path = []
                    path_total_weight = 0
                    path_min_confidence = 1.0
                    
                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)
                        
                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }
                        
                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)
                            
                            if edge_data:
                                edge_weight = edge_data.get('weight', 0.5)
                                edge_confidence = edge_data.get('confidence', 0.5)
                                
                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_weight,
                                    "confidence": edge_confidence
                                }
                                node_info["next_relation"] = relation_info
                                
                                path_total_weight += edge_weight
                                path_min_confidence = min(path_min_confidence, edge_confidence)
                        
                        formatted_path.append(node_info)
                    
                    # Add path metadata
                    formatted_path_with_meta = {
                        "path": formatted_path,
                        "avg_weight": path_total_weight / (len(path) - 1) if len(path) > 1 else 0,
                        "min_confidence": path_min_confidence,
                        "path_length": len(path) - 1
                    }
                    
                    weighted_paths.append(formatted_path_with_meta)
                    
                    # Limit to top 5 paths
                    if len(weighted_paths) >= 5:
                        break
                
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                # No path exists
                logging.warning(f"No path found in weighted search: {e}")
                pass
            
            # Format return value to match existing path_query
            return [wp["path"] for wp in weighted_paths]
            
        except Exception as e:
            logging.error(f"Error in weighted path query: {e}")
            return []
    
    def bidirectional_search(self, start_entity: str, end_entity: str, max_hops: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between entities using bidirectional search for efficiency.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length
            
        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for bidirectional search")
            return []
        
        paths = []
        
        try:
            # Get entity IDs
            source_id = None
            target_id = None
            
            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()
            
            if source_row:
                source_id = source_row[0]
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_id = target_row[0]
            
            if self.conn is None:
                conn.close()
            
            # Return empty if entities not found
            if not source_id or not target_id:
                logging.warning(f"Could not find entities: {start_entity} or {end_entity}")
                return []
                
            # Check for direct connection through a common node
            # First, look for common neighbors
            try:
                source_neighbors = set(nx.all_neighbors(self.graph, source_id))
                target_neighbors = set(nx.all_neighbors(self.graph, target_id))
                common = source_neighbors.intersection(target_neighbors)
                
                if common:
                    for connector in common:
                        # Get details about the connector node
                        connector_details = self.get_entity_metadata(connector)
                        connector_name = connector_details.get('entity', 'Unknown')
                        
                        # Create a simple path through the common node
                        formatted_path = []
                        
                        # Add source node
                        source_details = self.get_entity_metadata(source_id)
                        source_node = {
                            "id": source_id,
                            "entity": source_details.get('entity', 'Unknown'),
                            "type": source_details.get('entity_type', 'unknown')
                        }
                        
                        # Get edge data for source to connector
                        if self.graph.has_edge(source_id, connector):
                            edge_data = self.graph.get_edge_data(source_id, connector)
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            source_node["next_relation"] = relation_info
                        else:
                            # Check if the edge is in the reverse direction
                            edge_data = self.graph.get_edge_data(connector, source_id)
                            relation_info = {
                                "type": f"inverse_{edge_data.get('relation', 'is_related_to')}",
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            source_node["next_relation"] = relation_info
                        
                        formatted_path.append(source_node)
                        
                        # Add connector node
                        connector_node = {
                            "id": connector,
                            "entity": connector_name,
                            "type": connector_details.get('entity_type', 'unknown')
                        }
                        
                        # Get edge data for connector to target
                        if self.graph.has_edge(connector, target_id):
                            edge_data = self.graph.get_edge_data(connector, target_id)
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            connector_node["next_relation"] = relation_info
                        else:
                            # Check if the edge is in the reverse direction
                            edge_data = self.graph.get_edge_data(target_id, connector)
                            relation_info = {
                                "type": f"inverse_{edge_data.get('relation', 'is_related_to')}",
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            connector_node["next_relation"] = relation_info
                        
                        formatted_path.append(connector_node)
                        
                        # Add target node
                        target_details = self.get_entity_metadata(target_id)
                        target_node = {
                            "id": target_id,
                            "entity": target_details.get('entity', 'Unknown'),
                            "type": target_details.get('entity_type', 'unknown')
                        }
                        formatted_path.append(target_node)
                        
                        paths.append(formatted_path)
                        
                        # Only use the first common node for simplicity
                        break
                        
                if paths:
                    return paths
            except Exception as e:
                logging.error(f"Error checking for common neighbors: {e}")
            
            # If no direct connection through common neighbors, use bidirectional BFS
            # Implementation of bidirectional BFS
            max_distance = max_hops // 2 + max_hops % 2  # Split max hops between forward and backward searches
            
            # Forward search from source
            forward_paths = {source_id: [[source_id]]}
            forward_visited = {source_id}
            
            # Backward search from target
            backward_paths = {target_id: [[target_id]]}
            backward_visited = {target_id}
            
            # Intersection of paths
            intersection = set()
            
            # Bidirectional BFS
            for _ in range(max_distance):
                # If no paths to expand, break
                if not forward_paths and not backward_paths:
                    break
                
                # Expand forward paths
                new_forward_paths = {}
                for node, paths_to_node in forward_paths.items():
                    try:
                        for neighbor in self.graph.successors(node):
                            if neighbor not in forward_visited:
                                new_forward_paths.setdefault(neighbor, [])
                                for path in paths_to_node:
                                    new_forward_paths[neighbor].append(path + [neighbor])
                                forward_visited.add(neighbor)
                                
                                # Check for intersection
                                if neighbor in backward_visited:
                                    intersection.add(neighbor)
                    except Exception as e:
                        logging.error(f"Error expanding forward from node {node}: {e}")
                
                forward_paths = new_forward_paths
                
                # Expand backward paths
                new_backward_paths = {}
                for node, paths_to_node in backward_paths.items():
                    try:
                        for neighbor in self.graph.predecessors(node):
                            if neighbor not in backward_visited:
                                new_backward_paths.setdefault(neighbor, [])
                                for path in paths_to_node:
                                    new_backward_paths[neighbor].append([neighbor] + path)
                                backward_visited.add(neighbor)
                                
                                # Check for intersection
                                if neighbor in forward_visited:
                                    intersection.add(neighbor)
                    except Exception as e:
                        logging.error(f"Error expanding backward from node {node}: {e}")
                
                backward_paths = new_backward_paths
                
                # If we have intersections, construct the paths
                if intersection:
                    break
            
            # Construct complete paths
            complete_paths = []
            for node in intersection:
                # Get all forward paths to the intersection node
                if node in forward_paths:
                    forward_to_node = forward_paths[node]
                else:
                    # Check if this is the source
                    forward_to_node = [[source_id]] if node == source_id else []
                
                # Get all backward paths from the intersection node
                if node in backward_paths:
                    backward_from_node = backward_paths[node]
                else:
                    # Check if this is the target
                    backward_from_node = [[target_id]] if node == target_id else []
                
                # Connect forward and backward paths
                for f_path in forward_to_node:
                    for b_path in backward_from_node:
                        if node == f_path[-1] and node == b_path[0]:
                            # Merge paths, avoiding duplicate intersection node
                            complete_path = f_path + b_path[1:]
                            if len(complete_path) <= max_hops + 1:
                                complete_paths.append(complete_path)
            
            # If no paths found, try a direct connection search
            if not complete_paths:
                logging.info("No paths found using bidirectional BFS, trying direct path search")
                try:
                    # Look for direct paths using a higher max_hops
                    for path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops):
                        complete_paths.append(path)
                        # Only take the first few paths
                        if len(complete_paths) >= 3:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    logging.warning(f"No direct path exists: {e}")
            
            # Format paths
            for path in complete_paths[:5]:  # Limit to top 5 paths
                formatted_path = []
                
                for i, node_id in enumerate(path):
                    # Get node details
                    node_details = self.get_entity_metadata(node_id)
                    
                    node_info = {
                        "id": node_id,
                        "entity": node_details.get('entity', 'Unknown'),
                        "type": node_details.get('entity_type', 'unknown')
                    }
                    
                    # Add relation to next node if not the last node
                    if i < len(path) - 1:
                        next_node = path[i + 1]
                        edge_data = self.graph.get_edge_data(node_id, next_node)
                        
                        if edge_data:
                            relation_info = {
                                "type": edge_data.get('relation', 'is_related_to'),
                                "weight": edge_data.get('weight', 1.0),
                                "confidence": edge_data.get('confidence', 0.5)
                            }
                            node_info["next_relation"] = relation_info
                    
                    formatted_path.append(node_info)
                
                paths.append(formatted_path)
            
            # If we still haven't found a path, try a common connection through intermediate nodes
            if not paths:
                logging.info("No direct paths found, looking for connections through intermediate nodes")
                # Find all nodes that connect to the source
                source_connections = set()
                try:
                    for node in self.graph.nodes():
                        if nx.has_path(self.graph, source_id, node) or nx.has_path(self.graph, node, source_id):
                            source_connections.add(node)
                except Exception as e:
                    logging.error(f"Error finding source connections: {e}")
                
                # Find all nodes that connect to the target
                target_connections = set()
                try:
                    for node in self.graph.nodes():
                        if nx.has_path(self.graph, target_id, node) or nx.has_path(self.graph, node, target_id):
                            target_connections.add(node)
                except Exception as e:
                    logging.error(f"Error finding target connections: {e}")
                
                # Find common connections
                common_connections = source_connections.intersection(target_connections)
                if common_connections:
                    # Use the first common connection for simplicity
                    connector = next(iter(common_connections))
                    
                    # Try to find a path from source to connector
                    source_to_connector = None
                    try:
                        source_to_connector = next(nx.all_simple_paths(
                            self.graph, source_id, connector, cutoff=max_hops//2
                        ))
                    except (nx.NetworkXNoPath, StopIteration):
                        try:
                            source_to_connector = next(nx.all_simple_paths(
                                self.graph, connector, source_id, cutoff=max_hops//2
                            ))
                            # Reverse the path
                            source_to_connector = list(reversed(source_to_connector))
                        except (nx.NetworkXNoPath, StopIteration):
                            pass
                    
                    # Try to find a path from connector to target
                    connector_to_target = None
                    try:
                        connector_to_target = next(nx.all_simple_paths(
                            self.graph, connector, target_id, cutoff=max_hops//2
                        ))
                    except (nx.NetworkXNoPath, StopIteration):
                        try:
                            connector_to_target = next(nx.all_simple_paths(
                                self.graph, target_id, connector, cutoff=max_hops//2
                            ))
                            # Reverse the path
                            connector_to_target = list(reversed(connector_to_target))
                        except (nx.NetworkXNoPath, StopIteration):
                            pass
                    
                    # If we found both paths, combine them
                    if source_to_connector and connector_to_target:
                        # Combine paths, avoiding duplicate connector node
                        complete_path = source_to_connector
                        if connector_to_target[0] == complete_path[-1]:
                            complete_path.extend(connector_to_target[1:])
                        else:
                            complete_path.extend(connector_to_target)
                        
                        # Format the path
                        formatted_path = []
                        for i, node_id in enumerate(complete_path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)
                            
                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }
                            
                            # Add relation to next node if not the last node
                            if i < len(complete_path) - 1:
                                next_node = complete_path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)
                                
                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info
                            
                            formatted_path.append(node_info)
                        
                        paths.append(formatted_path)
            
            return paths
            
        except Exception as e:
            logging.error(f"Error in bidirectional search: {e}")
            return []
    
    def constrained_path_search(self, start_entity: str, end_entity: str, 
                              allowed_relations: List[str] = None,
                              forbidden_relations: List[str] = None,
                              max_hops: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths with constraints on relation types.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            allowed_relations: List of relation types to allow (if None, all are allowed)
            forbidden_relations: List of relation types to forbid (if None, none are forbidden)
            max_hops: Maximum path length
            
        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # NetworkX is required for efficient path finding
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for constrained path search")
            return []
        
        paths = []
        
        try:
            # Get entity IDs
            source_id = None
            target_id = None
            
            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()
            
            if source_row:
                source_id = source_row[0]
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_id = target_row[0]
            
            if self.conn is None:
                conn.close()
            
            # Return empty if entities not found
            if not source_id or not target_id:
                return []
            
            # Create a subgraph with only allowed relations
            constrained_graph = nx.DiGraph()
            
            # Copy nodes
            for node in self.graph.nodes():
                constrained_graph.add_node(node)
            
            # Copy edges that meet constraints
            for u, v, data in self.graph.edges(data=True):
                relation = data.get('relation', '')
                
                # Skip forbidden relations
                if forbidden_relations and relation in forbidden_relations:
                    continue
                
                # Check if relation is allowed
                if allowed_relations is None or relation in allowed_relations:
                    constrained_graph.add_edge(u, v, **data)
            
            # Find paths in constrained graph
            try:
                # Find all simple paths (can be slow for large graphs)
                simple_paths = nx.all_simple_paths(constrained_graph, source_id, target_id, cutoff=max_hops)
                
                # Convert paths to our format
                for path in list(simple_paths)[:5]:  # Limit to top 5 paths
                    formatted_path = []
                    
                    for i, node_id in enumerate(path):
                        # Get node details
                        node_details = self.get_entity_metadata(node_id)
                        
                        node_info = {
                            "id": node_id,
                            "entity": node_details.get('entity', 'Unknown'),
                            "type": node_details.get('entity_type', 'unknown')
                        }
                        
                        # Add relation to next node if not the last node
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_data = self.graph.get_edge_data(node_id, next_node)
                            
                            if edge_data:
                                relation_info = {
                                    "type": edge_data.get('relation', 'is_related_to'),
                                    "weight": edge_data.get('weight', 1.0),
                                    "confidence": edge_data.get('confidence', 0.5)
                                }
                                node_info["next_relation"] = relation_info
                        
                        formatted_path.append(node_info)
                    
                    paths.append(formatted_path)
                
            except nx.NetworkXNoPath:
                # No path exists
                pass
            
            return paths
            
        except Exception as e:
            logging.error(f"Error in constrained path search: {e}")
            return []
    
    def contract_graph(self, min_edge_weight: float = 0.2, 
                      min_confidence: float = 0.3,
                      combine_parallel_edges: bool = True) -> Dict[str, Any]:
        """
        Contract the graph to handle large knowledge graphs efficiently.
        Removes low-weight/confidence edges and combines parallel edges.
        
        Args:
            min_edge_weight: Minimum edge weight to keep
            min_confidence: Minimum confidence to keep
            combine_parallel_edges: Whether to combine parallel edges between nodes
            
        Returns:
            Dictionary with statistics about the contraction
        """
        # NetworkX is required for graph contraction
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for graph contraction")
            return {"success": False, "reason": "NetworkX not available"}
        
        stats = {
            "original_nodes": self.graph.number_of_nodes(),
            "original_edges": self.graph.number_of_edges(),
            "removed_edges": 0,
            "combined_edges": 0,
            "success": True
        }
        
        try:
            # Create a new graph for the contracted result
            contracted_graph = nx.DiGraph()
            
            # Copy all nodes
            for node, data in self.graph.nodes(data=True):
                contracted_graph.add_node(node, **data)
            
            # Filter edges by weight and confidence
            for u, v, data in self.graph.edges(data=True):
                edge_weight = data.get('weight', 0.5)
                edge_confidence = data.get('confidence', 0.5)
                
                if edge_weight >= min_edge_weight and edge_confidence >= min_confidence:
                    contracted_graph.add_edge(u, v, **data)
                else:
                    stats["removed_edges"] += 1
            
            # Combine parallel edges if requested
            if combine_parallel_edges:
                # Find all node pairs with multiple edges
                multi_edges = {}
                
                for u, v, key, data in contracted_graph.edges(data=True, keys=True):
                    multi_edges.setdefault((u, v), []).append((key, data))
                
                # Combine parallel edges
                for (u, v), edges in multi_edges.items():
                    if len(edges) > 1:
                        # Combine edges between the same nodes
                        combined_data = {
                            "relations": [],
                            "weight": 0,
                            "confidence": 0,
                            "is_combined": True
                        }
                        
                        for key, data in edges:
                            relation = data.get('relation', '')
                            if relation and relation not in combined_data["relations"]:
                                combined_data["relations"].append(relation)
                            
                            combined_data["weight"] += data.get('weight', 0.5)
                            combined_data["confidence"] += data.get('confidence', 0.5)
                        
                        # Average the weight and confidence
                        combined_data["weight"] /= len(edges)
                        combined_data["confidence"] /= len(edges)
                        
                        # Create a combined relation description
                        if combined_data["relations"]:
                            combined_data["relation"] = " & ".join(combined_data["relations"])
                        else:
                            combined_data["relation"] = "related_to"
                        
                        # Remove old edges
                        for key, _ in edges:
                            contracted_graph.remove_edge(u, v, key)
                        
                        # Add combined edge
                        contracted_graph.add_edge(u, v, **combined_data)
                        stats["combined_edges"] += len(edges) - 1
            
            # Update the graph
            self.graph = contracted_graph
            
            # Update stats
            stats["final_nodes"] = self.graph.number_of_nodes()
            stats["final_edges"] = self.graph.number_of_edges()
            
            return stats
            
        except Exception as e:
            logging.error(f"Error contracting graph: {e}")
            return {"success": False, "reason": str(e)}
    
    def create_graph_abstraction(self, community_resolution: float = 1.0, 
                               min_community_size: int = 3) -> Dict[str, Any]:
        """
        Create a hierarchical abstraction of the graph using community detection.
        Useful for navigating and querying large knowledge graphs.
        
        Args:
            community_resolution: Resolution parameter for community detection (higher values create smaller communities)
            min_community_size: Minimum size for a community to be represented as a supernode
            
        Returns:
            Dictionary with abstraction details and statistics
        """
        # NetworkX is required for graph abstraction
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX is required for graph abstraction")
            return {"success": False, "reason": "NetworkX not available"}
        
        # Additional libraries required
        try:
            import community as community_louvain
        except ImportError:
            logging.warning("python-louvain package not found. Install it for graph abstraction.")
            return {"success": False, "reason": "Required package 'python-louvain' not installed"}
        
        abstraction_stats = {
            "original_nodes": self.graph.number_of_nodes(),
            "original_edges": self.graph.number_of_edges(),
            "communities": 0,
            "supernodes": 0,
            "success": True
        }
        
        try:
            # Create an undirected copy of the graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Detect communities using Louvain method
            partition = community_louvain.best_partition(undirected_graph, 
                                                        resolution=community_resolution,
                                                        random_state=42)
            
            # Count communities
            communities = {}
            for node, community_id in partition.items():
                communities.setdefault(community_id, []).append(node)
            
            abstraction_stats["communities"] = len(communities)
            
            # Create abstracted graph
            abstracted_graph = nx.DiGraph()
            
            # Track community metadata
            community_metadata = {}
            
            # Create supernodes for sufficiently large communities
            for community_id, nodes in communities.items():
                if len(nodes) >= min_community_size:
                    # Create a supernode
                    supernode_id = f"community_{community_id}"
                    
                    # Determine the most representative entity for the community
                    # (highest degree or highest confidence)
                    representative = max(nodes, key=lambda n: self.graph.degree(n))
                    rep_metadata = self.get_entity_metadata(representative)
                    
                    # Get types of entities in this community
                    entity_types = {}
                    for node in nodes:
                        node_metadata = self.get_entity_metadata(node)
                        node_type = node_metadata.get('entity_type', 'unknown')
                        entity_types[node_type] = entity_types.get(node_type, 0) + 1
                    
                    # Get most common entity type
                    common_type = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else "mixed"
                    
                    supernode_attrs = {
                        "is_supernode": True,
                        "community_id": community_id,
                        "size": len(nodes),
                        "representative": rep_metadata.get('entity', 'Unknown'),
                        "entity_type": common_type,
                        "members": nodes  # Store member nodes for expansion
                    }
                    
                    abstracted_graph.add_node(supernode_id, **supernode_attrs)
                    community_metadata[community_id] = supernode_attrs
                    abstraction_stats["supernodes"] += 1
                else:
                    # Add individual nodes
                    for node in nodes:
                        node_data = self.graph.nodes[node]
                        abstracted_graph.add_node(node, **node_data)
            
            # Add edges between supernodes and regular nodes
            for u, v, data in self.graph.edges(data=True):
                u_community = partition.get(u)
                v_community = partition.get(v)
                
                if u_community == v_community:
                    # Skip intra-community edges unless the community is too small
                    if len(communities[u_community]) >= min_community_size:
                        continue
                
                # Determine source node or supernode
                if u_community is not None and len(communities[u_community]) >= min_community_size:
                    source = f"community_{u_community}"
                else:
                    source = u
                
                # Determine target node or supernode
                if v_community is not None and len(communities[v_community]) >= min_community_size:
                    target = f"community_{v_community}"
                else:
                    target = v
                
                # Add or update edge
                if abstracted_graph.has_edge(source, target):
                    # Update existing edge
                    edge_data = abstracted_graph.get_edge_data(source, target)
                    edge_data["weight"] = edge_data.get("weight", 0) + data.get("weight", 1.0)
                    edge_data["count"] = edge_data.get("count", 0) + 1
                else:
                    # Add new edge
                    abstracted_graph.add_edge(source, target, 
                                             weight=data.get("weight", 1.0),
                                             relation=data.get("relation", "related_to"),
                                             count=1)
            
            # Normalize edge weights for abstracted graph
            for u, v, data in abstracted_graph.edges(data=True):
                if "count" in data:
                    data["weight"] = data["weight"] / data["count"]
            
            # Store abstracted graph and metadata
            self.abstracted_graph = abstracted_graph
            self.community_metadata = community_metadata
            
            # Update stats
            abstraction_stats["abstracted_nodes"] = abstracted_graph.number_of_nodes()
            abstraction_stats["abstracted_edges"] = abstracted_graph.number_of_edges()
            abstraction_stats["compression_ratio"] = (abstraction_stats["original_nodes"] / 
                                                     abstraction_stats["abstracted_nodes"])
            
            return abstraction_stats
            
        except Exception as e:
            logging.error(f"Error creating graph abstraction: {e}")
            return {"success": False, "reason": str(e)}
    
    def path_query_with_abstraction(self, start_entity: str, end_entity: str, 
                                  max_hops: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Find paths between entities using graph abstraction for efficiency.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_hops: Maximum path length
            
        Returns:
            List of paths (each path is a list of node dictionaries)
        """
        # Check if abstraction is available
        if not hasattr(self, 'abstracted_graph') or self.abstracted_graph is None:
            logging.warning("Graph abstraction not available. Creating one with default settings.")
            abstraction_result = self.create_graph_abstraction()
            if not abstraction_result.get("success", False):
                logging.warning("Failed to create abstraction. Falling back to regular path query.")
                return self.path_query(start_entity, end_entity, max_hops)
        
        paths = []
        
        try:
            # Get entity IDs
            source_id = None
            target_id = None
            
            # Use SQLite for entity lookup
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            source_row = cursor.fetchone()
            
            if source_row:
                source_id = source_row[0]
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_id = target_row[0]
            
            if self.conn is None:
                conn.close()
            
            # Return empty if entities not found
            if not source_id or not target_id:
                return []
            
            # Find which community each entity belongs to
            source_community = None
            target_community = None
            
            for community_id, metadata in self.community_metadata.items():
                if source_id in metadata.get("members", []):
                    source_community = community_id
                if target_id in metadata.get("members", []):
                    target_community = community_id
            
            # Determine starting and ending nodes in abstracted graph
            if source_community is not None:
                abstracted_source = f"community_{source_community}"
            else:
                abstracted_source = source_id
            
            if target_community is not None:
                abstracted_target = f"community_{target_community}"
            else:
                abstracted_target = target_id
            
            # If source and target are in the same community, use regular path query
            if source_community is not None and source_community == target_community:
                # Filter the graph to only include nodes in this community
                community_nodes = self.community_metadata[source_community]["members"]
                subgraph = self.graph.subgraph(community_nodes)
                
                # Find paths in this community
                try:
                    simple_paths = nx.all_simple_paths(subgraph, source_id, target_id, cutoff=max_hops)
                    
                    # Convert paths to our format
                    for path in list(simple_paths)[:5]:  # Limit to top 5 paths
                        formatted_path = []
                        
                        for i, node_id in enumerate(path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)
                            
                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }
                            
                            # Add relation to next node if not the last node
                            if i < len(path) - 1:
                                next_node = path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)
                                
                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info
                            
                            formatted_path.append(node_info)
                        
                        paths.append(formatted_path)
                    
                except nx.NetworkXNoPath:
                    # No path exists
                    pass
                
                return paths
            
            # Find paths in abstracted graph
            try:
                abstracted_paths = list(nx.all_simple_paths(
                    self.abstracted_graph, 
                    abstracted_source, 
                    abstracted_target, 
                    cutoff=max_hops//2
                ))[:3]  # Limit to top 3 abstracted paths
                
                # Expand abstracted paths to detailed paths
                for abst_path in abstracted_paths:
                    # Build a list of segments to find paths for
                    segments = []
                    
                    for i in range(len(abst_path) - 1):
                        current = abst_path[i]
                        next_node = abst_path[i + 1]
                        
                        # Determine actual nodes to connect
                        if isinstance(current, str) and current.startswith("community_"):
                            if i == 0:  # Source community
                                start_node = source_id
                            else:
                                # Use representative node or random member
                                comm_id = int(current.split("_")[1])
                                start_node = self.community_metadata[comm_id].get("members", [])[0]
                        else:
                            start_node = current
                        
                        if isinstance(next_node, str) and next_node.startswith("community_"):
                            if i == len(abst_path) - 2:  # Target community
                                end_node = target_id
                            else:
                                # Use representative node or random member
                                comm_id = int(next_node.split("_")[1])
                                end_node = self.community_metadata[comm_id].get("members", [])[0]
                        else:
                            end_node = next_node
                        
                        segments.append((start_node, end_node))
                    
                    # Find paths for each segment and connect them
                    segment_paths = []
                    for start, end in segments:
                        try:
                            # Find a single path for this segment
                            segment_path = next(nx.all_simple_paths(self.graph, start, end, cutoff=2))
                            segment_paths.append(segment_path)
                        except (nx.NetworkXNoPath, StopIteration):
                            # No path exists for this segment
                            # Try using neighbors as connectors
                            try:
                                # Find common neighbors
                                start_neighbors = set(self.graph.successors(start))
                                end_neighbors = set(self.graph.predecessors(end))
                                common_neighbors = start_neighbors.intersection(end_neighbors)
                                
                                if common_neighbors:
                                    # Use first common neighbor
                                    connector = next(iter(common_neighbors))
                                    segment_path = [start, connector, end]
                                    segment_paths.append(segment_path)
                                else:
                                    # No common neighbor, skip this abstracted path
                                    segment_paths = []
                                    break
                            except Exception:
                                # Fail silently, skip this abstracted path
                                segment_paths = []
                                break
                    
                    # If all segments have paths, connect them
                    if segment_paths:
                        complete_path = segment_paths[0]
                        
                        for i in range(1, len(segment_paths)):
                            # Skip the first node of subsequent segments (already included)
                            complete_path.extend(segment_paths[i][1:])
                        
                        # Format the complete path
                        formatted_path = []
                        
                        for i, node_id in enumerate(complete_path):
                            # Get node details
                            node_details = self.get_entity_metadata(node_id)
                            
                            node_info = {
                                "id": node_id,
                                "entity": node_details.get('entity', 'Unknown'),
                                "type": node_details.get('entity_type', 'unknown')
                            }
                            
                            # Add relation to next node if not the last node
                            if i < len(complete_path) - 1:
                                next_node = complete_path[i + 1]
                                edge_data = self.graph.get_edge_data(node_id, next_node)
                                
                                if edge_data:
                                    relation_info = {
                                        "type": edge_data.get('relation', 'is_related_to'),
                                        "weight": edge_data.get('weight', 1.0),
                                        "confidence": edge_data.get('confidence', 0.5)
                                    }
                                    node_info["next_relation"] = relation_info
                            
                            formatted_path.append(node_info)
                        
                        paths.append(formatted_path)
                
            except nx.NetworkXNoPath:
                # No path exists in abstracted graph
                pass
            
            # If no paths found, fall back to regular path query
            if not paths:
                logging.info("No paths found using abstraction, falling back to regular path query")
                return self.path_query(start_entity, end_entity, max_hops)
            
            return paths
            
        except Exception as e:
            logging.error(f"Error in abstracted path query: {e}")
            # Fall back to regular path query
            return self.path_query(start_entity, end_entity, max_hops)
    
    def close(self):
        """Clean up resources."""
        if self.conn is not None:
            self.conn.close()
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close() 

class GraphMerger:
    """
    GraphMerger component for intelligently combining new information from multiple sources
    into the existing knowledge graph, with conflict detection and resolution strategies.
    """
    
    def __init__(self, graph_store):
        """
        Initialize the graph merger.
        
        Args:
            graph_store: The GraphStore instance to work with
        """
        self.graph_store = graph_store
        self.conn = graph_store.conn if graph_store.conn else sqlite3.connect(graph_store.db_path)
        self.cursor = self.conn.cursor()
        
        # Track statistics
        self.stats = {
            "entities_added": 0,
            "entities_updated": 0,
            "entities_merged": 0,
            "relations_added": 0,
            "relations_updated": 0,
            "relations_inferred": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0
        }
    
    def merge_entity(self, entity: str, entity_type: str = None, metadata: Dict[str, Any] = None,
                    provenance: str = None, confidence: float = 0.8,
                    temporal_start: str = None, temporal_end: str = None,
                    extraction_method: str = None) -> int:
        """
        Intelligently merge an entity with existing entities, handling duplicates and conflicts.
        
        Args:
            entity: Entity text to merge
            entity_type: Type of entity (e.g., PERSON, LOCATION)
            metadata: Additional entity metadata
            provenance: Source of the entity information
            confidence: Confidence score for this entity (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this entity
            
        Returns:
            ID of the merged entity
        """
        # Check for exact match
        self.cursor.execute('SELECT id, entity_type, metadata, confidence FROM graph_entities WHERE entity = ?', (entity,))
        exact_match = self.cursor.fetchone()
        
        # Check for fuzzy matches if no exact match
        fuzzy_matches = []
        if not exact_match and FUZZY_MATCHING_ENABLED:
            self.cursor.execute('SELECT id, entity, entity_type, metadata, confidence FROM graph_entities')
            all_entities = self.cursor.fetchall()
            
            # Find potential matches using fuzzy string matching
            for row in all_entities:
                similarity = fuzz.ratio(entity.lower(), row[1].lower())
                if similarity >= 85:  # Threshold for fuzzy matching
                    fuzzy_matches.append({
                        'id': row[0],
                        'entity': row[1],
                        'entity_type': row[2],
                        'metadata': json.loads(row[3]) if row[3] else {},
                        'confidence': row[4],
                        'similarity': similarity
                    })
            
            # Sort by similarity
            fuzzy_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if exact_match:
            entity_id = exact_match[0]
            existing_type = exact_match[1]
            existing_metadata_str = exact_match[2]
            existing_confidence = exact_match[3]
            
            # Parse existing metadata
            if existing_metadata_str:
                try:
                    existing_metadata = json.loads(existing_metadata_str)
                except (TypeError, json.JSONDecodeError):
                    existing_metadata = {}
            else:
                existing_metadata = {}
            
            # Determine if we need to update the existing entity
            should_update = False
            merged_metadata = dict(existing_metadata)  # Make a copy to modify
            
            # Only update if the new information has higher confidence or adds new metadata
            if confidence > existing_confidence:
                should_update = True
                
            if metadata:
                # Check if new metadata contains keys not in existing metadata or better values
                for key, value in metadata.items():
                    if key not in existing_metadata:
                        merged_metadata[key] = value
                        should_update = True
                    elif isinstance(value, list) and isinstance(existing_metadata[key], list):
                        # Merge lists
                        combined = list(set(existing_metadata[key] + value))
                        if len(combined) > len(existing_metadata[key]):
                            merged_metadata[key] = combined
                            should_update = True
                    elif confidence > existing_confidence:
                        # Replace with higher confidence value
                        merged_metadata[key] = value
                        should_update = True
                        # Record the conflict
                        self.stats["conflicts_detected"] += 1
                        self.stats["conflicts_resolved"] += 1
            
            if should_update:
                # Update the existing entity
                try:
                    result = self.graph_store.add_entity(
                        entity=entity,
                        entity_type=entity_type or existing_type,
                        metadata=merged_metadata,
                        provenance=provenance,
                        confidence=max(confidence, existing_confidence),
                        temporal_start=temporal_start,
                        temporal_end=temporal_end,
                        extraction_method=extraction_method,
                        changed_by="graph_merger"
                    )
                    self.stats["entities_updated"] += 1
                except Exception as e:
                    logging.error(f"Error updating entity: {e}")
            
            return entity_id
            
        elif fuzzy_matches:
            # Use the best fuzzy match
            best_match = fuzzy_matches[0]
            entity_id = best_match['id']
            
            # Also update the entity with any new metadata
            merged_metadata = dict(best_match['metadata'])  # Start with existing metadata
            
            if metadata:
                # Merge metadata
                for key, value in metadata.items():
                    if key not in merged_metadata:
                        merged_metadata[key] = value
                    elif isinstance(value, list) and isinstance(merged_metadata[key], list):
                        merged_metadata[key] = list(set(merged_metadata[key] + value))
                    elif confidence > best_match['confidence']:
                        merged_metadata[key] = value
            
            # Add the current entity name as an alias
            merged_metadata['aliases'] = list(set(merged_metadata.get('aliases', []) + [entity]))
            
            # Track entity merger event
            merged_metadata['merged_with'] = merged_metadata.get('merged_with', []) + [{
                'entity': entity,
                'similarity': best_match['similarity'],
                'timestamp': time.time(),
                'provenance': provenance
            }]
            
            # Add as an alias to the best match
            try:
                alias_added = self.graph_store.add_entity_alias(
                    entity_id=entity_id, 
                    alias=entity, 
                    confidence=confidence * (best_match['similarity'] / 100.0)
                )
            except Exception as e:
                logging.warning(f"Couldn't add alias, but continuing: {e}")
            
            # Update the entity with merged metadata
            try:
                self.graph_store.add_entity(
                    entity=best_match['entity'],
                    entity_type=entity_type or best_match['entity_type'],
                    metadata=merged_metadata,
                    provenance=provenance,
                    confidence=max(confidence, best_match['confidence']),
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger"
                )
            except Exception as e:
                logging.error(f"Error updating entity with merged metadata: {e}")
                
            self.stats["entities_merged"] += 1
            return entity_id
            
        else:
            # No match found, add as new entity
            try:
                entity_id = self.graph_store.add_entity(
                    entity=entity,
                    entity_type=entity_type,
                    metadata=metadata,
                    provenance=provenance,
                    confidence=confidence,
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger"
                )
                self.stats["entities_added"] += 1
                return entity_id
            except Exception as e:
                logging.error(f"Error adding new entity: {e}")
                return -1
    
    def merge_relation(self, source_entity: str, relation_type: str, target_entity: str,
                      weight: float = 1.0, metadata: Dict[str, Any] = None,
                      provenance: str = None, confidence: float = 0.5,
                      temporal_start: str = None, temporal_end: str = None,
                      extraction_method: str = None) -> bool:
        """
        Intelligently merge a relation with existing relations, handling duplicates and conflicts.
        
        Args:
            source_entity: Source entity text
            relation_type: Type of relation
            target_entity: Target entity text
            weight: Weight/importance of the relation
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this relation
            
        Returns:
            True if relation was merged successfully
        """
        # First, get or merge the source and target entities
        source_id = self.merge_entity(
            entity=source_entity,
            provenance=provenance,
            confidence=confidence,
            extraction_method=extraction_method
        )
        
        target_id = self.merge_entity(
            entity=target_entity,
            provenance=provenance,
            confidence=confidence,
            extraction_method=extraction_method
        )
        
        # Check if relation already exists
        self.cursor.execute('''
            SELECT id, weight, metadata, confidence 
            FROM graph_relationships 
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
        ''', (source_id, target_id, relation_type))
        existing = self.cursor.fetchone()
        
        if existing:
            relation_id = existing[0]
            existing_weight = existing[1]
            existing_metadata = json.loads(existing[2]) if existing[2] else {}
            existing_confidence = existing[3]
            
            # Detect conflicts and determine if we should update
            should_update = False
            
            # Higher confidence or new metadata
            if confidence > existing_confidence:
                should_update = True
            elif metadata:
                # Check for new keys in metadata
                new_keys = set(metadata.keys()) - set(existing_metadata.keys())
                if new_keys:
                    should_update = True
                    
                    # Merge metadata
                    for key in metadata:
                        if key in existing_metadata:
                            # If conflicting values, record conflict and resolve based on confidence
                            if metadata[key] != existing_metadata[key]:
                                self.stats["conflicts_detected"] += 1
                                
                                # If both are lists, merge them
                                if isinstance(metadata[key], list) and isinstance(existing_metadata[key], list):
                                    existing_metadata[key] = list(set(existing_metadata[key] + metadata[key]))
                                    self.stats["conflicts_resolved"] += 1
                                elif confidence > existing_confidence:
                                    existing_metadata[key] = metadata[key]
                                    self.stats["conflicts_resolved"] += 1
                                # Otherwise keep existing value (implicitly)
                        else:
                            existing_metadata[key] = metadata[key]
                    
                    metadata = existing_metadata
            
            if should_update:
                # Update with the merged information
                result = self.graph_store.add_relation(
                    source_entity=source_entity,
                    relation_type=relation_type,
                    target_entity=target_entity,
                    weight=max(weight, existing_weight),
                    metadata=metadata,
                    provenance=provenance,
                    confidence=max(confidence, existing_confidence),
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger"
                )
                self.stats["relations_updated"] += 1
            else:
                result = True  # No update needed
                
            return result
        else:
            # Add new relation
            result = self.graph_store.add_relation(
                source_entity=source_entity,
                relation_type=relation_type,
                target_entity=target_entity,
                weight=weight,
                metadata=metadata,
                provenance=provenance,
                confidence=confidence,
                temporal_start=temporal_start,
                temporal_end=temporal_end,
                extraction_method=extraction_method,
                changed_by="graph_merger"
            )
            self.stats["relations_added"] += 1
            return result
    
    def discover_taxonomic_relationships(self) -> int:
        """
        Automatically detect and extract hierarchical relationships (is_a, part_of) from existing entities.
        
        Returns:
            Number of taxonomic relationships discovered
        """
        # Find all entities
        self.cursor.execute('SELECT id, entity, entity_type FROM graph_entities')
        all_entities = self.cursor.fetchall()
        
        # Build entity type index
        type_index = {}
        for row in all_entities:
            ent_id, ent_text, ent_type = row
            if ent_type:
                if ent_type not in type_index:
                    type_index[ent_type] = []
                type_index[ent_type].append((ent_id, ent_text))
        
        discovered = 0
        
        # Look for patterns suggesting taxonomic relationships
        for row in all_entities:
            ent_id, ent_text, ent_type = row
            # Check if this entity name contains another entity name
            # e.g., "Machine Learning Algorithm" contains "Algorithm"
            words = ent_text.lower().split()
            if len(words) > 1:
                for other_row in all_entities:
                    other_id, other_text, other_type = other_row
                    if other_id != ent_id and len(other_text.split()) == 1 and other_text.lower() in words:
                        # Check if this looks like an is_a relationship
                        # If entity type matches, it's more likely to be an is_a relationship
                        if ent_type == other_text:
                            self._add_taxonomic_relation(
                                ent_id, ent_text, other_id, other_text, "instance_of",
                                confidence=0.85, provenance="taxonomic_discovery"
                            )
                            discovered += 1
                        else:
                            self._add_taxonomic_relation(
                                ent_id, ent_text, other_id, other_text, "is_a",
                                confidence=0.7, provenance="taxonomic_discovery"
                            )
                            discovered += 1
        
        # Discover instance_of relationships from entity types
        for ent_type, type_entities in type_index.items():
            # Find if the type exists as an entity
            self.cursor.execute('SELECT id, entity FROM graph_entities WHERE entity = ?', (ent_type,))
            type_entity = self.cursor.fetchone()
            
            if type_entity:
                type_id, type_text = type_entity
                # Add instance_of relationship for all entities of this type
                for ent_id, ent_text in type_entities:
                    self._add_taxonomic_relation(
                        ent_id, ent_text, type_id, type_text, "instance_of",
                        confidence=0.9, provenance="type_based_taxonomy"
                    )
                    discovered += 1
            else:
                # Create a new entity for this type
                type_id = self.graph_store.add_entity(
                    entity=ent_type,
                    entity_type="TYPE",
                    metadata={"automatic": True, "is_type": True},
                    confidence=0.85,
                    extraction_method="taxonomic_discovery",
                    changed_by="graph_merger"
                )
                
                # Add instance_of relationship for all entities of this type
                for ent_id, ent_text in type_entities:
                    self._add_taxonomic_relation(
                        ent_id, ent_text, type_id, ent_type, "instance_of",
                        confidence=0.85, provenance="type_based_taxonomy"
                    )
                    discovered += 1
        
        self.stats["relations_inferred"] += discovered
        return discovered
    
    def _add_taxonomic_relation(self, source_id: int, source_text: str, 
                              target_id: int, target_text: str, relation_type: str,
                              confidence: float, provenance: str) -> bool:
        """
        Helper method to add a taxonomic relationship if it doesn't exist.
        
        Args:
            source_id: Source entity ID
            source_text: Source entity text
            target_id: Target entity ID
            target_text: Target entity text
            relation_type: Type of taxonomic relation (is_a, part_of, etc.)
            confidence: Confidence score
            provenance: Source of this taxonomic relationship
            
        Returns:
            True if relation was added
        """
        # Check if this relation already exists
        self.cursor.execute('''
            SELECT id FROM graph_relationships 
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
        ''', (source_id, target_id, relation_type))
        
        if not self.cursor.fetchone():
            # Add the relation
            self.graph_store.add_relation(
                source_entity=source_text,
                relation_type=relation_type,
                target_entity=target_text,
                confidence=confidence,
                provenance=provenance,
                extraction_method="taxonomic_inference",
                changed_by="graph_merger",
                metadata={"automatic": True, "taxonomic": True}
            )
            return True
        
        return False
    
    def merge_from_text(self, text: str, source: str) -> Dict[str, int]:
        """
        Process text to extract entities and relations, then merge them into the graph.
        
        Args:
            text: Text to process
            source: Source of the text
            
        Returns:
            Dictionary with count of entities and relations merged
        """
        # Extract entities
        entities = self.graph_store.extract_entities(text)
        
        # Extract relations
        relations = self.graph_store.extract_relations(text)
        
        # Merge entities
        processed_entities = {}
        for entity in entities:
            entity_id = self.merge_entity(
                entity=entity['text'],
                entity_type=entity['type'],
                metadata=entity.get('metadata', {}),
                provenance=source,
                confidence=entity.get('confidence', 0.8),
                extraction_method=entity.get('source', 'text_extraction')
            )
            processed_entities[entity['text']] = entity_id
        
        # Merge relations
        processed_relations = 0
        for subj, pred, obj in relations:
            if self.merge_relation(
                source_entity=subj,
                relation_type=pred,
                target_entity=obj,
                provenance=source,
                confidence=0.7,  # Default confidence for extracted relations
                extraction_method='text_extraction'
            ):
                processed_relations += 1
        
        return {
            "entities": len(processed_entities),
            "relations": processed_relations
        }
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicting information in the knowledge graph.
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Find relationship conflicts (contradictory relationships)
        # e.g., A is_a B and B is_a A (cycle in taxonomy)
        self.cursor.execute('''
            SELECT r1.source_id, r1.target_id, r1.relation_type, r1.id,
                   r2.source_id, r2.target_id, r2.relation_type, r2.id
            FROM graph_relationships r1
            JOIN graph_relationships r2 ON r1.target_id = r2.source_id 
                                       AND r2.target_id = r1.source_id
                                       AND r1.relation_type = r2.relation_type
            WHERE r1.relation_type IN ('is_a', 'subclass_of', 'instance_of')
              AND r1.source_id < r2.target_id  -- Avoid duplicates
        ''')
        
        for row in self.cursor.fetchall():
            # Get entity names
            self.cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row[0],))
            source1 = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row[1],))
            target1 = self.cursor.fetchone()[0]
            
            conflicts.append({
                'type': 'cycle',
                'description': f"Taxonomic cycle detected: {source1} {row[2]} {target1} and {target1} {row[6]} {source1}",
                'relation_ids': [row[3], row[7]],
                'entities': [source1, target1],
                'relation_type': row[2]
            })
        
        # Find attribute conflicts (different values for the same attribute)
        self.cursor.execute('''
            SELECT r1.source_id, r1.relation_type, r1.target_id, r1.confidence, r1.id,
                   r2.target_id, r2.confidence, r2.id
            FROM graph_relationships r1
            JOIN graph_relationships r2 ON r1.source_id = r2.source_id 
                                       AND r1.relation_type = r2.relation_type
                                       AND r1.target_id != r2.target_id
            WHERE r1.relation_type LIKE 'has_%'
              AND r1.id < r2.id  -- Avoid duplicates
        ''')
        
        for row in self.cursor.fetchall():
            # Get entity names
            self.cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row[0],))
            source = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row[2],))
            target1 = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row[5],))
            target2 = self.cursor.fetchone()[0]
            
            conflicts.append({
                'type': 'attribute',
                'description': f"Attribute conflict: {source} {row[1]} {target1} (confidence: {row[3]}) vs {source} {row[1]} {target2} (confidence: {row[6]})",
                'relation_ids': [row[4], row[7]],
                'entities': [source, target1, target2],
                'relation_type': row[1]
            })
        
        return conflicts
    
    def resolve_conflicts(self, conflict_resolution: str = 'confidence') -> int:
        """
        Automatically resolve detected conflicts in the knowledge graph.
        
        Args:
            conflict_resolution: Strategy for resolving conflicts:
                - 'confidence': Keep the relation with higher confidence
                - 'recency': Keep the more recent relation
                - 'provenance': Prioritize by source reliability
                
        Returns:
            Number of conflicts resolved
        """
        conflicts = self.detect_conflicts()
        resolved = 0
        
        for conflict in conflicts:
            if conflict['type'] == 'cycle':
                # For taxonomy cycles, keep the relation with higher confidence or recency
                self.cursor.execute('''
                    SELECT id, confidence, timestamp, provenance FROM graph_relationships
                    WHERE id IN (?, ?)
                ''', (conflict['relation_ids'][0], conflict['relation_ids'][1]))
                
                relations = self.cursor.fetchall()
                
                if conflict_resolution == 'confidence':
                    # Keep the relation with higher confidence
                    if relations[0][1] >= relations[1][1]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]
                        
                elif conflict_resolution == 'recency':
                    # Keep the more recent relation
                    if relations[0][2] >= relations[1][2]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]
                        
                elif conflict_resolution == 'provenance':
                    # Keep the relation from more reliable source
                    # This is a placeholder - implement source reliability logic
                    relation_to_remove = relations[1][0]
                
                # Delete the relation to be removed
                self.cursor.execute('''
                    DELETE FROM graph_relationships WHERE id = ?
                ''', (relation_to_remove,))
                
                resolved += 1
                
            elif conflict['type'] == 'attribute':
                # For attribute conflicts, keep the attribute with higher confidence
                self.cursor.execute('''
                    SELECT id, confidence FROM graph_relationships
                    WHERE id IN (?, ?)
                ''', (conflict['relation_ids'][0], conflict['relation_ids'][1]))
                
                relations = self.cursor.fetchall()
                
                if conflict_resolution == 'confidence':
                    # Keep the relation with higher confidence
                    if relations[0][1] >= relations[1][1]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]
                        
                elif conflict_resolution == 'recency':
                    # Similar logic as above
                    self.cursor.execute('''
                        SELECT id, timestamp FROM graph_relationships
                        WHERE id IN (?, ?)
                    ''', (conflict['relation_ids'][0], conflict['relation_ids'][1]))
                    
                    recency_data = self.cursor.fetchall()
                    if recency_data[0][1] >= recency_data[1][1]:
                        relation_to_remove = recency_data[1][0]
                    else:
                        relation_to_remove = recency_data[0][0]
                        
                elif conflict_resolution == 'provenance':
                    # Placeholder for source reliability
                    relation_to_remove = relations[1][0]
                
                # Delete the relation to be removed
                self.cursor.execute('''
                    DELETE FROM graph_relationships WHERE id = ?
                ''', (relation_to_remove,))
                
                resolved += 1
        
        self.conn.commit()
        self.stats["conflicts_resolved"] += resolved
        return resolved
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the merging operations.
        
        Returns:
            Dictionary with counts of various operations
        """
        return self.stats