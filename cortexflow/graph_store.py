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

from cortexflow.config import CortexFlowConfig
try:
    from cortexflow.ontology import Ontology
    ONTOLOGY_ENABLED = True
except ImportError:
    ONTOLOGY_ENABLED = False
    logging.warning("Ontology module not found. Advanced knowledge graph capabilities will be limited.")

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
        
        logging.info(f"Graph store initialized with NetworkX: {NETWORKX_ENABLED}, Spacy: {SPACY_ENABLED}, Ontology: {ONTOLOGY_ENABLED}")
    
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
            UNIQUE(entity)
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
            FOREIGN KEY (source_id) REFERENCES graph_entities (id),
            FOREIGN KEY (target_id) REFERENCES graph_entities (id),
            UNIQUE(source_id, target_id, relation_type)
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
            timestamp REAL
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
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON graph_entities(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON graph_relationships(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON graph_relationships(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON graph_relationships(relation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_type ON nary_relationships(relation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_participant ON nary_participants(relationship_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_entity ON nary_participants(entity_id)')
        
        # Alter existing tables to add new metadata columns if they don't exist
        try:
            # Check if provenance column exists in graph_relationships
            cursor.execute("PRAGMA table_info(graph_relationships)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add provenance column if it doesn't exist
            if "provenance" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN provenance TEXT")
                
            # Add confidence column if it doesn't exist
            if "confidence" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN confidence REAL DEFAULT 0.5")
                
            # Add temporal_start column if it doesn't exist
            if "temporal_start" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN temporal_start TEXT")
                
            # Add temporal_end column if it doesn't exist
            if "temporal_end" not in columns:
                cursor.execute("ALTER TABLE graph_relationships ADD COLUMN temporal_end TEXT")
                
            # Check if similar columns need to be added to graph_entities
            cursor.execute("PRAGMA table_info(graph_entities)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add provenance column if it doesn't exist
            if "provenance" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN provenance TEXT")
                
            # Add confidence column if it doesn't exist
            if "confidence" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN confidence REAL DEFAULT 0.8")
                
            # Add temporal_start column if it doesn't exist
            if "temporal_start" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN temporal_start TEXT")
                
            # Add temporal_end column if it doesn't exist
            if "temporal_end" not in columns:
                cursor.execute("ALTER TABLE graph_entities ADD COLUMN temporal_end TEXT")
                
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
        NER, pattern matching, noun phrases, and domain-specific entity recognition.
        
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
                        'end': ent.end_char
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
                                    'mentions': [m.text for m in cluster.mentions]
                                })
                except ImportError:
                    logging.debug("neuralcoref not available, skipping coreference resolution")
                except Exception as e:
                    logging.error(f"Error in coreference resolution: {e}")
            except Exception as e:
                logging.error(f"Error in SpaCy NER: {e}")
        
        # 3. Add pattern-based entity extraction
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
                        'end': match.end()
                    })
        
        # 4. Add noun phrase extraction if no entities found yet or to supplement
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
                                'end': chunk.end_char
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
                                'end': token.idx + len(token.text)
                            })
            except Exception as e:
                logging.error(f"Error extracting noun phrases: {e}")

        # 5. Add domain-specific entity extraction
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
                    entities.append(entity)
        except Exception as e:
            logging.error(f"Error in domain-specific entity extraction: {e}")
                
        # 6. Add statistical keyword extraction (for domain-specific entities)
        if not entities or len(entities) < 3:
            # Simple statistical approach - find unusual words
            words = text.split()
            # Filter out common words
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "with", "for", "to", "of", "is", "are", "was", "were"}
            uncommon_words = [word for word in words if word.lower() not in common_words and len(word) > 3]
            
            # Find capitalized phrases (potential named entities)
            capitalized_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
            for match in re.finditer(capitalized_pattern, text):
                # Check for overlap with existing entities
                overlap = False
                for entity in entities:
                    if (match.start() >= entity['start'] and match.start() < entity['end']) or \
                       (match.end() > entity['start'] and match.end() <= entity['end']):
                        overlap = True
                        break
                        
                if not overlap:
                    entities.append({
                        'text': match.group(0),
                        'type': 'CAPITALIZED_PHRASE',
                        'start': match.start(),
                        'end': match.end()
                    })
            
            # Add remaining uncommon words as entities
            for word in uncommon_words:
                word_start = text.find(word)
                if word_start >= 0:
                    # Check for overlap
                    overlap = False
                    for entity in entities:
                        if (word_start >= entity['start'] and word_start < entity['end']) or \
                           (word_start + len(word) > entity['start'] and word_start + len(word) <= entity['end']):
                            overlap = True
                            break
                            
                    if not overlap:
                        entities.append({
                            'text': word,
                            'type': 'KEYWORD',
                            'start': word_start,
                            'end': word_start + len(word)
                        })
                    
        return entities
    
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
                  temporal_start: str = None, temporal_end: str = None) -> int:
        """
        Add an entity to the knowledge graph with enhanced metadata.
        
        Args:
            entity: Entity name/text
            entity_type: Type of entity (e.g., person, location, etc.)
            metadata: Additional entity metadata
            provenance: Source of the entity information
            confidence: Confidence score for this entity (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            
        Returns:
            Entity ID
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        timestamp = time.time()
        
        try:
            # Check if entity already exists
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity,))
            existing = cursor.fetchone()
            
            # If ontology is enabled, check the entity type in the ontology
            if ONTOLOGY_ENABLED and self.ontology and entity_type:
                if not self.ontology.get_class(entity_type):
                    # Try to suggest a new class
                    suggested_class = self.ontology.suggest_new_class(
                        entity_name=entity,
                        entity_type=entity_type,
                        entity_properties={}
                    )
                    
                    if suggested_class:
                        # Add the suggested class to the ontology
                        self.ontology.add_class(suggested_class)
                        logging.info(f"Added suggested ontology class: {entity_type}")
            
            if existing:
                entity_id = existing[0]
                # Update entity if needed
                cursor.execute('''
                    UPDATE graph_entities 
                    SET entity_type = ?, metadata = ?, timestamp = ?, 
                        provenance = ?, confidence = ?, 
                        temporal_start = ?, temporal_end = ?
                    WHERE id = ?
                ''', (
                    entity_type, 
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    entity_id
                ))
            else:
                # Insert new entity
                cursor.execute('''
                    INSERT INTO graph_entities 
                    (entity, entity_type, metadata, timestamp, provenance, confidence, temporal_start, temporal_end) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity, 
                    entity_type, 
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end
                ))
                entity_id = cursor.lastrowid
                
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
            if self.conn is None:
                conn.close()
    
    def add_relation(self, source_entity: str, relation_type: str, target_entity: str, 
                     weight: float = 1.0, metadata: Dict[str, Any] = None,
                     provenance: str = None, confidence: float = 0.5,
                     temporal_start: str = None, temporal_end: str = None) -> bool:
        """
        Add a relation between two entities with enhanced metadata.
        
        Args:
            source_entity: Source entity text
            relation_type: Type of relation
            target_entity: Target entity text
            weight: Weight/confidence of the relation
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            
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
            # Get source entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (source_entity,))
            source = cursor.fetchone()
            
            # Get target entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (target_entity,))
            target = cursor.fetchone()
            
            # If either entity doesn't exist, create them
            if not source:
                source_id = self.add_entity(source_entity, provenance=provenance, confidence=confidence)
            else:
                source_id = source[0]
                
            if not target:
                target_id = self.add_entity(target_entity, provenance=provenance, confidence=confidence)
            else:
                target_id = target[0]
            
            # If ontology is enabled, check the relation type in the ontology
            if ONTOLOGY_ENABLED and self.ontology:
                if not self.ontology.get_relation_type(relation_type):
                    # Try to suggest a new relation type
                    suggested_relation = self.ontology.suggest_new_relation_type(
                        source_entity=source_entity,
                        relation=relation_type,
                        target_entity=target_entity
                    )
                    
                    if suggested_relation:
                        # Add the suggested relation type to the ontology
                        self.ontology.add_relation_type(suggested_relation)
                        logging.info(f"Added suggested ontology relation type: {relation_type}")
            
            # Check if relation already exists
            cursor.execute('''
                SELECT id FROM graph_relationships 
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
            ''', (source_id, target_id, relation_type))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing relation
                cursor.execute('''
                    UPDATE graph_relationships 
                    SET weight = ?, metadata = ?, timestamp = ?,
                        provenance = ?, confidence = ?,
                        temporal_start = ?, temporal_end = ?
                    WHERE id = ?
                ''', (
                    weight,
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    provenance,
                    confidence,
                    temporal_start,
                    temporal_end,
                    existing[0]
                ))
            else:
                # Insert new relation
                cursor.execute('''
                    INSERT INTO graph_relationships 
                    (source_id, target_id, relation_type, weight, metadata, timestamp,
                     provenance, confidence, temporal_start, temporal_end) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    temporal_end
                ))
            
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
                    'timestamp': timestamp
                }
                
                if metadata:
                    edge_attrs.update(metadata)
                    
                self.graph.add_edge(source_id, target_id, **edge_attrs)
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding relation: {e}")
            conn.rollback()
            return False
            
        finally:
            if self.conn is None:
                conn.close()
    
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
                    if len(word) > 3 and word[0].isupper():
                        query_entities.append(word)
            
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
                
                # Add entity to nodes if new
                source_id = None
                for neighbor in neighbors:
                    if neighbor['direction'] == 'outgoing':
                        source_id = neighbor['source_id']
                        break
                    else:
                        source_id = neighbor['target_id']
                
                if source_id and source_id not in node_ids:
                    # Get entity details
                    entity_details = self.get_entity_metadata(source_id)
                    
                    # Create node
                    node = {
                        "id": source_id,
                        "label": entity_text,
                        "type": entity_details.get('entity_type', 'unknown'),
                        "confidence": entity_details.get('confidence', 0.5)
                    }
                    
                    subgraph["nodes"].append(node)
                    node_ids.add(source_id)
                
                # Process each neighbor
                for neighbor in neighbors:
                    neighbor_entity = neighbor['entity']
                    relation = neighbor['relation']
                    neighbor_id = neighbor['target_id'] if neighbor['direction'] == 'outgoing' else neighbor['source_id']
                    
                    # Add neighbor node if new
                    if neighbor_id not in node_ids:
                        # Get entity details
                        entity_details = self.get_entity_metadata(neighbor_id)
                        
                        # Create node
                        node = {
                            "id": neighbor_id,
                            "label": neighbor_entity,
                            "type": entity_details.get('entity_type', 'unknown'),
                            "confidence": entity_details.get('confidence', 0.5)
                        }
                        
                        subgraph["nodes"].append(node)
                        node_ids.add(neighbor_id)
                    
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
                
                # Check if we reached max nodes
                if len(node_ids) >= max_nodes:
                    break
            
            return subgraph
            
        except Exception as e:
            logging.error(f"Error building knowledge subgraph: {e}")
            return {"nodes": [], "edges": []}
    
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