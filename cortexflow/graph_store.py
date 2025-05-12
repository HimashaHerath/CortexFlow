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
        
        # Initialize graph database tables
        self._init_db()
        
        # Load existing graph from database
        self._load_graph_from_db()
        
        logging.info(f"Graph store initialized with NetworkX: {NETWORKX_ENABLED}, Spacy: {SPACY_ENABLED}")
    
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
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON graph_entities(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON graph_relationships(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON graph_relationships(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON graph_relationships(relation_type)')
        
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
    
    def add_entity(self, entity: str, entity_type: str = None, metadata: Dict[str, Any] = None) -> int:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: Entity name/text
            entity_type: Type of entity (e.g., person, location, etc.)
            metadata: Additional entity metadata
            
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
            
            if existing:
                entity_id = existing[0]
                # Update entity if needed
                cursor.execute('''
                    UPDATE graph_entities 
                    SET entity_type = ?, metadata = ?, timestamp = ?
                    WHERE id = ?
                ''', (
                    entity_type, 
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    entity_id
                ))
            else:
                # Insert new entity
                cursor.execute('''
                    INSERT INTO graph_entities (entity, entity_type, metadata, timestamp) 
                    VALUES (?, ?, ?, ?)
                ''', (
                    entity, 
                    entity_type, 
                    json.dumps(metadata) if metadata else None,
                    timestamp
                ))
                entity_id = cursor.lastrowid
                
            conn.commit()
            
            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                self.graph.add_node(
                    entity_id, 
                    name=entity,
                    entity_type=entity_type,
                    **(metadata or {})
                )
            
            return entity_id
            
        except Exception as e:
            logging.error(f"Error adding entity: {e}")
            conn.rollback()
            return -1
            
        finally:
            if self.conn is None:
                conn.close()
    
    def add_relation(self, source_entity: str, relation_type: str, target_entity: str, 
                     weight: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a relation between two entities.
        
        Args:
            source_entity: Source entity text
            relation_type: Type of relation
            target_entity: Target entity text
            weight: Weight/confidence of the relation
            metadata: Additional relation metadata
            
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
                source_id = self.add_entity(source_entity)
            else:
                source_id = source[0]
                
            if not target:
                target_id = self.add_entity(target_entity)
            else:
                target_id = target[0]
            
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
                    SET weight = ?, metadata = ?, timestamp = ?
                    WHERE id = ?
                ''', (
                    weight,
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                    existing[0]
                ))
            else:
                # Insert new relation
                cursor.execute('''
                    INSERT INTO graph_relationships 
                    (source_id, target_id, relation_type, weight, metadata, timestamp) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source_id, 
                    target_id, 
                    relation_type,
                    weight,
                    json.dumps(metadata) if metadata else None,
                    timestamp
                ))
            
            conn.commit()
            
            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                self.graph.add_edge(
                    source_id, 
                    target_id,
                    relation=relation_type,
                    weight=weight,
                    **(metadata or {})
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding relation: {e}")
            conn.rollback()
            return False
            
        finally:
            if self.conn is None:
                conn.close()
    
    def query_entities(self, entity_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities by type.
        
        Args:
            entity_type: Type of entities to query (None for all)
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if entity_type:
                cursor.execute('''
                    SELECT id, entity, entity_type, metadata, timestamp 
                    FROM graph_entities 
                    WHERE entity_type = ?
                    LIMIT ?
                ''', (entity_type, limit))
            else:
                cursor.execute('''
                    SELECT id, entity, entity_type, metadata, timestamp 
                    FROM graph_entities
                    LIMIT ?
                ''', (limit,))
                
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'entity': row['entity'],
                    'type': row['entity_type'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'timestamp': row['timestamp']
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error querying entities: {e}")
            return []
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_entity_neighbors(self, entity: str, relation_type: str = None, 
                            direction: str = "outgoing", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get neighboring entities connected to the specified entity.
        
        Args:
            entity: Entity text
            relation_type: Type of relations to consider (None for all)
            direction: "outgoing", "incoming", or "both"
            limit: Maximum number of results
            
        Returns:
            List of connected entity dictionaries with relation info
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # First get the entity ID
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity,))
            entity_row = cursor.fetchone()
            
            if not entity_row:
                return []  # Entity not found
                
            entity_id = entity_row['id']
            results = []
            
            # Query outgoing relations
            if direction in ["outgoing", "both"]:
                query = '''
                    SELECT e.id, e.entity, e.entity_type, e.metadata, 
                           r.relation_type, r.weight, r.metadata as rel_metadata 
                    FROM graph_relationships r
                    JOIN graph_entities e ON r.target_id = e.id
                    WHERE r.source_id = ? 
                '''
                params = [entity_id]
                
                if relation_type:
                    query += " AND r.relation_type = ? "
                    params.append(relation_type)
                    
                query += " LIMIT ? "
                params.append(limit)
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    results.append({
                        'id': row['id'],
                        'entity': row['entity'],
                        'type': row['entity_type'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'relation': row['relation_type'],
                        'weight': row['weight'],
                        'rel_metadata': json.loads(row['rel_metadata']) if row['rel_metadata'] else {},
                        'direction': 'outgoing'
                    })
            
            # Query incoming relations
            if direction in ["incoming", "both"] and len(results) < limit:
                remaining = limit - len(results)
                query = '''
                    SELECT e.id, e.entity, e.entity_type, e.metadata, 
                           r.relation_type, r.weight, r.metadata as rel_metadata 
                    FROM graph_relationships r
                    JOIN graph_entities e ON r.source_id = e.id
                    WHERE r.target_id = ? 
                '''
                params = [entity_id]
                
                if relation_type:
                    query += " AND r.relation_type = ? "
                    params.append(relation_type)
                    
                query += " LIMIT ? "
                params.append(remaining)
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    results.append({
                        'id': row['id'],
                        'entity': row['entity'],
                        'type': row['entity_type'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'relation': row['relation_type'],
                        'weight': row['weight'],
                        'rel_metadata': json.loads(row['rel_metadata']) if row['rel_metadata'] else {},
                        'direction': 'incoming'
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error getting entity neighbors: {e}")
            return []
            
        finally:
            if self.conn is None:
                conn.close()
    
    def path_query(self, start_entity: str, end_entity: str, max_hops: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities in the knowledge graph.
        
        Args:
            start_entity: Starting entity text
            end_entity: Target entity text
            max_hops: Maximum path length
            
        Returns:
            List of paths, where each path is a list of node dictionaries
        """
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX not available for path queries")
            return []
            
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Find entity IDs
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (start_entity,))
            start_row = cursor.fetchone()
            
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (end_entity,))
            end_row = cursor.fetchone()
            
            if not start_row or not end_row:
                return []  # One or both entities not found
                
            start_id = start_row['id']
            end_id = end_row['id']
            
            # Try to find all simple paths between the entities
            try:
                all_paths = list(nx.all_simple_paths(
                    self.graph, 
                    source=start_id, 
                    target=end_id, 
                    cutoff=max_hops
                ))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
            
            # Convert path node IDs to detailed information
            result_paths = []
            
            for path in all_paths:
                path_details = []
                
                for i in range(len(path)):
                    node_id = path[i]
                    
                    # Get node details
                    cursor.execute('''
                        SELECT id, entity, entity_type, metadata
                        FROM graph_entities WHERE id = ?
                    ''', (node_id,))
                    node = cursor.fetchone()
                    
                    if not node:
                        continue
                        
                    node_details = {
                        'id': node['id'],
                        'entity': node['entity'],
                        'type': node['entity_type'],
                        'metadata': json.loads(node['metadata']) if node['metadata'] else {}
                    }
                    
                    # Add edge details for connections
                    if i < len(path) - 1:
                        next_node_id = path[i + 1]
                        
                        cursor.execute('''
                            SELECT relation_type, weight, metadata
                            FROM graph_relationships 
                            WHERE source_id = ? AND target_id = ?
                        ''', (node_id, next_node_id))
                        edge = cursor.fetchone()
                        
                        if edge:
                            node_details['next_relation'] = {
                                'type': edge['relation_type'],
                                'weight': edge['weight'],
                                'metadata': json.loads(edge['metadata']) if edge['metadata'] else {}
                            }
                    
                    path_details.append(node_details)
                
                result_paths.append(path_details)
            
            return result_paths
            
        except Exception as e:
            logging.error(f"Error in path query: {e}")
            return []
            
        finally:
            if self.conn is None:
                conn.close()
    
    def build_knowledge_subgraph(self, query: str, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Build a knowledge subgraph relevant to the query.
        
        Args:
            query: The query text
            max_nodes: Maximum number of nodes in the subgraph
            
        Returns:
            Dictionary with nodes and edges for the subgraph
        """
        if not NETWORKX_ENABLED or self.graph is None:
            logging.warning("NetworkX not available for building subgraphs")
            return {"nodes": [], "edges": []}
            
        # Extract entities from query
        entities = self.extract_entities(query)
        query_entity_texts = [e["text"] for e in entities]
        
        # Add common words in query in case NER misses important concepts
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        
        # Find relevant entities in the database
        relevant_entities = set()
        
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Find exact matches for query entities
            for entity_text in query_entity_texts:
                cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (entity_text,))
                match = cursor.fetchone()
                if match:
                    relevant_entities.add(match["id"])
            
            # Find fuzzy matches for query words
            for word in query_words:
                cursor.execute('SELECT id FROM graph_entities WHERE entity LIKE ?', (f"%{word}%",))
                matches = cursor.fetchall()
                for match in matches:
                    relevant_entities.add(match["id"])
            
            # If we have relevant entities, expand subgraph
            subgraph_nodes = set(relevant_entities)
            subgraph_edges = set()
            
            if relevant_entities:
                # Expand neighborhood for each relevant entity
                for entity_id in relevant_entities:
                    if len(subgraph_nodes) >= max_nodes:
                        break
                        
                    # Get 1-hop neighbors
                    cursor.execute('''
                        SELECT source_id, target_id, relation_type
                        FROM graph_relationships
                        WHERE source_id = ? OR target_id = ?
                    ''', (entity_id, entity_id))
                    
                    neighbors = cursor.fetchall()
                    
                    for neighbor in neighbors:
                        source = neighbor["source_id"]
                        target = neighbor["target_id"]
                        relation = neighbor["relation_type"]
                        
                        subgraph_nodes.add(source)
                        subgraph_nodes.add(target)
                        subgraph_edges.add((source, target, relation))
                        
                        if len(subgraph_nodes) >= max_nodes:
                            break
            
            # Retrieve details for all nodes and edges
            nodes = []
            edges = []
            
            # Get node details
            for node_id in subgraph_nodes:
                cursor.execute('''
                    SELECT id, entity, entity_type, metadata
                    FROM graph_entities WHERE id = ?
                ''', (node_id,))
                node = cursor.fetchone()
                
                if node:
                    nodes.append({
                        'id': node['id'],
                        'label': node['entity'],
                        'type': node['entity_type'],
                        'metadata': json.loads(node['metadata']) if node['metadata'] else {},
                        'in_query': node['id'] in relevant_entities
                    })
            
            # Get edge details
            for edge in subgraph_edges:
                source, target, relation = edge
                edges.append({
                    'from': source,
                    'to': target,
                    'label': relation
                })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logging.error(f"Error building subgraph: {e}")
            return {"nodes": [], "edges": []}
            
        finally:
            if self.conn is None:
                conn.close()
    
    def process_text_to_graph(self, text: str, source: str = None) -> int:
        """
        Process text to extract entities and relations and add to the graph.
        Uses advanced NLP techniques for high-quality knowledge extraction.
        
        Args:
            text: Text to process
            source: Source of the text for metadata
            
        Returns:
            Number of relations added
        """
        relations_added = 0
        entities_added = 0
        
        logging.debug(f"Processing text for graph: '{text[:50]}...'")
        
        # Try coreference resolution first if available
        resolved_text = text
        coreference_clusters = []
        mentions_map = {}  # Maps mention text to canonical entity

        try:
            if SPACY_ENABLED:
                # Try to load and use neuralcoref for coreference resolution
                import neuralcoref
                if not hasattr(self, 'coref_nlp'):
                    # Initialize neuralcoref if not already done
                    self.coref_nlp = spacy.load('en_core_web_sm')
                    neuralcoref.add_to_pipe(self.coref_nlp)
                    logging.info("Coreference resolution model loaded successfully")
                
                # Process the text for coreference resolution
                coref_doc = self.coref_nlp(text)
                
                # Get resolved text with pronouns replaced
                resolved_text = coref_doc._.coref_resolved
                
                # Store coreference clusters for later entity merging
                coreference_clusters = coref_doc._.coref_clusters if hasattr(coref_doc._, 'coref_clusters') else []
                
                # Build a mention map for entity consolidation
                for cluster in coreference_clusters:
                    main_entity = cluster.main.text
                    for mention in cluster.mentions:
                        mentions_map[mention.text] = main_entity
                
                logging.debug(f"Performed coreference resolution. Clusters found: {len(coreference_clusters)}")
        except ImportError:
            logging.debug("neuralcoref not available, skipping coreference resolution")
        except Exception as e:
            logging.error(f"Error in coreference resolution: {e}")
            resolved_text = text  # Fall back to original text
        
        # Extract entities from the resolved text
        entities = self.extract_entities(resolved_text)
        logging.debug(f"Extracted {len(entities)} entities")
        
        # Extract domain-specific entities 
        try:
            domain_entities = self._extract_domain_specific_entities(resolved_text)
            # Add domain entities that don't overlap with already extracted entities
            for domain_entity in domain_entities:
                overlap = False
                for entity in entities:
                    if (domain_entity['start'] >= entity['start'] and domain_entity['start'] < entity['end']) or \
                       (domain_entity['end'] > entity['start'] and domain_entity['end'] <= entity['end']):
                        overlap = True
                        break
                if not overlap:
                    entities.append(domain_entity)
        except Exception as e:
            logging.error(f"Error extracting domain-specific entities: {e}")
        
        # Handle coreference clusters - ensure entities mentioned in different ways are tracked
        if coreference_clusters:
            # Add canonical entities from coreference clusters if not already present
            for cluster in coreference_clusters:
                main_mention = cluster.main
                main_text = main_mention.text
                
                # Check if this entity is already extracted
                already_exists = False
                for entity in entities:
                    if entity['text'] == main_text:
                        already_exists = True
                        # Add mentions as metadata
                        if 'mentions' not in entity:
                            entity['mentions'] = [m.text for m in cluster.mentions]
                        break
                
                # If not found, add it
                if not already_exists:
                    entities.append({
                        'text': main_text,
                        'type': 'COREF',
                        'start': main_mention.start_char,
                        'end': main_mention.end_char,
                        'mentions': [m.text for m in cluster.mentions]
                    })
        
        # Map of entity text to entity ID
        entity_ids = {}
        
        # Add all entities to graph
        for entity_info in entities:
            entity_text = entity_info["text"]
            entity_type = entity_info["type"]
            
            # Skip very short entities unless they are special types
            if len(entity_text) <= 2 and entity_type not in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY']:
                continue
                
            # Prepare metadata with source and any mentions
            entity_metadata = {"source": source} if source else {}
            
            # Add mention information from coreference resolution
            if 'mentions' in entity_info:
                entity_metadata['mentions'] = entity_info['mentions']
            
            # Add entity to graph
            entity_id = self.add_entity(
                entity=entity_text,
                entity_type=entity_type,
                metadata=entity_metadata
            )
            
            if entity_id >= 0:
                entity_ids[entity_text] = entity_id
                entities_added += 1
                
                # Also map all mentions of this entity to the same ID
                if entity_text in mentions_map.values():
                    # This is a canonical entity, map all its mentions
                    for mention, canonical in mentions_map.items():
                        if canonical == entity_text and mention != entity_text:
                            entity_ids[mention] = entity_id
        
        # Extract relations from the resolved text
        relations = self.extract_relations(resolved_text)
        logging.debug(f"Extracted {len(relations)} relations")
        
        # Try semantic role labeling for additional relation extraction
        try:
            srl_relations = self._extract_with_semantic_roles(resolved_text)
            if srl_relations:
                # Add new relations not already in the list
                existing_relations = set((s, p, o) for s, p, o in relations)
                for s, p, o in srl_relations:
                    if (s, p, o) not in existing_relations:
                        relations.append((s, p, o))
                logging.debug(f"Added {len(srl_relations)} relations from semantic role labeling")
        except Exception as e:
            logging.debug(f"Semantic role labeling extraction not available: {e}")
            
        # Try additional relation patterns based on syntactic dependency parsing
        if SPACY_ENABLED and self.nlp is not None and not relations:
            try:
                # Parse text
                doc = self.nlp(resolved_text)
                
                # Extract from each sentence
                for sent in doc.sents:
                    # Get SVO triples
                    svo_triples = self._extract_svo_from_dependency(sent)
                    relations.extend(svo_triples)
                    
                    # Get prepositional relations
                    prep_relations = self._extract_prep_relations(sent)
                    relations.extend(prep_relations)
                    
                logging.debug(f"Added {len(relations)} relations from dependency parsing")
            except Exception as e:
                logging.error(f"Error extracting relations from dependencies: {e}")
        
        # If no relations were found with standard extraction, try a simple subject-verb-object approach
        if not relations and len(entities) >= 2:
            try:
                # Create simple relations between co-occurring entities
                entity_pairs = []
                sorted_entities = sorted(entities, key=lambda e: e['start'])
                
                # Connect entities that appear close to each other
                for i in range(len(sorted_entities) - 1):
                    for j in range(i + 1, min(i + 3, len(sorted_entities))):
                        # If entities are close enough in text (within 50 chars)
                        if sorted_entities[j]['start'] - sorted_entities[i]['end'] < 50:
                            # Try to find a meaningful predicate between them
                            between_text = resolved_text[sorted_entities[i]['end']:sorted_entities[j]['start']].strip()
                            
                            if between_text:
                                # Clean up the between text to get a reasonable predicate
                                predicate = between_text
                                for stopword in [" the ", " a ", " an ", " and ", " or ", " but ", " of "]:
                                    predicate = predicate.replace(stopword, " ")
                                predicate = predicate.strip()
                                
                                if predicate:
                                    entity_pairs.append((
                                        sorted_entities[i]['text'],
                                        predicate,
                                        sorted_entities[j]['text']
                                    ))
                                else:
                                    entity_pairs.append((
                                        sorted_entities[i]['text'],
                                        "related_to",
                                        sorted_entities[j]['text']
                                    ))
                            else:
                                entity_pairs.append((
                                    sorted_entities[i]['text'],
                                    "related_to",
                                    sorted_entities[j]['text']
                                ))
                
                # Add these as backup relations
                relations.extend(entity_pairs)
                logging.debug(f"Added {len(entity_pairs)} proximity-based relations")
                
            except Exception as e:
                logging.error(f"Error creating backup relations: {e}")
        
        # Add relations to graph
        added_relation_keys = set()  # To avoid duplicates
        
        for subject, predicate, obj in relations:
            # Skip relations with empty components
            if not subject.strip() or not predicate.strip() or not obj.strip():
                continue
            
            # Skip self-relations (entity related to itself)
            if subject.strip() == obj.strip():
                continue
                
            # Normalize predicate to create consistent relation types
            predicate = predicate.lower().strip()
            
            # Create a key to detect duplicates
            relation_key = (subject.strip(), predicate, obj.strip())
            
            # Skip if already added
            if relation_key in added_relation_keys:
                continue
                
            # Handle coreference - if subject or object is a mention, use the canonical entity
            canonical_subject = mentions_map.get(subject, subject)
            canonical_object = mentions_map.get(obj, obj)
            
            success = self.add_relation(
                source_entity=canonical_subject,
                relation_type=predicate,
                target_entity=canonical_object,
                metadata={"source": source} if source else None
            )
            
            if success:
                relations_added += 1
                added_relation_keys.add(relation_key)
        
        logging.debug(f"Added {entities_added} entities and {relations_added} relations to graph")
        
        return relations_added
    
    def close(self):
        """Clean up resources."""
        if self.conn is not None:
            self.conn.close()
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close() 