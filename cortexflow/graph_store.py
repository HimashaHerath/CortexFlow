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
        Extract named entities from text using multiple techniques.
        
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
                
                # Get named entities
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            except Exception as e:
                logging.error(f"Error in SpaCy NER: {e}")
        
        # 2. Add pattern-based entity extraction
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'URL': r'https?://\S+',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'TIME': r'\b\d{1,2}:\d{2}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'NUMBER': r'\b\d+(?:\.\d+)?\b'
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
        
        # 3. Add noun phrase extraction if no entities found yet
        if not entities and SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:  # Skip very short chunks
                        entities.append({
                            'text': chunk.text,
                            'type': 'NOUN_PHRASE',
                            'start': chunk.start_char,
                            'end': chunk.end_char
                        })
                        
                # Add proper nouns
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
                
        # 4. Add statistical keyword extraction (for domain-specific entities)
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
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text.
        This uses dependency parsing to find relationships.
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        relations = []
        
        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Process each sentence separately
                for sent in doc.sents:
                    # Find root verb and its subject/object
                    root_verb = None
                    for token in sent:
                        if token.dep_ == "ROOT" and token.pos_ == "VERB":
                            root_verb = token
                            break
                    
                    # If we found a root verb, look for its subject and object
                    if root_verb:
                        subject = None
                        direct_object = None
                        
                        # Find subject and object
                        for child in root_verb.children:
                            # Find subject
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                # Get the full noun phrase
                                subject = self._get_span_text(child)
                                
                            # Find direct object
                            elif child.dep_ in ["dobj", "pobj"]:
                                # Get the full noun phrase
                                direct_object = self._get_span_text(child)
                                
                        # If we have both subject and object, create a relation
                        if subject and direct_object:
                            relations.append((subject, root_verb.lemma_, direct_object))
                    
                    # If no root verb relation found, try other patterns
                    if not relations:
                        # Try subject-verb-object patterns with any verb
                        for token in sent:
                            if token.pos_ == "VERB":
                                subject = None
                                direct_object = None
                                
                                # Find subject and object for this verb
                                for child in token.children:
                                    if child.dep_ in ["nsubj", "nsubjpass"] and not subject:
                                        subject = self._get_span_text(child)
                                        
                                    elif child.dep_ in ["dobj", "pobj"] and not direct_object:
                                        direct_object = self._get_span_text(child)
                                
                                # If we found both, add the relation
                                if subject and direct_object:
                                    relations.append((subject, token.lemma_, direct_object))
                
                # If no relations found with dependency parsing, try a simpler approach
                if not relations:
                    entities = self.extract_entities(text)
                    if len(entities) >= 2:
                        # Create relations between consecutive entities
                        for i in range(len(entities) - 1):
                            # Find words between entities that might be predicates
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
        
        Args:
            text: Text to process
            source: Source of the text for metadata
            
        Returns:
            Number of relations added
        """
        relations_added = 0
        entities_added = 0
        
        # Extract entities
        entities = self.extract_entities(text)
        
        logging.debug(f"Processing text for graph: '{text[:50]}...'")
        logging.debug(f"Extracted {len(entities)} entities")
        
        # If SpaCy didn't find any entities, try a simpler approach to extract potential entities
        if not entities and SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Extract noun phrases as potential entities
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:  # Skip very short chunks
                        entities.append({
                            'text': chunk.text,
                            'type': 'NOUN_PHRASE',  # Generic type for noun phrases
                            'start': chunk.start_char,
                            'end': chunk.end_char
                        })
                        
                # Extract proper names that might have been missed
                for token in doc:
                    if token.pos_ == "PROPN" and len(token.text) > 2:
                        # Check if this proper noun is already part of an entity
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
                            
                logging.debug(f"Added {len(entities)} additional entities using noun phrases and proper nouns")
                
            except Exception as e:
                logging.error(f"Error extracting additional entities: {e}")
        
        # Map of entity text to entity ID
        entity_ids = {}
        
        # Add all entities to graph
        for entity_info in entities:
            entity_id = self.add_entity(
                entity=entity_info["text"],
                entity_type=entity_info["type"],
                metadata={"source": source} if source else None
            )
            
            if entity_id >= 0:
                entity_ids[entity_info["text"]] = entity_id
                entities_added += 1
        
        # Extract relations using SpaCy
        relations = self.extract_relations(text)
        
        logging.debug(f"Extracted {len(relations)} relations")
        
        # If no relations were found with standard extraction, try a simple subject-verb-object approach
        if not relations and len(entities) >= 2:
            try:
                # Create simple relations between co-occurring entities
                entity_pairs = []
                sorted_entities = sorted(entities, key=lambda e: e['start'])
                
                # Connect entities that appear close to each other
                for i in range(len(sorted_entities) - 1):
                    for j in range(i + 1, min(i + 3, len(sorted_entities))):
                        # If entities are close enough in text
                        if sorted_entities[j]['start'] - sorted_entities[i]['end'] < 50:
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
        for subject, predicate, obj in relations:
            # Skip relations with empty components
            if not subject.strip() or not predicate.strip() or not obj.strip():
                continue
                
            # Normalize predicate to create consistent relation types
            predicate = predicate.lower().strip()
            
            success = self.add_relation(
                source_entity=subject,
                relation_type=predicate,
                target_entity=obj,
                metadata={"source": source} if source else None
            )
            
            if success:
                relations_added += 1
        
        logging.debug(f"Added {entities_added} entities and {relations_added} relations to graph")
        
        return relations_added
    
    def close(self):
        """Clean up resources."""
        if self.conn is not None:
            self.conn.close()
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close() 