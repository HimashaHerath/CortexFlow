"""
Uncertainty and contradiction handling mechanisms for CortexFlow.

This module provides capabilities to:
1. Represent uncertainty in knowledge graph data
2. Revise beliefs when contradictory information arrives
3. Resolve conflicts based on source reliability and recency
4. Reason with incomplete information
"""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import math

from .config import CortexFlowConfig

logger = logging.getLogger(__name__)

class UncertaintyHandler:
    """
    Handles uncertainty and contradictions in the knowledge graph.
    
    This class provides mechanisms to:
    1. Represent uncertainty using confidence scores and probability distributions
    2. Detect and handle contradictions in the knowledge graph
    3. Implement belief revision when new contradictory information arrives
    4. Provide reasoning capabilities with incomplete information
    """
    
    def __init__(
        self, 
        config: CortexFlowConfig,
        graph_store=None
    ):
        """
        Initialize the uncertainty handler.
        
        Args:
            config: CortexFlow configuration
            graph_store: Optional reference to the graph store
        """
        self.config = config
        self.graph_store = graph_store
        self.db_path = config.knowledge_store_path
        
        # For in-memory databases, we need to maintain a persistent connection
        self.conn = None
        if self.db_path == ':memory:':
            self.conn = sqlite3.connect("")
        
        # Initialize uncertainty database tables
        self._init_db()
        
        # Source reliability registry
        self.source_reliability = {}
        self._load_source_reliability()
        
        # Default values
        self.default_confidence = 0.5
        self.recency_weight = 0.6  # Weight given to more recent information
        self.reliability_weight = 0.4  # Weight given to source reliability
        
        logger.info("Uncertainty handler initialized")
    
    def _init_db(self):
        """Initialize database tables for uncertainty handling."""
        if self.conn is not None:
            cursor = self.conn.cursor()
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Create contradictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            relation_id INTEGER,
            contradiction_type TEXT NOT NULL,
            conflicting_items TEXT NOT NULL,
            resolution_strategy TEXT,
            resolved_value TEXT,
            confidence REAL,
            timestamp REAL,
            FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
            FOREIGN KEY (relation_id) REFERENCES graph_relationships (id)
        )
        ''')
        
        # Create belief revision history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS belief_revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            relation_id INTEGER,
            previous_value TEXT,
            new_value TEXT,
            revision_reason TEXT,
            confidence_before REAL,
            confidence_after REAL,
            timestamp REAL,
            FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
            FOREIGN KEY (relation_id) REFERENCES graph_relationships (id)
        )
        ''')
        
        # Create probability distributions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS probability_distributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            relation_id INTEGER,
            distribution_type TEXT NOT NULL,
            distribution_data TEXT NOT NULL,
            timestamp REAL,
            FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
            FOREIGN KEY (relation_id) REFERENCES graph_relationships (id)
        )
        ''')
        
        # Create source reliability table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS source_reliability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL UNIQUE,
            reliability_score REAL NOT NULL,
            decay_rate REAL,
            last_updated REAL,
            metadata TEXT
        )
        ''')
        
        # Create indices for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_contradict_entity ON contradictions(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_contradict_relation ON contradictions(relation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_revision_entity ON belief_revisions(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_revision_relation ON belief_revisions(relation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prob_entity ON probability_distributions(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_name ON source_reliability(source_name)')
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
    
    def _load_source_reliability(self):
        """Load source reliability scores from the database."""
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT source_name, reliability_score FROM source_reliability')
            sources = cursor.fetchall()
            
            for source in sources:
                self.source_reliability[source['source_name']] = source['reliability_score']
                
        except Exception as e:
            logger.error(f"Error loading source reliability scores: {e}")
            
        finally:
            if self.conn is None:
                conn.close()
    
    def detect_contradictions(self, entity_id=None, relation_type=None, max_results=100) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the knowledge graph.
        
        Args:
            entity_id: Optional entity ID to check for contradictions
            relation_type: Optional relation type to check
            max_results: Maximum number of contradictions to return
            
        Returns:
            List of detected contradictions
        """
        if not self.graph_store:
            logger.warning("Graph store not available. Cannot detect contradictions.")
            return []
        
        contradictions = []
        
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Query to find potentially contradictory relationships
            query = '''
            SELECT r1.id as id1, r1.source_id, r1.target_id, r1.relation_type, 
                   r1.confidence as conf1, r1.provenance as source1, r1.timestamp as time1,
                   r2.id as id2, r2.confidence as conf2, r2.provenance as source2, r2.timestamp as time2
            FROM graph_relationships r1
            JOIN graph_relationships r2 ON r1.source_id = r2.source_id 
                                      AND r1.relation_type = r2.relation_type
                                      AND r1.id < r2.id
            WHERE r1.target_id != r2.target_id
            '''
            
            params = []
            
            if entity_id:
                query += " AND r1.source_id = ?"
                params.append(entity_id)
                
            if relation_type:
                query += " AND r1.relation_type = ?"
                params.append(relation_type)
                
            query += " LIMIT ?"
            params.append(max_results)
            
            cursor.execute(query, params)
            potential_contradictions = cursor.fetchall()
            
            # Process each potential contradiction
            for row in potential_contradictions:
                # Get entity and target information
                cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row['source_id'],))
                entity_row = cursor.fetchone()
                entity_text = entity_row['entity'] if entity_row else "Unknown"
                
                cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (row['target_id'],))
                target1_row = cursor.fetchone()
                target1_text = target1_row['entity'] if target1_row else "Unknown"
                
                # Get second target
                cursor.execute(
                    'SELECT target_id FROM graph_relationships WHERE id = ?', 
                    (row['id2'],)
                )
                target2_id_row = cursor.fetchone()
                target2_id = target2_id_row['target_id'] if target2_id_row else None
                
                if target2_id:
                    cursor.execute('SELECT entity FROM graph_entities WHERE id = ?', (target2_id,))
                    target2_row = cursor.fetchone()
                    target2_text = target2_row['entity'] if target2_row else "Unknown"
                    
                    # Add to contradictions list
                    contradictions.append({
                        "entity": entity_text,
                        "entity_id": row['source_id'],
                        "relation": row['relation_type'],
                        "target1": target1_text,
                        "target2": target2_text,
                        "relation1_id": row['id1'],
                        "relation2_id": row['id2'],
                        "confidence1": row['conf1'],
                        "confidence2": row['conf2'],
                        "source1": row['source1'],
                        "source2": row['source2'],
                        "timestamp1": row['time1'],
                        "timestamp2": row['time2']
                    })
            
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            
        finally:
            if self.conn is None:
                conn.close()
        
        return contradictions
    
    def resolve_contradiction(self, contradiction: Dict[str, Any], 
                           strategy: str = "auto") -> Dict[str, Any]:
        """
        Resolve a contradiction using the specified strategy.
        
        Args:
            contradiction: The contradiction to resolve
            strategy: Resolution strategy (auto, recency, confidence, reliability, or keep_both)
            
        Returns:
            Resolution result including resolved value and confidence
        """
        if strategy == "auto":
            # Determine best strategy based on available information
            if contradiction.get("source1") and contradiction.get("source2"):
                strategy = "reliability"
            elif contradiction.get("timestamp1") and contradiction.get("timestamp2"):
                strategy = "recency"
            elif contradiction.get("confidence1") is not None and contradiction.get("confidence2") is not None:
                strategy = "confidence"
            else:
                strategy = "keep_both"
        
        resolution_result = {
            "entity_id": contradiction.get("entity_id"),
            "relation1_id": contradiction.get("relation1_id"),
            "relation2_id": contradiction.get("relation2_id"),
            "strategy_used": strategy,
            "resolved_value": None,
            "confidence": None,
            "kept_both": False
        }
        
        # Apply the selected strategy
        if strategy == "recency":
            # Prefer more recent information
            if contradiction.get("timestamp1", 0) > contradiction.get("timestamp2", 0):
                resolution_result["resolved_value"] = contradiction.get("target1")
                resolution_result["confidence"] = contradiction.get("confidence1", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation1_id")
            else:
                resolution_result["resolved_value"] = contradiction.get("target2")
                resolution_result["confidence"] = contradiction.get("confidence2", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation2_id")
        
        elif strategy == "confidence":
            # Prefer information with higher confidence
            if contradiction.get("confidence1", 0) > contradiction.get("confidence2", 0):
                resolution_result["resolved_value"] = contradiction.get("target1")
                resolution_result["confidence"] = contradiction.get("confidence1", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation1_id")
            else:
                resolution_result["resolved_value"] = contradiction.get("target2")
                resolution_result["confidence"] = contradiction.get("confidence2", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation2_id")
        
        elif strategy == "reliability":
            # Prefer information from more reliable sources
            source1 = contradiction.get("source1", "unknown")
            source2 = contradiction.get("source2", "unknown")
            
            reliability1 = self.source_reliability.get(source1, 0.5)
            reliability2 = self.source_reliability.get(source2, 0.5)
            
            if reliability1 > reliability2:
                resolution_result["resolved_value"] = contradiction.get("target1")
                resolution_result["confidence"] = contradiction.get("confidence1", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation1_id")
            else:
                resolution_result["resolved_value"] = contradiction.get("target2")
                resolution_result["confidence"] = contradiction.get("confidence2", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation2_id")
                
        elif strategy == "weighted":
            # Use a weighted approach considering recency and reliability
            source1 = contradiction.get("source1", "unknown")
            source2 = contradiction.get("source2", "unknown")
            
            reliability1 = self.source_reliability.get(source1, 0.5)
            reliability2 = self.source_reliability.get(source2, 0.5)
            
            # Calculate recency scores (more recent = higher score)
            timestamp1 = contradiction.get("timestamp1", 0)
            timestamp2 = contradiction.get("timestamp2", 0)
            
            # Normalize timestamps to 0-1 range
            max_timestamp = max(timestamp1, timestamp2)
            min_timestamp = min(timestamp1, timestamp2)
            
            recency1 = 1.0 if max_timestamp == min_timestamp else (timestamp1 - min_timestamp) / (max_timestamp - min_timestamp)
            recency2 = 1.0 if max_timestamp == min_timestamp else (timestamp2 - min_timestamp) / (max_timestamp - min_timestamp)
            
            # Calculate weighted scores
            score1 = (recency1 * self.recency_weight) + (reliability1 * self.reliability_weight)
            score2 = (recency2 * self.recency_weight) + (reliability2 * self.reliability_weight)
            
            if score1 > score2:
                resolution_result["resolved_value"] = contradiction.get("target1")
                resolution_result["confidence"] = contradiction.get("confidence1", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation1_id")
            else:
                resolution_result["resolved_value"] = contradiction.get("target2")
                resolution_result["confidence"] = contradiction.get("confidence2", self.default_confidence)
                resolution_result["relation_to_keep"] = contradiction.get("relation2_id")
        
        else:  # keep_both
            # Keep both values with uncertainty representation
            resolution_result["resolved_value"] = f"{contradiction.get('target1')} OR {contradiction.get('target2')}"
            
            # Average the confidences and reduce slightly to reflect uncertainty
            conf1 = contradiction.get("confidence1", self.default_confidence)
            conf2 = contradiction.get("confidence2", self.default_confidence)
            resolution_result["confidence"] = (conf1 + conf2) / 2 * 0.9  # Reduce confidence to reflect uncertainty
            resolution_result["kept_both"] = True
        
        # Record the resolution in the database
        self._record_contradiction_resolution(contradiction, resolution_result)
        
        # Apply the resolution if not keeping both
        if not resolution_result["kept_both"] and self.graph_store:
            self._apply_resolution(resolution_result)
        
        return resolution_result
    
    def _record_contradiction_resolution(self, contradiction: Dict[str, Any], 
                                      resolution: Dict[str, Any]):
        """Record a contradiction resolution in the database."""
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        
        try:
            # Serialize the contradiction and resolution data
            conflicting_items = json.dumps({
                "entity": contradiction.get("entity"),
                "relation": contradiction.get("relation"),
                "target1": contradiction.get("target1"),
                "target2": contradiction.get("target2")
            })
            
            # Insert into contradictions table
            cursor.execute('''
            INSERT INTO contradictions 
            (entity_id, relation_id, contradiction_type, conflicting_items, 
             resolution_strategy, resolved_value, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                contradiction.get("entity_id"),
                contradiction.get("relation1_id"),
                "value_conflict",
                conflicting_items,
                resolution.get("strategy_used"),
                resolution.get("resolved_value"),
                resolution.get("confidence"),
                datetime.now().timestamp()
            ))
            
            # Record the belief revision
            cursor.execute('''
            INSERT INTO belief_revisions
            (entity_id, relation_id, previous_value, new_value, 
             revision_reason, confidence_before, confidence_after, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                contradiction.get("entity_id"),
                contradiction.get("relation1_id"),
                f"{contradiction.get('target1')} OR {contradiction.get('target2')}",
                resolution.get("resolved_value"),
                f"Contradiction resolved using {resolution.get('strategy_used')} strategy",
                contradiction.get("confidence1", self.default_confidence),
                resolution.get("confidence"),
                datetime.now().timestamp()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording contradiction resolution: {e}")
            conn.rollback()
            
        finally:
            if self.conn is None:
                conn.close()
    
    def _apply_resolution(self, resolution: Dict[str, Any]):
        """Apply the resolution by updating the graph store."""
        if not self.graph_store:
            logger.warning("Graph store not available. Cannot apply resolution.")
            return
        
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        
        try:
            # Get relation details for the one to keep
            cursor.execute('''
            SELECT source_id, target_id, relation_type 
            FROM graph_relationships
            WHERE id = ?
            ''', (resolution.get("relation_to_keep"),))
            
            relation = cursor.fetchone()
            if not relation:
                logger.error(f"Could not find relation with ID {resolution.get('relation_to_keep')}")
                return
                
            # Get the relation to remove
            relation_to_remove = resolution.get("relation1_id") if resolution.get("relation_to_keep") == resolution.get("relation2_id") else resolution.get("relation2_id")
            
            # Update the confidence of the kept relation
            cursor.execute('''
            UPDATE graph_relationships
            SET confidence = ?
            WHERE id = ?
            ''', (resolution.get("confidence"), resolution.get("relation_to_keep")))
            
            # Remove the other relation if not keeping both
            if not resolution.get("kept_both"):
                cursor.execute('''
                DELETE FROM graph_relationships
                WHERE id = ?
                ''', (relation_to_remove,))
            
            conn.commit()
            
            # Update the NetworkX graph if available
            if hasattr(self.graph_store, 'graph') and self.graph_store.graph:
                source_id = relation[0]
                target_id = relation[1]
                relation_type = relation[2]
                
                # Update the edge in the graph
                if self.graph_store.graph.has_edge(source_id, target_id):
                    self.graph_store.graph[source_id][target_id]['confidence'] = resolution.get("confidence")
            
        except Exception as e:
            logger.error(f"Error applying contradiction resolution: {e}")
            conn.rollback()
            
        finally:
            if self.conn is None:
                conn.close()
    
    def add_probability_distribution(self, entity_id: int, relation_id: int, 
                                  distribution_type: str, distribution_data: Dict[str, Any]):
        """
        Add a probability distribution to represent uncertainty about a fact.
        
        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            distribution_type: Type of distribution (discrete, gaussian, etc.)
            distribution_data: Data representing the distribution
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        
        try:
            # Serialize the distribution data
            data_json = json.dumps(distribution_data)
            
            # Insert or replace the probability distribution
            cursor.execute('''
            INSERT OR REPLACE INTO probability_distributions
            (entity_id, relation_id, distribution_type, distribution_data, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                entity_id,
                relation_id,
                distribution_type,
                data_json,
                datetime.now().timestamp()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error adding probability distribution: {e}")
            conn.rollback()
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_probability_distribution(self, entity_id: int, relation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the probability distribution for a fact.
        
        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            
        Returns:
            Probability distribution data or None if not found
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT distribution_type, distribution_data, timestamp
            FROM probability_distributions
            WHERE entity_id = ? AND relation_id = ?
            ''', (entity_id, relation_id))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "type": row['distribution_type'],
                    "data": json.loads(row['distribution_data']),
                    "timestamp": row['timestamp']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting probability distribution: {e}")
            return None
            
        finally:
            if self.conn is None:
                conn.close()
    
    def update_source_reliability(self, source_name: str, reliability_score: float, 
                              metadata: Dict[str, Any] = None):
        """
        Update the reliability score for a source.
        
        Args:
            source_name: Name of the source
            reliability_score: Reliability score (0.0-1.0)
            metadata: Optional metadata about the source
        """
        if reliability_score < 0 or reliability_score > 1:
            logger.error(f"Reliability score must be between 0 and 1, got {reliability_score}")
            return
        
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        cursor = conn.cursor()
        
        try:
            # Update in-memory cache
            self.source_reliability[source_name] = reliability_score
            
            # Serialize metadata
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert or replace the source reliability
            cursor.execute('''
            INSERT OR REPLACE INTO source_reliability
            (source_name, reliability_score, last_updated, metadata)
            VALUES (?, ?, ?, ?)
            ''', (
                source_name,
                reliability_score,
                datetime.now().timestamp(),
                metadata_json
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating source reliability: {e}")
            conn.rollback()
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_source_reliability(self, source_name: str) -> float:
        """
        Get the reliability score for a source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            Reliability score (0.0-1.0)
        """
        # First check in-memory cache
        if source_name in self.source_reliability:
            return self.source_reliability[source_name]
        
        # Default value if not found
        return 0.5
    
    def revise_belief(self, entity_id: int, relation_id: int, new_value: str, 
                    confidence: float, revision_reason: str):
        """
        Revise a belief with new information.
        
        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            new_value: New value for the relation
            confidence: Confidence in the new value
            revision_reason: Reason for the revision
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get current relation details
            cursor.execute('''
            SELECT r.target_id, r.confidence, e.entity
            FROM graph_relationships r
            JOIN graph_entities e ON r.target_id = e.id
            WHERE r.id = ?
            ''', (relation_id,))
            
            current = cursor.fetchone()
            
            if not current:
                logger.error(f"Could not find relation with ID {relation_id}")
                return
            
            current_value = current['entity']
            current_confidence = current['confidence']
            
            # Record the belief revision
            cursor.execute('''
            INSERT INTO belief_revisions
            (entity_id, relation_id, previous_value, new_value, 
             revision_reason, confidence_before, confidence_after, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity_id,
                relation_id,
                current_value,
                new_value,
                revision_reason,
                current_confidence,
                confidence,
                datetime.now().timestamp()
            ))
            
            # Update the relation with the new value
            # This requires finding or creating the target entity and updating the relation
            cursor.execute('SELECT id FROM graph_entities WHERE entity = ?', (new_value,))
            target_row = cursor.fetchone()
            
            if target_row:
                new_target_id = target_row['id']
            else:
                # Create a new entity
                cursor.execute('''
                INSERT INTO graph_entities (entity, confidence, timestamp)
                VALUES (?, ?, ?)
                ''', (new_value, confidence, datetime.now().timestamp()))
                new_target_id = cursor.lastrowid
            
            # Update the relation
            cursor.execute('''
            UPDATE graph_relationships
            SET target_id = ?, confidence = ?
            WHERE id = ?
            ''', (new_target_id, confidence, relation_id))
            
            conn.commit()
            
            # Update the NetworkX graph if available
            if hasattr(self.graph_store, 'graph') and self.graph_store.graph:
                # Get the source entity ID
                cursor.execute('SELECT source_id, relation_type FROM graph_relationships WHERE id = ?', (relation_id,))
                rel_row = cursor.fetchone()
                
                if rel_row:
                    source_id = rel_row['source_id']
                    relation_type = rel_row['relation_type']
                    
                    # Remove the old edge
                    if self.graph_store.graph.has_edge(source_id, current['target_id']):
                        self.graph_store.graph.remove_edge(source_id, current['target_id'])
                    
                    # Add the new edge
                    self.graph_store.graph.add_edge(
                        source_id, 
                        new_target_id,
                        relation=relation_type,
                        confidence=confidence
                    )
            
        except Exception as e:
            logger.error(f"Error revising belief: {e}")
            conn.rollback()
            
        finally:
            if self.conn is None:
                conn.close()
    
    def get_belief_revision_history(self, entity_id: int = None, relation_id: int = None, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the revision history for a belief.
        
        Args:
            entity_id: Optional entity ID filter
            relation_id: Optional relation ID filter
            limit: Maximum number of revisions to return
            
        Returns:
            List of belief revisions
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        revisions = []
        
        try:
            query = '''
            SELECT br.id, br.entity_id, br.relation_id, br.previous_value, br.new_value,
                   br.revision_reason, br.confidence_before, br.confidence_after, br.timestamp,
                   e.entity as entity_name
            FROM belief_revisions br
            JOIN graph_entities e ON br.entity_id = e.id
            WHERE 1=1
            '''
            
            params = []
            
            if entity_id:
                query += " AND br.entity_id = ?"
                params.append(entity_id)
                
            if relation_id:
                query += " AND br.relation_id = ?"
                params.append(relation_id)
                
            query += " ORDER BY br.timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                revisions.append({
                    "id": row['id'],
                    "entity_id": row['entity_id'],
                    "entity": row['entity_name'],
                    "relation_id": row['relation_id'],
                    "previous_value": row['previous_value'],
                    "new_value": row['new_value'],
                    "revision_reason": row['revision_reason'],
                    "confidence_before": row['confidence_before'],
                    "confidence_after": row['confidence_after'],
                    "timestamp": row['timestamp']
                })
            
        except Exception as e:
            logger.error(f"Error getting belief revision history: {e}")
            
        finally:
            if self.conn is None:
                conn.close()
                
        return revisions
    
    def reason_with_incomplete_information(self, query: Dict[str, Any], 
                                       available_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reason with incomplete information to provide best possible answers.
        
        Args:
            query: The query to answer
            available_knowledge: Available knowledge to reason with
            
        Returns:
            Reasoning result with confidence and explanation
        """
        # Initialize the result
        result = {
            "answer": None,
            "confidence": 0,
            "explanation": [],
            "missing_information": []
        }
        
        # Identify what information is missing
        missing_info = self._identify_missing_information(query, available_knowledge)
        result["missing_information"] = missing_info
        
        # If critical information is missing, apply incomplete information reasoning
        if missing_info:
            # Apply default reasoning technique - use most reliable partial match
            partial_matches = self._find_partial_matches(query, available_knowledge)
            
            if partial_matches:
                # Sort by confidence
                partial_matches.sort(key=lambda x: x.get("match_confidence", 0), reverse=True)
                best_match = partial_matches[0]
                
                result["answer"] = best_match.get("answer")
                result["confidence"] = best_match.get("match_confidence", 0) * 0.8  # Reduce confidence for incomplete info
                result["explanation"].append(f"Used best partial match with {best_match.get('match_confidence', 0):.2f} confidence")
                result["explanation"].append(f"Missing information: {', '.join(missing_info)}")
                
                # Add all partial matches for reference
                result["partial_matches"] = partial_matches[:3]  # Top 3 matches
            else:
                result["answer"] = "Unknown"
                result["confidence"] = 0.1
                result["explanation"].append("No partial matches found with the available information")
        else:
            # All information is available, use the most confident knowledge
            best_knowledge = max(available_knowledge, key=lambda x: x.get("confidence", 0), default=None)
            
            if best_knowledge:
                result["answer"] = best_knowledge.get("answer")
                result["confidence"] = best_knowledge.get("confidence", 0.5)
                result["explanation"].append(f"Used complete information with {best_knowledge.get('confidence', 0.5):.2f} confidence")
            else:
                result["answer"] = "Unknown"
                result["confidence"] = 0
                result["explanation"].append("No knowledge available to answer the query")
        
        return result
    
    def _identify_missing_information(self, query: Dict[str, Any], 
                                   available_knowledge: List[Dict[str, Any]]) -> List[str]:
        """Identify what information is missing to answer a query."""
        missing = []
        
        # Check required fields in the query
        required_fields = query.get("required_fields", [])
        
        for field in required_fields:
            # Check if this field is covered in any knowledge item
            field_covered = False
            
            for item in available_knowledge:
                if field in item:
                    field_covered = True
                    break
            
            if not field_covered:
                missing.append(field)
        
        return missing
    
    def _find_partial_matches(self, query: Dict[str, Any], 
                           available_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find partial matches for a query with incomplete information."""
        matches = []
        
        required_fields = query.get("required_fields", [])
        if not required_fields:
            return matches
        
        # Calculate matches for each knowledge item
        for item in available_knowledge:
            # Count how many required fields are matched
            matched_fields = 0
            total_fields = len(required_fields)
            
            for field in required_fields:
                if field in item:
                    matched_fields += 1
            
            # Calculate match confidence
            if total_fields > 0:
                match_confidence = matched_fields / total_fields
                
                # Apply confidence scaling based on the item's own confidence
                item_confidence = item.get("confidence", 0.5)
                scaled_confidence = match_confidence * item_confidence
                
                # Only include if it's a meaningful partial match
                if match_confidence > 0.3:
                    matches.append({
                        "answer": item.get("answer"),
                        "matched_fields": matched_fields,
                        "total_fields": total_fields,
                        "match_confidence": scaled_confidence,
                        "original_confidence": item_confidence,
                        "source": item.get("source", "unknown")
                    })
        
        return matches
        
    def close(self):
        """Close database connections."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None 