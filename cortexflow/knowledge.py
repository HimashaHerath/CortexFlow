"""
CortexFlow Knowledge module.

This module provides the knowledge storage and retrieval system for CortexFlow.

Migration Guide for CortexFlow 0.5.0 Knowledge Module Changes:
-----------------------------------------------------------------
1. API Changes:
   - Added `add_knowledge()` method which replaces `remember_knowledge()`
   - The `remember_knowledge()` method is now deprecated and will be removed in a future version
   
2. Migration steps:
   - Replace `knowledge_store.remember_knowledge(text, source)` with `knowledge_store.add_knowledge(text, source)`
   - Replace `manager.remember_knowledge(text, source)` with `manager.add_knowledge(text, source)`
   
3. Benefits of new API:
   - More intuitive naming that better reflects the method's purpose
   - Better documentation of low-level vs high-level methods
   - Clearer distinction between interface methods and implementation details

For any issues or questions, please refer to the documentation or open an issue on the GitHub repository.
"""

import os
import time
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, ContextManager, Protocol, runtime_checkable
from sqlitedict import SqliteDict
from datetime import datetime
import logging
from collections import defaultdict
import random
from contextlib import contextmanager
import warnings

# Import sentence-transformers for vector embeddings
try:
    from sentence_transformers import SentenceTransformer, util
    VECTOR_ENABLED = True
except ImportError:
    VECTOR_ENABLED = False
    logging.warning("sentence-transformers not found. Vector-based retrieval will be disabled.")

# Import BM25 for keyword-based scoring
try:
    from rank_bm25 import BM25Okapi
    BM25_ENABLED = True
except ImportError:
    BM25_ENABLED = False
    logging.warning("rank_bm25 not found. BM25 keyword scoring will be disabled.")

from cortexflow.config import CortexFlowConfig
from cortexflow.graph_store import GraphStore
from cortexflow.interfaces import KnowledgeStoreInterface

@runtime_checkable
class SearchStrategy(Protocol):
    """Strategy interface for search methods."""
    
    def search(self, knowledge_store: 'KnowledgeStore', **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search strategy.
        
        Args:
            knowledge_store: The knowledge store instance
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of search results
        """
        pass


class BM25SearchStrategy:
    """BM25 search strategy for keyword-based retrieval."""
    
    def search(self, knowledge_store: 'KnowledgeStore', query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search for keyword-based retrieval.
        
        Args:
            knowledge_store: The knowledge store instance
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of results with BM25 scores
        """
        if not BM25_ENABLED or not knowledge_store.bm25_index:
            return []
            
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = knowledge_store.bm25_index.get_scores(tokenized_query)
        
        # Sort document IDs by score
        sorted_indices = np.argsort(bm25_scores)[::-1]
        
        results = []
        for i in sorted_indices[:max_results]:
            if i in knowledge_store.id_to_doc_mapping and bm25_scores[i] > 0:
                doc = knowledge_store.id_to_doc_mapping[i]
                doc_type = doc.get('type', 'fact')
                
                if doc_type == 'summary':
                    results.append({
                        'type': 'summary',
                        'content': f"From previous conversation: {doc['text']}",
                        'keywords': doc.get('keywords', []),
                        'bm25_score': float(bm25_scores[i]),
                        'id': doc['id']
                    })
                elif doc_type == 'knowledge_item':
                    results.append({
                        'type': 'knowledge_item',
                        'text': doc['text'],
                        'score': float(bm25_scores[i]),
                        'content': doc['text'],
                        'bm25_score': float(bm25_scores[i]),
                        'id': doc['id']
                    })
                else:
                    results.append({
                        'type': 'fact',
                        'text': f"{doc['subject']} {doc['predicate']} {doc['object']}",
                        'content': f"{knowledge_store.trust_marker} {doc['subject']} {doc['predicate']} {doc['object']}",
                        'bm25_score': float(bm25_scores[i]),
                        'score': float(bm25_scores[i]),
                        'id': doc['id']
                    })
                    
        return results


class DenseVectorSearchStrategy:
    """Dense vector search strategy using embeddings."""
    
    def search(self, knowledge_store: 'KnowledgeStore', query_embedding: np.ndarray, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for embeddings.
        
        Args:
            knowledge_store: The knowledge store instance
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
        
        Returns:
            List of matching items with similarity scores
        """
        if not VECTOR_ENABLED or query_embedding is None:
            return []
        
        results = []
        
        with knowledge_store.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all facts with embeddings
            cursor.execute('SELECT id, subject, predicate, object, confidence, embedding FROM fact_triples WHERE embedding IS NOT NULL')
            facts = cursor.fetchall()
            
            # Get all knowledge items with embeddings
            cursor.execute('SELECT id, text, confidence, embedding FROM knowledge_items WHERE embedding IS NOT NULL')
            knowledge_items = cursor.fetchall()
            
            # Calculate similarity scores for facts
            for fact in facts:
                # Convert blob to numpy array
                embedding_bytes = fact['embedding']
                if embedding_bytes:
                    fact_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Skip if the embedding is the wrong shape
                    if fact_embedding.shape[0] != knowledge_store.embedding_dimension:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = knowledge_store._vector_similarity(query_embedding, fact_embedding)
                    
                    result = {
                        'id': fact['id'],
                        'text': f"{fact['subject']} {fact['predicate']} {fact['object']}",
                        'score': float(similarity),
                        'type': 'fact',
                        'confidence': fact['confidence']
                    }
                    
                    results.append(result)
            
            # Calculate similarity scores for knowledge items
            for item in knowledge_items:
                # Convert blob to numpy array
                embedding_bytes = item['embedding']
                if embedding_bytes:
                    item_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Skip if the embedding is the wrong shape
                    if item_embedding.shape[0] != knowledge_store.embedding_dimension:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = knowledge_store._vector_similarity(query_embedding, item_embedding)
                    
                    result = {
                        'id': item['id'],
                        'text': item['text'],
                        'score': float(similarity),
                        'type': 'knowledge',
                        'confidence': item['confidence']
                    }
                    
                    results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top results
        return results[:max_results]


class HybridSearchStrategy:
    """Hybrid search strategy combining dense vector and sparse BM25 results."""
    
    def search(self, knowledge_store: 'KnowledgeStore', query: str, query_embedding: np.ndarray, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense vector and sparse BM25 results.
        
        Args:
            knowledge_store: The knowledge store instance
            query: Text query for sparse search
            query_embedding: Query embedding for dense search
            max_results: Maximum number of results
            
        Returns:
            List of results with combined scores
        """
        # Get sparse BM25 results
        sparse_results = knowledge_store.search_strategy(
            strategy="bm25",
            query=query, 
            max_results=knowledge_store.rerank_top_k
        )
        
        # Get dense vector results
        dense_results = knowledge_store.search_strategy(
            strategy="dense_vector",
            query_embedding=query_embedding, 
            max_results=knowledge_store.rerank_top_k
        )
        
        # Combine results
        combined_results = {}
        
        # Normalize sparse scores
        max_sparse_score = max([r['bm25_score'] for r in sparse_results]) if sparse_results else 1.0
        min_sparse_score = min([r['bm25_score'] for r in sparse_results]) if sparse_results else 0.0
        sparse_range = max(1e-6, max_sparse_score - min_sparse_score)
        
        # Add sparse results with normalized scores
        for result in sparse_results:
            result_id = result['id']
            normalized_score = (result['bm25_score'] - min_sparse_score) / sparse_range if sparse_range > 0 else 0
            combined_results[result_id] = {
                **result,
                'sparse_score': result['bm25_score'],
                'normalized_sparse_score': normalized_score,
                'combined_score': (1 - knowledge_store.hybrid_alpha) * normalized_score
            }
            
        # Add dense results, combining with sparse scores when available
        for result in dense_results:
            result_id = result['id']
            if result_id in combined_results:
                # Update existing result
                combined_results[result_id].update({
                    'similarity': result.get('score', 0),
                    'dense_score': result.get('score', 0),
                    'combined_score': combined_results[result_id]['combined_score'] + knowledge_store.hybrid_alpha * result.get('score', 0)
                })
            else:
                # Add new result
                combined_results[result_id] = {
                    **result,
                    'dense_score': result.get('score', 0),
                    'combined_score': knowledge_store.hybrid_alpha * result.get('score', 0)
                }
                
        # Convert to list and sort by combined score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results_list[:max_results]


class KeywordSearchStrategy:
    """Basic keyword search strategy."""
    
    def search(self, knowledge_store: 'KnowledgeStore', query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Basic keyword search as fallback when vector/BM25 is not available.
        
        Args:
            knowledge_store: The knowledge store instance
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of results
        """
        results = []
        
        if knowledge_store.conn is not None:
            # Using in-memory database
            conn = knowledge_store.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(knowledge_store.db_path)
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            query_terms = query.lower().split()
            
            # Build a query that searches across all fields
            sql_query = '''
            SELECT * FROM fact_triples 
            WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
            '''
            
            # Improve search by expanding query with synonyms and related terms
            expanded_query = knowledge_store._expand_query(query)
            for term in expanded_query:
                param = f"%{term}%"
                cursor.execute(sql_query, (param, param, param, max_results))
                fact_results = [dict(row) for row in cursor.fetchall()]
                
                # Add facts to results with trust marker
                for fact in fact_results:
                    results.append({
                        'type': 'fact',
                        'content': f"{knowledge_store.trust_marker} {fact['subject']} {fact['predicate']} {fact['object']}",
                        'confidence': fact['confidence'],
                        'timestamp': fact['timestamp'],
                        'source': fact['source'] if fact['source'] else "memory"
                    })
                
                if len(results) >= max_results:
                    break
        except sqlite3.OperationalError as e:
            logging.error(f"SQLite error in keyword search: {e}")
        
        if knowledge_store.conn is None:
            conn.close()
            
        # Search summaries if needed
        if len(results) < max_results and hasattr(knowledge_store, 'vector_store'):
            summary_scores = []
            for summary_id, keywords in knowledge_store.vector_store.items():
                if not isinstance(keywords, str):
                    continue
                    
                # Improved keyword matching with partial matches
                score = 0
                for term in query_terms:
                    if term in keywords.lower():
                        score += 1
                    else:
                        # Check for partial matches
                        for keyword in keywords.lower().split():
                            if term in keyword or keyword in term:
                                score += 0.5
                                break
                
                if score > 0:
                    summary_scores.append((summary_id, score))
            
            # Sort by score and get top results
            summary_scores.sort(key=lambda x: x[1], reverse=True)
            for summary_id, score in summary_scores[:max_results - len(results)]:
                if hasattr(knowledge_store, 'summaries') and summary_id in knowledge_store.summaries:
                    summary = knowledge_store.summaries[summary_id]
                    results.append({
                        'type': 'summary',
                        'content': f"From previous conversation: {summary['text']}",
                        'keywords': summary['keywords'],
                        'timestamp': summary['timestamp'],
                        'score': score
                    })
                    
        return results

class KnowledgeStore(KnowledgeStoreInterface):
    """Persistent storage for important facts and retrievable context."""
    
    def __init__(self, config: CortexFlowConfig):
        """
        Initialize the knowledge store with configuration.
        
        Args:
            config: Configuration for the knowledge store
        """
        self.config = config
        self.db_path = config.knowledge_store_path
        
        # Trust marker for high-confidence facts
        self.trust_marker = config.trust_marker if hasattr(config, 'trust_marker') else "📚"
        
        # Configure retrieval settings
        self.use_reranking = config.use_reranking if hasattr(config, 'use_reranking') else True
        self.rerank_top_k = config.rerank_top_k if hasattr(config, 'rerank_top_k') else 15
        
        # GraphRAG settings
        self.use_graph_rag = config.use_graph_rag if hasattr(config, 'use_graph_rag') else False
        self.graph_weight = config.graph_weight if hasattr(config, 'graph_weight') else 0.3
        
        # Ontology settings
        self.use_ontology = config.use_ontology if hasattr(config, 'use_ontology') else False
        
        # Metadata tracking
        self.track_provenance = config.track_provenance if hasattr(config, 'track_provenance') else True
        self.track_confidence = config.track_confidence if hasattr(config, 'track_confidence') else True
        self.track_temporal = config.track_temporal if hasattr(config, 'track_temporal') else True
        
        # Hybrid search parameters
        self.hybrid_alpha = 0.7  # Weight for dense vector search (1-alpha is weight for sparse BM25)
        
        # Inference engine settings
        self.use_inference_engine = config.use_inference_engine if hasattr(config, 'use_inference_engine') else False
        self.inference_engine = None
        
        if self.use_inference_engine:
            try:
                from .inference import InferenceEngine
                self.inference_engine = InferenceEngine(self, config)
                logging.info(f"Inference engine initialized successfully")
            except ImportError:
                logging.error(f"Inference module not found, disabling inference features")
                self.use_inference_engine = False
            except Exception as e:
                logging.error(f"Error initializing inference engine: {e}")
                self.use_inference_engine = False
        
        # Initialize database connection
        self._init_db()
        
        # Initialize vector embedding model
        self.model = None
        self.embedding_dimension = 384
        
        try:
            if VECTOR_ENABLED:
                from sentence_transformers import SentenceTransformer
                model_name = config.embedding_model if hasattr(config, 'embedding_model') else "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(model_name)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logging.info(f"Loaded embedding model: {model_name} with dimension {self.embedding_dimension}")
        except Exception as e:
            logging.error(f"Error loading vector model: {e}")
        
        # Initialize BM25 indexer
        self.bm25_index = None
        if BM25_ENABLED:
            # Initialize BM25 related attributes without creating the index yet
            self.bm25_corpus = []
            self.id_to_doc_mapping = {}
            self.doc_to_id_mapping = {}
            self._update_bm25_index(force_rebuild=True)
        
        # Initialize graph store if enabled
        self.graph_store = None
        
        if self.use_graph_rag:
            try:
                from .graph_store import GraphStore
                self.graph_store = GraphStore(config)
                logging.info(f"Graph store initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing graph store: {e}")
                self.use_graph_rag = False
            
        # Initialize ontology if enabled
        self.ontology = None
        if self.use_ontology:
            try:
                from .ontology import Ontology
                self.ontology = Ontology(self.db_path)
                logging.info(f"Ontology system initialized successfully")
                
                # Connect the ontology to the graph store if both are enabled
                if self.use_graph_rag and self.graph_store:
                    self.graph_store.ontology = self.ontology
                    logging.info(f"Connected ontology to graph store")
            except ImportError:
                logging.error(f"Ontology module not found, disabling ontology features")
                self.use_ontology = False
            except Exception as e:
                logging.error(f"Error initializing ontology: {e}")
                self.use_ontology = False
            
        # In-memory connection for better performance
        self.conn = None
        try:
            self.conn = sqlite3.connect("")
            self._copy_db_to_memory()
        except Exception as e:
            logging.error(f"Error creating in-memory database: {e}")
            self.conn = None
        
        logging.info(f"Knowledge store initialized with vector retrieval: {VECTOR_ENABLED}, BM25: {BM25_ENABLED}")
        
        # Add embedding cache
        self.embedding_cache = {}
        self.max_cache_size = 1000  # Limit cache size
        
        # Add tracking for index updates
        self.last_indexed_fact_id = 0
        self.last_indexed_knowledge_id = 0
        
        # Initialize search strategies
        self._search_strategies = {
            "bm25": BM25SearchStrategy(),
            "dense_vector": DenseVectorSearchStrategy(),
            "hybrid": HybridSearchStrategy(),
            "keyword": KeywordSearchStrategy()
        }
    
    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """
        Context manager for database connections.
        
        Returns a connection to the database, either the persistent in-memory connection
        or a new connection to the file-based database if in-memory is not available.
        
        The connection is automatically closed if it's a new connection.
        """
        if self.conn is not None:
            # Using in-memory database
            yield self.conn
        else:
            # Create a new connection to the file-based database
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_db(self):
        """Initialize the database for storing structured knowledge."""
        # Create tables if they don't exist
        try:
            # Handle in-memory database in a platform-independent way
            if self.db_path == ":memory:":
                # For Windows compatibility, use empty string instead of ":memory:"
                conn = sqlite3.connect("")
                conn.row_factory = sqlite3.Row
            else:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            
            # Create fact triples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fact_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL,
                    timestamp REAL,
                    source TEXT,
                    embedding BLOB
                )
            ''')
            
            # Create knowledge items table for direct text storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    confidence REAL,
                    timestamp REAL,
                    source TEXT,
                    embedding BLOB
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON fact_triples(subject)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicate ON fact_triples(predicate)')
            
            # Create graph entities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL UNIQUE,
                    entity_type TEXT,
                    metadata TEXT,
                    timestamp REAL
                )
            ''')
            
            # Create graph relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS graph_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER,
                    target_id INTEGER,
                    relation_type TEXT,
                    weight REAL,
                    metadata TEXT,
                    timestamp REAL,
                    FOREIGN KEY (source_id) REFERENCES graph_entities (id),
                    FOREIGN KEY (target_id) REFERENCES graph_entities (id)
                )
            ''')
            
            # Create indices for graph queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON graph_relationships(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON graph_relationships(target_id)')
            
            conn.commit()
            conn.close()
            
            logging.debug(f"Database initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logging.error(f"SQLite error initializing database: {e}")
    
    def _copy_db_to_memory(self):
        """Copy the database to memory for faster access."""
        try:
            # Open the file database
            if self.db_path == ":memory:":
                # If we're already using in-memory database, no need to copy
                return
            
            disk_conn = sqlite3.connect(self.db_path)
            
            # Create in-memory connection if it doesn't exist
            if self.conn is None:
                self.conn = sqlite3.connect("")
            
            # Copy to in-memory database
            disk_conn.backup(self.conn)
            disk_conn.close()
            
            logging.debug("Database copied to memory")
            
        except sqlite3.Error as e:
            logging.error(f"Error copying database to memory: {e}")
            
        except AttributeError:
            logging.error("In-memory connection not initialized")
            self.conn = None
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for text with caching.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if embedding fails
        """
        if not VECTOR_ENABLED or self.model is None:
            return None
            
        # Use hash of text as cache key (or another suitable unique identifier)
        cache_key = hash(text)
        
        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Manage cache size
            if len(self.embedding_cache) >= self.max_cache_size:
                # Remove a random item to prevent cache from growing too large
                # A more sophisticated LRU implementation would be better
                self.embedding_cache.pop(random.choice(list(self.embedding_cache.keys())))
            
            # Store in cache
            self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def _update_bm25_index(self, force_rebuild=False):
        """
        Update the BM25 index with the latest facts and knowledge items.
        
        Args:
            force_rebuild: Force complete rebuild of the index
        """
        # Skip if BM25 is not enabled
        if not BM25_ENABLED:
            return
        
        # Only update if we have the required libraries
        try:
            import rank_bm25
            from nltk.tokenize import word_tokenize
        except ImportError:
            logging.error("BM25 indexing requires rank_bm25 and nltk")
            return
        
        # Check if we need to rebuild based on elapsed time or new entries
        current_time = time.time()
        needs_rebuild = (
            force_rebuild or
            not hasattr(self, 'bm25_last_update') or
            current_time - getattr(self, 'bm25_last_update', 0) > 600  # 10 minutes
        )
        
        if needs_rebuild:
            # Get latest document IDs from the database
            max_fact_id = 0
            max_knowledge_id = 0
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get max fact ID
                cursor.execute('SELECT MAX(id) FROM fact_triples')
                result = cursor.fetchone()
                if result and result[0]:
                    max_fact_id = result[0]
                
                # Get max knowledge ID
                cursor.execute('SELECT MAX(id) FROM knowledge_items')
                result = cursor.fetchone()
                if result and result[0]:
                    max_knowledge_id = result[0]
            
            # Check if we need to update based on new entries
            if (not force_rebuild and
                hasattr(self, 'last_indexed_fact_id') and
                hasattr(self, 'last_indexed_knowledge_id') and
                max_fact_id == self.last_indexed_fact_id and
                max_knowledge_id == self.last_indexed_knowledge_id):
                # No new documents, skip update
                return
            
            try:
                # Rebuild the corpus from the database
                if force_rebuild:
                    self.bm25_corpus = []
                    self.id_to_doc_mapping = {}
                    self.doc_to_id_mapping = {}
                    doc_id = 0
                
                with self.get_connection() as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Add fact triples to the corpus
                    last_id = getattr(self, 'last_indexed_fact_id', 0)
                    cursor.execute('SELECT id, subject, predicate, object FROM fact_triples WHERE id > ?', (last_id,))
                    
                    for item in cursor.fetchall():
                        # Create document text and tokenize
                        doc_text = f"{item['subject']} {item['predicate']} {item['object']}"
                        try:
                            tokenized_doc = word_tokenize(doc_text.lower())
                        except:
                            # Fallback to simple whitespace tokenization
                            tokenized_doc = doc_text.lower().split()
                        
                        self.bm25_corpus.append(tokenized_doc)
                        self.id_to_doc_mapping[doc_id] = {
                            'id': item['id'],
                            'text': doc_text,
                            'type': 'fact_triple'
                        }
                        self.doc_to_id_mapping[doc_text] = doc_id
                        doc_id += 1
                    
                    # Add knowledge items to the corpus
                    last_id = getattr(self, 'last_indexed_knowledge_id', 0)
                    cursor.execute('SELECT id, text FROM knowledge_items WHERE id > ?', (last_id,))
                    
                    for item in cursor.fetchall():
                        # Create document text and tokenize
                        doc_text = item['text']
                        try:
                            tokenized_doc = word_tokenize(doc_text.lower())
                        except:
                            # Fallback to simple whitespace tokenization
                            tokenized_doc = doc_text.lower().split()
                        
                        self.bm25_corpus.append(tokenized_doc)
                        self.id_to_doc_mapping[doc_id] = {
                            'id': item['id'],
                            'text': doc_text,
                            'type': 'knowledge_item'
                        }
                        self.doc_to_id_mapping[doc_text] = doc_id
                        doc_id += 1
                        
                    # Update the latest IDs we've indexed
                    self.last_indexed_fact_id = max_fact_id
                    self.last_indexed_knowledge_id = max_knowledge_id
                    
            except Exception as e:
                logging.error(f"Error updating BM25 index: {e}")
        
            # Recreate the BM25 index with the updated corpus
            if self.bm25_corpus:
                import rank_bm25
                self.bm25_index = rank_bm25.BM25Okapi(self.bm25_corpus)
            
            self.bm25_last_update = time.time()
            logging.debug(f"BM25 index updated with {len(self.bm25_corpus)} documents")
    
    def store_fact_triple(self, subject: str, predicate: str, obj: str, 
                          confidence: float = 1.0, source: str = None) -> int:
        """
        Store a fact triple in the knowledge store.
        
        Args:
            subject: Subject of the fact
            predicate: Predicate/relation
            obj: Object of the fact
            confidence: Confidence score (0.0-1.0)
            source: Source of the fact
            
        Returns:
            ID of the stored fact
        """
        # Generate embedding for the fact if possible
        fact_text = f"{subject} {predicate} {obj}"
        embedding = self._generate_embedding(fact_text)
        embedding_blob = None
        
        if embedding is not None:
            # Convert embedding to binary blob
            embedding_blob = embedding.tobytes()
        
        fact_id = None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if embedding_blob is not None:
                # Store with embedding
                cursor.execute(
                    'INSERT INTO fact_triples (subject, predicate, object, confidence, timestamp, source, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (subject, predicate, obj, confidence, time.time(), source, embedding_blob)
                )
            else:
                # Store without embedding
                cursor.execute(
                    'INSERT INTO fact_triples (subject, predicate, object, confidence, timestamp, source) VALUES (?, ?, ?, ?, ?, ?)',
                    (subject, predicate, obj, confidence, time.time(), source)
                )
            
            fact_id = cursor.lastrowid
            conn.commit()
        
        # Update BM25 index after adding a new fact
        self._update_bm25_index()
        
        return fact_id
    
    def get_facts_about(self, subject: str) -> List[Dict[str, Any]]:
        """
        Retrieve facts about a subject.
        
        Args:
            subject: The subject to query
            
        Returns:
            List of fact dictionaries
        """
        facts = []
        
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM fact_triples WHERE subject = ? ORDER BY confidence DESC',
                (subject,)
            )
            
            facts = [dict(row) for row in cursor.fetchall()]
        
        return facts
    
    def get_facts_by_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Retrieve facts by predicate.
        
        Args:
            predicate: The predicate to query
            
        Returns:
            List of fact dictionaries
        """
        facts = []
        
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM fact_triples WHERE predicate = ? ORDER BY confidence DESC',
                (predicate,)
            )
            
            facts = [dict(row) for row in cursor.fetchall()]
        
        return facts
    
    def update_fact_confidence(self, fact_id: int, new_confidence: float) -> bool:
        """
        Update the confidence of a fact.
        
        Args:
            fact_id: ID of the fact
            new_confidence: New confidence score
            
        Returns:
            True if update was successful
        """
        success = False
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE fact_triples SET confidence = ? WHERE id = ?',
                (new_confidence, fact_id)
            )
            
            success = cursor.rowcount > 0
            conn.commit()
        
        return success
    
    def forget_old_facts(self, threshold_days: int = 30, max_confidence: float = 0.7) -> int:
        """
        Remove facts older than threshold with confidence below max_confidence.
        
        Args:
            threshold_days: Age threshold in days
            max_confidence: Maximum confidence threshold for removal
            
        Returns:
            Number of facts removed
        """
        threshold_time = time.time() - (threshold_days * 86400)
        count = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'DELETE FROM fact_triples WHERE timestamp < ? AND confidence < ?',
                (threshold_time, max_confidence)
            )
            
            count = cursor.rowcount
            conn.commit()
        
        return count
    
    def store_conversation_summary(self, summary: str, keywords: List[str], 
                                 timestamp: float = None) -> str:
        """
        Store a conversation summary with keywords.
        
        Args:
            summary: Conversation summary text
            keywords: List of keywords
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            ID of the stored summary
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Generate a unique ID based on timestamp
        summary_id = f"summary_{int(timestamp)}"
        
        # Generate embedding for the summary
        embedding = self._generate_embedding(summary)
        
        # Store the summary with metadata
        self.summaries[summary_id] = {
            'text': summary,
            'keywords': keywords,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'embedding': embedding.tolist() if embedding is not None else None
        }
        
        # Add to vector store for retrieval
        if embedding is not None:
            self.vector_store[summary_id] = embedding
        else:
            # Fall back to keyword storage if embedding fails
            self.vector_store[summary_id] = ' '.join(keywords)
            
        # Update BM25 index
        self._update_bm25_index()
        
        return summary_id
    
    def _bm25_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search for keyword-based retrieval.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of results with BM25 scores
        """
        return self.search_strategy(strategy="bm25", query=query, max_results=max_results)
    
    def _dense_vector_search(self, query_embedding: np.ndarray, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for embeddings.
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
        
        Returns:
            List of matching items with similarity scores
        """
        return self.search_strategy(strategy="dense_vector", query_embedding=query_embedding, max_results=max_results)
    
    def _hybrid_search(self, query: str, query_embedding: np.ndarray, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense vector and sparse BM25 results.
        
        Args:
            query: Text query for sparse search
            query_embedding: Query embedding for dense search
            max_results: Maximum number of results
            
        Returns:
            List of results with combined scores
        """
        return self.search_strategy(strategy="hybrid", query=query, query_embedding=query_embedding, max_results=max_results)
    
    def _keyword_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Basic keyword search as fallback when vector/BM25 is not available.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of results
        """
        return self.search_strategy(strategy="keyword", query=query, max_results=max_results)
    
    def search_strategy(self, strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Use a specific search strategy.
        
        Args:
            strategy: The name of the strategy to use
            **kwargs: Arguments to pass to the strategy
            
        Returns:
            Search results from the selected strategy
        """
        if strategy not in self._search_strategies:
            logging.warning(f"Unknown search strategy: {strategy}, falling back to keyword search")
            strategy = "keyword"
            
        return self._search_strategies[strategy].search(self, **kwargs)
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank search results for improved relevance.
        
        Args:
            query: Original search query
            results: Initial search results
            max_results: Maximum number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results or len(results) <= 1:
            return results[:max_results]
            
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Extract content texts
            texts = []
            for result in results:
                if 'text' in result:
                    texts.append(result['text'])
                elif 'content' in result:
                    texts.append(result['content'])
                else:
                    # If no text content found, use a placeholder
                    texts.append("")
                    
            # Generate embeddings for all texts
            text_embeddings = [self._generate_embedding(text) for text in texts]
            
            # Calculate similarity scores
            similarities = []
            for embed in text_embeddings:
                if embed is not None and query_embedding is not None:
                    # Calculate cosine similarity
                    similarity = self._vector_similarity(query_embedding, embed)
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)
                
            # Re-weight scores based on similarity
            for i, result in enumerate(results):
                # Get current score or default to 0.5
                orig_score = result.get("score", 0.5)
                
                # Combine with similarity (if available)
                if i < len(similarities):
                    sim_score = similarities[i]
                    # Weighted combination of original score and similarity
                    new_score = orig_score * 0.7 + sim_score * 0.3
                    result["score"] = new_score
                    result["similarity"] = sim_score
            
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            return reranked_results[:max_results]
            
        except Exception as e:
            logging.error(f"Error in re-ranking: {e}")
            # Fall back to original results on error
            return results[:max_results]
        
    def get_relevant_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant knowledge for a query using hybrid search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        # Update the BM25 index to ensure it's fresh
        self._update_bm25_index()
        
        results = []
        
        # Use vector search if available
        if VECTOR_ENABLED and self.model is not None:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Use hybrid search (vector + keyword)
            if BM25_ENABLED and self.bm25_index is not None:
                results = self._hybrid_search(query, query_embedding, max_results=self.rerank_top_k)
            else:
                # Fall back to vector-only search
                results = self._dense_vector_search(query_embedding, max_results=self.rerank_top_k)
        else:
            # Fall back to keyword search if vector search is not available
            if BM25_ENABLED:
                results = self._bm25_search(query, max_results=self.rerank_top_k)
            else:
                # Last resort: simple keyword search
                results = self._keyword_search(query, max_results=self.rerank_top_k)
        
        # Get graph-based results if enabled
        graph_results = []
        if self.use_graph_rag:
            graph_results = self._graph_search(query, max_results=self.rerank_top_k)
        
        # Merge results from different retrieval methods
        combined_results = []
        
        # Create a set to track unique items
        seen_texts = set()
        
        # Add standard retrieval results
        for result in results:
            text = result.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                combined_results.append(result)
        
        # Add graph-based results with a factor to boost their scores
        for result in graph_results:
            text = result.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                # Apply graph weight to score
                if "score" in result:
                    result["score"] = result["score"] * (1.0 + self.graph_weight)
                combined_results.append(result)
        
        # Apply re-ranking if enabled
        if self.use_reranking:
            combined_results = self._rerank_results(query, combined_results, max_results)
        else:
            # Sort by score and limit results
            combined_results = sorted(combined_results, key=lambda x: x.get("score", 0), reverse=True)[:max_results]
        
        # Format results for the context
        final_results = []
        for item in combined_results:
            # Add trust marker if confidence is high
            if item.get("confidence", 0) > 0.8:
                text = f"{self.trust_marker} {item.get('text', '')}"
            else:
                text = item.get("text", "")
                
            # Add provenance if available
            if "source" in item:
                text = f"{text} [Source: {item['source']}]"
                
            final_results.append({
                "text": text,
                "score": item.get("score", 0),
                "confidence": item.get("confidence", 0.5),
                "source": item.get("source", "knowledge_store"),
                "type": item.get("type", "fact")
            })
            
        return final_results
    
    def _graph_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get knowledge from graph store relevant to the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of relevant knowledge items from graph
        """
        results = []
        
        try:
            # Extract entities from query
            entities = self.graph_store.extract_entities(query)
            
            # Track entities found in query
            query_entities = [entity["text"] for entity in entities]
            
            logging.debug(f"Graph search for query: '{query}'")
            logging.debug(f"Extracted entities: {query_entities}")
            
            # If we don't have entities, try using noun phrases and proper nouns
            if not query_entities:
                # Extract noun phrases from query
                try:
                    doc = self.graph_store.nlp(query) if hasattr(self.graph_store, 'nlp') else None
                    if doc:
                        # Find noun phrases
                        for chunk in doc.noun_chunks:
                            if len(chunk.text) > 2:  # Skip very short chunks
                                query_entities.append(chunk.text)
                                
                        # Find proper nouns
                        for token in doc:
                            if token.pos_ == "PROPN" and len(token.text) > 2:
                                if token.text not in query_entities:
                                    query_entities.append(token.text)
                                    
                        logging.debug(f"Added potential entities from query: {query_entities}")
                except Exception as e:
                    logging.error(f"Error extracting additional entities: {e}")
            
            # If we still don't have entities, use important words from the query
            if not query_entities:
                # Use words that might be important
                for word in query.split():
                    if len(word) > 3 and word.lower() not in ["what", "where", "when", "how", "why", "who", "does", "is", "are", "the", "and", "that", "this"]:
                        query_entities.append(word)
                        
                logging.debug(f"Using important words as entities: {query_entities}")
            
            # If we have entities in the query
            if query_entities:
                # Get subgraph relevant to the query
                subgraph = self.graph_store.build_knowledge_subgraph(query, max_nodes=30)
                
                logging.debug(f"Built subgraph with {len(subgraph['nodes'])} nodes and {len(subgraph['edges'])} edges")
                
                # For each entity in the query, explore its neighbors
                for entity_text in query_entities:
                    neighbors = self.graph_store.get_entity_neighbors(
                        entity=entity_text,
                        direction="both",
                        limit=5
                    )
                    
                    logging.debug(f"Found {len(neighbors)} neighbors for entity '{entity_text}'")
                    
                    # Convert neighbors to knowledge items
                    for neighbor in neighbors:
                        direction = neighbor.get("direction", "")
                        relation = neighbor.get("relation", "")
                        entity = neighbor.get("entity", "")
                        
                        if direction == "outgoing":
                            # Entity → Relation → Neighbor
                            fact_text = f"{entity_text} {relation} {entity}"
                        else:
                            # Neighbor → Relation → Entity
                            fact_text = f"{entity} {relation} {entity_text}"
                        
                        results.append({
                            "text": fact_text,
                            "score": neighbor.get("weight", 0.5),
                            "confidence": 0.7,  # Default confidence for graph-based facts
                            "source": "knowledge_graph",
                            "type": "graph_fact",
                            "entity_type": neighbor.get("type")
                        })
                
                # Try to find paths between different entities in the query
                if len(query_entities) >= 2:
                    # Check paths between pairs of entities
                    for i in range(len(query_entities)):
                        for j in range(i + 1, len(query_entities)):
                            start_entity = query_entities[i]
                            end_entity = query_entities[j]
                            
                            paths = self.graph_store.path_query(
                                start_entity=start_entity,
                                end_entity=end_entity,
                                max_hops=4  # Increased max hops for path finding
                            )
                            
                            logging.debug(f"Found {len(paths)} paths between '{start_entity}' and '{end_entity}'")
                            
                            # Convert paths to knowledge items
                            for path in paths:
                                path_text = self._format_path_as_text(path)
                                if path_text:
                                    results.append({
                                        "text": path_text,
                                        "score": 0.8,  # Paths connecting query entities are highly relevant
                                        "confidence": 0.75,
                                        "source": "knowledge_graph",
                                        "type": "graph_path"
                                    })
            
            # If no results were found but we have query terms, try fuzzy matching
            if not results and len(query.split()) > 0:
                # Try fuzzy matching with common terms in query
                for word in query.split():
                    if len(word) > 3:  # Only use meaningful words
                        cursor = None
                        try:
                            if hasattr(self, 'conn') and self.conn is not None:
                                conn = self.conn
                            else:
                                conn = sqlite3.connect(self.graph_store.db_path)
                                
                            conn.row_factory = sqlite3.Row
                            cursor = conn.cursor()
                            
                            # Search for entities that might match
                            cursor.execute('''
                                SELECT id, entity, entity_type FROM graph_entities 
                                WHERE entity LIKE ? LIMIT 5
                            ''', (f"%{word}%",))
                            
                            entity_matches = cursor.fetchall()
                            
                            for entity in entity_matches:
                                entity_id = entity['id']
                                entity_text = entity['entity']
                                
                                # Look for relationships involving this entity
                                cursor.execute('''
                                    SELECT r.relation_type, e2.entity 
                                    FROM graph_relationships r
                                    JOIN graph_entities e2 ON r.target_id = e2.id
                                    WHERE r.source_id = ?
                                    LIMIT 3
                                ''', (entity_id,))
                                
                                relations = cursor.fetchall()
                                
                                for relation in relations:
                                    relation_type = relation['relation_type']
                                    target_entity = relation['entity']
                                    
                                    fact_text = f"{entity_text} {relation_type} {target_entity}"
                                    
                                    results.append({
                                        "text": fact_text,
                                        "score": 0.4,  # Lower score for fuzzy matches
                                        "confidence": 0.5,
                                        "source": "knowledge_graph",
                                        "type": "graph_fact"
                                    })
                        except Exception as e:
                            logging.error(f"Error in fuzzy entity matching: {e}")
                        finally:
                            if not hasattr(self, 'conn') or self.conn is None:
                                if conn is not None:
                                    conn.close()
            
            # Sort results by score
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:max_results]
            
            logging.debug(f"Graph search returned {len(results)} results")
            
        except Exception as e:
            logging.error(f"Error in graph search: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
        
        return results
    
    def _format_path_as_text(self, path: List[Dict[str, Any]]) -> str:
        """
        Format a graph path as readable text.
        
        Args:
            path: List of path nodes with their details
            
        Returns:
            Formatted text representation of the path
        """
        if not path or len(path) < 2:
            return ""
            
        path_segments = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            current_entity = current.get("entity", "")
            relation = current.get("next_relation", {}).get("type", "is related to")
            
            path_segments.append(f"{current_entity} {relation}")
            
            # Add the final entity for the last segment
            if i == len(path) - 2:
                path_segments.append(next_node.get("entity", ""))
        
        return " → ".join(path_segments)
    
    def extract_facts_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract fact triples from text.
        
        Args:
            text: Input text to extract facts from
            
        Returns:
            List of (subject, predicate, object) triples
        """
        # Extract graph-based relations if available
        if hasattr(self, 'graph_store') and self.graph_store is not None:
            try:
                return self.graph_store.extract_relations(text)
            except Exception as e:
                logging.error(f"Error extracting relations from graph store: {e}")
                # Fall back to simple extraction
        
        # Simple pattern-based extraction fallback
        facts = []
        
        # Split text into sentences (simplified)
        sentences = text.replace("! ", ". ").replace("? ", ". ").split(". ")
        
        for sentence in sentences:
            # Skip short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Very simple SVO (subject-verb-object) extraction
            words = sentence.split()
            for i in range(1, len(words) - 1):
                # This is a very basic heuristic and not linguistically accurate
                subject = words[i-1]
                predicate = words[i]
                obj = words[i+1]
                
                # Skip common words and punctuation
                if (len(subject) > 3 and len(predicate) > 3 and len(obj) > 3 and
                    subject not in ["this", "that", "these", "those"] and
                    obj not in ["this", "that", "these", "those"]):
                    facts.append((subject, predicate, obj))
        
        return facts
    
    def remember_explicit(self, text: str, source: str = "user_command", confidence: float = 0.95) -> List[int]:
        """
        Explicitly add knowledge to the knowledge store, with special handling for facts marked with a trust marker.
        
        This is a low-level implementation method that handles:
        1. Parsing for facts prefixed with trust markers to extract triples
        2. Building knowledge graph relations
        3. Storing raw text as knowledge items when no structured data is extracted
        
        Most callers should use remember() or add_knowledge() instead, which provide a simpler interface.
        
        Args:
            text: The text to add
            source: Source of the knowledge (for provenance)
            confidence: Confidence level in the knowledge (0.0 to 1.0)
            
        Returns:
            IDs of the facts added
        """
        fact_ids = []
        
        # Parse for facts prefixed with trust marker
        lines = text.strip().split('\n')
        for line in lines:
            if line.startswith(self.trust_marker):
                fact = line[len(self.trust_marker):].strip()
                if fact:
                    try:
                        fact_id = self.store_fact_triple(
                            subject=fact.split()[0],
                            predicate=fact.split()[1],
                            obj=fact.split()[2],
                            confidence=confidence,
                            source=source
                        )
                        if fact_id:
                            fact_ids.append(fact_id)
                    except Exception as e:
                        logging.error(f"Error storing fact: {e}")
        
        # Process text to build knowledge graph with enhanced metadata
        if hasattr(self, 'graph_store') and self.graph_store is not None:
            try:
                # Add temporal information if available
                current_time = datetime.now().isoformat()
                
                # Process text to extract entities and relations with metadata
                relations_added = self.graph_store.process_text_to_graph(
                    text=text, 
                    source=source
                )
                logging.debug(f"Added {relations_added} relations to knowledge graph")
                
                # Manually add the text as a single fact if no relations were extracted
                if not relations_added and not fact_ids:
                    # Try to extract simple subject-verb-object
                    words = text.split()
                    if len(words) >= 3:
                        subject = words[0]
                        predicate = "states"
                        obj = " ".join(words[1:])
                        
                        # Add to graph with metadata
                        success = self.graph_store.add_relation(
                            source_entity=subject,
                            relation_type=predicate,
                            target_entity=obj,
                            provenance=source if self.track_provenance else None,
                            confidence=confidence if self.track_confidence else 0.5,
                            temporal_start=current_time if self.track_temporal else None
                        )
                        
                        if success:
                            logging.debug(f"Added backup relation: {subject} {predicate} {obj}")
            except Exception as e:
                logging.error(f"Error processing text for knowledge graph: {e}")
        else:
            logging.debug("No graph_store available for graph-based storage")
        
        # If no triples were extracted, store the text as a regular knowledge item
        if not fact_ids:
            # Store the text directly
            try:
                # Store as a direct knowledge item
                result = self.store_knowledge_item(text, confidence, source)
                if result:
                    fact_ids.append(result)
            except Exception as e:
                logging.error(f"Error storing direct knowledge: {e}")
                
        return fact_ids
    
    def store_knowledge_item(self, text: str, confidence: float = 0.9, source: str = None) -> int:
        """
        Store a knowledge item in the database.
        
        Args:
            text: Text of the knowledge item
            confidence: Confidence score (0.0-1.0)
            source: Source of the knowledge item
        
        Returns:
            ID of the stored knowledge item
        """
        # Generate embedding for the knowledge item if possible
        embedding = self._generate_embedding(text)
        embedding_blob = None
        
        if embedding is not None:
            # Convert embedding to binary blob
            embedding_blob = embedding.tobytes()
        
        item_id = None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if embedding_blob is not None:
                # Store with embedding
                cursor.execute(
                    'INSERT INTO knowledge_items (text, confidence, timestamp, source, embedding) VALUES (?, ?, ?, ?, ?)',
                    (text, confidence, time.time(), source, embedding_blob)
                )
            else:
                # Store without embedding
                cursor.execute(
                    'INSERT INTO knowledge_items (text, confidence, timestamp, source) VALUES (?, ?, ?, ?)',
                    (text, confidence, time.time(), source)
                )
            
            item_id = cursor.lastrowid
            conn.commit()
        
        # Update BM25 index after adding a new knowledge item
        self._update_bm25_index()
        
        return item_id
    
    def add_knowledge(self, text: str, source: str = None, confidence: float = 0.95) -> List[int]:
        """
        Add knowledge to the system, the primary method for adding knowledge programmatically.
        
        Args:
            text: Text to remember
            source: Source of the knowledge, defaults to "system" if None
            confidence: Confidence score for the facts
            
        Returns:
            List of IDs for stored facts
        """
        return self.remember_explicit(text, source or "system", confidence)
    
    def remember_knowledge(self, text: str, source: str = None, confidence: float = 0.95) -> List[int]:
        """
        Remember knowledge from text.
        
        DEPRECATED: Use add_knowledge() instead.
        
        Args:
            text: Text to remember
            source: Source of the knowledge
            confidence: Confidence score for the facts
            
        Returns:
            List of IDs for stored facts
        """
        import warnings
        warnings.warn(
            "remember_knowledge() is deprecated; use add_knowledge() instead",
            DeprecationWarning, stacklevel=2
        )
        return self.add_knowledge(text, source, confidence)
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand a query with synonyms and related terms to improve search.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded search terms
        """
        expanded = [query.lower()]
        
        # Add original terms
        terms = query.lower().split()
        expanded.extend(terms)
        
        # Add variations for common question patterns
        question_words = ["what", "where", "who", "when", "how", "why"]
        for word in question_words:
            if word in terms:
                # Remove question words to focus on entities
                clean_terms = [t for t in terms if t not in question_words and len(t) > 2]
                expanded.extend(clean_terms)
                
                # Add special handling for common questions
                if "name" in query.lower():
                    expanded.append("is")
                    expanded.append("called")
                if "live" in query.lower() or "location" in query.lower():
                    expanded.append("in")
                    expanded.append("from")
                    expanded.append("at")
                if "dog" in query.lower() or "pet" in query.lower():
                    expanded.append("named")
                    expanded.append("has")
                if "color" in query.lower() or "favourite" in query.lower() or "favorite" in query.lower():
                    expanded.append("likes")
                    expanded.append("prefers")
                    expanded.append("is")
                
                break
        
        # Remove duplicates and very short terms
        expanded = list(set([term for term in expanded if len(term) > 1]))
        return expanded
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'facts') and self.facts is not None:
                self.facts.close()
        except Exception as e:
            logging.error(f"Error closing facts: {e}")
            
        try:
            if hasattr(self, 'summaries') and self.summaries is not None:
                self.summaries.close()
        except Exception as e:
            logging.error(f"Error closing summaries: {e}")
            
        try:
            if hasattr(self, 'graph_store') and self.graph_store is not None:
                self.graph_store.close()
        except Exception as e:
            logging.error(f"Error closing graph store: {e}")
            
        try:
            if hasattr(self, 'conn') and self.conn is not None:
                self.conn.close()
                self.conn = None
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.close()
        except Exception as e:
            logging.error(f"Error in __del__: {e}")
            
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get snapshots of the knowledge store for consistency evaluation.
        
        Returns:
            List of knowledge snapshots, each with a timestamp
        """
        snapshots = []
        
        try:
            # Check if we have a history of snapshots in the database
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Try to get snapshots from the database
                try:
                    cursor.execute('''
                    SELECT id, snapshot_data, timestamp 
                    FROM knowledge_snapshots
                    ORDER BY timestamp DESC
                    LIMIT 10
                    ''')
                    
                    rows = cursor.fetchall()
                    
                    if rows:
                        for row in rows:
                            snapshot_data = json.loads(row['snapshot_data'])
                            snapshot_data['timestamp'] = row['timestamp']
                            snapshots.append(snapshot_data)
                
                except sqlite3.OperationalError:
                    # Table might not exist, we'll create it later
                    pass
                
            # If no snapshots found, return current state as the only snapshot
            if not snapshots:
                current_snapshot = self.take_snapshot()
                snapshots.append(current_snapshot)
                
        except Exception as e:
            logging.error(f"Error getting snapshots: {e}")
            
            # Return an empty snapshot as fallback
            snapshots.append({
                "timestamp": datetime.now().timestamp(),
                "entities": [],
                "relations": []
            })
            
        return snapshots
    
    def take_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current knowledge state.
        
        Returns:
            Dictionary representing the current knowledge state
        """
        snapshot = {
            "timestamp": datetime.now().timestamp(),
            "entities": [],
            "relations": []
        }
        
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Ensure snapshot table exists
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                ''')
                
                # Get entities
                cursor.execute("SELECT * FROM graph_entities")
                entities = [dict(row) for row in cursor.fetchall()]
                snapshot["entities"] = entities
                
                # Get relations
                cursor.execute('''
                SELECT r.*, e1.entity as source_entity, e2.entity as target_entity, 
                       r.relation_type as relation
                FROM graph_relationships r
                JOIN graph_entities e1 ON r.source_id = e1.id
                JOIN graph_entities e2 ON r.target_id = e2.id
                ''')
                relations = []
                for row in cursor.fetchall():
                    relation = dict(row)
                    # Add formatted relation for easier analysis
                    relation["formatted"] = f"{relation['source_entity']} {relation['relation']} {relation['target_entity']}"
                    relations.append(relation)
                
                snapshot["relations"] = relations
                
                # Store snapshot in database
                cursor.execute(
                    'INSERT INTO knowledge_snapshots (snapshot_data, timestamp) VALUES (?, ?)',
                    (json.dumps(snapshot), snapshot["timestamp"])
                )
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error taking knowledge snapshot: {e}")
            
        return snapshot

    # Implement methods required by KnowledgeStoreInterface
    def remember(self, text: str, source: Optional[str] = None) -> List[int]:
        """
        Store knowledge in the system as required by KnowledgeStoreInterface.
        
        Args:
            text: Text to remember
            source: Optional source information
            
        Returns:
            List of IDs for the stored knowledge
        """
        return self.add_knowledge(text, source)
    
    def retrieve(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to the query as required by KnowledgeStoreInterface.
        
        Args:
            query: Query text
            max_results: Maximum number of results
            
        Returns:
            List of relevant knowledge items
        """
        return self.get_relevant_knowledge(query, max_results)
    
    def clear(self) -> None:
        """Clear all stored knowledge."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete all fact triples
            cursor.execute('DELETE FROM fact_triples')
            
            # Delete all knowledge items
            cursor.execute('DELETE FROM knowledge_items')
            
            # Delete all graph entities and relationships
            cursor.execute('DELETE FROM graph_entities')
            cursor.execute('DELETE FROM graph_relationships')
            
            # Delete all snapshots
            try:
                cursor.execute('DELETE FROM knowledge_snapshots')
            except sqlite3.OperationalError:
                # Table might not exist
                pass
            
            conn.commit()
        
        # Reset BM25 index
        if BM25_ENABLED:
            self.bm25_corpus = []
            self.id_to_doc_mapping = {}
            self.doc_to_id_mapping = {}
            self.bm25_index = None
            self.last_indexed_fact_id = 0
            self.last_indexed_knowledge_id = 0
        
        # Clear embedding cache
        self.embedding_cache = {}
    
    def retrieve_context(self, query: str, max_results: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The query to retrieve context for
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            
        Returns:
            List of relevant context items
        """
        results = []
        
        # Get vector search results
        vector_results = self._vector_search(query, max_results * 2) if VECTOR_ENABLED else []
        
        # Get BM25 search results
        bm25_results = self._bm25_search(query, max_results * 2) if BM25_ENABLED else []
        
        # Get graph search results if enabled
        graph_results = []
        if self.use_graph_rag and self.graph_store:
            graph_results = self._graph_search(query, max_results * 2)
        
        # Combine results from different sources with hybrid ranking
        if vector_results and bm25_results:
            # Map IDs to maintain uniqueness
            combined_results = {}
            
            # Add vector results with their scores
            for item in vector_results:
                item_id = item.get("id", 0)
                if item_id:
                    combined_results[item_id] = {
                        "item": item,
                        "vector_score": item.get("score", 0),
                        "bm25_score": 0
                    }
            
            # Update or add BM25 results
            for item in bm25_results:
                item_id = item.get("id", 0)
                if item_id:
                    if item_id in combined_results:
                        combined_results[item_id]["bm25_score"] = item.get("score", 0)
                    else:
                        combined_results[item_id] = {
                            "item": item,
                            "vector_score": 0,
                            "bm25_score": item.get("score", 0)
                        }
            
            # Calculate hybrid scores
            for item_id, data in combined_results.items():
                # Normalize scores to 0-1 range (they should already be in this range)
                vector_score = data["vector_score"]
                bm25_score = data["bm25_score"]
                
                # Calculate hybrid score
                hybrid_score = self.hybrid_alpha * vector_score + (1 - self.hybrid_alpha) * bm25_score
                
                # Update the item with hybrid score
                data["item"]["score"] = hybrid_score
                results.append(data["item"])
                
            # Sort by hybrid score
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        elif vector_results:
            results = vector_results
        elif bm25_results:
            results = bm25_results
        
        # Add graph results if available, weighing them appropriately
        if graph_results:
            existing_ids = {r.get("id", 0) for r in results if "id" in r}
            
            for graph_item in graph_results:
                graph_id = graph_item.get("id", 0)
                
                # Only add if not already in results
                if graph_id and graph_id not in existing_ids:
                    # Adjust score by graph weight
                    graph_item["score"] = graph_item.get("score", 0.5) * self.graph_weight
                    results.append(graph_item)
            
            # Resort results
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        # If ontology is enabled, enhance results with subclass/superclass information
        if self.use_ontology and self.ontology:
            try:
                # Extract entity mentions from the query
                entities = []
                if self.graph_store:
                    extracted_entities = self.graph_store.extract_entities(query)
                    entities = [entity["text"] for entity in extracted_entities]
                
                # For each entity, check if it belongs to a class in the ontology
                enhanced_results = []
                for entity in entities:
                    # Get entity type from graph store
                    entity_type = None
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT entity_type FROM graph_entities WHERE entity = ?", (entity,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        entity_type = row[0]
                    
                    if entity_type and self.ontology.get_class(entity_type):
                        # Get superclasses
                        for parent_class in self.ontology.get_class(entity_type).parent_classes:
                            if parent_class in self.ontology.classes:
                                # Find entities of the parent class
                                parent_entities = self.graph_store.query_entities(entity_type=parent_class, limit=3)
                                for parent_entity in parent_entities:
                                    enhanced_results.append({
                                        "text": f"{entity} is a {entity_type}, which is a type of {parent_class}. {parent_entity['entity']} is also a {parent_class}.",
                                        "score": 0.75,
                                        "confidence": 0.9,
                                        "source": "ontology_reasoning",
                                        "type": "class_hierarchy"
                                    })
                
                # Add the enhanced results
                results.extend(enhanced_results)
                
                # Resort results
                results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logging.error(f"Error applying ontology reasoning: {e}")
        
        # If inference engine is enabled, enhance results with inferences
        if self.use_inference_engine and self.inference_engine:
            try:
                # Check if this is a "why" question for backward chaining
                if query.lower().strip().startswith("why "):
                    explanations = self.inference_engine.answer_why_question(query)
                    
                    if explanations and not any("error" in exp for exp in explanations):
                        for explanation in explanations:
                            explanation["score"] = 0.95  # High score for logical explanations
                            explanation["source"] = "inference_engine"
                            
                        # Add explanations to results
                        results.extend(explanations)
                        
                        # Resort results to prioritize explanations
                        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
                else:
                    # Try to infer new facts through forward chaining
                    # Only do this occasionally to avoid performance impact
                    should_run_forward_chain = (hash(query) % 5 == 0)  # ~20% of queries
                    
                    if should_run_forward_chain:
                        inferred_facts = self.inference_engine.forward_chain(
                            iterations=self.config.max_forward_chain_iterations
                        )
                        
                        # Convert inferred facts to result format
                        for fact in inferred_facts:
                            # Create a natural language representation
                            fact_text = f"{fact.get('source', '')} {fact.get('relation', '')} {fact.get('target', '')}"
                            
                            results.append({
                                "text": fact_text,
                                "score": 0.85,  # High but lower than direct answers
                                "confidence": fact.get("confidence", 0.8),
                                "source": "inference_engine",
                                "type": "inferred_fact",
                                "rule": fact.get("rule")
                            })
                        
                        # Resort results
                        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logging.error(f"Error applying inference engine: {e}")
        
        # Filter by minimum score
        results = [r for r in results if r.get("score", 0) >= min_score]
        
        # Deduplicate and limit results
        unique_texts = set()
        deduplicated_results = []
        
        for result in results:
            text = result.get("text", "")
            if text and text not in unique_texts:
                unique_texts.add(text)
                deduplicated_results.append(result)
                
                if len(deduplicated_results) >= max_results:
                    break
        
        # Rerank if enabled and we have enough results
        if self.use_reranking and len(deduplicated_results) > 1 and self.model:
            try:
                reranked_results = self._rerank_results(query, deduplicated_results[:self.rerank_top_k])
                
                # Keep track of which results were reranked for debugging
                for result in reranked_results:
                    result["reranked"] = True
                    
                # Add any results that weren't in the reranking pool
                if len(deduplicated_results) > self.rerank_top_k:
                    for result in deduplicated_results[self.rerank_top_k:]:
                        reranked_results.append(result)
                        
                deduplicated_results = reranked_results[:max_results]
            except Exception as e:
                logging.error(f"Error during reranking: {e}")
        
        # Add metadata about the retrieval process
        for result in deduplicated_results:
            if "provenance" not in result and "source" in result:
                result["provenance"] = result["source"]
                
            # Include timestamp information if available but no explicit timestamp in result
            if "timestamp" not in result and self.track_temporal:
                result["timestamp"] = datetime.now().isoformat()
        
        return deduplicated_results[:max_results]
    
    def generate_hypotheses(self, observation: str, max_hypotheses: int = 3) -> List[Dict[str, Any]]:
        """
        Generate hypotheses to explain an observation using abductive reasoning.
        
        Args:
            observation: The observation to explain
            max_hypotheses: Maximum number of hypotheses to generate
            
        Returns:
            List of hypothesis explanations
        """
        if not self.use_inference_engine or not self.inference_engine:
            logging.warning("Inference engine not available for hypothesis generation")
            return []
            
        if not self.config.abductive_reasoning_enabled:
            logging.info("Abductive reasoning is disabled in configuration")
            return []
            
        try:
            # Extract a fact pattern from the observation
            fact_pattern = None
            
            # Try to extract entities and relations from the graph store
            if self.graph_store:
                entities = self.graph_store.extract_entities(observation)
                if len(entities) >= 1:
                    # Get the first entity as the observation subject
                    subject = entities[0]["text"]
                    
                    # Try to extract relations
                    relations = self.graph_store.extract_relations(observation)
                    if relations:
                        # Use first relation
                        s, p, o = relations[0]
                        fact_pattern = {
                            "source": s,
                            "relation": p,
                            "target": o
                        }
                    else:
                        # Create a simple is_a or has_property fact pattern
                        words = observation.split()
                        if "is" in words or "are" in words:
                            # Find what comes after "is" or "are"
                            for i, word in enumerate(words):
                                if word in ["is", "are"] and i < len(words) - 1:
                                    fact_pattern = {
                                        "source": subject,
                                        "relation": "is_a",
                                        "target": words[i+1]
                                    }
                                    break
                        elif "has" in words or "have" in words:
                            # Find what comes after "has" or "have"
                            for i, word in enumerate(words):
                                if word in ["has", "have"] and i < len(words) - 1:
                                    fact_pattern = {
                                        "source": subject,
                                        "relation": "has_property",
                                        "target": words[i+1]
                                    }
                                    break
            
            # If we couldn't extract a fact pattern, return empty
            if not fact_pattern:
                logging.warning(f"Could not extract a fact pattern from observation: {observation}")
                return []
                
            # Apply abductive reasoning
            hypotheses = self.inference_engine.abductive_reasoning(
                fact_pattern, 
                max_hypotheses=max_hypotheses
            )
            
            # Format the hypotheses
            formatted_hypotheses = []
            for hypothesis in hypotheses:
                hypothesis_fact = hypothesis.get("hypothesis", {})
                
                # Create a natural language representation
                if hypothesis_fact:
                    source = hypothesis_fact.get("source", "")
                    relation = hypothesis_fact.get("relation", "")
                    target = hypothesis_fact.get("target", "")
                    
                    # Format relation for readability
                    if relation == "is_a":
                        relation_text = "is a"
                    elif relation == "has_property":
                        relation_text = "has"
                    else:
                        relation_text = relation.replace("_", " ")
                    
                    hypothesis_text = f"{source} {relation_text} {target}"
                    
                    formatted_hypotheses.append({
                        "text": hypothesis_text,
                        "confidence": hypothesis.get("confidence", 0.5),
                        "is_known": hypothesis.get("is_known", False),
                        "source": "abductive_reasoning",
                        "reasoning_path": f"Rule: {hypothesis.get('rule', 'unknown')}"
                    })
            
            return formatted_hypotheses
            
        except Exception as e:
            logging.error(f"Error generating hypotheses: {e}")
            return [] 