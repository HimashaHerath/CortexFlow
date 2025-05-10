import os
import time
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlitedict import SqliteDict
from datetime import datetime
import logging
from collections import defaultdict

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

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.graph_store import GraphStore

class KnowledgeStore:
    """Persistent storage for important facts and retrievable context."""
    
    def __init__(self, config: AdaptiveContextConfig):
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
        
        # Hybrid search parameters
        self.hybrid_alpha = 0.7  # Weight for dense vector search (1-alpha is weight for sparse BM25)
        
        # Initialize database connection
        self._init_db()
        
        # Initialize vector embedding model
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        if VECTOR_ENABLED:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Get vector model from config
                vector_model = config.vector_model if hasattr(config, 'vector_model') else "all-MiniLM-L6-v2"
                
                self.model = SentenceTransformer(vector_model)
                logging.info("Vector embedding model loaded successfully")
            except ImportError:
                logging.warning("SentenceTransformer not available. Vector search will be disabled.")
            except Exception as e:
                logging.error(f"Error loading vector model: {e}")
                
        # Initialize BM25 index
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        self.bm25_last_update = 0
        
        # Storage for in-memory caches
        self.vector_index = None
        self.vector_ids = []
        self.vector_store = {}  # For summaries
        
        # Create SQLiteDicts for persistence
        try:
            from sqlitedict import SqliteDict
            self.summaries = SqliteDict(self.db_path, tablename='summaries', autocommit=True)
        except ImportError:
            logging.warning("SqliteDict not available. In-memory storage will be used.")
            self.summaries = {}
        except Exception as e:
            logging.error(f"Error initializing summary store: {e}")
            self.summaries = {}
            
        # Initialize the graph store if GraphRAG is enabled
        self.graph_store = None
        if self.use_graph_rag:
            try:
                from .graph_store import GraphStore
                self.graph_store = GraphStore(config)
                logging.info(f"Graph store initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing graph store: {e}")
                self.use_graph_rag = False
                
        # In-memory connection for better performance
        self.conn = None
        try:
            self.conn = sqlite3.connect(":memory:")
            self._copy_db_to_memory()
        except Exception as e:
            logging.error(f"Error creating in-memory database: {e}")
            self.conn = None
            
        logging.info(f"Knowledge store initialized with vector retrieval: {VECTOR_ENABLED}, BM25: {BM25_ENABLED}")
    
    def _init_db(self):
        """Initialize the database for storing structured knowledge."""
        # Initialize connection
        self.conn = None
        
        try:
            # Create tables if they don't exist
            conn = sqlite3.connect(self.db_path)
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
            disk_conn = sqlite3.connect(self.db_path)
            
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
        Generate embedding vector for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if embedding fails
        """
        if not VECTOR_ENABLED or self.model is None:
            return None
            
        try:
            # Generate embedding
            embedding = self.model.encode(text)
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
    
    def _update_bm25_index(self):
        """
        Update BM25 index with all current documents.
        """
        if not BM25_ENABLED:
            return
            
        self.bm25_corpus = []
        self.id_to_doc_mapping = {}
        self.doc_to_id_mapping = {}
        doc_id = 0
        
        # Get all facts from database
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)
            
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, subject, predicate, object FROM fact_triples')
            facts = cursor.fetchall()
            
            for fact in facts:
                # Create document text
                doc_text = f"{fact['subject']} {fact['predicate']} {fact['object']}"
                # Tokenize document
                tokenized_doc = doc_text.lower().split()
                
                self.bm25_corpus.append(tokenized_doc)
                self.id_to_doc_mapping[doc_id] = {
                    'id': fact['id'],
                    'text': doc_text, 
                    'subject': fact['subject'],
                    'predicate': fact['predicate'],
                    'object': fact['object']
                }
                self.doc_to_id_mapping[doc_text] = doc_id
                doc_id += 1
                
            # Also add summaries
            for summary_id, summary_data in self.summaries.items():
                doc_text = summary_data.get('text', '')
                if doc_text:
                    tokenized_doc = doc_text.lower().split()
                    self.bm25_corpus.append(tokenized_doc)
                    self.id_to_doc_mapping[doc_id] = {
                        'id': summary_id,
                        'text': doc_text,
                        'type': 'summary',
                        'keywords': summary_data.get('keywords', [])
                    }
                    self.doc_to_id_mapping[doc_text] = doc_id
                    doc_id += 1
        except sqlite3.OperationalError as e:
            logging.error(f"Error updating BM25 index: {e}")
        finally:
            if self.conn is None:
                conn.close()
        
        # Create BM25 index if we have documents
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            logging.info(f"BM25 index updated with {len(self.bm25_corpus)} documents")
    
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
        if self.conn is not None:
            # Using in-memory database
            cursor = self.conn.cursor()
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Generate embedding for the fact if possible
        fact_text = f"{subject} {predicate} {obj}"
        embedding = self._generate_embedding(fact_text)
        embedding_blob = None
        
        if embedding is not None:
            # Convert embedding to binary blob
            embedding_blob = embedding.tobytes()
            
            # Store in vector store for in-memory search
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
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
        
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
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM fact_triples WHERE subject = ? ORDER BY confidence DESC',
            (subject,)
        )
        
        facts = [dict(row) for row in cursor.fetchall()]
        
        if self.conn is None:
            conn.close()
        
        return facts
    
    def get_facts_by_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Retrieve facts by predicate.
        
        Args:
            predicate: The predicate to query
            
        Returns:
            List of fact dictionaries
        """
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM fact_triples WHERE predicate = ? ORDER BY confidence DESC',
            (predicate,)
        )
        
        facts = [dict(row) for row in cursor.fetchall()]
        
        if self.conn is None:
            conn.close()
        
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
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE fact_triples SET confidence = ? WHERE id = ?',
            (new_confidence, fact_id)
        )
        
        success = cursor.rowcount > 0
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
        
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
        
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM fact_triples WHERE timestamp < ? AND confidence < ?',
            (threshold_time, max_confidence)
        )
        
        removed_count = cursor.rowcount
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
            
        # Update BM25 index after removing facts
        if removed_count > 0:
            self._update_bm25_index()
        
        return removed_count
    
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
        if not BM25_ENABLED or not self.bm25_index:
            return []
            
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Sort document IDs by score
        sorted_indices = np.argsort(bm25_scores)[::-1]
        
        results = []
        for i in sorted_indices[:max_results]:
            if i in self.id_to_doc_mapping and bm25_scores[i] > 0:
                doc = self.id_to_doc_mapping[i]
                if 'type' in doc and doc['type'] == 'summary':
                    results.append({
                        'type': 'summary',
                        'content': f"From previous conversation: {doc['text']}",
                        'keywords': doc.get('keywords', []),
                        'bm25_score': float(bm25_scores[i]),
                        'id': doc['id']
                    })
                else:
                    results.append({
                        'type': 'fact',
                        'content': f"{self.trust_marker} {doc['subject']} {doc['predicate']} {doc['object']}",
                        'bm25_score': float(bm25_scores[i]),
                        'id': doc['id']
                    })
                    
        return results
    
    def _dense_vector_search(self, query_embedding: np.ndarray, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform dense vector search using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
            
        Returns:
            List of results with vector similarity scores
        """
        results = []
        
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get all facts with embeddings
            cursor.execute('SELECT * FROM fact_triples WHERE embedding IS NOT NULL')
            facts_with_embeddings = [dict(row) for row in cursor.fetchall()]
            
            # Calculate similarity and sort
            similarity_scores = []
            for fact in facts_with_embeddings:
                embedding_bytes = fact.get('embedding')
                if embedding_bytes:
                    # Convert binary blob back to numpy array
                    fact_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # Reshape if needed
                    if len(fact_embedding) != self.embedding_dimension:
                        fact_embedding = fact_embedding[:self.embedding_dimension]
                    
                    # Calculate similarity
                    similarity = self._vector_similarity(query_embedding, fact_embedding)
                    similarity_scores.append((fact, similarity))
            
            # Sort by similarity (descending)
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add top results
            for fact, similarity in similarity_scores[:max_results]:
                results.append({
                    'type': 'fact',
                    'content': f"{self.trust_marker} {fact['subject']} {fact['predicate']} {fact['object']}",
                    'confidence': fact['confidence'],
                    'timestamp': fact['timestamp'],
                    'source': fact['source'] if fact['source'] else "memory",
                    'similarity': similarity,
                    'id': fact['id']
                })
                
        except sqlite3.OperationalError as e:
            logging.error(f"SQLite error in vector search: {e}")
            
        if self.conn is None:
            conn.close()
            
        # Also search in summaries
        summary_scores = []
        
        # Check all summaries with vector embeddings
        for summary_id, summary_data in self.summaries.items():
            summary_embedding = None
            
            # Get embedding from vector store or summary data
            if summary_id in self.vector_store and isinstance(self.vector_store[summary_id], np.ndarray):
                summary_embedding = self.vector_store[summary_id]
            elif summary_data.get('embedding') is not None:
                if isinstance(summary_data['embedding'], list):
                    # Convert list to numpy array
                    summary_embedding = np.array(summary_data['embedding'])
            
            # Calculate similarity if we have an embedding
            if summary_embedding is not None:
                similarity = self._vector_similarity(query_embedding, summary_embedding)
                summary_scores.append((summary_id, similarity))
        
        # Sort by similarity
        summary_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add top summaries
        for summary_id, score in summary_scores[:max_results - len(results)]:
            if summary_id in self.summaries:
                summary = self.summaries[summary_id]
                results.append({
                    'type': 'summary',
                    'content': f"From previous conversation: {summary['text']}",
                    'keywords': summary['keywords'],
                    'timestamp': summary['timestamp'],
                    'similarity': score,
                    'id': summary_id
                })
                
        return results
        
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
        # Get sparse BM25 results
        sparse_results = self._bm25_search(query, max_results=self.rerank_top_k)
        
        # Get dense vector results
        dense_results = self._dense_vector_search(query_embedding, max_results=self.rerank_top_k)
        
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
                'combined_score': (1 - self.hybrid_alpha) * normalized_score
            }
            
        # Add dense results, combining with sparse scores when available
        for result in dense_results:
            result_id = result['id']
            if result_id in combined_results:
                # Update existing result
                combined_results[result_id].update({
                    'similarity': result['similarity'],
                    'dense_score': result['similarity'],
                    'combined_score': combined_results[result_id]['combined_score'] + self.hybrid_alpha * result['similarity']
                })
            else:
                # Add new result
                combined_results[result_id] = {
                    **result,
                    'dense_score': result['similarity'],
                    'combined_score': self.hybrid_alpha * result['similarity']
                }
                
        # Convert to list and sort by combined score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results_list[:max_results]
    
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
        if hasattr(self, 'graph_store'):
            return self.graph_store.extract_relations(text)
        
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
        Explicitly remember important facts from text.
        
        Args:
            text: Text to remember
            source: Source of the knowledge
            confidence: Confidence score for the facts
            
        Returns:
            List of IDs for stored facts
        """
        fact_ids = []
        
        logging.debug(f"Remembering text explicitly: '{text[:50]}...'")
        
        # Extract triples from text
        triples = self.extract_facts_from_text(text)
        
        logging.debug(f"Extracted {len(triples)} fact triples")
        
        # Store each triple as a fact
        for subject, predicate, obj in triples:
            fact_id = self.store_fact_triple(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,
                source=source
            )
            if fact_id > 0:
                fact_ids.append(fact_id)
        
        # Process text to build knowledge graph
        if hasattr(self, 'graph_store'):
            # Make sure graph_store is initialized
            if not hasattr(self.graph_store, 'extract_entities'):
                logging.error("GraphStore is not properly initialized")
                return fact_ids
                
            # Process text to extract entities and relations
            relations_added = self.graph_store.process_text_to_graph(text, source=source)
            logging.debug(f"Added {relations_added} relations to knowledge graph")
            
            # Manually add the text as a single fact if no relations were extracted
            if not relations_added and not fact_ids:
                # Try to extract simple subject-verb-object
                words = text.split()
                if len(words) >= 3:
                    subject = words[0]
                    predicate = "states"
                    obj = " ".join(words[1:])
                    
                    # Add to graph
                    success = self.graph_store.add_relation(
                        source_entity=subject,
                        relation_type=predicate,
                        target_entity=obj,
                        metadata={"source": source} if source else None
                    )
                    
                    if success:
                        logging.debug(f"Added backup relation: {subject} {predicate} {obj}")
        else:
            logging.warning("No graph_store available for graph-based storage")
        
        return fact_ids
    
    def _keyword_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Basic keyword search as fallback when vector/BM25 is not available.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of results
        """
        results = []
        
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
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
            expanded_query = self._expand_query(query)
            for term in expanded_query:
                param = f"%{term}%"
                cursor.execute(sql_query, (param, param, param, max_results))
                fact_results = [dict(row) for row in cursor.fetchall()]
                
                # Add facts to results with trust marker
                for fact in fact_results:
                    results.append({
                        'type': 'fact',
                        'content': f"{self.trust_marker} {fact['subject']} {fact['predicate']} {fact['object']}",
                        'confidence': fact['confidence'],
                        'timestamp': fact['timestamp'],
                        'source': fact['source'] if fact['source'] else "memory"
                    })
                
                if len(results) >= max_results:
                    break
        except sqlite3.OperationalError as e:
            logging.error(f"SQLite error in keyword search: {e}")
        
        if self.conn is None:
            conn.close()
            
        # Search summaries if needed
        if len(results) < max_results:
            summary_scores = []
            for summary_id, keywords in self.vector_store.items():
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
                if summary_id in self.summaries:
                    summary = self.summaries[summary_id]
                    results.append({
                        'type': 'summary',
                        'content': f"From previous conversation: {summary['text']}",
                        'keywords': summary['keywords'],
                        'timestamp': summary['timestamp'],
                        'score': score
                    })
                    
        return results
    
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
        if hasattr(self, 'facts'):
            self.facts.close()
        if hasattr(self, 'summaries'):
            self.summaries.close()
        if hasattr(self, 'graph_store'):
            self.graph_store.close()
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close() 