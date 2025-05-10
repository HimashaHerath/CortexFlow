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

class KnowledgeStore:
    """Persistent storage for important facts and retrievable context."""
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize knowledge store.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.db_path = config.knowledge_store_path
        
        # For in-memory databases, we need to maintain a persistent connection
        self.conn = None
        if self.db_path == ':memory:':
            self.conn = sqlite3.connect(self.db_path)
        
        # Initialize SQLite storage
        self.facts = SqliteDict(self.db_path, tablename='facts', autocommit=True)
        self.summaries = SqliteDict(self.db_path, tablename='summaries', autocommit=True)
        
        # Initialize embedding model if available
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        if VECTOR_ENABLED:
            try:
                # Use a small efficient model by default
                self.model = SentenceTransformer(config.vector_embedding_model)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logging.info("Vector embedding model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading vector model: {e}")
        
        # Vector store for embeddings
        self.vector_store = {}
        
        # BM25 index
        self.bm25_corpus = []  # List of tokenized documents for BM25
        self.bm25_index = None  # BM25 index
        self.id_to_doc_mapping = {}  # Map document IDs to their content
        self.doc_to_id_mapping = {}  # Map document content to their IDs
        
        # Initialize SQLite for structured data
        self._init_db()
        
        # Trust marker for retrieved knowledge
        self.trust_marker = "[VERIFIED FACT]"
        
        # Hybrid search parameters
        self.hybrid_alpha = 0.7  # Weight for dense vector search (1-alpha is weight for sparse BM25)
        
        # Re-ranking parameters 
        self.use_reranking = config.use_reranking if hasattr(config, 'use_reranking') else True
        self.rerank_top_k = 20  # Number of initial candidates to re-rank
        
        logging.info(f"Knowledge store initialized with vector retrieval: {VECTOR_ENABLED}, BM25: {BM25_ENABLED}")
    
    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        if self.conn is not None:
            # We're using an in-memory database
            cursor = self.conn.cursor()
        else:
            # Using a file-based database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Create facts table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL,
            timestamp REAL,
            source TEXT
        )
        ''')
        
        # Add embedding columns if vector-based retrieval is enabled
        if VECTOR_ENABLED:
            try:
                cursor.execute('ALTER TABLE fact_triples ADD COLUMN embedding BLOB')
            except sqlite3.OperationalError:
                # Column may already exist
                pass
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON fact_triples(subject)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicate ON fact_triples(predicate)')
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
    
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
        Re-rank search results using cross-encoder if available.
        
        Args:
            query: Original query
            results: List of initial search results
            max_results: Maximum results to return after re-ranking
            
        Returns:
            Re-ranked results
        """
        # If there are no results or only a few, skip re-ranking
        if not results or len(results) <= 1 or not VECTOR_ENABLED:
            return results[:max_results]
            
        try:
            # Use sentence-transformers for re-ranking
            pairs = []
            for result in results:
                # Remove the trust marker for better matching
                content = result['content']
                if content.startswith(self.trust_marker):
                    content = content[len(self.trust_marker):].strip()
                pairs.append((query, content))
                
            # Calculate cross-encoder scores using cosine similarity of query and content embeddings
            # In a production system, you would use a proper cross-encoder model here
            query_embedding = self._generate_embedding(query)
            rerank_scores = []
            
            for i, (_, content) in enumerate(pairs):
                content_embedding = self._generate_embedding(content)
                if query_embedding is not None and content_embedding is not None:
                    # Calculate cosine similarity
                    sim_score = self._vector_similarity(query_embedding, content_embedding)
                    rerank_scores.append((i, sim_score))
                else:
                    # Fallback if embedding fails
                    rerank_scores.append((i, 0.0))
                    
            # Sort by re-ranking scores
            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Re-rank results
            reranked_results = []
            for i, score in rerank_scores[:max_results]:
                if i < len(results):
                    result = results[i].copy()
                    result['rerank_score'] = score
                    reranked_results.append(result)
                    
            return reranked_results
            
        except Exception as e:
            logging.error(f"Error in re-ranking: {e}")
            # Fall back to original ranking
            return results[:max_results]
        
    def get_relevant_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to the query using hybrid search and re-ranking.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        # Initialize BM25 index if not already done
        if BM25_ENABLED and (self.bm25_index is None):
            self._update_bm25_index()
            
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        # Strategy selection based on available components
        if VECTOR_ENABLED and query_embedding is not None and BM25_ENABLED and self.bm25_index is not None:
            # Use hybrid search (dense + sparse)
            logging.info("Using hybrid search (dense vectors + BM25)")
            results = self._hybrid_search(query, query_embedding, max_results=self.rerank_top_k)
        elif VECTOR_ENABLED and query_embedding is not None:
            # Use dense vector search
            logging.info("Using dense vector search")
            results = self._dense_vector_search(query_embedding, max_results=self.rerank_top_k)
        elif BM25_ENABLED and self.bm25_index is not None:
            # Use BM25 search
            logging.info("Using BM25 sparse search")
            results = self._bm25_search(query, max_results=self.rerank_top_k)
        else:
            # Fall back to keyword search
            logging.info("Using basic keyword search")
            results = self._keyword_search(query, max_results=self.rerank_top_k)
            
        # Apply re-ranking if enabled and there are enough results
        if self.use_reranking and len(results) > 1:
            logging.info("Applying re-ranking to search results")
            results = self._rerank_results(query, results, max_results=max_results)
        else:
            # Just take top results without re-ranking
            results = results[:max_results]
            
        return results
        
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

    def extract_facts_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract potential fact triples from text.
        
        Args:
            text: Text to extract facts from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        # In a real implementation, this would use NLP techniques
        # For now, we'll use some simple pattern matching
        
        facts = []
        
        # Look for simple patterns like "X is Y"
        is_pattern = r'([\w\s]+) is ([\w\s]+)'
        import re
        is_matches = re.findall(is_pattern, text)
        for subject, obj in is_matches:
            subject = subject.strip()
            obj = obj.strip()
            if subject and obj:
                facts.append((subject, "is", obj))
        
        # Look for "X has Y"
        has_pattern = r'([\w\s]+) has ([\w\s]+)'
        has_matches = re.findall(has_pattern, text)
        for subject, obj in has_matches:
            subject = subject.strip()
            obj = obj.strip()
            if subject and obj:
                facts.append((subject, "has", obj))
        
        return facts
    
    def remember_explicit(self, text: str, source: str = "user_command", confidence: float = 0.95) -> List[int]:
        """
        Process an explicit remember command from the user.
        
        Args:
            text: Text to remember
            source: Source identifier
            confidence: Confidence score (default higher for explicit memory)
            
        Returns:
            List of fact IDs that were stored
        """
        # Extract facts
        facts = self.extract_facts_from_text(text)
        
        # Store with high confidence
        fact_ids = []
        for subject, predicate, obj in facts:
            fact_id = self.store_fact_triple(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,  # Higher confidence for explicit memory
                source=source
            )
            fact_ids.append(fact_id)
        
        # If no structured facts were found, store as a general note
        if not facts:
            fact_id = self.store_fact_triple(
                subject="note",
                predicate="contains",
                obj=text,
                confidence=confidence,
                source=source
            )
            fact_ids.append(fact_id)
        
        return fact_ids
    
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
        """Close all database connections."""
        try:
            self.facts.close()
            self.summaries.close()
            if self.conn is not None:
                self.conn.close()
        except Exception as e:
            logging.error(f"Error closing knowledge store: {e}")
    
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            self.close()
        except:
            pass 