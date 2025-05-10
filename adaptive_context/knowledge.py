import os
import time
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlitedict import SqliteDict
from datetime import datetime

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
        
        # Simple in-memory vector store
        # In a production system, this would use FAISS or similar
        self.vector_store = {}
        
        # Initialize SQLite for structured data
        self._init_db()
        
        # Trust marker for retrieved knowledge
        self.trust_marker = "[VERIFIED FACT]"
    
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
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON fact_triples(subject)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicate ON fact_triples(predicate)')
        
        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()
    
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
        
        # Store the summary with metadata
        self.summaries[summary_id] = {
            'text': summary,
            'keywords': keywords,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }
        
        # Add to vector store for retrieval
        # In a real system, this would generate embeddings using a model
        # For simplicity, we'll use keyword matching
        self.vector_store[summary_id] = ' '.join(keywords)
        
        return summary_id
    
    def get_relevant_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to the query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        results = []
        
        # First check fact triples
        query_terms = query.lower().split()
        
        # Search for facts containing query terms
        if self.conn is not None:
            # Using in-memory database
            conn = self.conn
        else:
            # Using file-based database
            conn = sqlite3.connect(self.db_path)
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
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
            # Table might not exist yet or other SQLite error
            print(f"SQLite error in get_relevant_knowledge: {e}")
        
        if self.conn is None:
            conn.close()
        
        # Search conversation summaries
        # In a real system, this would use vector similarity
        if len(results) < max_results:
            summary_scores = []
            for summary_id, keywords in self.vector_store.items():
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
        
        # Sort all results by relevance (confidence or score)
        results.sort(key=lambda x: x.get('confidence', 0) if x['type'] == 'fact' else x.get('score', 0), 
                     reverse=True)
        
        return results[:max_results]
    
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
            print(f"Error closing knowledge store: {e}")
    
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            self.close()
        except:
            pass 