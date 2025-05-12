"""
Reasoning Path Logger for CortexFlow.

This module provides functionality to log and analyze reasoning paths
in the CortexFlow knowledge graph system.
"""

import logging
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import uuid
import traceback

class ReasoningLogger:
    """
    Logger for recording and analyzing reasoning paths in CortexFlow.
    
    This class provides functionality to:
    1. Record detailed reasoning steps during inference
    2. Store path information for later analysis
    3. Visualize reasoning paths
    4. Export reasoning data for evaluation
    """
    
    def __init__(
        self,
        db_path: str = None,
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        log_dir: str = "logs",
        max_path_length: int = 1000
    ):
        """
        Initialize the reasoning logger.
        
        Args:
            db_path: Path to the SQLite database for storing reasoning paths
            log_level: Logging level
            enable_file_logging: Whether to enable logging to files
            log_dir: Directory for log files
            max_path_length: Maximum path length to store
        """
        self.db_path = db_path
        self.max_path_length = max_path_length
        
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger = logging.getLogger("cortexflow.reasoning")
        self.logger.setLevel(numeric_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if enabled
        if enable_file_logging:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Create database for storing reasoning paths
        if db_path:
            self._init_db()
        
        # Current reasoning path context
        self.current_path = None
        self.current_reasoning_id = None
        
        self.logger.info("Reasoning logger initialized")
    
    def _init_db(self):
        """Initialize database tables for reasoning path logging."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create reasoning_sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_sessions (
                id TEXT PRIMARY KEY,
                query TEXT,
                start_time REAL,
                end_time REAL,
                success BOOLEAN,
                metadata TEXT
            )
            ''')
            
            # Create reasoning_steps table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_number INTEGER,
                step_type TEXT,
                description TEXT,
                entities TEXT,
                relations TEXT,
                confidence REAL,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES reasoning_sessions (id)
            )
            ''')
            
            # Create reasoning_paths table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                path TEXT,
                score REAL,
                hop_count INTEGER,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES reasoning_sessions (id)
            )
            ''')
            
            # Create indices for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reasoning_session_id ON reasoning_sessions(id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reasoning_step_session ON reasoning_steps(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reasoning_path_session ON reasoning_paths(session_id)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.db_path = None
    
    def start_reasoning(self, query: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new reasoning session.
        
        Args:
            query: The query that initiated the reasoning
            metadata: Optional metadata about the reasoning session
            
        Returns:
            Reasoning session ID
        """
        # Generate a unique ID for this reasoning session
        reasoning_id = str(uuid.uuid4())
        self.current_reasoning_id = reasoning_id
        self.current_path = []
        
        # Log the start of reasoning
        self.logger.info(f"Starting reasoning session {reasoning_id} for query: {query}")
        
        # Store in database if enabled
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Store session info
                cursor.execute(
                    '''
                    INSERT INTO reasoning_sessions (id, query, start_time, metadata)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (reasoning_id, query, datetime.now().timestamp(), json.dumps(metadata or {}))
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to record reasoning session: {e}")
        
        return reasoning_id
    
    def log_reasoning_step(
        self, 
        step_type: str, 
        description: str,
        entities: List[str] = None,
        relations: List[str] = None,
        confidence: float = None
    ):
        """
        Log a reasoning step.
        
        Args:
            step_type: Type of reasoning step (e.g., "entity_extraction", "relation_inference")
            description: Description of the reasoning step
            entities: List of entities involved in this step
            relations: List of relations involved in this step
            confidence: Confidence score for this step
        """
        if not self.current_reasoning_id:
            self.logger.warning("No active reasoning session. Call start_reasoning() first.")
            return
        
        # Log the reasoning step
        if entities or relations:
            entity_str = f", entities: {entities}" if entities else ""
            relation_str = f", relations: {relations}" if relations else ""
            conf_str = f", confidence: {confidence:.2f}" if confidence is not None else ""
            self.logger.info(f"Reasoning step [{step_type}]: {description}{entity_str}{relation_str}{conf_str}")
        else:
            self.logger.info(f"Reasoning step [{step_type}]: {description}")
        
        # Add to current path
        if len(self.current_path) < self.max_path_length:
            step = {
                "step_type": step_type,
                "description": description,
                "entities": entities or [],
                "relations": relations or [],
                "confidence": confidence,
                "timestamp": datetime.now().timestamp()
            }
            self.current_path.append(step)
        
        # Store in database if enabled
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Store step info
                cursor.execute(
                    '''
                    INSERT INTO reasoning_steps 
                    (session_id, step_number, step_type, description, entities, relations, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        self.current_reasoning_id, 
                        len(self.current_path), 
                        step_type, 
                        description,
                        json.dumps(entities or []),
                        json.dumps(relations or []),
                        confidence,
                        datetime.now().timestamp()
                    )
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to record reasoning step: {e}")
    
    def log_path(self, path: List[str], score: float = None, hop_count: int = None):
        """
        Log a reasoning path.
        
        Args:
            path: The reasoning path as a list of nodes
            score: Optional score for this path
            hop_count: Number of hops in this path
        """
        if not self.current_reasoning_id:
            self.logger.warning("No active reasoning session. Call start_reasoning() first.")
            return
        
        # Calculate hop count if not provided
        if hop_count is None and path:
            hop_count = len(path) - 1
        
        # Log the path
        path_str = " → ".join(path) if path else "Empty path"
        score_str = f", score: {score:.2f}" if score is not None else ""
        self.logger.info(f"Reasoning path: {path_str}{score_str}")
        
        # Store in database if enabled
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Store path info
                cursor.execute(
                    '''
                    INSERT INTO reasoning_paths 
                    (session_id, path, score, hop_count, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        self.current_reasoning_id, 
                        json.dumps(path),
                        score,
                        hop_count,
                        datetime.now().timestamp()
                    )
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to record reasoning path: {e}")
    
    def end_reasoning(self, success: bool = True, metadata: Dict[str, Any] = None):
        """
        End the current reasoning session.
        
        Args:
            success: Whether the reasoning was successful
            metadata: Optional metadata about the reasoning results
        """
        if not self.current_reasoning_id:
            self.logger.warning("No active reasoning session to end.")
            return
        
        # Log the end of reasoning
        self.logger.info(f"Ending reasoning session {self.current_reasoning_id}, success: {success}")
        
        # Store in database if enabled
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Update session info
                cursor.execute(
                    '''
                    UPDATE reasoning_sessions
                    SET end_time = ?, success = ?, metadata = ?
                    WHERE id = ?
                    ''',
                    (
                        datetime.now().timestamp(),
                        success,
                        json.dumps(metadata or {}),
                        self.current_reasoning_id
                    )
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to update reasoning session: {e}")
        
        # Reset current path context
        self.current_path = None
        self.current_reasoning_id = None
    
    def get_reasoning_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a reasoning session.
        
        Args:
            session_id: ID of the reasoning session
            
        Returns:
            Dictionary with session information
        """
        if not self.db_path:
            self.logger.warning("Database logging not enabled.")
            return {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute(
                '''
                SELECT * FROM reasoning_sessions
                WHERE id = ?
                ''',
                (session_id,)
            )
            
            session = cursor.fetchone()
            
            if not session:
                return {}
            
            # Convert to dictionary
            session_dict = dict(session)
            
            # Get steps
            cursor.execute(
                '''
                SELECT * FROM reasoning_steps
                WHERE session_id = ?
                ORDER BY step_number
                ''',
                (session_id,)
            )
            
            steps = cursor.fetchall()
            session_dict["steps"] = [dict(step) for step in steps]
            
            # Get paths
            cursor.execute(
                '''
                SELECT * FROM reasoning_paths
                WHERE session_id = ?
                ORDER BY timestamp
                ''',
                (session_id,)
            )
            
            paths = cursor.fetchall()
            session_dict["paths"] = [dict(path) for path in paths]
            
            # Parse JSON fields
            if "metadata" in session_dict and session_dict["metadata"]:
                session_dict["metadata"] = json.loads(session_dict["metadata"])
            
            for step in session_dict["steps"]:
                if "entities" in step and step["entities"]:
                    step["entities"] = json.loads(step["entities"])
                if "relations" in step and step["relations"]:
                    step["relations"] = json.loads(step["relations"])
            
            for path in session_dict["paths"]:
                if "path" in path and path["path"]:
                    path["path"] = json.loads(path["path"])
            
            conn.close()
            
            return session_dict
            
        except Exception as e:
            self.logger.error(f"Failed to get reasoning session: {e}")
            return {}
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reasoning sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        if not self.db_path:
            self.logger.warning("Database logging not enabled.")
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent sessions
            cursor.execute(
                '''
                SELECT id, query, start_time, end_time, success
                FROM reasoning_sessions
                ORDER BY start_time DESC
                LIMIT ?
                ''',
                (limit,)
            )
            
            sessions = cursor.fetchall()
            sessions_list = [dict(session) for session in sessions]
            
            # Get step counts for each session
            for session in sessions_list:
                cursor.execute(
                    '''
                    SELECT COUNT(*) as step_count
                    FROM reasoning_steps
                    WHERE session_id = ?
                    ''',
                    (session["id"],)
                )
                
                step_count = cursor.fetchone()
                session["step_count"] = step_count["step_count"] if step_count else 0
                
                # Get path counts
                cursor.execute(
                    '''
                    SELECT COUNT(*) as path_count
                    FROM reasoning_paths
                    WHERE session_id = ?
                    ''',
                    (session["id"],)
                )
                
                path_count = cursor.fetchone()
                session["path_count"] = path_count["path_count"] if path_count else 0
            
            conn.close()
            
            return sessions_list
            
        except Exception as e:
            self.logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    def export_session_data(self, session_id: str, file_path: str) -> bool:
        """
        Export a reasoning session to a JSON file.
        
        Args:
            session_id: ID of the reasoning session
            file_path: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_reasoning_session(session_id)
        
        if not session:
            self.logger.warning(f"Session {session_id} not found.")
            return False
        
        try:
            with open(file_path, 'w') as f:
                json.dump(session, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export session data: {e}")
            return False
    
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a reasoning session for insights.
        
        Args:
            session_id: ID of the reasoning session
            
        Returns:
            Dictionary with analysis results
        """
        session = self.get_reasoning_session(session_id)
        
        if not session:
            self.logger.warning(f"Session {session_id} not found.")
            return {}
        
        # Initialize analysis
        analysis = {
            "query": session.get("query", ""),
            "duration": 0,
            "step_count": len(session.get("steps", [])),
            "path_count": len(session.get("paths", [])),
            "success": session.get("success", False),
            "max_path_length": 0,
            "avg_path_length": 0,
            "max_confidence": 0,
            "min_confidence": 1,
            "avg_confidence": 0,
            "entity_frequencies": {},
            "relation_frequencies": {},
            "step_type_distribution": {}
        }
        
        # Calculate duration
        if "start_time" in session and "end_time" in session:
            analysis["duration"] = session["end_time"] - session["start_time"]
        
        # Analyze paths
        if "paths" in session:
            path_lengths = [len(p.get("path", [])) for p in session["paths"]]
            if path_lengths:
                analysis["max_path_length"] = max(path_lengths) if path_lengths else 0
                analysis["avg_path_length"] = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        
        # Analyze steps
        if "steps" in session:
            confidences = [s.get("confidence") for s in session["steps"] if s.get("confidence") is not None]
            if confidences:
                analysis["max_confidence"] = max(confidences)
                analysis["min_confidence"] = min(confidences)
                analysis["avg_confidence"] = sum(confidences) / len(confidences)
            
            # Analyze entity and relation frequencies
            for step in session["steps"]:
                # Count entities
                for entity in step.get("entities", []):
                    if entity not in analysis["entity_frequencies"]:
                        analysis["entity_frequencies"][entity] = 0
                    analysis["entity_frequencies"][entity] += 1
                
                # Count relations
                for relation in step.get("relations", []):
                    if relation not in analysis["relation_frequencies"]:
                        analysis["relation_frequencies"][relation] = 0
                    analysis["relation_frequencies"][relation] += 1
                
                # Count step types
                step_type = step.get("step_type", "unknown")
                if step_type not in analysis["step_type_distribution"]:
                    analysis["step_type_distribution"][step_type] = 0
                analysis["step_type_distribution"][step_type] += 1
        
        # Sort frequency dictionaries
        analysis["entity_frequencies"] = dict(sorted(
            analysis["entity_frequencies"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        analysis["relation_frequencies"] = dict(sorted(
            analysis["relation_frequencies"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return analysis
    
    def clear_session(self, session_id: str) -> bool:
        """
        Delete a reasoning session from the database.
        
        Args:
            session_id: ID of the reasoning session
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_path:
            self.logger.warning("Database logging not enabled.")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete steps
            cursor.execute(
                "DELETE FROM reasoning_steps WHERE session_id = ?",
                (session_id,)
            )
            
            # Delete paths
            cursor.execute(
                "DELETE FROM reasoning_paths WHERE session_id = ?",
                (session_id,)
            )
            
            # Delete session
            cursor.execute(
                "DELETE FROM reasoning_sessions WHERE id = ?",
                (session_id,)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Deleted reasoning session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete reasoning session: {e}")
            return False

    def create_context_manager(self, query: str, metadata: Dict[str, Any] = None):
        """
        Create a context manager for automatic reasoning session management.
        
        Args:
            query: The query that initiated the reasoning
            metadata: Optional metadata about the reasoning session
            
        Returns:
            Context manager for the reasoning session
        """
        return ReasoningContext(self, query, metadata)

class ReasoningContext:
    """Context manager for automatic reasoning session management."""
    
    def __init__(self, logger: ReasoningLogger, query: str, metadata: Dict[str, Any] = None):
        """
        Initialize the reasoning context.
        
        Args:
            logger: The reasoning logger
            query: The query that initiated the reasoning
            metadata: Optional metadata about the reasoning session
        """
        self.logger = logger
        self.query = query
        self.metadata = metadata
        self.session_id = None
        self.success = True
    
    def __enter__(self):
        """Start the reasoning session when entering the context."""
        self.session_id = self.logger.start_reasoning(self.query, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the reasoning session when exiting the context."""
        if exc_type is not None:
            # An exception occurred
            self.success = False
            
            # Log the exception as a reasoning step
            error_msg = traceback.format_exception_only(exc_type, exc_val)[0].strip()
            self.logger.log_reasoning_step("error", f"Exception occurred: {error_msg}")
        
        # End reasoning session
        result_metadata = self.metadata.copy() if self.metadata else {}
        result_metadata["exception"] = str(exc_val) if exc_val else None
        
        self.logger.end_reasoning(self.success, result_metadata)
        
        # Don't suppress the exception
        return False 