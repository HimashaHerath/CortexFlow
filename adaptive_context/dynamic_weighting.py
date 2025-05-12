"""
Dynamic Memory Tier Weighting for AdaptiveContext.

This module implements adaptive token allocation between memory tiers based on:
1. Query complexity
2. Document type
3. Historical patterns
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque

from adaptive_context.config import AdaptiveContextConfig

logger = logging.getLogger('adaptive_context')

class DynamicWeightingEngine:
    """
    Engine for dynamically weighting memory tier allocations based on various factors.
    
    This class provides mechanisms to:
    1. Analyze query complexity and determine optimal memory allocation
    2. Track document types and adjust weightings accordingly
    3. Monitor usage patterns and adapt over time
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize the dynamic weighting engine.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        
        # Default tier ratios (active:working:archive)
        self.default_ratios = {
            "active": 0.25,    # 25% of total tokens
            "working": 0.35,   # 35% of total tokens
            "archive": 0.40    # 40% of total tokens
        }
        
        # Total token budget across all tiers
        self.total_token_budget = (
            config.active_token_limit + 
            config.working_token_limit + 
            config.archive_token_limit
        )
        
        # Track recent queries for pattern analysis
        self.recent_queries = deque(maxlen=20)  # Store last 20 queries
        self.query_complexity_history = deque(maxlen=50)  # Store complexity scores
        
        # Track document type distributions
        self.document_type_counts = {
            "code": 0,
            "text": 0,
            "data": 0,
            "mixed": 0
        }
        
        # Tier allocation history
        self.allocation_history = []
        
        # Current tier weights (will be dynamically adjusted)
        self.current_tier_weights = self.default_ratios.copy()
        
        # Learning rate for weight adjustments
        self.learning_rate = 0.1
        
        # Initialize tier limits with current config values
        self.current_tier_limits = {
            "active": config.active_token_limit,
            "working": config.working_token_limit,
            "archive": config.archive_token_limit
        }
        
        # Add pattern tracking
        self.pattern_weights = {
            "code_heavy": {"active": 0.35, "working": 0.35, "archive": 0.30},
            "data_heavy": {"active": 0.20, "working": 0.50, "archive": 0.30},
            "qa_conversation": {"active": 0.30, "working": 0.30, "archive": 0.40},
            "complex_reasoning": {"active": 0.40, "working": 0.40, "archive": 0.20}
        }
        
        # Track detected patterns
        self.detected_patterns = {pattern: 0 for pattern in self.pattern_weights}
        self.conversation_type = "general"  # Default pattern
        
        logger.info(f"DynamicWeightingEngine initialized with total token budget: {self.total_token_budget}")
    
    def analyze_query_complexity(self, query: str) -> float:
        """
        Analyze the complexity of a user query.
        
        Args:
            query: The user's query text
            
        Returns:
            Complexity score (0.0-1.0)
        """
        if not query:
            return 0.0
            
        # Store query for pattern analysis
        self.recent_queries.append(query)
        
        # Feature extraction for complexity
        features = {}
        
        # 1. Length-based features
        features["length"] = min(len(query) / 500.0, 1.0)  # Normalize by 500 chars
        features["word_count"] = min(len(query.split()) / 100.0, 1.0)  # Normalize by 100 words
        
        # 2. Question complexity
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        features["is_question"] = 1.0 if "?" in query else 0.0
        features["question_type"] = 0.0
        
        # Assign complexity scores based on question type
        if features["is_question"]:
            lower_query = query.lower()
            if "why" in lower_query:
                features["question_type"] = 0.9  # Why questions are complex
            elif "how" in lower_query:
                features["question_type"] = 0.8  # How questions are relatively complex
            elif "what is the relationship" in lower_query or "connection between" in lower_query:
                features["question_type"] = 0.9  # Relationship questions are complex
            elif "compare" in lower_query or "difference between" in lower_query:
                features["question_type"] = 0.8  # Comparison questions are complex
            elif "define" in lower_query or "what is" in lower_query:
                features["question_type"] = 0.5  # Definition questions are medium complexity
            else:
                features["question_type"] = 0.6  # Other questions are medium complexity
        
        # 3. Count entities, numbers, and domain-specific terms
        features["entity_count"] = min(len(re.findall(r'\b[A-Z][a-z]+\b', query)) / 5.0, 1.0)
        features["number_count"] = min(len(re.findall(r'\b\d+(?:\.\d+)?\b', query)) / 5.0, 1.0)
        
        # 4. Check for code-related queries
        code_indicators = ["code", "function", "class", "method", 
                          "variable", "algorithm", "implement", 
                          "programming", "debug", "error"]
        
        features["code_related"] = 0.0
        for indicator in code_indicators:
            if indicator in query.lower():
                features["code_related"] += 0.2
        features["code_related"] = min(features["code_related"], 1.0)
        
        # 5. Check for multi-part questions
        features["multi_part"] = 0.0
        if query.count("?") > 1:
            features["multi_part"] = min(query.count("?") / 3.0, 1.0)
        
        # Calculate weighted complexity score
        weights = {
            "length": 0.1,
            "word_count": 0.15,
            "is_question": 0.1,
            "question_type": 0.25,
            "entity_count": 0.1,
            "number_count": 0.05,
            "code_related": 0.15,
            "multi_part": 0.1
        }
        
        complexity = sum(features[key] * weights[key] for key in weights)
        
        # Ensure score is between 0 and 1
        complexity = max(0.0, min(complexity, 1.0))
        
        # Add to history
        self.query_complexity_history.append(complexity)
        
        logger.debug(f"Query complexity score: {complexity:.2f} for query: {query[:50]}...")
        return complexity
    
    def analyze_document_type(self, content: str) -> str:
        """
        Analyze the type of content in a document.
        
        Args:
            content: Document content to analyze
            
        Returns:
            Document type ("code", "text", "data", or "mixed")
        """
        if not content:
            return "text"
            
        # Simple heuristics to identify document type
        code_indicators = [
            "def ", "class ", "function", "import ", "from ", "return ", 
            "var ", "const ", "let ", "function(", "{", "}", "=>", "->",
            "public ", "private ", "protected ", "#include", "<div", "<script"
        ]
        
        data_indicators = [
            '":', '": ', '",', '{', '}', '[', ']', 'null', 'true', 'false',
            "<table", "<row", "<column", "dataframe", "df.", "pd.", "np."
        ]
        
        # Count indicators
        code_count = sum(1 for indicator in code_indicators if indicator in content)
        data_count = sum(1 for indicator in data_indicators if indicator in content)
        
        # Additional code detection: indentation patterns and code blocks
        if re.search(r'```python|```java|```js|```c|```cpp|```html|```css', content):
            code_count += 5
            
        # Additional data detection: CSV-like patterns, tabular data
        if re.search(r'\b\w+,\w+,\w+,\w+\b', content) or re.search(r'\|\s*\w+\s*\|\s*\w+\s*\|', content):
            data_count += 3
            
        # Make determination
        if code_count > 5 and data_count > 5:
            doc_type = "mixed"
        elif code_count > 5:
            doc_type = "code"
        elif data_count > 5:
            doc_type = "data"
        else:
            doc_type = "text"
            
        # Update counts
        self.document_type_counts[doc_type] += 1
        
        return doc_type
    
    def calculate_optimal_weights(self, query_complexity: float, document_type: str) -> Dict[str, float]:
        """
        Calculate optimal tier weights based on query complexity and document type.
        
        Args:
            query_complexity: Complexity score (0.0-1.0)
            document_type: Type of document ("code", "text", "data", or "mixed")
            
        Returns:
            Dictionary of tier weights (active, working, archive)
        """
        # Start with default ratios
        weights = self.default_ratios.copy()
        
        # 1. Adjust based on query complexity
        if query_complexity > 0.7:  # High complexity
            # Increase active and working memory for complex queries
            weights["active"] += 0.10
            weights["working"] += 0.05
            weights["archive"] -= 0.15
        elif query_complexity < 0.3:  # Low complexity
            # Decrease active memory for simple queries
            weights["active"] -= 0.05
            weights["archive"] += 0.05
            
        # 2. Adjust based on document type
        if document_type == "code":
            # Code benefits from more active memory
            weights["active"] += 0.05
            weights["working"] += 0.05
            weights["archive"] -= 0.10
        elif document_type == "data":
            # Data benefits from more working memory
            weights["active"] -= 0.05
            weights["working"] += 0.10
            weights["archive"] -= 0.05
        elif document_type == "mixed":
            # Mixed content needs balanced memory
            weights["active"] += 0.05
            weights["working"] += 0.05
            weights["archive"] -= 0.10
            
        # 3. Adjust based on historical patterns
        if len(self.query_complexity_history) > 5:
            avg_complexity = sum(self.query_complexity_history) / len(self.query_complexity_history)
            if avg_complexity > 0.6:  # Sustained high complexity
                weights["active"] += 0.05
                weights["working"] += 0.05
                weights["archive"] -= 0.10
        
        # Ensure weights are positive
        weights = {k: max(0.1, v) for k, v in weights.items()}
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def update_tier_allocations(self) -> Dict[str, int]:
        """
        Update memory tier allocations based on current weights.
        
        Returns:
            Dictionary with updated token limits for each tier
        """
        # Calculate token allocations based on weights
        new_limits = {
            "active": int(self.total_token_budget * self.current_tier_weights["active"]),
            "working": int(self.total_token_budget * self.current_tier_weights["working"]),
            "archive": int(self.total_token_budget * self.current_tier_weights["archive"])
        }
        
        # Ensure minimum sizes for each tier
        min_tier_size = 1000  # Minimum 1000 tokens per tier
        for tier in new_limits:
            new_limits[tier] = max(new_limits[tier], min_tier_size)
            
        # Adjust to fit within total budget if needed
        total_allocated = sum(new_limits.values())
        if total_allocated > self.total_token_budget:
            # Proportionally scale down to fit budget
            scale_factor = self.total_token_budget / total_allocated
            for tier in new_limits:
                new_limits[tier] = int(new_limits[tier] * scale_factor)
                
        # Update current limits
        self.current_tier_limits = new_limits
        
        # Record allocation in history
        self.allocation_history.append({
            "timestamp": time.time(),
            "allocations": new_limits.copy(),
            "weights": self.current_tier_weights.copy()
        })
        
        # Trim history if it gets too long
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]
            
        logger.info(f"Updated tier allocations: Active={new_limits['active']}, "
                   f"Working={new_limits['working']}, Archive={new_limits['archive']}")
        
        return new_limits
    
    def update_weights_from_history(self):
        """Analyze historical patterns and adjust weights accordingly."""
        if len(self.query_complexity_history) < 10:
            return  # Need more history to detect patterns
            
        # Analyze document type distribution
        doc_types = self.document_type_counts
        total_docs = sum(doc_types.values())
        
        if total_docs == 0:
            return
            
        # Calculate document type ratios
        code_ratio = doc_types.get("code", 0) / total_docs if total_docs > 0 else 0
        data_ratio = doc_types.get("data", 0) / total_docs if total_docs > 0 else 0
        
        # Calculate average complexity
        avg_complexity = sum(self.query_complexity_history) / len(self.query_complexity_history)
        
        # Detect conversation patterns
        if code_ratio > 0.3:
            # Code-heavy conversation
            self.detected_patterns["code_heavy"] += 1
            self.conversation_type = "code_heavy"
        elif data_ratio > 0.3:
            # Data-heavy conversation
            self.detected_patterns["data_heavy"] += 1
            self.conversation_type = "data_heavy"
        elif avg_complexity > 0.7:
            # Complex reasoning conversation
            self.detected_patterns["complex_reasoning"] += 1
            self.conversation_type = "complex_reasoning"
        elif avg_complexity < 0.4:
            # Simple Q&A conversation
            self.detected_patterns["qa_conversation"] += 1
            self.conversation_type = "qa_conversation"
        
        # Apply detected pattern weights with learning rate
        pattern_weights = self.pattern_weights.get(self.conversation_type)
        if pattern_weights:
            for tier in self.current_tier_weights:
                # Blend current weights with pattern weights using learning rate
                target = pattern_weights[tier]
                current = self.current_tier_weights[tier]
                self.current_tier_weights[tier] += self.learning_rate * 0.5 * (target - current)
                
        # Log the detected pattern
        logger.info(f"Detected conversation pattern: {self.conversation_type}")
        logger.info(f"Updated weights based on pattern: {self.current_tier_weights}")

    def process_query(self, query: str, context_content: str = None) -> Dict[str, int]:
        """
        Process a user query and update memory tier allocations.
        
        Args:
            query: The user's query
            context_content: Optional context content to analyze
            
        Returns:
            Dictionary with updated token limits for each tier
        """
        # Analyze query complexity
        complexity = self.analyze_query_complexity(query)
        
        # Analyze document type if context provided
        doc_type = "text"  # Default
        if context_content:
            doc_type = self.analyze_document_type(context_content)
            
        # Calculate optimal weights
        optimal_weights = self.calculate_optimal_weights(complexity, doc_type)
        
        # Blend current weights with optimal weights using learning rate
        for tier in self.current_tier_weights:
            current = self.current_tier_weights[tier]
            optimal = optimal_weights[tier]
            self.current_tier_weights[tier] += self.learning_rate * (optimal - current)
            
        # After multiple queries, analyze patterns
        if len(self.query_complexity_history) % 5 == 0 and len(self.query_complexity_history) >= 10:
            self.update_weights_from_history()
            
        # Update tier allocations based on current weights
        return self.update_tier_allocations()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dynamic weighting engine.
        
        Returns:
            Dictionary with statistics and current state
        """
        return {
            "current_weights": self.current_tier_weights,
            "current_limits": self.current_tier_limits,
            "document_type_distribution": self.document_type_counts,
            "recent_query_complexity": list(self.query_complexity_history)[-5:] if self.query_complexity_history else [],
            "total_token_budget": self.total_token_budget,
            "allocation_history_count": len(self.allocation_history)
        }
    
    def reset_to_defaults(self):
        """Reset weighting to default values."""
        self.current_tier_weights = self.default_ratios.copy()
        self.update_tier_allocations()
        logger.info("Reset tier weights to default values") 