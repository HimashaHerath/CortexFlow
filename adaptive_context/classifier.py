import os
import pickle
import time
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import requests

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.memory import ContextSegment

class RuleBasedClassifier:
    """
    Rule-based classifier for message importance.
    """
    
    def __init__(self):
        """Initialize rule-based classifier."""
        # Patterns that suggest higher importance
        self.important_patterns = [
            r'\b(?:important|critical|urgent|crucial|key|essential|remember|note|significant)\b',
            r'\b(?:deadline|schedule|appointment|meeting|interview)\b',
            r'\b(?:password|credential|account|login|security)\b',
            r'\b(?:address|phone|email|contact)\b',
            r'(?:\d{1,2}[:./-]\d{1,2}(?:[:./-]\d{2,4})?)',  # Date/time patterns
            r'(?:\$\d+(?:\.\d{2})?)',  # Money patterns
            r'(?:https?://\S+)',  # URLs
            r'(?:\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)',  # Email
            r'(?:\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)'  # Phone numbers
        ]
        
        # Patterns that suggest lower importance
        self.unimportant_patterns = [
            r'\b(?:lol|haha|hmm|oh|ah|um|uh)\b',
            r'(?:[\U0001F600-\U0001F64F])',  # Emojis
            r'(?:^(?:ok|okay|sure|yes|no|maybe)$)',  # Single-word responses
        ]
        
        # Keywords that influence importance
        self.importance_keywords = {
            'high': ['urgent', 'important', 'critical', 'emergency', 'deadline', 'required',
                   'necessary', 'essential', 'vital', 'crucial', 'key', 'significant',
                   'remember', 'note', 'attention', 'action', 'decision'],
            'medium': ['meeting', 'project', 'task', 'update', 'information', 'report', 
                     'review', 'consider', 'discuss', 'option', 'alternative', 'suggestion'],
            'low': ['fyi', 'maybe', 'perhaps', 'sometime', 'whenever', 'thought', 
                  'random', 'just', 'btw', 'by the way', 'off topic']
        }
        
        # Message type weights
        self.type_weights = {
            'system': 0.9,
            'user': 0.7,
            'assistant': 0.5,
            'function': 0.6,
            'data': 0.8,
            'summary': 0.7,
            'unknown': 0.5
        }
    
    def classify(self, segment: ContextSegment) -> float:
        """
        Classify importance using rule-based heuristics.
        
        Args:
            segment: Context segment to classify
            
        Returns:
            Importance score (0-1)
        """
        if not segment or not segment.content:
            return 0.0
            
        content = segment.content
        segment_type = segment.segment_type
        metadata = segment.metadata or {}
        
        # Base score from message type
        score = self.type_weights.get(segment_type, 0.5)
        
        # Check for important patterns
        for pattern in self.important_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.1
                
        # Check for unimportant patterns
        for pattern in self.unimportant_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.1
                
        # Check for keywords
        for keyword in self.importance_keywords['high']:
            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                score += 0.15
                
        for keyword in self.importance_keywords['medium']:
            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                score += 0.07
                
        for keyword in self.importance_keywords['low']:
            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                score -= 0.05
        
        # Message length factor (longer messages might be more important)
        length_factor = min(len(content.split()) / 100.0, 0.2)
        score += length_factor
        
        # Content recency
        if hasattr(segment, 'timestamp'):
            recency = (time.time() - segment.timestamp) / 86400.0  # Days
            recency_factor = max(0.0, 0.1 - min(recency, 10) / 100.0)
            score += recency_factor
            
        # Explicit importance from metadata
        if 'importance' in metadata:
            explicit_importance = float(metadata['importance'])
            score = score * 0.6 + explicit_importance * 0.4
            
        # Ensure score is within range
        return max(0.0, min(score, 1.0))


class MLClassifier:
    """
    Machine learning-based classifier for message importance.
    Requires a pre-trained model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML classifier.
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model = None
        self.vectorizer = None
        
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.vectorizer = model_data.get('vectorizer')
                logging.info(f"ML classifier model loaded from {model_path}")
            except Exception as e:
                logging.error(f"Error loading ML model: {e}")
        else:
            logging.warning("ML model path not provided or does not exist; ML classification disabled")
    
    def classify(self, segment: ContextSegment) -> float:
        """
        Classify importance using ML model.
        
        Args:
            segment: Context segment to classify
            
        Returns:
            Importance score (0-1) or 0.5 if model unavailable
        """
        if not self.model or not self.vectorizer:
            return 0.5
            
        try:
            content = segment.content
            # Extract features (text and metadata)
            features = {
                'text': content,
                'segment_type': segment.segment_type,
                'token_count': segment.token_count,
                'metadata': json.dumps(segment.metadata or {})
            }
            
            # Transform with vectorizer
            X = self.vectorizer.transform([features])
            
            # Predict probability
            proba = self.model.predict_proba(X)[0][1]  # Probability of positive class
            
            return float(proba)
        except Exception as e:
            logging.error(f"Error in ML classification: {e}")
            return 0.5


class LLMClassifier:
    """
    LLM-based classifier for message importance.
    Uses external LLM to determine importance.
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize LLM classifier.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.cache = {}  # Simple cache to avoid repeated queries
        
        # Base prompt for importance classification
        self.base_prompt = """
        You are an AI assistant that determines the importance of conversation messages.
        Rate this message's importance on a scale from 0.0 to 1.0, where:
        - 1.0: Critical, essential information that must be remembered
        - 0.7-0.9: Very important information that should be prioritized
        - 0.4-0.6: Moderately important information
        - 0.1-0.3: Background information with limited relevance
        - 0.0: Completely unimportant, could be forgotten
        
        Consider factors like:
        - Information density and uniqueness
        - Presence of facts, data, references
        - Questions or requests requiring follow-up
        - Overall contribution to the conversation
        
        Message: {message}
        
        Rate the importance as a single number between 0.0 and 1.0:
        """
    
    def classify(self, segment: ContextSegment, context: List[ContextSegment] = None) -> float:
        """
        Classify importance using LLM.
        
        Args:
            segment: Context segment to classify
            context: Recent conversation context
            
        Returns:
            Importance score (0-1) or 0.5 if LLM fails
        """
        # Simple feature-based fallback
        if not segment.content:
            return 0.1
        
        content = segment.content
        
        # Check cache first
        cache_key = content[:100]  # Use first 100 chars as key
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Extract context for better classification
        context_text = ""
        if context:
            context_samples = context[-2:]  # Just use the last 2 messages
            context_text = "\n".join([f"[{c.segment_type}] {c.content[:100]}..." for c in context_samples])
        
        # Truncate long messages
        max_length = 500  # Character limit for LLM
        if len(content) > max_length:
            content = content[:max_length] + "..."
            
        # Prepare prompt
        prompt = self.base_prompt.format(message=content)
        if context_text:
            prompt = prompt.replace("Message:", f"Recent context:\n{context_text}\n\nMessage:")
            
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.config.ollama_host}/api/generate",
                json={
                    "model": self.config.default_model,
                    "prompt": prompt,
                    "temperature": 0.1,
                    "stream": False
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                
                # Extract score from response
                score_match = re.search(r'0\.\d+', result)
                if score_match:
                    score = float(score_match.group(0))
                    # Cache the result
                    self.cache[cache_key] = score
                    return score
                    
                # If no decimal found, look for integer
                int_match = re.search(r'\b[0-9]\b', result)
                if int_match:
                    score = float(int_match.group(0)) / 10.0
                    self.cache[cache_key] = score
                    return score
                
        except Exception as e:
            logging.error(f"Error in LLM classification: {e}")
        
        # Fallback importance
        return 0.5


class ImportanceClassifier:
    """
    Classifier that determines the importance of context segments.
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize importance classifier.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        
        # Initialize component classifiers
        self.rule_classifier = RuleBasedClassifier()
        
        # Initialize ML classifier if enabled
        self.ml_classifier = None
        if hasattr(config, 'use_ml') and config.use_ml:
            model_path = config.ml_model_path if hasattr(config, 'ml_model_path') else None
            self.ml_classifier = MLClassifier(model_path)
        
        # Initialize LLM classifier
        self.llm_classifier = LLMClassifier(config)
        
        # Set up weights for ensemble
        self.rule_weight = 0.5  # Default weight for rule-based classifier
        self.ml_weight = 0.3    # Default weight for ML-based classifier
        self.llm_weight = 0.7   # Default weight for LLM-based classifier
        
        # Override with config if available
        if hasattr(config, 'rule_weight'):
            self.rule_weight = config.rule_weight
        if hasattr(config, 'ml_weight'):
            self.ml_weight = config.ml_weight
        if hasattr(config, 'llm_weight'):
            self.llm_weight = config.llm_weight
            
    def classify(self, segment: ContextSegment, context: List[ContextSegment] = None) -> float:
        """
        Classify the importance of a context segment.
        
        Args:
            segment: The segment to classify
            context: Recent conversation context (optional)
            
        Returns:
            Importance score between 0 and 1
        """
        if segment is None:
            return 0.0
        
        # Always use rule-based classification
        rule_score = self.rule_classifier.classify(segment)
        scores = [rule_score]
        weights = [self.rule_weight]
        
        # Use ML classifier if available
        if self.ml_classifier is not None:
            ml_score = self.ml_classifier.classify(segment)
            scores.append(ml_score)
            weights.append(self.ml_weight)
        
        # Use LLM classifier for more complex content
        use_llm = hasattr(self.config, 'use_llm_classification') and self.config.use_llm_classification
        if use_llm:
            max_llm_length = getattr(self.config, 'max_llm_classification_length', 250)
            if len(segment.content) < max_llm_length:
                llm_score = self.llm_classifier.classify(segment, context)
                scores.append(llm_score)
                weights.append(self.llm_weight)
            else:
                # For very long content, adjust rule score slightly higher
                scores[0] = min(rule_score * 1.2, 1.0)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        importance = sum(s * w for s, w in zip(scores, weights))
        
        # Ensure importance is within valid range
        importance = max(0.0, min(importance, 1.0))
        
        # Store importance on segment
        segment.importance = importance
        
        return importance 

class ContentClassifier:
    """
    Classifier for content types in messages.
    Used to identify content categories, questions, commands, etc.
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize the content classifier.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Define classification categories
        self.categories = {
            "question": ["what", "why", "how", "when", "where", "who", "which", "?"],
            "command": ["do", "please", "can you", "could you", "make", "create", "find", "search"],
            "factual": ["is", "are", "was", "were", "fact", "knowledge", "information"],
            "opinion": ["think", "believe", "feel", "opinion", "perspective"],
            "greeting": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "talk later", "thanks"]
        }
        
        # Initialize vector model if ML classification is enabled
        self.model = None
        if hasattr(config, 'use_ml_classifier') and config.use_ml_classifier:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(config.classifier_model)
                logger.info(f"Content classifier model loaded: {config.classifier_model}")
            except ImportError:
                logger.warning("SentenceTransformer not available for ML classification")
            except Exception as e:
                logger.error(f"Error loading classifier model: {e}")
    
    def classify(self, content: str) -> Dict[str, Any]:
        """
        Classify the content of a message.
        
        Args:
            content: Text content to classify
            
        Returns:
            Classification results
        """
        result = {
            "categories": {},
            "primary_category": None,
            "is_question": False,
            "sentiment": "neutral",
            "confidence": 0.0
        }
        
        # Skip classification for very short content
        if len(content) < 3:
            return result
            
        # Normalize content for classification
        normalized = content.lower().strip()
        
        # Simple rule-based classification
        for category, keywords in self.categories.items():
            score = 0.0
            for keyword in keywords:
                if keyword in normalized:
                    score += 0.2
                    
            # Scale to 0-1
            score = min(1.0, score)
            result["categories"][category] = score
        
        # Determine primary category
        if result["categories"]:
            primary = max(result["categories"].items(), key=lambda x: x[1])
            if primary[1] > 0.2:  # Minimum threshold
                result["primary_category"] = primary[0]
                result["confidence"] = primary[1]
        
        # Check if content is a question
        result["is_question"] = normalized.endswith("?") or any(q in normalized for q in ["what", "why", "how", "when", "where", "who", "which"])
        
        # Simple sentiment detection
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "happy", "thanks"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "angry", "upset", "disappointed"]
        
        positive_score = sum(1 for word in positive_words if word in normalized)
        negative_score = sum(1 for word in negative_words if word in normalized)
        
        if positive_score > negative_score:
            result["sentiment"] = "positive"
        elif negative_score > positive_score:
            result["sentiment"] = "negative"
            
        # Use ML classification if available
        if self.model is not None:
            try:
                # For this example, we use predefined categories
                # A real implementation would have a trained classifier
                result["ml_classification"] = {
                    "enabled": True,
                    "model": self.config.classifier_model
                }
            except Exception as e:
                logger.error(f"Error in ML classification: {e}")
                
        return result 