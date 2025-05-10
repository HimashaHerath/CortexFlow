import re
import time
from typing import List, Dict, Any, Optional, Tuple
import json
import requests

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.memory import ContextSegment

class RuleBasedClassifier:
    """Simple rule-based importance classifier."""
    
    def __init__(self):
        # Patterns indicating important information
        self.important_patterns = [
            r"\b(remember|note|important|crucial|key|essential|critical|significant)\b",
            r"\bmy name is\b",
            r"\b(address|phone|email) is\b",
            r"\b(don't|do not|never)\b",
            r"\b(always|must|should|need to)\b",
            r"\?{1,}$",  # Questions
            r"^\d+\.",   # Numbered lists
            r"```[\s\S]*```",  # Code blocks
            r"\$\$[\s\S]*\$\$"  # Equations
        ]
        
        # Content types with inherent importance
        self.important_content_types = [
            "instruction",
            "question",
            "code",
            "equation",
            "system"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.important_patterns]
    
    def score(self, segment: ContextSegment) -> float:
        """
        Score the importance of a segment on a scale of 0-10.
        
        Args:
            segment: The context segment to score
            
        Returns:
            Importance score (0-10)
        """
        content = segment.content
        base_score = 5.0  # Start with neutral importance
        
        # Check content type
        if segment.segment_type in self.important_content_types:
            base_score += 2.0
        
        # Check for important patterns
        pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(content))
        pattern_score = min(3.0, pattern_matches * 0.5)
        
        # Length considerations - very short or very long content may be less important
        length_score = 0.0
        if 10 <= len(content.split()) <= 100:
            length_score = 1.0
        
        # Recency factor - newer content is generally more important
        age_hours = segment.age / 3600
        recency_score = max(0, 2.0 - (age_hours / 12) * 2.0)
        
        # Calculate final score (capped at 0-10)
        final_score = base_score + pattern_score + length_score + recency_score
        return max(0.0, min(10.0, final_score))


class MLClassifier:
    """Machine learning based importance classifier."""
    
    def __init__(self, model_path: str):
        """
        Initialize ML classifier.
        
        Args:
            model_path: Path to pretrained model
        """
        # In a real implementation, load the model here
        # For this implementation, we'll use a simplified approach
        self.model_path = model_path
        
        # Features we'd extract in a real implementation
        self.features = [
            "length",
            "contains_question",
            "contains_code",
            "contains_instruction",
            "sentiment_score",
            "named_entity_count"
        ]
    
    def extract_features(self, segment: ContextSegment) -> Dict[str, float]:
        """
        Extract features from the segment for ML scoring.
        
        Args:
            segment: The context segment
            
        Returns:
            Dictionary of feature values
        """
        content = segment.content
        features = {}
        
        # Simple feature extraction
        features["length"] = len(content)
        features["contains_question"] = 1.0 if "?" in content else 0.0
        features["contains_code"] = 1.0 if "```" in content else 0.0
        features["contains_instruction"] = 1.0 if any(word in content.lower() 
                                                  for word in ["do", "please", "must", "should"]) else 0.0
        
        # In a real implementation, we'd use NLP libraries for these
        features["sentiment_score"] = 0.5  # Placeholder
        features["named_entity_count"] = content.count("[") + content.count("]")  # Very rough approximation
        
        return features
    
    def score(self, segment: ContextSegment, context: List[ContextSegment] = None) -> float:
        """
        Score the importance of a segment using ML.
        
        Args:
            segment: The context segment to score
            context: Optional context segments for contextual scoring
            
        Returns:
            Importance score (0-10)
        """
        # Extract features
        features = self.extract_features(segment)
        
        # In a real implementation, we'd use the model for prediction
        # Here, we'll use a simple heuristic based on the features
        
        # Simple weighted average of features
        weights = {
            "length": 0.1,
            "contains_question": 2.0,
            "contains_code": 2.5,
            "contains_instruction": 2.0,
            "sentiment_score": 1.0,
            "named_entity_count": 0.5
        }
        
        raw_score = sum(features[key] * weights[key] for key in weights)
        
        # Normalize to 0-10 scale
        return min(10.0, max(0.0, raw_score / 50 * 10))


class LLMClassifier:
    """LLM-based importance classifier."""
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize LLM classifier.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.ollama_url = f"{config.ollama_host}/api/generate"
        self.model = config.default_model
    
    def score(self, segment: ContextSegment, context: List[ContextSegment] = None) -> float:
        """
        Score the importance of a segment using LLM.
        
        Args:
            segment: The context segment to score
            context: Optional context segments for contextual scoring
            
        Returns:
            Importance score (0-10)
        """
        if len(segment.content) > self.config.max_llm_classification_length:
            # Fall back to rule-based for long content
            return RuleBasedClassifier().score(segment)
        
        # Prepare context if provided
        context_text = ""
        if context and len(context) > 0:
            context_text = "Previous conversation context:\n"
            context_text += "\n".join([f"- {seg.content[:100]}..." for seg in context[:3]])
        
        # Construct prompt
        prompt = f"""
        You are an importance classifier for conversational AI. 
        Rate the importance of the following message on a scale of 0 to 10, where:
        - 0: Completely unimportant, can be forgotten
        - 5: Moderately important
        - 10: Critically important, must be remembered
        
        {context_text}
        
        Message to classify:
        {segment.content}
        
        Factors to consider:
        - Contains key information like names, facts, or instructions
        - Establishes context that will be important later
        - Asks a question or requests an action
        - Contains code, formulas, or specific data
        
        Return only a single number between 0 and 10.
        """
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Extract the score from the response
                try:
                    # First try to parse as a simple number
                    score = float(response_text)
                except ValueError:
                    # If that fails, try to find a number in the text
                    match = re.search(r'\b(\d+(\.\d+)?)\b', response_text)
                    if match:
                        score = float(match.group(1))
                    else:
                        # Fallback
                        score = 5.0
                
                return max(0.0, min(10.0, score))
            
        except Exception as e:
            # Fall back to rule-based on error
            pass
        
        # Fallback to rule-based
        return RuleBasedClassifier().score(segment)


class ImportanceClassifier:
    """Ensemble importance classifier."""
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize the importance classifier.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLClassifier(config.ml_model_path) if config.use_ml else None
        self.llm_classifier = LLMClassifier(config) if config.use_llm_classification else None
    
    def classify(self, segment: ContextSegment, context: List[ContextSegment] = None) -> float:
        """
        Classify the importance of a segment using available classifiers.
        
        Args:
            segment: The context segment to classify
            context: Optional context segments for contextual classification
            
        Returns:
            Importance score (0-10)
        """
        # Get scores from available classifiers
        scores = []
        weights = []
        
        # Rule-based (always available)
        rule_score = self.rule_classifier.score(segment)
        scores.append(rule_score)
        weights.append(self.config.rule_weight)
        
        # ML-based (if enabled)
        if self.ml_classifier:
            ml_score = self.ml_classifier.score(segment, context)
            scores.append(ml_score)
            weights.append(self.config.ml_weight)
            
        # LLM-based (if enabled and within budget)
        if self.llm_classifier and len(segment.content) < self.config.max_llm_classification_length:
            llm_score = self.llm_classifier.score(segment, context)
            scores.append(llm_score)
            weights.append(self.config.llm_weight)
            
        # Return weighted average
        if not scores:
            return 5.0  # Default neutral importance
            
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights) 