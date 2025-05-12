"""
Self-Reflection and Self-Correction module for AdaptiveContext.

This module implements mechanisms for verifying knowledge relevance,
checking response consistency, and revising answers based on detected issues.
"""

import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union

import requests

from adaptive_context.config import CortexFlowConfig
from adaptive_context.knowledge import KnowledgeStore

logger = logging.getLogger('cortexflow')

class ReflectionEngine:
    """
    Engine for self-reflection and self-correction capabilities in AdaptiveContext.
    
    This class provides mechanisms to:
    1. Verify the relevance of retrieved knowledge
    2. Check for inconsistencies in generated responses
    3. Revise answers based on detected issues
    """
    
    def __init__(
        self, 
        config: CortexFlowConfig,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        """
        Initialize the reflection engine with configuration.
        
        Args:
            config: AdaptiveContext configuration
            knowledge_store: Optional knowledge store for verification
        """
        self.config = config
        self.knowledge_store = knowledge_store
        self.ollama_host = config.ollama_host
        self.default_model = config.default_model
        
        # Default reflection thresholds
        self.relevance_threshold = 0.6  # Minimum score for knowledge relevance
        self.confidence_threshold = 0.7  # Minimum confidence for answers
        
        # Configure from config if available
        if hasattr(config, 'reflection_relevance_threshold'):
            self.relevance_threshold = config.reflection_relevance_threshold
            
        if hasattr(config, 'reflection_confidence_threshold'):
            self.confidence_threshold = config.reflection_confidence_threshold
            
        logger.info(f"Initialized ReflectionEngine with relevance threshold {self.relevance_threshold} and confidence threshold {self.confidence_threshold}")
    
    def verify_knowledge_relevance(
        self,
        query: str,
        knowledge_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify the relevance of knowledge items to the query.
        
        Args:
            query: The user's query
            knowledge_items: List of knowledge items to verify
            
        Returns:
            Filtered list of knowledge items with relevance scores
        """
        if not knowledge_items:
            return []
        
        # Create a prompt for relevance verification
        prompt = self._create_relevance_prompt(query, knowledge_items)
        
        # Process with LLM
        response = self._process_with_llm(prompt)
        
        try:
            # Parse the relevance scores
            relevant_items = self._parse_relevance_response(response, knowledge_items)
            
            # Filter by relevance threshold
            filtered_items = [
                item for item in relevant_items 
                if item.get('relevance_score', 0) >= self.relevance_threshold
            ]
            
            logger.info(f"Knowledge relevance verification: {len(filtered_items)}/{len(knowledge_items)} items passed threshold")
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error parsing relevance response: {e}")
            # Fall back to original items if parsing fails
            return knowledge_items
    
    def check_response_consistency(
        self,
        query: str,
        response: str,
        knowledge_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for inconsistencies between response and knowledge items.
        
        Args:
            query: The user's query
            response: Generated response to check
            knowledge_items: Knowledge items used for response
            
        Returns:
            Dictionary with consistency assessment and identified issues
        """
        # Create a prompt for consistency checking
        prompt = self._create_consistency_prompt(query, response, knowledge_items)
        
        # Process with LLM
        check_result = self._process_with_llm(prompt)
        
        try:
            # Parse the consistency check result
            consistency_result = self._parse_consistency_response(check_result)
            logger.info(f"Consistency check: {consistency_result.get('is_consistent', False)}")
            return consistency_result
            
        except Exception as e:
            logger.error(f"Error parsing consistency response: {e}")
            # Return a default result on error
            return {
                "is_consistent": True,  # Assume consistent on error
                "confidence": 0.5,
                "issues": [],
                "reasoning": "Failed to perform consistency check due to error."
            }
    
    def revise_response(
        self,
        query: str,
        original_response: str,
        knowledge_items: List[Dict[str, Any]],
        consistency_result: Dict[str, Any]
    ) -> str:
        """
        Revise the response based on detected inconsistencies.
        
        Args:
            query: The user's query
            original_response: Original response to revise
            knowledge_items: Knowledge items for reference
            consistency_result: Result of consistency check
            
        Returns:
            Revised response
        """
        # Only revise if inconsistent
        if consistency_result.get('is_consistent', True):
            return original_response
        
        # Create a prompt for response revision
        prompt = self._create_revision_prompt(
            query, 
            original_response, 
            knowledge_items,
            consistency_result
        )
        
        # Process with LLM
        revised_response = self._process_with_llm(prompt)
        
        logger.info("Response revised based on consistency check")
        return revised_response
    
    def _create_relevance_prompt(
        self, 
        query: str, 
        knowledge_items: List[Dict[str, Any]]
    ) -> str:
        """Create a prompt for knowledge relevance verification."""
        knowledge_texts = []
        for i, item in enumerate(knowledge_items):
            text = item.get('text', '')
            knowledge_texts.append(f"[{i+1}] {text}")
            
        knowledge_context = "\n".join(knowledge_texts)
        
        return f"""As an AI assistant, your task is to evaluate the relevance of retrieved knowledge to a user's query.
For each knowledge item, assign a relevance score between 0.0 and 1.0, where:
- 1.0: Directly answers the query
- 0.8: Highly relevant and important for answering
- 0.6: Moderately relevant, contains useful context
- 0.4: Somewhat relevant but not essential
- 0.2: Tangentially related
- 0.0: Not relevant at all

USER QUERY: {query}

RETRIEVED KNOWLEDGE ITEMS:
{knowledge_context}

For each knowledge item, provide:
1. Item number
2. Relevance score (0.0-1.0)
3. Brief explanation (1 sentence)

Format your response as a JSON array of objects with these fields:
[
  {{"item": 1, "score": 0.8, "explanation": "Directly addresses..."}},
  ...
]

RELEVANCE ASSESSMENT:"""
    
    def _create_consistency_prompt(
        self, 
        query: str, 
        response: str, 
        knowledge_items: List[Dict[str, Any]]
    ) -> str:
        """Create a prompt for response consistency checking."""
        knowledge_texts = []
        for i, item in enumerate(knowledge_items):
            text = item.get('text', '')
            knowledge_texts.append(f"[{i+1}] {text}")
            
        knowledge_context = "\n".join(knowledge_texts)
        
        return f"""As an AI assistant with self-reflection capabilities, your task is to identify any inconsistencies or factual errors in a generated response compared to the knowledge base.

USER QUERY: {query}

GENERATED RESPONSE:
{response}

KNOWLEDGE BASE:
{knowledge_context}

Analyze the response for:
1. Factual inconsistencies with the knowledge base
2. Unsupported claims not present in the knowledge base
3. Logical contradictions within the response
4. Hallucinations or made-up information

Provide your assessment as a JSON object with these fields:
- is_consistent: Boolean indicating if the response is consistent
- confidence: Number between 0.0 and 1.0 indicating your confidence in this assessment
- issues: Array of specific issues found (empty if consistent)
- reasoning: Brief explanation of your reasoning

CONSISTENCY ASSESSMENT:"""
    
    def _create_revision_prompt(
        self, 
        query: str, 
        original_response: str, 
        knowledge_items: List[Dict[str, Any]],
        consistency_result: Dict[str, Any]
    ) -> str:
        """Create a prompt for response revision."""
        knowledge_texts = []
        for i, item in enumerate(knowledge_items):
            text = item.get('text', '')
            knowledge_texts.append(f"[{i+1}] {text}")
            
        knowledge_context = "\n".join(knowledge_texts)
        
        issues = consistency_result.get('issues', [])
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        
        return f"""As an AI assistant with self-correction capabilities, your task is to revise a response that contains inconsistencies or errors.

USER QUERY: {query}

ORIGINAL RESPONSE:
{original_response}

IDENTIFIED ISSUES:
{issues_text}

KNOWLEDGE BASE:
{knowledge_context}

Please revise the response to:
1. Fix all identified issues
2. Ensure factual accuracy based on the knowledge base
3. Remove any unsupported claims
4. Maintain a helpful, informative tone

Provide a complete revised response that directly answers the user's query while addressing all identified issues.

REVISED RESPONSE:"""
    
    def _process_with_llm(self, prompt: str) -> str:
        """Process the prompt with an LLM."""
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.default_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Error from LLM: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return ""
    
    def _parse_relevance_response(
        self, 
        response: str, 
        knowledge_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse the relevance assessment response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                relevance_data = json.loads(json_match.group(0))
            else:
                # Try to extract with more lenient parsing
                in_json = False
                json_lines = []
                for line in response.split('\n'):
                    if line.strip().startswith('['):
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if in_json and line.strip().endswith(']'):
                        break
                        
                relevance_json = ''.join(json_lines)
                relevance_data = json.loads(relevance_json)
            
            # Map relevance scores to knowledge items
            updated_items = []
            for i, item in enumerate(knowledge_items):
                item_copy = item.copy()
                
                # Find the corresponding relevance data
                for rel_item in relevance_data:
                    if rel_item.get('item') == i + 1:
                        item_copy['relevance_score'] = rel_item.get('score', 0.0)
                        item_copy['relevance_explanation'] = rel_item.get('explanation', '')
                        break
                else:
                    # Default if not found
                    item_copy['relevance_score'] = 0.0
                    item_copy['relevance_explanation'] = 'Not assessed'
                    
                updated_items.append(item_copy)
                
            return updated_items
            
        except Exception as e:
            logger.error(f"Error parsing relevance response: {e}")
            # Return original items with default scores on error
            return [
                {**item, 'relevance_score': 0.5, 'relevance_explanation': 'Error in assessment'} 
                for item in knowledge_items
            ]
    
    def _parse_consistency_response(self, response: str) -> Dict[str, Any]:
        """Parse the consistency check response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{\s*".*"\s*:.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Try to extract with more lenient parsing
                in_json = False
                json_lines = []
                for line in response.split('\n'):
                    if line.strip().startswith('{'):
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if in_json and line.strip().endswith('}'):
                        break
                        
                consistency_json = ''.join(json_lines)
                return json.loads(consistency_json)
                
        except Exception as e:
            logger.error(f"Error parsing consistency response: {e}")
            # Return default result on error
            return {
                "is_consistent": True,
                "confidence": 0.5,
                "issues": [],
                "reasoning": "Failed to parse consistency check result."
            } 