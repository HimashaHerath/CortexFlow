"""
CortexFlow Reflection module.

This module provides self-reflection capabilities for CortexFlow.
"""

import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union

from cortexflow.config import CortexFlowConfig
from cortexflow.knowledge import KnowledgeStore
from cortexflow.llm_client import create_llm_client

logger = logging.getLogger('cortexflow')

class ReflectionEngine:
    """
    Engine for self-reflection and self-correction capabilities in CortexFlow.
    
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
            config: CortexFlow configuration
            knowledge_store: Optional knowledge store for verification
        """
        self.config = config
        self.knowledge_store = knowledge_store
        self.llm_client = create_llm_client(config)
        
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
            filtered_items = []
            removed_items = []
            for item in relevant_items:
                if item.get('relevance_score', 0) >= self.relevance_threshold:
                    filtered_items.append(item)
                else:
                    removed_items.append(item)

            # Log what was removed and why
            if removed_items:
                for item in removed_items:
                    text_preview = item.get('text', '')[:80]
                    score = item.get('relevance_score', 0)
                    reason = item.get('relevance_explanation', 'below threshold')
                    logger.info(
                        f"Reflection filtered out knowledge item (score={score:.2f}, "
                        f"threshold={self.relevance_threshold}): '{text_preview}...' - {reason}"
                    )

            logger.info(f"Knowledge relevance verification: {len(filtered_items)}/{len(knowledge_items)} items passed threshold")
            return filtered_items

        except Exception as e:
            logger.error(f"Error parsing relevance response: {e}")
            # Fall back to original items if parsing fails
            return knowledge_items
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract key claims from a response using sentence splitting.

        Splits the response into sentences and filters out very short or
        non-substantive sentences to identify verifiable claims.

        Args:
            response: The generated response text

        Returns:
            List of claim strings extracted from the response.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences and non-substantive fragments
            if len(sentence.split()) >= 4 and not sentence.startswith(('I ', 'Let me', 'Here')):
                claims.append(sentence)
        return claims

    def _compute_kb_support_ratio(
        self,
        claims: List[str],
        knowledge_items: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Compute what fraction of claims have supporting evidence in the KB.

        For each claim, checks whether any knowledge item contains overlapping
        key terms (3+ word overlap), which serves as a lightweight signal for
        whether the claim is grounded in the knowledge base.

        Args:
            claims: List of extracted claim strings
            knowledge_items: Knowledge items to check against

        Returns:
            Tuple of (support_ratio, per_claim_details) where support_ratio
            is the fraction of claims with KB backing (0.0-1.0), and
            per_claim_details is a list of dicts with claim text and whether
            it was supported.
        """
        if not claims:
            return 1.0, []

        kb_texts = [item.get('text', '').lower() for item in knowledge_items]
        supported_count = 0
        claim_details = []

        for claim in claims:
            claim_words = set(claim.lower().split())
            # Remove common stopwords for matching
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                        'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of',
                        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'it', 'this',
                        'that', 'and', 'or', 'but', 'not', 'so', 'if', 'as'}
            claim_keywords = claim_words - stopwords

            has_support = False
            for kb_text in kb_texts:
                kb_words = set(kb_text.split())
                overlap = claim_keywords & kb_words
                # Require at least 3 keyword overlaps for support
                if len(overlap) >= 3:
                    has_support = True
                    break

            if has_support:
                supported_count += 1

            claim_details.append({
                "claim": claim[:100],
                "has_kb_support": has_support
            })

        support_ratio = supported_count / len(claims)
        return support_ratio, claim_details

    def check_response_consistency(
        self,
        query: str,
        response: str,
        knowledge_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for inconsistencies between response and knowledge items.

        Performs two levels of verification:
        1. KB-based: Extracts claims from the response and checks what fraction
           have supporting evidence in the knowledge items (ground-truth signal).
        2. LLM-based: Asks the LLM to identify factual inconsistencies,
           unsupported claims, and logical contradictions.

        Args:
            query: The user's query
            response: Generated response to check
            knowledge_items: Knowledge items used for response

        Returns:
            Dictionary with consistency assessment, identified issues,
            and kb_support_ratio indicating ground-truth KB coverage.
        """
        # Step 1: KB-based verification (lightweight, no LLM call)
        claims = self._extract_claims(response)
        kb_support_ratio, claim_details = self._compute_kb_support_ratio(claims, knowledge_items)
        logger.info(
            f"KB-based verification: {kb_support_ratio:.1%} of {len(claims)} "
            f"claims have knowledge base support"
        )

        # Step 2: LLM-based consistency checking
        prompt = self._create_consistency_prompt(query, response, knowledge_items)
        check_result = self._process_with_llm(prompt)

        try:
            # Parse the consistency check result
            consistency_result = self._parse_consistency_response(check_result)

            # Enrich with KB-based verification results
            consistency_result['kb_support_ratio'] = kb_support_ratio
            consistency_result['claim_details'] = claim_details

            is_consistent = consistency_result.get('is_consistent', False)
            if not is_consistent:
                issues = consistency_result.get('issues', [])
                logger.warning(
                    f"Consistency check FAILED: {len(issues)} issue(s) found - "
                    f"{'; '.join(str(i) for i in issues[:3])}"
                )
            else:
                logger.info("Consistency check passed")

            return consistency_result

        except Exception as e:
            logger.error(f"Error parsing consistency response: {e}")
            # Return a default result on error, still including KB signal
            return {
                "is_consistent": True,  # Assume consistent on error
                "confidence": 0.5,
                "issues": [],
                "reasoning": "Failed to perform LLM consistency check due to error.",
                "kb_support_ratio": kb_support_ratio,
                "claim_details": claim_details
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
            Revised response, or the original if no revision was needed.
        """
        # Only revise if inconsistent
        if consistency_result.get('is_consistent', True):
            logger.debug("Response is consistent, no revision needed")
            return original_response

        issues = consistency_result.get('issues', [])
        logger.warning(
            f"Revising response due to {len(issues)} consistency issue(s): "
            f"{'; '.join(str(i) for i in issues[:3])}"
        )

        # Create a prompt for response revision
        prompt = self._create_revision_prompt(
            query,
            original_response,
            knowledge_items,
            consistency_result
        )

        try:
            # Process with LLM
            revised_response = self._process_with_llm(prompt)
            logger.info(
                f"Response revised based on consistency check "
                f"(original: {len(original_response)} chars, revised: {len(revised_response)} chars)"
            )
            return revised_response
        except Exception as e:
            logger.error(f"Error during response revision: {e}")
            return original_response
    
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
        """Process the prompt with an LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response text

        Raises:
            Exception: If the LLM call fails after logging the error
        """
        try:
            return self.llm_client.generate_from_prompt(prompt)
        except Exception as e:
            logger.error(f"LLM processing failed in ReflectionEngine: {e}")
            raise
    
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