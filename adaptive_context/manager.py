import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.memory import (
    ContextSegment, 
    MemoryTier, 
    ActiveTier, 
    WorkingTier, 
    ArchiveTier
)
from adaptive_context.classifier import ImportanceClassifier
from adaptive_context.compressor import ContextCompressor
from adaptive_context.knowledge import KnowledgeStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('adaptive_context')

class AdaptiveContextManager:
    """
    Main orchestrator for the AdaptiveContext system that manages the memory tiers,
    importance classification, context compression, and knowledge store.
    """
    
    def __init__(self, config: AdaptiveContextConfig = None):
        """
        Initialize the adaptive context manager.
        
        Args:
            config: Configuration object (optional, uses defaults if not provided)
        """
        self.config = config or AdaptiveContextConfig()
        logger.info(f"Initializing AdaptiveContextManager with {self.config.active_tier_tokens} active tokens, "
                   f"{self.config.working_tier_tokens} working tokens, "
                   f"{self.config.archive_tier_tokens} archive tokens")
        
        # Initialize memory tiers
        self.active_tier = ActiveTier(self.config.active_tier_tokens)
        self.working_tier = WorkingTier(self.config.working_tier_tokens)
        self.archive_tier = ArchiveTier(self.config.archive_tier_tokens)
        
        # Initialize components
        self.importance_classifier = ImportanceClassifier(self.config)
        self.compressor = ContextCompressor(self.config)
        self.knowledge_store = KnowledgeStore(self.config)
        
        # Track token counting and state
        self.total_tokens = 0
        self.token_counting_method = "basic"  # Could be "basic", "ollama", "tiktoken"
        self.session_start_time = time.time()
        
        logger.info("AdaptiveContextManager initialized successfully")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the selected method.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if self.token_counting_method == "basic":
            # Simple approximation: words + punctuation
            return len(text.split()) + text.count(".") + text.count(",") + text.count("!") + text.count("?")
        
        elif self.token_counting_method == "ollama":
            # Use Ollama API to count tokens (more accurate but higher latency)
            try:
                response = requests.post(
                    f"{self.config.ollama_host}/api/tokenize",
                    json={"model": self.config.default_model, "text": text},
                    timeout=2
                )
                if response.status_code == 200:
                    return len(response.json().get("tokens", []))
            except Exception as e:
                logger.warning(f"Error counting tokens with Ollama: {e}")
                # Fall back to basic counting on error
        
        # Default basic counting
        return len(text.split())
    
    def add_message(self, content: str, segment_type: str = "user", metadata: Dict[str, Any] = None) -> bool:
        """
        Add a new message to the context.
        
        Args:
            content: Message content
            segment_type: Type of message ('user', 'assistant', 'system', etc.)
            metadata: Optional metadata for the message
            
        Returns:
            True if message was added successfully
        """
        if not content:
            return False
            
        # Count tokens
        token_count = self._count_tokens(content)
        
        # Create context segment
        segment = ContextSegment(
            content=content,
            importance=0.0,  # Will be set by classifier
            timestamp=time.time(),
            token_count=token_count,
            segment_type=segment_type,
            metadata=metadata or {}
        )
        
        # Get current context for importance classification
        current_context = self._get_recent_context()
        
        # Classify importance
        importance = self.importance_classifier.classify(segment, current_context)
        segment.importance = importance
        
        logger.info(f"Adding {segment_type} message with {token_count} tokens and importance {importance:.2f}")
        
        # Try to add to active tier first
        if self.active_tier.add_segment(segment):
            self.total_tokens += token_count
            logger.debug(f"Added to active tier, now at {self.active_tier.current_token_count}/{self.active_tier.max_tokens} tokens")
            
            # Check if we need to manage tiers after adding
            if self.active_tier.fullness_ratio > self.config.compression_threshold:
                logger.info(f"Active tier exceeds threshold ({self.active_tier.fullness_ratio:.2f}), managing tiers")
                self._manage_tiers()
                
            return True
        
        # If active tier is full, we need to make room
        logger.info("Active tier full, making room")
        self._manage_tiers()
        
        # Try again after managing tiers
        if self.active_tier.add_segment(segment):
            self.total_tokens += token_count
            return True
            
        # If still can't add, segment is too large
        logger.warning(f"Message too large ({token_count} tokens) to add to context")
        return False
    
    def _manage_tiers(self):
        """
        Manage memory tiers to maintain efficient context window usage.
        This is the core of the adaptive context management algorithm.
        """
        # Step 1: Move less important items from active to working tier
        if self.active_tier.fullness_ratio > self.config.compression_threshold:
            logger.info("Moving less important segments from active to working tier")
            
            # Find less important segments
            active_segments = self.active_tier.segments.copy()
            
            # Sort by importance and age (less important and older first)
            active_segments.sort(key=lambda s: (s.importance, -s.age))
            
            # Move segments until below threshold or working tier is full
            segments_to_move = []
            tokens_to_move = 0
            target_tokens = int(self.active_tier.current_token_count * 0.7)  # Target 70% fullness
            
            for segment in active_segments:
                if (self.active_tier.current_token_count - tokens_to_move > target_tokens and 
                    self.working_tier.current_token_count + segment.token_count <= self.working_tier.max_tokens):
                    segments_to_move.append(segment)
                    tokens_to_move += segment.token_count
                    
                if self.active_tier.current_token_count - tokens_to_move <= target_tokens:
                    break
            
            # Move identified segments
            for segment in segments_to_move:
                index = self.active_tier.segments.index(segment)
                removed = self.active_tier.remove_segment(index)
                if removed:
                    self.working_tier.add_segment(removed)
                    logger.debug(f"Moved segment with importance {removed.importance:.2f} to working tier")
        
        # Step 2: Compress working tier if it's getting full
        if self.working_tier.fullness_ratio > self.config.compression_threshold:
            logger.info("Compressing working tier segments")
            
            # Get working segments
            working_segments = self.working_tier.segments.copy()
            
            # Calculate target token count (80% of capacity)
            target_tokens = int(self.working_tier.max_tokens * 0.8)
            
            # Compress segments to meet target
            compressed_segments = self.compressor.progressive_compress(
                working_segments, target_tokens
            )
            
            # Clear and refill working tier with compressed segments
            self.working_tier = WorkingTier(self.config.working_tier_tokens)
            for segment in compressed_segments:
                self.working_tier.add_segment(segment)
                
            logger.debug(f"Working tier compressed to {self.working_tier.current_token_count}/{self.working_tier.max_tokens} tokens")
        
        # Step 3: Move least important items from working to archive tier
        if (self.working_tier.fullness_ratio > self.config.compression_threshold and 
            self.archive_tier.current_token_count < self.archive_tier.max_tokens):
            logger.info("Moving segments from working to archive tier")
            
            # Find less important segments
            working_segments = self.working_tier.segments.copy()
            
            # Sort by importance and age
            working_segments.sort(key=lambda s: (s.importance, -s.age))
            
            # Calculate how many tokens to move
            target_working_tokens = int(self.working_tier.max_tokens * 0.7)  # Target 70% fullness
            tokens_to_move = max(0, self.working_tier.current_token_count - target_working_tokens)
            available_archive_tokens = self.archive_tier.max_tokens - self.archive_tier.current_token_count
            tokens_to_move = min(tokens_to_move, available_archive_tokens)
            
            # Move segments
            moved_tokens = 0
            for segment in working_segments:
                if moved_tokens >= tokens_to_move:
                    break
                
                # Compress before moving to archive
                archive_ratio = 0.5  # More aggressive compression for archive
                compressed = self.compressor.compress_segment(segment, archive_ratio)
                
                # Remove from working tier
                index = self.working_tier.segments.index(segment)
                self.working_tier.remove_segment(index)
                
                # Add to archive tier
                self.archive_tier.add_segment(compressed)
                moved_tokens += segment.token_count
                logger.debug(f"Moved and compressed segment with importance {segment.importance:.2f} to archive tier")
        
        # Step 4: Extract facts for knowledge store from archive tier
        if self.archive_tier.fullness_ratio > 0.9:
            logger.info("Extracting facts from archive tier to knowledge store")
            
            # Get oldest and least important segments
            archive_segments = self.archive_tier.segments.copy()
            archive_segments.sort(key=lambda s: (s.importance, -s.age))
            
            # Extract facts from oldest 20% of segments
            segments_to_process = archive_segments[:max(1, len(archive_segments) // 5)]
            
            fact_count = 0
            for segment in segments_to_process:
                # Only process segments older than 1 hour
                if segment.age < 3600:
                    continue
                
                # Extract facts
                facts = self.knowledge_store.extract_facts_from_text(segment.content)
                
                # Store in knowledge store with confidence based on importance
                confidence = min(0.8, segment.importance / 10)
                for subject, predicate, obj in facts:
                    self.knowledge_store.store_fact_triple(
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        confidence=confidence,
                        source=f"archive_segment_{segment.timestamp}"
                    )
                    fact_count += 1
                
                # Remove processed segment if we've extracted facts
                if facts:
                    index = self.archive_tier.segments.index(segment)
                    self.archive_tier.remove_segment(index)
                    logger.debug(f"Removed segment from archive after extracting {len(facts)} facts")
            
            logger.info(f"Extracted {fact_count} facts from archive tier")
            
            # If we still need more space, remove oldest segments
            if self.archive_tier.fullness_ratio > 0.9:
                # Remove oldest 10%
                archive_segments = sorted(self.archive_tier.segments, key=lambda s: s.timestamp)
                segments_to_remove = archive_segments[:max(1, len(archive_segments) // 10)]
                
                for segment in segments_to_remove:
                    index = self.archive_tier.segments.index(segment)
                    self.archive_tier.remove_segment(index)
                    
                logger.info(f"Removed {len(segments_to_remove)} oldest segments from archive tier")
    
    def _get_recent_context(self, max_segments: int = 5) -> List[ContextSegment]:
        """
        Get recent context segments for use in classification.
        
        Args:
            max_segments: Maximum segments to return
            
        Returns:
            List of recent segments
        """
        all_segments = self.active_tier.segments.copy()
        all_segments.sort(key=lambda s: s.timestamp, reverse=True)
        return all_segments[:max_segments]
    
    def get_full_context(self) -> str:
        """
        Get the full context across all tiers for sending to the LLM.
        
        Returns:
            Concatenated context text
        """
        # Get content from each tier
        active_content = self.active_tier.get_content()
        working_content = self.working_tier.get_content()
        archive_content = self.archive_tier.get_content()
        
        # Assemble with tier markers
        full_context = active_content
        
        if working_content:
            full_context += "\n\n[Working Memory]:\n" + working_content
            
        if archive_content:
            full_context += "\n\n[Long-term Memory]:\n" + archive_content
        
        # Add relevant knowledge from knowledge store if available
        if active_content:
            # Use recent user queries to find relevant knowledge
            recent_segments = [s for s in self.active_tier.segments if s.segment_type == "user"]
            if recent_segments:
                recent_query = recent_segments[0].content
                relevant_knowledge = self.knowledge_store.get_relevant_knowledge(recent_query, max_results=3)
                
                if relevant_knowledge:
                    knowledge_text = "\n\n[Relevant Knowledge]:\n"
                    for item in relevant_knowledge:
                        knowledge_text += f"- {item['content']}\n"
                    
                    full_context += knowledge_text
        
        return full_context
    
    def flush(self):
        """Reset all memory tiers for a new conversation."""
        logger.info("Flushing all memory tiers")
        
        # Summarize current conversation before flushing
        if self.active_tier.segments:
            active_content = self.active_tier.get_content()
            
            # Extract keywords from content
            words = active_content.lower().split()
            keywords = list(set(word for word in words if len(word) > 3))[:20]
            
            # Store summary in knowledge store
            self.knowledge_store.store_conversation_summary(
                summary=f"Conversation from {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.session_start_time))}: {active_content[:200]}...",
                keywords=keywords
            )
            logger.info(f"Stored conversation summary with {len(keywords)} keywords")
        
        # Reset tiers
        self.active_tier = ActiveTier(self.config.active_tier_tokens)
        self.working_tier = WorkingTier(self.config.working_tier_tokens)
        self.archive_tier = ArchiveTier(self.config.archive_tier_tokens)
        
        # Reset session time
        self.session_start_time = time.time()
        self.total_tokens = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current context state.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "active_tier": {
                "tokens": self.active_tier.current_token_count,
                "capacity": self.active_tier.max_tokens,
                "fullness": self.active_tier.fullness_ratio,
                "segments": len(self.active_tier.segments)
            },
            "working_tier": {
                "tokens": self.working_tier.current_token_count,
                "capacity": self.working_tier.max_tokens,
                "fullness": self.working_tier.fullness_ratio,
                "segments": len(self.working_tier.segments)
            },
            "archive_tier": {
                "tokens": self.archive_tier.current_token_count,
                "capacity": self.archive_tier.max_tokens,
                "fullness": self.archive_tier.fullness_ratio,
                "segments": len(self.archive_tier.segments)
            },
            "session_duration": time.time() - self.session_start_time
        }
    
    def explicitly_remember(self, text: str) -> bool:
        """
        Explicitly store something in the knowledge base.
        
        Args:
            text: Text to remember
            
        Returns:
            True if successfully stored
        """
        try:
            fact_ids = self.knowledge_store.remember_explicit(text)
            logger.info(f"Explicitly stored {len(fact_ids)} facts")
            return len(fact_ids) > 0
        except Exception as e:
            logger.error(f"Error explicitly remembering: {e}")
            return False
    
    def close(self):
        """Clean up resources."""
        try:
            self.knowledge_store.close()
            logger.info("AdaptiveContextManager resources closed")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        self.close() 