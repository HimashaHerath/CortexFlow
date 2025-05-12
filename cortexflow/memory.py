"""
CortexFlow Memory module.

This module provides the memory management system for the CortexFlow.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .config import CortexFlowConfig
from .interfaces import MemoryTierInterface

logger = logging.getLogger('cortexflow')

@dataclass
class ContextSegment:
    """Represents a segment of conversation context with metadata."""
    content: str
    importance: float
    timestamp: float
    token_count: int
    segment_type: str  # 'user', 'assistant', 'system', etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def age(self) -> float:
        """Returns the age of this segment in seconds."""
        return time.time() - self.timestamp


class MemoryTier(MemoryTierInterface):
    """Base class for memory tiers implementing the MemoryTierInterface."""
    
    def __init__(self, name: str, max_tokens: int):
        """
        Initialize a memory tier.
        
        Args:
            name: The name of this tier
            max_tokens: Maximum token capacity
        """
        self.name = name
        self.max_tokens = max_tokens
        self.segments: List[ContextSegment] = []
        self.current_token_count = 0
    
    def add_content(self, content: Any, importance: float) -> bool:
        """
        Add content to this tier as required by MemoryTierInterface.
        
        Args:
            content: Content to add (typically a ContextSegment)
            importance: Importance of the content
            
        Returns:
            True if content was added successfully
        """
        if isinstance(content, ContextSegment):
            return self.add_segment(content)
        else:
            # Create a segment from generic content
            segment = ContextSegment(
                content=str(content),
                importance=importance,
                timestamp=time.time(),
                token_count=len(str(content).split()),  # Simple token count estimation
                segment_type="generic"
            )
            return self.add_segment(segment)
    
    def add_segment(self, segment: ContextSegment) -> bool:
        """
        Add a segment to this tier.
        
        Args:
            segment: The segment to add
            
        Returns:
            True if the segment was added, False if there wasn't enough space
        """
        if segment.token_count > self.available_tokens:
            return False
        
        self.segments.append(segment)
        self.current_token_count += segment.token_count
        return True
    
    def remove_segment(self, index: int) -> Optional[ContextSegment]:
        """
        Remove a segment at the specified index.
        
        Args:
            index: Index of segment to remove
            
        Returns:
            The removed segment or None if index was invalid
        """
        if 0 <= index < len(self.segments):
            segment = self.segments.pop(index)
            self.current_token_count -= segment.token_count
            return segment
        return None
    
    def get_content(self) -> str:
        """
        Get the full content of this tier as a single string.
        
        Returns:
            Concatenated content of all segments
        """
        return "\n".join(segment.content for segment in self.segments)
    
    def update_size(self, new_size: int) -> bool:
        """
        Update the size/capacity of this tier as required by MemoryTierInterface.
        
        Args:
            new_size: New size/capacity
            
        Returns:
            True if size was updated successfully
        """
        return self.update_token_limit(new_size)
        
    def get_segments_by_importance(self, threshold: float) -> List[ContextSegment]:
        """
        Get segments with importance greater than or equal to the threshold.
        
        Args:
            threshold: Minimum importance score (0-10)
            
        Returns:
            List of segments meeting the threshold
        """
        return [segment for segment in self.segments if segment.importance >= threshold]
    
    def get_least_important_segment(self) -> Optional[int]:
        """
        Find the index of the least important segment.
        
        Returns:
            Index of least important segment or None if tier is empty
        """
        if not self.segments:
            return None
        
        # Factor in both importance and age
        # Older and less important segments are more likely to be removed
        return min(range(len(self.segments)), 
                   key=lambda i: (self.segments[i].importance, -self.segments[i].age))
    
    def update_token_limit(self, new_limit: int) -> bool:
        """
        Update the token limit for this tier.
        
        Args:
            new_limit: New token limit
            
        Returns:
            True if update was successful, False if not
        """
        # Don't allow reducing below current usage or below minimum (1000 tokens)
        if new_limit < self.current_token_count or new_limit < 1000:
            logger.warning(f"Cannot update {self.name} tier limit to {new_limit} tokens "
                          f"(current usage: {self.current_token_count})")
            return False
            
        logger.debug(f"Updating {self.name} tier limit from {self.max_tokens} to {new_limit} tokens")
        self.max_tokens = new_limit
        return True
    
    @property
    def is_full(self) -> bool:
        """Returns True if tier is at or near capacity."""
        return self.current_token_count >= self.max_tokens
    
    @property
    def available_tokens(self) -> int:
        """Returns the number of available tokens in this tier."""
        return max(0, self.max_tokens - self.current_token_count)
    
    @property
    def fullness_ratio(self) -> float:
        """Returns the fullness ratio (0.0 to 1.0)."""
        if self.max_tokens == 0:
            return 1.0
        return self.current_token_count / self.max_tokens


class ActiveTier(MemoryTier):
    """Active memory tier - most recent and important context."""
    
    def __init__(self, max_tokens: int):
        super().__init__("active", max_tokens)
    
    def get_content(self) -> str:
        """
        Get the full content with most recent segments first.
        
        Returns:
            Concatenated content of all segments
        """
        # Order by recency (timestamp)
        ordered_segments = sorted(self.segments, key=lambda s: s.timestamp, reverse=True)
        return "\n".join(segment.content for segment in ordered_segments)


class WorkingTier(MemoryTier):
    """Working memory tier - medium-term storage."""
    
    def __init__(self, max_tokens: int):
        super().__init__("working", max_tokens)
    
    def get_content(self) -> str:
        """
        Get the full content with most important segments first.
        
        Returns:
            Concatenated content of all segments
        """
        # Order by importance
        ordered_segments = sorted(self.segments, key=lambda s: s.importance, reverse=True)
        return "\n".join(segment.content for segment in ordered_segments)


class ArchiveTier(MemoryTier):
    """Archive memory tier - compressed long-term storage."""
    
    def __init__(self, max_tokens: int):
        super().__init__("archive", max_tokens)
    
    def get_content(self) -> str:
        """
        Get archive content with compressed segments.
        
        Returns:
            Concatenated content of all segments
        """
        # In the archive tier, content is already compressed
        # Order by importance and then by recency
        ordered_segments = sorted(
            self.segments, 
            key=lambda s: (s.importance, s.timestamp), 
            reverse=True
        )
        return "\n".join(segment.content for segment in ordered_segments)


class ConversationMemory:
    """
    Main memory management system for CortexFlow.
    
    Manages a multi-tier memory system with active, working, and archive tiers.
    Handles the movement of context between tiers based on importance and recency.
    """
    
    def __init__(self, config: CortexFlowConfig):
        """
        Initialize the ConversationMemory with provided configuration.
        
        Args:
            config: Configuration for the memory system
        """
        self.config = config
        self.active_token_limit = config.active_token_limit if hasattr(config, 'active_token_limit') else 4096
        self.working_token_limit = config.working_token_limit if hasattr(config, 'working_token_limit') else 8192
        self.archive_token_limit = config.archive_token_limit if hasattr(config, 'archive_token_limit') else 16384
        
        # Initialize memory tiers
        self.active_tier = ActiveTier(self.active_token_limit)
        self.working_tier = WorkingTier(self.working_token_limit)
        self.archive_tier = ArchiveTier(self.archive_token_limit)
        
        # Track memory statistics
        self.tier_stats = {
            "active": {
                "original_limit": self.active_token_limit,
                "current_limit": self.active_token_limit,
                "usage_history": []
            },
            "working": {
                "original_limit": self.working_token_limit,
                "current_limit": self.working_token_limit,
                "usage_history": []
            },
            "archive": {
                "original_limit": self.archive_token_limit,
                "current_limit": self.archive_token_limit,
                "usage_history": []
            }
        }
        
        # Initialize message storage
        self.messages = []
        self.next_message_id = 1
        
    def update_tier_limits(self, active_limit: int = None, working_limit: int = None, archive_limit: int = None) -> bool:
        success = True
        
        # Update active tier if specified
        if active_limit is not None:
            # Ensure minimum size and check if we can accommodate current content
            if active_limit >= max(self.active_tier.current_token_count, 1000):
                self.active_token_limit = active_limit
                self.active_tier.max_tokens = active_limit
                self.tier_stats["active"]["current_limit"] = active_limit
            else:
                success = False
                logger.warning(f"Cannot update active tier limit to {active_limit} (current usage: {self.active_tier.current_token_count})")
                
        # Update working tier if specified
        if working_limit is not None:
            if working_limit >= max(self.working_tier.current_token_count, 1000):
                self.working_token_limit = working_limit
                self.working_tier.max_tokens = working_limit
                self.tier_stats["working"]["current_limit"] = working_limit
            else:
                success = False
                logger.warning(f"Cannot update working tier limit to {working_limit} (current usage: {self.working_tier.current_token_count})")
                
        # Update archive tier if specified
        if archive_limit is not None:
            if archive_limit >= max(self.archive_tier.current_token_count, 1000):
                self.archive_token_limit = archive_limit
                self.archive_tier.max_tokens = archive_limit
                self.tier_stats["archive"]["current_limit"] = archive_limit
            else:
                success = False
                logger.warning(f"Cannot update archive tier limit to {archive_limit} (current usage: {self.archive_tier.current_token_count})")
                
        # Update tier usage history
        self._update_tier_usage_stats()
                
        return success
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a new message to the conversation memory.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata for the message
            
        Returns:
            The message that was added
        """
        if not content:
            logger.warning("Attempted to add empty message content")
            return {}
            
        # Create the message
        message = {
            "id": self.next_message_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Increment message ID
        self.next_message_id += 1
        
        # Add message to memory
        self.messages.append(message)
        logger.debug(f"Added {role} message: {content[:50]}...")
        
        # Trim memory if necessary
        if len(self.messages) > 100:  # Simple limit for now
            self._trim_old_messages()
            
        return message
    
    def _trim_old_messages(self):
        """Trim old messages to maintain reasonable memory size."""
        # Keep all system messages and last 50 messages
        system_messages = [m for m in self.messages if m["role"] == "system"]
        recent_messages = self.messages[-50:]
        
        # Combine and deduplicate
        kept_ids = set(m["id"] for m in system_messages + recent_messages)
        self.messages = [m for m in self.messages if m["id"] in kept_ids]
        
        logger.debug(f"Trimmed messages to {len(self.messages)} entries")
    
    def get_context_messages(self) -> List[Dict[str, Any]]:
        """
        Get messages for context window.
        
        Returns:
            List of messages with role and content
        """
        # Convert to format expected by LLM APIs
        formatted_messages = []
        for message in self.messages:
            formatted = {
                "role": message["role"],
                "content": message["content"]
            }
            formatted_messages.append(formatted)
            
        return formatted_messages
    
    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get messages with specified role.
        
        Args:
            role: Role to filter by
            
        Returns:
            List of messages with matching role
        """
        return [m for m in self.messages if m["role"] == role]
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent message.
        
        Returns:
            Most recent message or None if no messages exist
        """
        return self.messages[-1] if self.messages else None
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            String summary
        """
        if not self.messages:
            return "No conversation history."
            
        user_messages = self.get_messages_by_role("user")
        assistant_messages = self.get_messages_by_role("assistant")
        
        summary_lines = [
            f"Conversation with {len(self.messages)} messages",
            f"- User messages: {len(user_messages)}",
            f"- Assistant messages: {len(assistant_messages)}",
        ]
        
        # Add first and last message summaries
        if self.messages:
            first_msg = self.messages[0]
            summary_lines.append(f"- First message ({first_msg['role']}): {first_msg['content'][:50]}...")
            
            last_msg = self.messages[-1]
            summary_lines.append(f"- Latest message ({last_msg['role']}): {last_msg['content'][:50]}...")
            
        return "\n".join(summary_lines)
    
    def _update_tier_usage_stats(self):
        """Update tier usage statistics for tracking."""
        timestamp = time.time()
        
        # Update active tier stats
        self.tier_stats["active"]["usage_history"].append({
            "timestamp": timestamp,
            "tokens_used": self.active_tier.current_token_count,
            "fullness_ratio": self.active_tier.fullness_ratio
        })
        
        # Update working tier stats
        self.tier_stats["working"]["usage_history"].append({
            "timestamp": timestamp,
            "tokens_used": self.working_tier.current_token_count,
            "fullness_ratio": self.working_tier.fullness_ratio
        })
        
        # Update archive tier stats
        self.tier_stats["archive"]["usage_history"].append({
            "timestamp": timestamp,
            "tokens_used": self.archive_tier.current_token_count,
            "fullness_ratio": self.archive_tier.fullness_ratio
        })
        
        # Keep history size manageable
        for tier in self.tier_stats:
            if len(self.tier_stats[tier]["usage_history"]) > 50:
                self.tier_stats[tier]["usage_history"] = self.tier_stats[tier]["usage_history"][-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Update stats before returning
        self._update_tier_usage_stats()
        
        return {
            "message_count": len(self.messages),
            "tiers": {
                "active": {
                    "limit": self.active_token_limit,
                    "used": self.active_tier.current_token_count,
                    "fullness": self.active_tier.fullness_ratio,
                    "segment_count": len(self.active_tier.segments)
                },
                "working": {
                    "limit": self.working_token_limit,
                    "used": self.working_tier.current_token_count,
                    "fullness": self.working_tier.fullness_ratio,
                    "segment_count": len(self.working_tier.segments)
                },
                "archive": {
                    "limit": self.archive_token_limit,
                    "used": self.archive_tier.current_token_count,
                    "fullness": self.archive_tier.fullness_ratio,
                    "segment_count": len(self.archive_tier.segments)
                }
            },
            "tier_stats": self.tier_stats
        }
    
    def clear_memory(self):
        """Clear all conversation memory."""
        self.messages = []
        
        # Clear memory tiers
        self.active_tier = ActiveTier(self.active_token_limit)
        self.working_tier = WorkingTier(self.working_token_limit) 
        self.archive_tier = ArchiveTier(self.archive_token_limit)
        
        # Reset message ID counter
        self.next_message_id = 1
        
        # Update stats
        self._update_tier_usage_stats()
        
        logger.info("Conversation memory cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to serializable dictionary.
        
        Returns:
            Memory as dictionary
        """
        return {
            "messages": self.messages,
            "tiers": {
                "active_limit": self.active_token_limit,
                "working_limit": self.working_token_limit,
                "archive_limit": self.archive_token_limit
            },
            "next_message_id": self.next_message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: CortexFlowConfig) -> 'ConversationMemory':
        """
        Create from serialized dictionary.
        
        Args:
            data: Dictionary with memory data
            config: Configuration
            
        Returns:
            New ConversationMemory instance
        """
        memory = cls(config)
        
        # Restore messages
        memory.messages = data.get("messages", [])
        memory.next_message_id = data.get("next_message_id", 1)
        
        # Restore tier limits if available
        tiers = data.get("tiers", {})
        if tiers:
            memory.active_token_limit = tiers.get("active_limit", config.active_token_limit)
            memory.working_token_limit = tiers.get("working_limit", config.working_token_limit)
            memory.archive_token_limit = tiers.get("archive_limit", config.archive_token_limit)
            
            # Recreate tiers with restored limits
            memory.active_tier = ActiveTier(memory.active_token_limit)
            memory.working_tier = WorkingTier(memory.working_token_limit)
            memory.archive_tier = ArchiveTier(memory.archive_token_limit)
        
        return memory 