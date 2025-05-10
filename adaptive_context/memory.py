import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .config import AdaptiveContextConfig

logger = logging.getLogger('adaptive_context')

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


class MemoryTier:
    """Base class for memory tiers."""
    
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
    Manages conversation history and memory for the adaptive context system.
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize the conversation memory with configuration.
        
        Args:
            config: Configuration for the conversation memory
        """
        self.config = config
        self.active_token_limit = config.active_token_limit if hasattr(config, 'active_token_limit') else 4096
        self.messages = []
        self.next_message_id = 1
        
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
        non_system = [m for m in self.messages if m["role"] != "system"]
        
        # Keep most recent messages
        recent = non_system[-50:] if len(non_system) > 50 else non_system
        
        # Update messages list
        self.messages = system_messages + recent
        logger.debug(f"Trimmed messages to {len(self.messages)} total")
    
    def get_context_messages(self) -> List[Dict[str, Any]]:
        """
        Get formatted messages for the context.
        
        Returns:
            List of formatted messages for LLM context
        """
        # Format messages for LLM
        formatted = []
        for message in self.messages:
            formatted.append({
                "role": message["role"],
                "content": message["content"]
            })
            
        return formatted
    
    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get messages with a specific role.
        
        Args:
            role: Role to filter by
            
        Returns:
            List of messages with the specified role
        """
        return [m for m in self.messages if m["role"] == role]
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the conversation.
        
        Returns:
            Last message or None if no messages
        """
        return self.messages[-1] if self.messages else None
    
    def get_conversation_summary(self) -> str:
        """
        Get a simple summary of the conversation.
        
        Returns:
            String summary of conversation
        """
        if not self.messages:
            return "No conversation history"
            
        # Count messages by role
        roles = {}
        for message in self.messages:
            role = message["role"]
            if role in roles:
                roles[role] += 1
            else:
                roles[role] = 1
                
        # Get first and last timestamps
        start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.messages[0]["timestamp"]))
        end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.messages[-1]["timestamp"]))
        
        # Build summary
        summary = f"Conversation from {start_time} to {end_time}\n"
        summary += f"Total messages: {len(self.messages)}\n"
        
        for role, count in roles.items():
            summary += f"- {role}: {count} messages\n"
            
        return summary
    
    def clear_memory(self):
        """Clear all conversation memory."""
        # Keep system messages
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages
        logger.info(f"Memory cleared, kept {len(system_messages)} system messages")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary for serialization.
        
        Returns:
            Dictionary representation of memory
        """
        return {
            "messages": self.messages,
            "next_message_id": self.next_message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: AdaptiveContextConfig) -> 'ConversationMemory':
        """
        Create memory from dictionary.
        
        Args:
            data: Dictionary data
            config: Configuration
            
        Returns:
            ConversationMemory instance
        """
        memory = cls(config)
        memory.messages = data.get("messages", [])
        memory.next_message_id = data.get("next_message_id", 1)
        return memory 