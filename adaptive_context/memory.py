import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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