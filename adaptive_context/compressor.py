import re
import requests
from typing import List, Dict, Any, Optional, Tuple

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.memory import ContextSegment

class TruncationCompressor:
    """Simple truncation-based context compressor."""
    
    def compress(self, content: str, target_ratio: float) -> str:
        """
        Compress content by simple truncation.
        
        Args:
            content: The content to compress
            target_ratio: Target compression ratio (0.0-1.0)
            
        Returns:
            Compressed content
        """
        if not content:
            return ""
            
        # Calculate target length
        target_length = int(len(content) * target_ratio)
        if target_length >= len(content):
            return content
            
        # Simple truncation with ellipsis
        return content[:target_length] + "..."


class ExtractiveSummarizer:
    """Keyword-based extractive summarization."""
    
    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "with", "by", "about", "as", "of",
            "that", "this", "these", "those", "it", "they", "them", "their",
            "he", "she", "his", "her", "i", "you", "we", "my", "your", "our"
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Remove code blocks before processing
        text_without_code = re.sub(r'```[\s\S]*?```', '', text)
        
        # Tokenize and convert to lowercase
        words = re.findall(r'\b\w+\b', text_without_code.lower())
        
        # Filter out stop words and count frequencies
        word_freq = {}
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    
    def rank_sentences(self, text: str, keywords: List[str]) -> List[Tuple[str, float]]:
        """
        Rank sentences by keyword presence.
        
        Args:
            text: Input text
            keywords: List of important keywords
            
        Returns:
            List of (sentence, score) tuples
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Preserve code blocks and specific patterns as important
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        equations = re.findall(r'\$\$[\s\S]*?\$\$', text)
        
        # Score each sentence based on keyword presence
        scored_sentences = []
        for sentence in sentences:
            # Code blocks and equations get high scores
            if any(block in sentence for block in code_blocks) or any(eq in sentence for eq in equations):
                scored_sentences.append((sentence, 10.0))
                continue
                
            # Otherwise score based on keywords
            score = sum(1.0 for keyword in keywords if keyword.lower() in sentence.lower())
            scored_sentences.append((sentence, score))
        
        return scored_sentences
    
    def compress(self, content: str, target_ratio: float) -> str:
        """
        Compress content using extractive summarization.
        
        Args:
            content: The content to compress
            target_ratio: Target compression ratio (0.0-1.0)
            
        Returns:
            Compressed content
        """
        if not content or target_ratio >= 1.0:
            return content
            
        # Extract keywords
        keywords = self.extract_keywords(content)
        
        # Rank sentences
        scored_sentences = self.rank_sentences(content, keywords)
        
        # Sort by score (highest first)
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Select top sentences to meet target ratio
        target_length = int(len(content) * target_ratio)
        compressed = []
        current_length = 0
        
        for sentence, _ in sorted_sentences:
            if current_length + len(sentence) <= target_length:
                compressed.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        # Join and return
        if not compressed:
            # Fallback to truncation if no sentences were selected
            return TruncationCompressor().compress(content, target_ratio)
            
        return " ".join(compressed)


class LLMSummarizer:
    """LLM-based abstractive summarization."""
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize LLM summarizer.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.ollama_url = f"{config.ollama_host}/api/generate"
        self.model = config.default_model
    
    def compress(self, content: str, target_ratio: float) -> str:
        """
        Compress content using LLM-based summarization.
        
        Args:
            content: The content to compress
            target_ratio: Target compression ratio (0.0-1.0)
            
        Returns:
            Compressed content
        """
        if not content or target_ratio >= 1.0:
            return content
        
        # Calculate target token count (approximate)
        approx_tokens = len(content.split())
        target_tokens = int(approx_tokens * target_ratio)
        
        # Don't compress very short content
        if approx_tokens < 50:
            return content
        
        # Create prompt
        prompt = f"""
        Compress the following text to approximately {target_tokens} words while preserving key information.
        Maintain all crucial details, facts, names, and coded instructions.
        
        Text to compress:
        {content}
        
        Compressed version:
        """
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                compressed = result.get("response", "").strip()
                
                # Verify compression was achieved
                if len(compressed.split()) <= approx_tokens * 1.1:  # Allow 10% margin
                    return compressed
        except Exception as e:
            # Fall back to extractive summarization on error
            pass
        
        # Fallback to extractive summarization
        return ExtractiveSummarizer().compress(content, target_ratio)


class ContextCompressor:
    """Context compressor that manages different compression strategies."""
    
    def __init__(self, config: AdaptiveContextConfig):
        """
        Initialize context compressor.
        
        Args:
            config: AdaptiveContext configuration
        """
        self.config = config
        self.truncation = TruncationCompressor()
        self.extractive = ExtractiveSummarizer()
        self.abstractive = LLMSummarizer(config)
    
    def compress_segment(self, segment: ContextSegment, target_ratio: float) -> ContextSegment:
        """
        Compress a single context segment.
        
        Args:
            segment: The segment to compress
            target_ratio: Target compression ratio (0.0-1.0)
            
        Returns:
            Compressed segment (new instance)
        """
        content = segment.content
        
        # Preserve critical segments
        if segment.importance >= 8.0:
            # Minimal compression for very important content
            target_ratio = max(0.9, target_ratio)
        
        # Use appropriate compression method based on segment type and length
        compressed_content = ""
        if segment.segment_type == "code":
            # Use truncation for code to avoid breaking syntax
            compressed_content = self.truncation.compress(content, target_ratio)
        elif len(content.split()) < 30:
            # Use truncation for very short segments
            compressed_content = self.truncation.compress(content, target_ratio)
        elif segment.importance < 4.0:
            # Use more aggressive compression for unimportant content
            compressed_content = self.extractive.compress(content, target_ratio)
        else:
            # Use abstractive summarization for important content
            compressed_content = self.abstractive.compress(content, target_ratio)
        
        # Create new segment with compressed content
        # Estimate token count based on original ratio
        new_token_count = int(segment.token_count * (len(compressed_content) / len(content)))
        
        return ContextSegment(
            content=compressed_content,
            importance=segment.importance,
            timestamp=segment.timestamp,
            token_count=new_token_count,
            segment_type=segment.segment_type,
            metadata={**segment.metadata, "compressed": True, "original_length": len(content)}
        )
    
    def progressive_compress(self, segments: List[ContextSegment], target_token_count: int) -> List[ContextSegment]:
        """
        Progressively compress segments to meet target token count.
        
        Args:
            segments: List of segments to compress
            target_token_count: Target total token count
            
        Returns:
            Compressed segments
        """
        # If already under target, no compression needed
        current_token_count = sum(segment.token_count for segment in segments)
        if current_token_count <= target_token_count:
            return segments
        
        # Sort by importance (least important first)
        sorted_segments = sorted(segments, key=lambda s: (s.importance, -s.age))
        
        # Calculate compression needed
        compression_ratio = target_token_count / current_token_count
        
        # Compress segments progressively
        compressed_segments = []
        remaining_tokens = current_token_count
        
        for segment in sorted_segments:
            # Start with less important segments
            if remaining_tokens <= target_token_count:
                # We've reached our target, no need to compress further
                compressed_segments.append(segment)
                continue
            
            # Calculate compression ratio for this segment
            # Adjust based on importance - compress less important segments more
            segment_ratio = min(1.0, max(0.3, compression_ratio + (segment.importance / 20)))
            
            # More aggressive compression for older segments
            if segment.age > 3600:  # Older than 1 hour
                segment_ratio *= 0.8
            
            # Compress the segment
            compressed = self.compress_segment(segment, segment_ratio)
            compressed_segments.append(compressed)
            
            # Update remaining tokens
            token_reduction = segment.token_count - compressed.token_count
            remaining_tokens -= token_reduction
        
        # Sort back to original order for return
        return sorted(compressed_segments, key=lambda s: s.timestamp, reverse=True) 