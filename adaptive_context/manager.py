import time
import json
import logging
import requests
import traceback
from typing import List, Dict, Any, Optional, Union

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.memory import (
    ContextSegment, 
    MemoryTier, 
    ActiveTier, 
    WorkingTier, 
    ArchiveTier,
    ConversationMemory
)
from adaptive_context.classifier import ImportanceClassifier, ContentClassifier
from adaptive_context.compressor import ContextCompressor
from adaptive_context.knowledge import KnowledgeStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('adaptive_context')

def configure_logging(verbose: bool = False):
    """Configure logging for the adaptive_context module."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

class AdaptiveContextManager:
    """
    Main manager class for AdaptiveContext system.
    Coordinates between components for memory, knowledge, and external integrations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the AdaptiveContextManager with provided configuration.
        
        Args:
            config: Configuration for the system, if None, default config is used
        """
        from .config import AdaptiveContextConfig
        
        # Initialize config if not provided
        self.config = config if config is not None else AdaptiveContextConfig()
        
        # Set up logging
        verbose = self.config.verbose_logging if hasattr(self.config, "verbose_logging") else False
        configure_logging(verbose)
        
        # Log initialization
        logger.info(f"Initializing AdaptiveContextManager with {self.config.active_token_limit} active tokens, "
                   f"{self.config.working_token_limit} working tokens, {self.config.archive_token_limit} archive tokens")
        
        try:
            # Initialize components
            self.knowledge_store = KnowledgeStore(self.config)
            self.memory = ConversationMemory(self.config)
            
            # Initialize content classifier if enabled
            if hasattr(self.config, "use_ml_classifier") and self.config.use_ml_classifier:
                self.classifier = ContentClassifier(self.config)
            else:
                self.classifier = None
                
            logger.info("AdaptiveContextManager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AdaptiveContextManager: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role (e.g., user, assistant, system)
            content: Message content
            metadata: Optional metadata for the message
            
        Returns:
            Message object that was added
        """
        message = self.memory.add_message(role, content, metadata)
        
        # Perform classification only for user messages
        if role == "user" and self.classifier is not None:
            try:
                # Classify content
                result = self.classifier.classify(content)
                message["classification"] = result
                logger.debug(f"Message classified as: {result}")
            except Exception as e:
                logger.error(f"Error classifying message: {e}")
                
        return message
    
    def get_conversation_context(self, max_tokens: int = None) -> Dict[str, Any]:
        """
        Get the full conversation context for generating a response.
        
        Args:
            max_tokens: Maximum tokens for the context
            
        Returns:
            Context with messages and knowledge
        """
        context = {
            "messages": self.memory.get_context_messages(),
            "knowledge": []
        }
        
        # Get the most recent user message for knowledge retrieval
        user_messages = self.memory.get_messages_by_role("user")
        if user_messages:
            last_user_message = user_messages[-1]["content"]
            
            # Retrieve relevant knowledge for the user's message
            knowledge_items = self.knowledge_store.get_relevant_knowledge(last_user_message)
            context["knowledge"] = knowledge_items
        
        return context
    
    def generate_response(self, prompt: str = None, model: str = None) -> str:
        """
        Generate a response using the conversation context.
        
        Args:
            prompt: Optional prompt to use instead of the conversation context
            model: Model to use for generation
            
        Returns:
            Generated response
        """
        try:
            import requests
            
            # Use model from config if not specified
            if model is None:
                model = self.config.default_model
                
            # Get conversation context if no prompt provided
            if prompt is None:
                context = self.get_conversation_context()
                
                # Extract messages
                messages = context["messages"]
                
                # Add knowledge as system message if available
                knowledge = context.get("knowledge", [])
                if knowledge:
                    knowledge_text = "\n".join(item["text"] for item in knowledge)
                    
                    # Add knowledge context as a system message
                    messages = [{"role": "system", "content": f"Use this knowledge to answer the question:\n{knowledge_text}"}] + messages
                
                # Format as prompt if needed
                if not messages:
                    prompt = "Hello! How can I assist you today?"
            else:
                messages = [{"role": "user", "content": prompt}]
                
            # Get Ollama URL
            ollama_url = f"{self.config.ollama_host}/api/chat"
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            logger.debug(f"Sending request to Ollama: {ollama_url}")
            response = requests.post(ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["message"]["content"]
                
                # Add the response to memory
                self.add_message("assistant", generated_text)
                
                return generated_text
            else:
                error_message = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_message)
                return f"Error generating response: {error_message}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def remember_knowledge(self, text: str, source: str = None) -> List[int]:
        """
        Explicitly remember knowledge from text.
        
        Args:
            text: Text to remember
            source: Source of the knowledge
            
        Returns:
            List of IDs for the stored knowledge
        """
        # Store the knowledge
        fact_ids = self.knowledge_store.remember_explicit(text, source=source)
        
        return fact_ids
    
    def get_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant knowledge items
        """
        return self.knowledge_store.get_relevant_knowledge(query)
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear_memory()
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'knowledge_store'):
                self.knowledge_store.close()
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
            
    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.close() 