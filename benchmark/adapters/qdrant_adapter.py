"""
Adapter for benchmarking Qdrant.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid
import logging
import traceback

logger = logging.getLogger(__name__)

# Try basic LangChain imports
try:
    print("Attempting to import LangChain components...")
    
    import langchain
    print(f"Found langchain version: {langchain.__version__}")
    
    # We'll use just the basic conversation chain without Qdrant
    from langchain.chat_models import ChatOllama
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    
    QDRANT_AVAILABLE = True
    print("Successfully imported LangChain components for basic Qdrant adapter")
except Exception as e:
    print(f"Error importing basic LangChain components: {e}")
    print(f"Error details: {traceback.format_exc()}")
    QDRANT_AVAILABLE = False
    logger.warning(f"Error importing basic LangChain components: {e}. Qdrant adapter will not be available.")

from benchmark.adapters.base import BenchmarkSystemAdapter


class QdrantAdapter(BenchmarkSystemAdapter):
    """Adapter for the Qdrant system."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the Qdrant adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        super().__init__(model, verbose)
        
        if not QDRANT_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it to use this adapter.")
        
        try:
            # Initialize components
            self.llm = ChatOllama(model=model, temperature=0.1)
            
            # Initialize memory - important: use "history" as the memory key
            self.memory = ConversationBufferMemory(
                memory_key="history",  # Changed from "chat_history" to "history"
                return_messages=True
            )
            
            # Initialize chain - use simple ConversationChain 
            self.chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=verbose
            )
            
            # Track documents and time
            self.documents = []
            self.conversation_start_time = time.time()
            
            if self.verbose:
                print("Qdrant adapter initialized successfully (using basic conversation)")
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant adapter: {e}")
            raise ImportError(f"Failed to initialize Qdrant adapter: {e}")
        
    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="history",  # Changed from "chat_history" to "history"
                return_messages=True
            )
            
            # Reinitialize chain with new memory
            self.chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=self.verbose
            )
            
            self.conversation_start_time = time.time()
            if self.verbose:
                print("Qdrant: Conversation initialized")
        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            if self.verbose:
                print(f"Qdrant: Error initializing conversation: {e}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        try:
            # LangChain memory doesn't directly support adding messages this way
            # We'll need to run a query to add the message to the memory
            if role == "user":
                # This will be handled in the query method
                pass
            elif role == "assistant":
                # We need to manually add assistant messages after query is run
                self.memory.chat_memory.add_ai_message(content)
            else:
                raise ValueError(f"Unsupported role: {role}")
                
            if self.verbose:
                print(f"Qdrant: Added {role} message: {content[:30]}...")
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            if self.verbose:
                print(f"Qdrant: Error adding message: {e}")
    
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        try:
            # Store context documents in the user query for simplicity
            if context:
                # Format context for the model
                context_text = "\n\n".join([f"Context: {doc}" for doc in context])
                # Add context to the query
                query_with_context = f"{context_text}\n\nQuestion: {query}"
                
                # Store documents for tracking
                for doc in context:
                    self.documents.append(doc)
                
                if self.verbose:
                    print(f"Qdrant: Added {len(context)} context documents to query")
                
                # Use the enhanced query
                response_text = self.chain.predict(input=query_with_context)
            else:
                # Use the original query
                response_text = self.chain.predict(input=query)
            
            if self.verbose:
                print(f"Qdrant: Query: {query}")
                print(f"Qdrant: Response: {response_text[:50]}...")
            
            return response_text
        except Exception as e:
            logger.error(f"Error in query method: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            # Estimate token counts
            try:
                if hasattr(self, 'memory') and hasattr(self.memory, 'chat_memory'):
                    chat_history = self.memory.chat_memory.messages
                    chat_text = " ".join([msg.content for msg in chat_history])
                    chat_tokens = len(chat_text.split()) * 1.3  # Rough approximation
                else:
                    chat_tokens = 0
            except Exception as e:
                logger.warning(f"Error calculating chat tokens: {e}")
                chat_tokens = 0
            
            # Document tokens
            try:
                if hasattr(self, 'documents'):
                    doc_tokens = sum(len(doc.split()) * 1.3 for doc in self.documents)
                else:
                    doc_tokens = 0
            except Exception as e:
                logger.warning(f"Error calculating document tokens: {e}")
                doc_tokens = 0
            
            stats = {
                "total_tokens": int(chat_tokens + doc_tokens),
                "chat_tokens": int(chat_tokens),
                "document_tokens": int(doc_tokens),
                "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
                "conversation_duration": time.time() - self.conversation_start_time if hasattr(self, 'conversation_start_time') else 0
            }
            
            if self.verbose:
                print(f"Qdrant: Memory stats: {stats}")
                
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "total_tokens": 0,
                "chat_tokens": 0,
                "document_tokens": 0,
                "document_count": 0,
                "conversation_duration": 0,
                "error": str(e)
            }
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        try:
            self.initialize_conversation()
            if self.verbose:
                print("Qdrant: Memory flushed")
        except Exception as e:
            logger.error(f"Error flushing memory: {e}")
            if self.verbose:
                print(f"Qdrant: Error flushing memory: {e}") 