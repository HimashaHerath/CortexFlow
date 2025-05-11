"""
Adapter for benchmarking LlamaIndex.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid
import logging
import traceback

logger = logging.getLogger(__name__)

try:
    print("Attempting to import LlamaIndex components...")
    # First, try to check which version of LlamaIndex is installed
    try:
        import llama_index
        print(f"Found llama_index version: {llama_index.__version__}")
        
        # Import appropriate components for version 0.9.x
        from llama_index import (
            VectorStoreIndex,
            Document,
            StorageContext,
            ServiceContext,
            LLMPredictor,
        )
        from llama_index.memory import ChatMemoryBuffer
        
        # Import langchain components for LLM
        from langchain.llms import Ollama
        
        LLAMAINDEX_AVAILABLE = True
        print("Successfully imported LlamaIndex components")
        
    except ImportError as e:
        print(f"Error importing LlamaIndex: {e}")
        print(f"Import error details: {traceback.format_exc()}")
        LLAMAINDEX_AVAILABLE = False
        
except Exception as e:
    print(f"Unexpected error while importing LlamaIndex: {e}")
    print(f"Error details: {traceback.format_exc()}")
    LLAMAINDEX_AVAILABLE = False
    logger.warning(f"LlamaIndex import error: {e}. LlamaIndex adapter will not be available.")

from benchmark.adapters.base import BenchmarkSystemAdapter


class LlamaIndexAdapter(BenchmarkSystemAdapter):
    """Adapter for the LlamaIndex system."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the LlamaIndex adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        super().__init__(model, verbose)
        
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex is not installed. Please install it to use this adapter.")
        
        try:
            # Initialize LlamaIndex components
            self.model = model
            self.verbose = verbose
            
            # Create LangChain Ollama LLM
            try:
                langchain_llm = Ollama(model=model)
                llm_predictor = LLMPredictor(llm=langchain_llm)
                if self.verbose:
                    print(f"Created LangChain Ollama LLM")
            except Exception as e:
                logger.error(f"Error creating Langchain Ollama: {e}")
                raise ImportError(f"Failed to create Ollama: {e}")
                
            # Configure service context
            try:
                self.service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
                if self.verbose:
                    print(f"Created ServiceContext with LLMPredictor")
            except Exception as e:
                logger.error(f"Error creating ServiceContext: {e}")
                raise ImportError(f"Failed to create ServiceContext: {e}")
                
            # Create storage context
            try:
                self.storage_context = StorageContext.from_defaults()
            except Exception as e:
                logger.error(f"Error creating StorageContext: {e}")
                raise ImportError(f"Failed to create StorageContext: {e}")
            
            # Create memory
            try:
                self.memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
            except Exception as e:
                logger.error(f"Error creating ChatMemoryBuffer: {e}")
                raise ImportError(f"Failed to create ChatMemoryBuffer: {e}")
            
            # Initialize document storage
            self.documents = []
            self.index = None
            self.conversation_start_time = time.time()
            
            # Initialize empty index
            self._initialize_index()
            
            if self.verbose:
                print("LlamaIndex adapter initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex adapter: {e}")
            raise ImportError(f"Failed to initialize LlamaIndex components: {e}")
        
    def _initialize_index(self):
        """Initialize the vector store index with current documents."""
        try:
            if self.documents:
                self.index = VectorStoreIndex.from_documents(
                    self.documents,
                    storage_context=self.storage_context,
                    service_context=self.service_context
                )
            else:
                # Create an empty index if no documents
                self.index = VectorStoreIndex(
                    [],
                    storage_context=self.storage_context,
                    service_context=self.service_context
                )
            
            if self.verbose:
                print(f"LlamaIndex: Index initialized with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            # Create a simple dummy index if there's an error
            self.index = None
            if self.verbose:
                print(f"LlamaIndex: Failed to initialize index. Error: {e}")

    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        try:
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=10000
            )
            self.conversation_start_time = time.time()
            if self.verbose:
                print("LlamaIndex: Conversation initialized")
        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            if self.verbose:
                print(f"LlamaIndex: Failed to initialize conversation. Error: {e}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        try:
            if role == "user":
                self.memory.put("user", content)
            elif role == "assistant":
                self.memory.put("assistant", content)
            else:
                raise ValueError(f"Unsupported role: {role}")
                
            if self.verbose:
                print(f"LlamaIndex: Added {role} message: {content[:30]}...")
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            if self.verbose:
                print(f"LlamaIndex: Failed to add message. Error: {e}")
    
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
            # Add any context documents to the index
            if context:
                new_documents = []
                for i, doc in enumerate(context):
                    doc_id = f"benchmark_context_{i}_{uuid.uuid4()}"
                    document = Document(text=doc, id=doc_id)
                    self.documents.append(document)
                    new_documents.append(document)
                    if self.verbose:
                        print(f"LlamaIndex: Added context document {i}: {doc[:30]}...")
                
                # Reinitialize the index with new documents
                self._initialize_index()
            
            # Check if index was successfully initialized
            if self.index is None:
                return f"Error: Index not initialized. Unable to process query: {query}"
            
            # Create query engine with chat memory
            query_engine = self.index.as_query_engine(
                memory=self.memory
            )
            
            # Get response
            response = query_engine.query(query)
            response_text = str(response)
            
            # Add to chat memory
            self.memory.put("user", query)
            self.memory.put("assistant", response_text)
            
            if self.verbose:
                print(f"LlamaIndex: Query: {query}")
                print(f"LlamaIndex: Response: {response_text[:50]}...")
            
            return response_text
        except Exception as e:
            logger.error(f"Error during query: {e}")
            error_message = f"Error processing query: {e}"
            if self.verbose:
                print(f"LlamaIndex: {error_message}")
            return error_message
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            # Approximate token counts
            try:
                chat_tokens = self.memory.get_stats().get("token_count", 0)
            except:
                # Fallback if get_stats() doesn't work as expected
                chat_tokens = sum(len(msg.split()) * 1.3 for msg in self.memory.chat_history) if hasattr(self.memory, 'chat_history') else 0
                
            doc_tokens = sum(len(doc.text.split()) * 1.3 for doc in self.documents) if self.documents else 0
            
            stats = {
                "total_tokens": int(chat_tokens + doc_tokens),
                "chat_tokens": int(chat_tokens),
                "document_tokens": int(doc_tokens),
                "document_count": len(self.documents),
                "conversation_duration": time.time() - self.conversation_start_time
            }
            
            if self.verbose:
                print(f"LlamaIndex: Memory stats: {stats}")
                
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            if self.verbose:
                print(f"LlamaIndex: Failed to get memory stats. Error: {e}")
            return {
                "total_tokens": 0,
                "chat_tokens": 0,
                "document_tokens": 0,
                "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
                "conversation_duration": time.time() - self.conversation_start_time if hasattr(self, 'conversation_start_time') else 0,
                "error": str(e)
            }
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        try:
            self.initialize_conversation()
            if self.verbose:
                print("LlamaIndex: Memory flushed")
        except Exception as e:
            logger.error(f"Error flushing memory: {e}")
            if self.verbose:
                print(f"LlamaIndex: Failed to flush memory. Error: {e}") 