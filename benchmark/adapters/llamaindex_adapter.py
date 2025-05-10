"""
Adapter for benchmarking LlamaIndex.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid

try:
    from llama_index.core import (
        Settings,
        VectorStoreIndex,
        Document,
        SimpleDirectoryReader,
        StorageContext,
    )
    from llama_index.llms.ollama import Ollama
    from llama_index.core.memory import ChatMemoryBuffer
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("Warning: LlamaIndex not installed. LlamaIndex adapter will not be available.")

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
        
        # Initialize LlamaIndex components
        self.llm = Ollama(model=model, request_timeout=120)
        
        # Configure settings
        Settings.llm = self.llm
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 20
        
        # Memory for chat history
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=10000
        )
        
        # Storage for documents
        self.storage_context = StorageContext.from_defaults()
        self.documents = []
        self.index = None
        self.conversation_start_time = time.time()
        
        # Initialize empty index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the vector store index with current documents."""
        if self.documents:
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=self.storage_context,
            )
        else:
            # Create an empty index if no documents
            self.index = VectorStoreIndex(
                [],
                storage_context=self.storage_context,
            )
        
        if self.verbose:
            print(f"LlamaIndex: Index initialized with {len(self.documents)} documents")

    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=10000
        )
        self.conversation_start_time = time.time()
        if self.verbose:
            print("LlamaIndex: Conversation initialized")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        if role == "user":
            self.memory.put("user", content)
        elif role == "assistant":
            self.memory.put("assistant", content)
        else:
            raise ValueError(f"Unsupported role: {role}")
            
        if self.verbose:
            print(f"LlamaIndex: Added {role} message: {content[:30]}...")
    
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        # Add any context documents to the index
        if context:
            for i, doc in enumerate(context):
                doc_id = f"benchmark_context_{i}_{uuid.uuid4()}"
                document = Document(text=doc, id=doc_id)
                self.documents.append(document)
                if self.verbose:
                    print(f"LlamaIndex: Added context document {i}: {doc[:30]}...")
            
            # Reinitialize the index with new documents
            self._initialize_index()
        
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Approximate token counts
        chat_tokens = self.memory.get_stats().get("token_count", 0)
        doc_tokens = sum(len(doc.text.split()) * 1.3 for doc in self.documents)  # Rough approximation
        
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
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        self.initialize_conversation()
        if self.verbose:
            print("LlamaIndex: Memory flushed") 