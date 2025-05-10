"""
Adapter for benchmarking Activeloop DeepMemory.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid

try:
    import deeplake
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chat_models import ChatOllama
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationSummaryMemory
    from langchain_community.vectorstores import DeepLake
    ACTIVELOOP_AVAILABLE = True
except ImportError:
    ACTIVELOOP_AVAILABLE = False
    print("Warning: Activeloop DeepMemory not installed. DeepMemory adapter will not be available.")

from benchmark.adapters.base import BenchmarkSystemAdapter


class ActiveloopAdapter(BenchmarkSystemAdapter):
    """Adapter for the Activeloop DeepMemory system."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the Activeloop DeepMemory adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        super().__init__(model, verbose)
        
        if not ACTIVELOOP_AVAILABLE:
            raise ImportError("Activeloop DeepMemory is not installed. Please install it to use this adapter.")
        
        # Initialize components
        self.llm = ChatOllama(model=model, temperature=0.1)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create temp directory for vector store
        os.makedirs("./temp_deep_memory", exist_ok=True)
        
        # Initialize vector store
        self.vector_store_path = "./temp_deep_memory/benchmark_vector_store"
        
        # Create LangChain DeepLake vector store
        self.vector_store = DeepLake(
            dataset_path=self.vector_store_path,
            embedding_function=self.embeddings,
            read_only=False
        )
        
        # Initialize memory
        self.memory = ConversationSummaryMemory(llm=self.llm)
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=verbose
        )
        
        # Track documents and time
        self.documents = []
        self.conversation_start_time = time.time()
    
    def _add_documents_to_store(self, documents):
        """Add documents to the vector store."""
        # Using LangChain's DeepLake add_texts method
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            texts.append(doc)
            metadatas.append({
                "source": f"benchmark_context_{i}",
                "id": f"doc_{uuid.uuid4()}"
            })
        
        if texts:
            self.vector_store.add_texts(texts, metadatas=metadatas)
    
    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=self.verbose
        )
        self.conversation_start_time = time.time()
        if self.verbose:
            print("Activeloop: Conversation initialized")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        if role == "user":
            # User messages will be added during query
            pass
        elif role == "assistant":
            # Store in memory - need to simulate the conversation history
            self.memory.save_context({"input": "temp_input"}, {"output": content})
        else:
            raise ValueError(f"Unsupported role: {role}")
            
        if self.verbose:
            print(f"Activeloop: Added {role} message: {content[:30]}...")
    
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        # Add any context documents to the vector store
        if context:
            for i, doc in enumerate(context):
                self.documents.append(doc)
                if self.verbose:
                    print(f"Activeloop: Added context document {i}: {doc[:30]}...")
            
            # Add all documents to the vector store
            self._add_documents_to_store(context)
        
        # Search for relevant documents
        relevant_docs = []
        if self.documents:
            # Use similarity search from DeepLake
            results = self.vector_store.similarity_search(query, k=3)
            relevant_docs = [doc.page_content for doc in results]
        
        # Create prompt with context
        context_prompt = ""
        if relevant_docs:
            context_prompt = f"Based on the following information:\n{' '.join(relevant_docs)}\n"
        
        # Run query through conversation chain
        full_query = context_prompt + query if context_prompt else query
        response = self.conversation.predict(input=full_query)
        
        if self.verbose:
            print(f"Activeloop: Query: {query}")
            print(f"Activeloop: Response: {response[:50]}...")
        
        return response
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Try to get vector store size 
        try:
            vector_store_size = len(self.documents)
        except:
            vector_store_size = 0
        
        # Approximate token counts
        chat_tokens = 0
        if hasattr(self.memory, "buffer") and self.memory.buffer:
            chat_tokens = len(self.memory.buffer.split()) * 1.3  # Rough approximation
        
        # Document tokens
        doc_tokens = sum(len(doc.split()) * 1.3 for doc in self.documents)
        
        stats = {
            "total_tokens": int(chat_tokens + doc_tokens),
            "chat_tokens": int(chat_tokens),
            "document_tokens": int(doc_tokens),
            "document_count": len(self.documents),
            "vector_store_size": vector_store_size,
            "conversation_duration": time.time() - self.conversation_start_time
        }
        
        if self.verbose:
            print(f"Activeloop: Memory stats: {stats}")
            
        return stats
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        self.initialize_conversation()
        if self.verbose:
            print("Activeloop: Memory flushed")
    
    def __del__(self):
        """Clean up resources."""
        # No explicit cleanup needed for DeepLake via langchain 