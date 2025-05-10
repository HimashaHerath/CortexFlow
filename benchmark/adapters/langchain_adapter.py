"""
Adapter for benchmarking LangChain.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid

try:
    from langchain.chat_models import ChatOllama
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. LangChain adapter will not be available.")

from benchmark.adapters.base import BenchmarkSystemAdapter


class LangChainAdapter(BenchmarkSystemAdapter):
    """Adapter for the LangChain system."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the LangChain adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        super().__init__(model, verbose)
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it to use this adapter.")
        
        # Initialize LangChain components
        self.llm = ChatOllama(model=model, temperature=0.1)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize document storage
        self.documents = []
        self.vector_store = None
        self.conversation_start_time = time.time()
        
        # Create text splitter for documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Initialize empty vector store
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize the vector store with current documents."""
        if self.documents:
            # Create vector store from documents
            texts = [doc.page_content for doc in self.documents]
            # Generate unique IDs for each document to avoid duplication
            unique_ids = [f"doc_{uuid.uuid4()}" for _ in range(len(texts))]
            
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                ids=unique_ids
            )
        else:
            # Create an empty vector store with unique empty doc ID
            self.vector_store = Chroma.from_texts(
                texts=["Empty initial document"],
                embedding=self.embeddings,
                ids=[f"empty_doc_{uuid.uuid4()}"]
            )
        
        # Create retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
        
        if self.verbose:
            print(f"LangChain: Vector store initialized with {len(self.documents)} documents")

    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation_start_time = time.time()
        
        # Reinitialize chain with new memory
        if self.vector_store:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory
            )
        
        if self.verbose:
            print("LangChain: Conversation initialized")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
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
            print(f"LangChain: Added {role} message: {content[:30]}...")
    
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
            new_documents = []
            for i, doc_text in enumerate(context):
                doc = Document(
                    page_content=doc_text,
                    metadata={"id": f"benchmark_context_{i}_{uuid.uuid4()}"}
                )
                new_documents.append(doc)
                if self.verbose:
                    print(f"LangChain: Added context document {i}: {doc_text[:30]}...")
            
            # Add to document collection
            self.documents.extend(new_documents)
            
            # Update vector store
            self._initialize_vector_store()
        
        # Get response
        result = self.chain({"question": query})
        response_text = result.get("answer", "")
        
        if self.verbose:
            print(f"LangChain: Query: {query}")
            print(f"LangChain: Response: {response_text[:50]}...")
        
        return response_text
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Estimate token counts based on text length
        chat_history = self.memory.chat_memory.messages
        chat_text = " ".join([msg.content for msg in chat_history])
        chat_tokens = len(chat_text.split()) * 1.3  # Rough approximation
        
        # Document tokens
        doc_tokens = sum(len(doc.page_content.split()) * 1.3 for doc in self.documents)
        
        stats = {
            "total_tokens": int(chat_tokens + doc_tokens),
            "chat_tokens": int(chat_tokens),
            "document_tokens": int(doc_tokens),
            "document_count": len(self.documents),
            "conversation_duration": time.time() - self.conversation_start_time
        }
        
        if self.verbose:
            print(f"LangChain: Memory stats: {stats}")
            
        return stats
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        self.initialize_conversation()
        if self.verbose:
            print("LangChain: Memory flushed") 