"""
Adapter for benchmarking Qdrant.
"""
from typing import Dict, Any, List, Optional
import time
import os
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chat_models import ChatOllama
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain_community.retrievers.qdrant import QdrantRetriever
    from langchain_community.schema import Document
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant not installed. Qdrant adapter will not be available.")

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
            raise ImportError("Qdrant is not installed. Please install it to use this adapter.")
        
        # Initialize components
        self.llm = ChatOllama(model=model, temperature=0.1)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Qdrant client (local, in-memory)
        self.client = QdrantClient(":memory:")
        
        try:
            # Create collection for documents
            self.collection_name = "benchmark_docs"
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embeddings.client.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            
            # Initialize retriever
            self.retriever = QdrantRetriever(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error initializing Qdrant collection: {e}")
            # Create a fallback retriever that doesn't do anything
            self.retriever = None
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize chain
        if self.retriever:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory
            )
        else:
            # Create a conversation chain without retrieval
            from langchain.chains import ConversationChain
            self.chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=verbose
            )
        
        # Track documents and time
        self.documents = []
        self.conversation_start_time = time.time()
        
    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Reinitialize chain with new memory
        if self.retriever:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory
            )
        else:
            # Create a conversation chain without retrieval
            from langchain.chains import ConversationChain
            self.chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=self.verbose
            )
        
        self.conversation_start_time = time.time()
        if self.verbose:
            print("Qdrant: Conversation initialized")
    
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
            print(f"Qdrant: Added {role} message: {content[:30]}...")
    
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        # Add any context documents to Qdrant
        if context and self.retriever:
            try:
                for i, doc_text in enumerate(context):
                    # Generate document ID
                    doc_id = str(uuid.uuid4())
                    
                    # Get embedding for document
                    embedding = self.embeddings.embed_query(doc_text)
                    
                    # Add to Qdrant
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            models.PointStruct(
                                id=doc_id,
                                vector=embedding,
                                payload={"text": doc_text, "source": f"benchmark_context_{i}"}
                            )
                        ]
                    )
                    
                    # Track document
                    self.documents.append(doc_text)
                    
                    if self.verbose:
                        print(f"Qdrant: Added context document {i}: {doc_text[:30]}...")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error adding documents to Qdrant: {e}")
        
        # Get response
        try:
            if isinstance(self.chain, ConversationalRetrievalChain):
                result = self.chain({"question": query})
                response_text = result.get("answer", "")
            else:
                response_text = self.chain.predict(input=query)
        except Exception as e:
            if self.verbose:
                print(f"Error getting response: {e}")
            response_text = f"I encountered an error retrieving information: {str(e)}"
        
        if self.verbose:
            print(f"Qdrant: Query: {query}")
            print(f"Qdrant: Response: {response_text[:50]}...")
        
        return response_text
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Get collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vector_count = collection_info.points_count
        except:
            vector_count = 0
        
        # Estimate token counts
        chat_history = self.memory.chat_memory.messages
        chat_text = " ".join([msg.content for msg in chat_history])
        chat_tokens = len(chat_text.split()) * 1.3  # Rough approximation
        
        # Document tokens
        doc_tokens = sum(len(doc.split()) * 1.3 for doc in self.documents)
        
        stats = {
            "total_tokens": int(chat_tokens + doc_tokens),
            "chat_tokens": int(chat_tokens),
            "document_tokens": int(doc_tokens),
            "document_count": len(self.documents),
            "vector_count": vector_count,
            "conversation_duration": time.time() - self.conversation_start_time
        }
        
        if self.verbose:
            print(f"Qdrant: Memory stats: {stats}")
            
        return stats
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        self.initialize_conversation()
        if self.verbose:
            print("Qdrant: Memory flushed") 