import os
from typing import Optional

class AdaptiveContextConfig:
    """Configuration for the AdaptiveContext system."""
    
    def __init__(self, 
                 active_tier_tokens=2000,
                 working_tier_tokens=4000, 
                 archive_tier_tokens=6000,
                 use_ml=False,
                 use_llm_classification=True,
                 ml_model_path=None,
                 rule_weight=0.5,
                 ml_weight=0.3,
                 llm_weight=0.7,
                 max_llm_classification_length=250,
                 compression_threshold=0.8,
                 knowledge_store_path=':memory:',
                 ollama_host='http://localhost:11434',
                 default_model='llama3',
                 vector_embedding_model='all-MiniLM-L6-v2',
                 use_vector_search=True,
                 use_bm25_search=True,
                 hybrid_search_alpha=0.7,
                 use_reranking=True,
                 rerank_top_k=20):
        """
        Initialize AdaptiveContext configuration.
        
        Args:
            active_tier_tokens: Maximum tokens in active memory tier
            working_tier_tokens: Maximum tokens in working memory tier
            archive_tier_tokens: Maximum tokens in archive memory tier
            use_ml: Whether to use ML-based importance classifier
            use_llm_classification: Whether to use LLM for classification
            ml_model_path: Path to ML classifier model file
            rule_weight: Weight for rule-based classifier
            ml_weight: Weight for ML-based classifier
            llm_weight: Weight for LLM-based classifier
            max_llm_classification_length: Maximum text length for LLM classification
            compression_threshold: Tier fullness threshold to trigger compression
            knowledge_store_path: Path to knowledge store database
            ollama_host: Ollama API host URL
            default_model: Default Ollama model to use
            vector_embedding_model: Model to use for vector embeddings
            use_vector_search: Whether to use vector-based search
            use_bm25_search: Whether to use BM25 keyword search
            hybrid_search_alpha: Weight for vector search in hybrid search (0-1)
            use_reranking: Whether to use result re-ranking
            rerank_top_k: Number of candidates to consider for re-ranking
        """
        # Memory tier settings
        self.active_tier_tokens = active_tier_tokens
        self.working_tier_tokens = working_tier_tokens
        self.archive_tier_tokens = archive_tier_tokens
        
        # Classification settings
        self.use_ml = use_ml
        self.use_llm_classification = use_llm_classification
        self.ml_model_path = ml_model_path
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.llm_weight = llm_weight
        self.max_llm_classification_length = max_llm_classification_length
        
        # Compression settings
        self.compression_threshold = compression_threshold
        
        # Knowledge store settings
        self.knowledge_store_path = knowledge_store_path
        self.vector_embedding_model = vector_embedding_model
        self.use_vector_search = use_vector_search
        
        # Ollama settings
        self.ollama_host = ollama_host
        self.default_model = default_model
        
        # Advanced retrieval settings
        self.use_bm25_search = use_bm25_search
        self.hybrid_search_alpha = hybrid_search_alpha
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        
        # Calculate weights for importance scoring
        self.weights = [self.rule_weight]
        if self.use_ml:
            self.weights.append(self.ml_weight)
        if self.use_llm_classification:
            self.weights.append(self.llm_weight) 