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
                 default_model='llama3'):
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
        
        # Ollama settings
        self.ollama_host = ollama_host
        self.default_model = default_model
        
        # Calculate weights for importance scoring
        self.weights = [self.rule_weight]
        if self.use_ml:
            self.weights.append(self.ml_weight)
        if self.use_llm_classification:
            self.weights.append(self.llm_weight) 