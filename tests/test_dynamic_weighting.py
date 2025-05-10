import pytest
import time
from unittest.mock import MagicMock, patch
from adaptive_context.dynamic_weighting import DynamicWeightingEngine
from adaptive_context.config import AdaptiveContextConfig

class TestDynamicWeightingEngine:
    """Tests for the DynamicWeightingEngine class"""
    
    def test_init(self):
        """Test initialization of DynamicWeightingEngine"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000,
            use_dynamic_weighting=True,
            dynamic_weighting_learning_rate=0.1
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Check initialization values
        assert engine.total_token_budget == 6000  # Sum of all tier limits
        assert engine.default_ratios["active"] == 0.25
        assert engine.default_ratios["working"] == 0.35
        assert engine.default_ratios["archive"] == 0.40
        assert engine.learning_rate == 0.1
        assert engine.current_tier_weights == engine.default_ratios
        
    def test_analyze_query_complexity(self):
        """Test query complexity analysis"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Test various query types
        simple_query = "What time is it?"
        medium_query = "Can you explain how photosynthesis works?"
        complex_query = "What is the relationship between quantum mechanics and general relativity?"
        code_query = "Write a Python function to implement quicksort algorithm with detailed comments."
        multi_part_query = "What is the capital of France? Also, what's the population? And when was the Eiffel Tower built?"
        
        # Check relative complexity scores
        simple_score = engine.analyze_query_complexity(simple_query)
        medium_score = engine.analyze_query_complexity(medium_query)
        complex_score = engine.analyze_query_complexity(complex_query)
        code_score = engine.analyze_query_complexity(code_query)
        multi_part_score = engine.analyze_query_complexity(multi_part_query)
        
        # Simple query should have the lowest score
        assert simple_score < medium_score
        assert simple_score < complex_score
        assert simple_score < code_score
        
        # Complex and code queries should have higher scores
        assert complex_score > medium_score
        assert code_score > medium_score
        
        # Multi-part should have a higher score due to multiple questions
        assert multi_part_score > simple_score
        
        # Empty query edge case
        assert engine.analyze_query_complexity("") == 0.0
        
    def test_analyze_document_type(self):
        """Test document type analysis"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Test various document types
        code_content = """
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)
                
        class Calculator:
            def __init__(self):
                self.value = 0
                
            def add(self, x):
                self.value += x
                return self
        """
        
        data_content = """
        {
            "users": [
                {"id": 1, "name": "Alice", "age": 28},
                {"id": 2, "name": "Bob", "age": 35},
                {"id": 3, "name": "Charlie", "age": 42}
            ],
            "settings": {
                "darkMode": true,
                "notifications": false
            }
        }
        """
        
        text_content = """
        The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet.
        Science is the systematic study of the structure and behaviour of the physical and natural world through
        observation and experimentation. Philosophy is the study of general and fundamental questions about
        existence, knowledge, values, reason, mind, and language.
        """
        
        mixed_content = """
        # Data Analysis Report
        
        ## Introduction
        This report analyzes user data from our application.
        
        ## Code Used
        ```python
        import pandas as pd
        
        df = pd.read_csv('users.csv')
        avg_age = df['age'].mean()
        print(f"Average age: {avg_age}")
        ```
        
        ## Results
        The analysis revealed the following JSON structure:
        ```json
        {"average_age": 34.5, "active_users": 2341}
        ```
        """
        
        # Check document type detection
        assert engine.analyze_document_type(code_content) == "code"
        assert engine.analyze_document_type(data_content) == "data"
        assert engine.analyze_document_type(text_content) == "text"
        assert engine.analyze_document_type(mixed_content) == "mixed"
        
        # Empty content edge case
        assert engine.analyze_document_type("") == "text"  # Default is text
        assert engine.analyze_document_type(None) == "text"  # None should default to text
        
    def test_calculate_optimal_weights(self):
        """Test optimal weight calculation"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Test various combinations of complexity and document type
        weights_simple_text = engine.calculate_optimal_weights(0.2, "text")
        weights_complex_text = engine.calculate_optimal_weights(0.8, "text")
        weights_medium_code = engine.calculate_optimal_weights(0.5, "code")
        weights_medium_data = engine.calculate_optimal_weights(0.5, "data")
        
        # Simple text should have less active memory
        assert weights_simple_text["active"] < engine.default_ratios["active"]
        
        # Complex text should have more active memory
        assert weights_complex_text["active"] > engine.default_ratios["active"]
        
        # Code should have more active memory compared to data
        assert weights_medium_code["active"] > weights_medium_data["active"]
        
        # Data should have more working memory
        assert weights_medium_data["working"] > engine.default_ratios["working"]
        
        # Weights should sum to 1.0
        assert abs(sum(weights_simple_text.values()) - 1.0) < 0.00001
        assert abs(sum(weights_complex_text.values()) - 1.0) < 0.00001
        assert abs(sum(weights_medium_code.values()) - 1.0) < 0.00001
        assert abs(sum(weights_medium_data.values()) - 1.0) < 0.00001
        
    def test_update_tier_allocations(self):
        """Test updating tier allocations"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Initial allocation should match default weights
        initial_limits = engine.update_tier_allocations()
        assert initial_limits["active"] == 1500  # 25% of 6000
        assert initial_limits["working"] == 2100  # 35% of 6000
        assert initial_limits["archive"] == 2400  # 40% of 6000
        
        # Modify weights and check new allocations
        engine.current_tier_weights = {
            "active": 0.4,    # Increased
            "working": 0.4,   # Increased
            "archive": 0.2    # Decreased
        }
        
        new_limits = engine.update_tier_allocations()
        assert new_limits["active"] == 2400  # 40% of 6000
        assert new_limits["working"] == 2400  # 40% of 6000
        assert new_limits["archive"] == 1200  # 20% of 6000
        
        # Test minimum tier size enforcement
        engine.current_tier_weights = {
            "active": 0.01,   # Very small
            "working": 0.01,  # Very small
            "archive": 0.98   # Almost everything
        }
        
        min_limits = engine.update_tier_allocations()
        min_tier_size = 1000  # Default minimum
        
        # Should enforce minimum tier size
        assert min_limits["active"] >= min_tier_size
        assert min_limits["working"] >= min_tier_size
        
    def test_process_query(self):
        """Test processing a query and updating allocations"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000,
            dynamic_weighting_learning_rate=0.5  # High learning rate for testing
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Initial weights
        initial_weights = engine.current_tier_weights.copy()
        
        # Process a simple query - should decrease active tier
        simple_context = "Just some simple text for testing."
        new_limits = engine.process_query("What time is it?", simple_context)
        
        # Check that weights were updated in the expected direction
        assert engine.current_tier_weights["active"] < initial_weights["active"]
        assert engine.current_tier_weights["archive"] > initial_weights["archive"]
        
        # Now process a complex query - should increase active tier
        initial_weights_2 = engine.current_tier_weights.copy()
        complex_context = """
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)
        """
        
        new_limits_2 = engine.process_query(
            "Explain how quantum computing differs from classical computing with detailed examples.", 
            complex_context
        )
        
        # Check that weights were updated in the expected direction
        assert engine.current_tier_weights["active"] > initial_weights_2["active"]
        
    def test_get_stats(self):
        """Test getting statistics"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Process a query to generate some stats
        engine.process_query("Test query")
        
        # Get stats
        stats = engine.get_stats()
        
        # Check stat structure
        assert "current_weights" in stats
        assert "current_limits" in stats
        assert "document_type_distribution" in stats
        assert "recent_query_complexity" in stats
        assert "total_token_budget" in stats
        assert stats["total_token_budget"] == 6000
        
    def test_reset_to_defaults(self):
        """Test resetting weights to defaults"""
        config = AdaptiveContextConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000,
            dynamic_weighting_learning_rate=0.5  # High learning rate for testing
        )
        
        engine = DynamicWeightingEngine(config)
        
        # Change weights by processing queries
        engine.process_query("What time is it?")
        engine.process_query("Explain quantum computing.")
        
        # Weights should be different from defaults now
        assert engine.current_tier_weights != engine.default_ratios
        
        # Reset to defaults
        engine.reset_to_defaults()
        
        # Should be back to default values
        assert engine.current_tier_weights == engine.default_ratios


@pytest.fixture
def mock_engine():
    """Create a mock DynamicWeightingEngine for testing with AdaptiveContextManager"""
    engine = MagicMock()
    engine.process_query.return_value = {
        "active": 1500,
        "working": 2500,
        "archive": 2000
    }
    engine.get_stats.return_value = {
        "current_weights": {"active": 0.25, "working": 0.35, "archive": 0.4},
        "current_limits": {"active": 1500, "working": 2500, "archive": 2000},
        "enabled": True
    }
    engine.current_tier_limits = {
        "active": 1500,
        "working": 2500,
        "archive": 2000
    }
    return engine


@patch("adaptive_context.dynamic_weighting.DynamicWeightingEngine")
def test_manager_integration(mock_engine_class, mock_engine):
    """Test integration with AdaptiveContextManager"""
    from adaptive_context.manager import AdaptiveContextManager
    
    # Set up the mock engine
    mock_engine_class.return_value = mock_engine
    
    # Create configuration
    config = AdaptiveContextConfig(
        active_token_limit=1000,
        working_token_limit=2000,
        archive_token_limit=3000,
        use_dynamic_weighting=True
    )
    
    # Create manager (should initialize engine)
    manager = AdaptiveContextManager(config)
    
    # Add a message (should process query)
    manager.add_message("user", "Test query")
    
    # Verify engine was used
    mock_engine.process_query.assert_called_once()
    args, kwargs = mock_engine.process_query.call_args
    assert args[0] == "Test query"  # First arg should be the query
    
    # Check stats method
    stats = manager.get_dynamic_weighting_stats()
    assert stats["enabled"] == True
    mock_engine.get_stats.assert_called_once()
    
    # Test reset method
    manager.reset_dynamic_weighting()
    mock_engine.reset_to_defaults.assert_called_once() 