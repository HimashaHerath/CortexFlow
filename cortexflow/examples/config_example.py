"""
Example demonstrating the CortexFlow configuration system usage.

This example shows how to use the refactored configuration system with the builder pattern.
"""

from cortexflow.config import ConfigBuilder, CortexFlowConfig

def basic_config_example():
    """Example of creating a basic configuration with defaults."""
    # Creating a configuration with all defaults
    config = CortexFlowConfig()
    print("Basic configuration with defaults:")
    config.log_config()
    print("\n")

def builder_example():
    """Example of using the builder pattern to create a configuration."""
    # Using the builder pattern to customize configuration
    config = (ConfigBuilder()
              .with_memory(
                  active_token_limit=8192, 
                  working_token_limit=16384
              )
              .with_knowledge_store(
                  knowledge_store_path="custom_db.sqlite",
                  retrieval_type="vector"
              )
              .with_llm(
                  default_model="gemma3:8b",
                  conversation_style="professional"
              )
              .with_debug(verbose_logging=True)
              .build())
    
    print("Configuration created with builder pattern:")
    config.log_config()
    print("\n")

def section_update_example():
    """Example of updating specific sections of a configuration."""
    # Start with default config
    config = CortexFlowConfig()
    
    # Update specific sections
    config.memory.active_token_limit = 10000
    config.llm.default_model = "gemma3:8b"
    config.graph_rag.use_graph_rag = True
    config.graph_rag.max_graph_hops = 5
    
    print("Configuration after section updates:")
    config.log_config()
    print("\n")

def configuration_serialization():
    """Example of serializing and deserializing a configuration."""
    # Create a configuration with some custom settings
    config = (ConfigBuilder()
              .with_memory(active_token_limit=8192)
              .with_graph_rag(use_graph_rag=True, max_graph_hops=4)
              .build())
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print("Configuration as dictionary:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    print("\n")
    
    # Create new configuration from dictionary
    new_config = CortexFlowConfig.from_dict(config_dict)
    print("Configuration recreated from dictionary:")
    new_config.log_config()

if __name__ == "__main__":
    print("CortexFlow Configuration Examples\n" + "="*30 + "\n")
    basic_config_example()
    builder_example()
    section_update_example()
    configuration_serialization() 