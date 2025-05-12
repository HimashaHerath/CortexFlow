import time
import json
import logging
import requests
import traceback
from typing import List, Dict, Any, Optional, Union, Iterator

from adaptive_context.config import CortexFlowConfig
from adaptive_context.memory import (
    ContextSegment, 
    MemoryTier, 
    ActiveTier, 
    WorkingTier, 
    ArchiveTier,
    ConversationMemory
)
from adaptive_context.classifier import ImportanceClassifier, ContentClassifier
from adaptive_context.compressor import ContextCompressor
from adaptive_context.knowledge import KnowledgeStore

# Add import for Chain of Agents
try:
    from adaptive_context.agent_chain import AgentChainManager
    AGENT_CHAIN_ENABLED = True
except ImportError:
    AGENT_CHAIN_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("agent_chain module not found. Chain of Agents functionality will be disabled.")

# Add import for Self-Reflection
try:
    from adaptive_context.reflection import ReflectionEngine
    REFLECTION_ENABLED = True
except ImportError:
    REFLECTION_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("reflection module not found. Self-Reflection functionality will be disabled.")

# Add import for Dynamic Weighting
try:
    from adaptive_context.dynamic_weighting import DynamicWeightingEngine
    DYNAMIC_WEIGHTING_ENABLED = True
except ImportError:
    DYNAMIC_WEIGHTING_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("dynamic_weighting module not found. Dynamic Weighting functionality will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cortexflow')

def configure_logging(verbose: bool = False):
    """Configure logging for the adaptive_context module."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

class CortexFlowManager:
    """
    Main manager class for AdaptiveContext system.
    Coordinates between components for memory, knowledge, and external integrations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the CortexFlowManager with provided configuration.
        
        Args:
            config: Configuration for the system, if None, default config is used
        """
        from .config import CortexFlowConfig
        
        # Initialize config if not provided
        self.config = config if config is not None else CortexFlowConfig()
        
        # Set up logging
        verbose = self.config.verbose_logging if hasattr(self.config, "verbose_logging") else False
        configure_logging(verbose)
        
        # Log initialization
        logger.info(f"Initializing CortexFlowManager with {self.config.active_token_limit} active tokens, "
                   f"{self.config.working_token_limit} working tokens, {self.config.archive_token_limit} archive tokens")
        
        try:
            # Initialize components
            self.knowledge_store = KnowledgeStore(self.config)
            self.memory = ConversationMemory(self.config)
            
            # Initialize content classifier if enabled
            if hasattr(self.config, "use_ml_classifier") and self.config.use_ml_classifier:
                self.classifier = ContentClassifier(self.config)
            else:
                self.classifier = None

            # Initialize Agent Chain Manager if enabled
            self.agent_chain_manager = None
            if hasattr(self.config, "use_chain_of_agents") and self.config.use_chain_of_agents and AGENT_CHAIN_ENABLED:
                try:
                    self.agent_chain_manager = AgentChainManager(self.config, self.knowledge_store)
                    logger.info("Chain of Agents initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Chain of Agents: {e}")
            
            # Initialize Reflection Engine if enabled
            self.reflection_engine = None
            if hasattr(self.config, "use_self_reflection") and self.config.use_self_reflection and REFLECTION_ENABLED:
                try:
                    self.reflection_engine = ReflectionEngine(self.config, self.knowledge_store)
                    logger.info("Self-Reflection Engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Self-Reflection Engine: {e}")
                    
            # Initialize Dynamic Weighting Engine if enabled
            self.weighting_engine = None
            if hasattr(self.config, "use_dynamic_weighting") and self.config.use_dynamic_weighting and DYNAMIC_WEIGHTING_ENABLED:
                try:
                    self.weighting_engine = DynamicWeightingEngine(self.config)
                    logger.info("Dynamic Weighting Engine initialized successfully")
                    
                    # Apply initial dynamic weighting if enabled
                    if self.weighting_engine:
                        initial_limits = self.weighting_engine.update_tier_allocations()
                        # Update memory tier limits - will be implemented in the memory module
                        self._update_memory_tier_limits(initial_limits)
                except Exception as e:
                    logger.error(f"Failed to initialize Dynamic Weighting Engine: {e}")
                
            logger.info("CortexFlowManager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing CortexFlowManager: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _update_memory_tier_limits(self, new_limits: Dict[str, int]) -> None:
        """
        Update memory tier token limits based on dynamic weighting.
        
        Args:
            new_limits: Dictionary with new token limits for each tier
        """
        if not hasattr(self.memory, "update_tier_limits"):
            logger.warning("Memory module doesn't support dynamic tier limit updates")
            return
        
        try:
            self.memory.update_tier_limits(
                active_limit=new_limits.get("active", self.config.active_token_limit),
                working_limit=new_limits.get("working", self.config.working_token_limit),
                archive_limit=new_limits.get("archive", self.config.archive_token_limit)
            )
            logger.info(f"Updated memory tier limits: {new_limits}")
        except Exception as e:
            logger.error(f"Error updating memory tier limits: {e}")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role (e.g., user, assistant, system)
            content: Message content
            metadata: Optional metadata for the message
            
        Returns:
            Message object that was added
        """
        message = self.memory.add_message(role, content, metadata)
        
        # Perform classification only for user messages
        if role == "user" and self.classifier is not None:
            try:
                # Classify content
                result = self.classifier.classify(content)
                message["classification"] = result
                logger.debug(f"Message classified as: {result}")
            except Exception as e:
                logger.error(f"Error classifying message: {e}")
                
        # Apply dynamic weighting for user queries
        if role == "user" and self.weighting_engine and self.config.use_dynamic_weighting:
            try:
                # Get recent context for document type analysis
                context_messages = self.memory.get_context_messages()
                context_content = "\n".join([m.get("content", "") for m in context_messages[-5:]])
                
                # Process query and update memory tier allocation
                new_limits = self.weighting_engine.process_query(content, context_content)
                
                # Update memory tier limits
                self._update_memory_tier_limits(new_limits)
            except Exception as e:
                logger.error(f"Error in dynamic weighting: {e}")
                
        return message
    
    def get_conversation_context(self, max_tokens: int = None) -> Dict[str, Any]:
        """
        Get the full conversation context for generating a response.
        
        Args:
            max_tokens: Maximum tokens for the context
            
        Returns:
            Context with messages and knowledge
        """
        context = {
            "messages": self.memory.get_context_messages(),
            "knowledge": []
        }
        
        # Get the most recent user message for knowledge retrieval
        user_messages = self.memory.get_messages_by_role("user")
        if user_messages:
            last_user_message = user_messages[-1]["content"]
            
            # Retrieve relevant knowledge for the user's message
            knowledge_items = self.knowledge_store.get_relevant_knowledge(last_user_message)
            
            # Apply self-reflection to verify knowledge relevance if enabled
            if self.reflection_engine and hasattr(self.config, "use_self_reflection") and self.config.use_self_reflection:
                try:
                    knowledge_items = self.reflection_engine.verify_knowledge_relevance(
                        last_user_message, 
                        knowledge_items
                    )
                    logger.info(f"Knowledge relevance verification applied: {len(knowledge_items)} items retained")
                except Exception as e:
                    logger.error(f"Error in knowledge relevance verification: {e}")
                    
            context["knowledge"] = knowledge_items
        
        return context
    
    def get_dynamic_weighting_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dynamic weighting engine.
        
        Returns:
            Dictionary with dynamic weighting statistics or None if not enabled
        """
        if not self.weighting_engine or not self.config.use_dynamic_weighting:
            return {"enabled": False}
            
        try:
            stats = self.weighting_engine.get_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting dynamic weighting stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def reset_dynamic_weighting(self) -> None:
        """Reset dynamic weighting to default values."""
        if not self.weighting_engine or not self.config.use_dynamic_weighting:
            return
            
        try:
            self.weighting_engine.reset_to_defaults()
            new_limits = self.weighting_engine.current_tier_limits
            self._update_memory_tier_limits(new_limits)
            logger.info("Reset dynamic weighting to default values")
        except Exception as e:
            logger.error(f"Error resetting dynamic weighting: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the CortexFlowManager.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Include memory stats
        if hasattr(self.memory, "get_stats"):
            stats["memory"] = self.memory.get_stats()
            
        # Include dynamic weighting stats if enabled
        if self.config.use_dynamic_weighting and self.weighting_engine:
            stats["dynamic_weighting"] = self.get_dynamic_weighting_stats()
            
        return stats
    
    def generate_response(self, prompt: str = None, model: str = None) -> str:
        """
        Generate a response using the conversation context.
        
        Args:
            prompt: Optional prompt to use instead of the conversation context
            model: Model to use for generation
            
        Returns:
            Generated response
        """
        try:
            import requests
            
            # Use model from config if not specified
            if model is None:
                model = self.config.default_model
                
            # Get conversation context if no prompt provided
            if prompt is None:
                context = self.get_conversation_context()
                
                # Extract messages
                messages = context["messages"]
                
                # Add knowledge as system message if available
                knowledge = context.get("knowledge", [])
                if knowledge:
                    knowledge_text = "\n".join(item["text"] for item in knowledge)
                    
                    # Add knowledge context as a system message
                    messages = [{"role": "system", "content": f"Use this knowledge to answer the question:\n{knowledge_text}"}] + messages
                
                # Format as prompt if needed
                if not messages:
                    prompt = "Hello! How can I assist you today?"
                    
                # Get the most recent user query for Chain of Agents processing
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                query = user_messages[-1]["content"] if user_messages else ""
                
                # Use Chain of Agents for complex queries if enabled
                if (self.agent_chain_manager is not None and 
                    hasattr(self.config, "use_chain_of_agents") and 
                    self.config.use_chain_of_agents):
                    
                    try:
                        # Check if query is complex enough to warrant Chain of Agents
                        # For now, use a simple length-based heuristic
                        if len(query.split()) > 5 or "?" in query:
                            logger.info(f"Processing query with Chain of Agents: {query[:50]}...")
                            
                            # Process with Chain of Agents
                            coa_result = self.agent_chain_manager.process_query(
                                query=query,
                                context={"messages": messages, "knowledge": knowledge}
                            )
                            
                            # Get the answer from the Chain of Agents
                            generated_text = coa_result.get("answer", "")
                            
                            if generated_text:
                                # Apply self-reflection if enabled
                                if (self.reflection_engine and 
                                    hasattr(self.config, "use_self_reflection") and 
                                    self.config.use_self_reflection):
                                    
                                    try:
                                        # Check response consistency
                                        consistency_result = self.reflection_engine.check_response_consistency(
                                            query,
                                            generated_text,
                                            knowledge
                                        )
                                        
                                        # Revise if needed
                                        if not consistency_result.get("is_consistent", True):
                                            generated_text = self.reflection_engine.revise_response(
                                                query,
                                                generated_text,
                                                knowledge,
                                                consistency_result
                                            )
                                            logger.info("Response revised through self-reflection")
                                    except Exception as e:
                                        logger.error(f"Error in self-reflection: {e}")
                                
                                # Add the response to memory
                                self.add_message("assistant", generated_text)
                                logger.info(f"Chain of Agents generated response in {coa_result.get('total_processing_time', 0):.2f} seconds")
                                return generated_text
                            # If Chain of Agents fails, fall back to standard processing
                            logger.warning("Chain of Agents failed to generate response, falling back to standard processing")
                    except Exception as e:
                        logger.error(f"Error processing with Chain of Agents: {e}")
                        logger.error(traceback.format_exc())
                        # Continue with standard processing on error
            else:
                messages = [{"role": "user", "content": prompt}]
                
            # Get Ollama URL
            ollama_url = f"{self.config.ollama_host}/api/chat"
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            logger.debug(f"Sending request to Ollama: {ollama_url}")
            response = requests.post(ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["message"]["content"]
                
                # Apply self-reflection if enabled
                if (self.reflection_engine and 
                    hasattr(self.config, "use_self_reflection") and 
                    self.config.use_self_reflection and
                    len(user_messages) > 0):  # Need a user query for reflection
                    
                    try:
                        # Check response consistency
                        consistency_result = self.reflection_engine.check_response_consistency(
                            query,
                            generated_text,
                            knowledge
                        )
                        
                        # Revise if needed
                        if not consistency_result.get("is_consistent", True):
                            generated_text = self.reflection_engine.revise_response(
                                query,
                                generated_text,
                                knowledge,
                                consistency_result
                            )
                            logger.info("Response revised through self-reflection")
                    except Exception as e:
                        logger.error(f"Error in self-reflection: {e}")
                
                # Add the response to memory
                self.add_message("assistant", generated_text)
                
                return generated_text
            else:
                error_message = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_message)
                return f"Error generating response: {error_message}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    def generate_response_stream(self, prompt: str = None, model: str = None) -> Iterator[str]:
        """
        Generate a streaming response using the conversation context.
        
        Args:
            prompt: Optional prompt to use instead of the conversation context
            model: Model to use for generation
            
        Yields:
            Chunks of the generated response
        """
        try:
            import requests
            
            # Use model from config if not specified
            if model is None:
                model = self.config.default_model
                
            # Get conversation context if no prompt provided
            if prompt is None:
                context = self.get_conversation_context()
                
                # Extract messages
                messages = context["messages"]
                
                # Add knowledge as system message if available
                knowledge = context.get("knowledge", [])
                if knowledge:
                    knowledge_text = "\n".join(item["text"] for item in knowledge)
                    
                    # Add knowledge context as a system message
                    messages = [{"role": "system", "content": f"Use this knowledge to answer the question:\n{knowledge_text}"}] + messages
                    
                # Format as prompt if needed
                if not messages:
                    prompt = "Hello! How can I assist you today?"
                    
                # Skip Chain of Agents for streaming (would require more complex implementation)
            else:
                messages = [{"role": "user", "content": prompt}]
                
            # Get Ollama URL
            ollama_url = f"{self.config.ollama_host}/api/chat"
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            
            logger.debug(f"Sending streaming request to Ollama: {ollama_url}")
            response = requests.post(ollama_url, json=payload, timeout=30, stream=True)
            
            if response.status_code == 200:
                # Store the full response as we stream
                full_response = ""
                
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line)
                        if "message" in chunk_data:
                            chunk = chunk_data["message"].get("content", "")
                        else:
                            chunk = chunk_data.get("response", "")
                            
                        if chunk:
                            full_response += chunk
                            yield chunk
                
                # After streaming completes, add the response to memory
                self.add_message("assistant", full_response)
                
            else:
                error_message = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_message)
                yield f"Error generating response: {error_message}"
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            logger.error(traceback.format_exc())
            yield f"Error generating streaming response: {str(e)}"
    
    def remember_knowledge(self, text: str, source: str = None) -> List[int]:
        """
        Remember knowledge from text.
        
        Args:
            text: Text to remember
            source: Optional source information
            
        Returns:
            List of IDs for the stored knowledge
        """
        return self.knowledge_store.remember_knowledge(text, source)
    
    def get_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Get relevant knowledge for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of relevant knowledge items
        """
        return self.knowledge_store.get_relevant_knowledge(query)
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        
        # Reset dynamic weighting to defaults if enabled
        if self.config.use_dynamic_weighting and self.weighting_engine:
            self.reset_dynamic_weighting()
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'knowledge_store'):
                self.knowledge_store.close()
        except Exception as e:
            logger.error(f"Error closing knowledge store: {e}")
    
    def __del__(self) -> None:
        """Clean up on deletion."""
        self.close() 