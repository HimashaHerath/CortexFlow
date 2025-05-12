"""
CortexFlow Manager module.

This module provides the main manager class for the CortexFlow system.
"""

import time
import json
import logging
import requests
import traceback
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
import re

from cortexflow.interfaces import ContextProvider
from cortexflow.config import CortexFlowConfig
from cortexflow.memory import (
    ContextSegment, 
    MemoryTier, 
    ActiveTier, 
    WorkingTier, 
    ArchiveTier,
    ConversationMemory
)
from cortexflow.classifier import ImportanceClassifier, ContentClassifier
from cortexflow.compressor import ContextCompressor
from cortexflow.knowledge import KnowledgeStore

# Add import for Chain of Agents
try:
    from cortexflow.agent_chain import AgentChainManager
    AGENT_CHAIN_ENABLED = True
except ImportError:
    AGENT_CHAIN_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("agent_chain module not found. Chain of Agents functionality will be disabled.")

# Add import for Self-Reflection
try:
    from cortexflow.reflection import ReflectionEngine
    REFLECTION_ENABLED = True
except ImportError:
    REFLECTION_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("reflection module not found. Self-Reflection functionality will be disabled.")

# Add import for Dynamic Weighting
try:
    from cortexflow.dynamic_weighting import DynamicWeightingEngine
    DYNAMIC_WEIGHTING_ENABLED = True
except ImportError:
    DYNAMIC_WEIGHTING_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("dynamic_weighting module not found. Dynamic Weighting functionality will be disabled.")

# Add import for Uncertainty Handling
try:
    from cortexflow.uncertainty_handler import UncertaintyHandler
    UNCERTAINTY_HANDLING_ENABLED = True
except ImportError:
    UNCERTAINTY_HANDLING_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("uncertainty_handler module not found. Uncertainty handling functionality will be disabled.")

# Add import for Performance Optimization
try:
    from cortexflow.performance_optimizer import PerformanceOptimizer
    PERFORMANCE_OPTIMIZATION_ENABLED = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_ENABLED = False
    logger = logging.getLogger('cortexflow')
    logger.warning("performance_optimizer module not found. Performance optimization functionality will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cortexflow')

def configure_logging(verbose: bool = False):
    """Configure logging for the cortexflow module."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

class CortexFlowManager(ContextProvider):
    """
    Main manager class for CortexFlow system.
    Coordinates between components for memory, knowledge, and external integrations.
    Implements the ContextProvider interface.
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

            # Initialize Uncertainty Handler if enabled
            self.uncertainty_handler = None
            if hasattr(self.config, "use_uncertainty_handling") and self.config.use_uncertainty_handling and UNCERTAINTY_HANDLING_ENABLED:
                try:
                    # Pass the graph store to the uncertainty handler
                    graph_store = self.knowledge_store.graph_store if hasattr(self.knowledge_store, 'graph_store') else None
                    self.uncertainty_handler = UncertaintyHandler(self.config, graph_store)
                    logger.info("Uncertainty Handler initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Uncertainty Handler: {e}")
                
            # Initialize Performance Optimizer if enabled
            self.performance_optimizer = None
            if hasattr(self.config, "use_performance_optimization") and self.config.use_performance_optimization and PERFORMANCE_OPTIMIZATION_ENABLED:
                try:
                    # Pass the graph store to the performance optimizer
                    graph_store = self.knowledge_store.graph_store if hasattr(self.knowledge_store, 'graph_store') else None
                    self.performance_optimizer = PerformanceOptimizer(self.config, graph_store)
                    logger.info("Performance Optimizer initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Performance Optimizer: {e}")
                
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
        Get system-wide statistics.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            "memory": {
                "messages": self.memory.get_message_count() if hasattr(self.memory, "get_message_count") else None,
                "total_tokens": self.memory.get_total_tokens() if hasattr(self.memory, "get_total_tokens") else None,
                "tiers": self.memory.get_tier_stats() if hasattr(self.memory, "get_tier_stats") else None
            },
            "knowledge": {
                "facts": self.knowledge_store.get_fact_count() if hasattr(self.knowledge_store, "get_fact_count") else None,
                "embeddings": self.knowledge_store.get_embedding_count() if hasattr(self.knowledge_store, "get_embedding_count") else None,
                "sources": self.knowledge_store.get_source_count() if hasattr(self.knowledge_store, "get_source_count") else None
            }
        }
        
        # Add dynamic weighting stats if available
        if self.weighting_engine:
            try:
                stats["dynamic_weighting"] = self.get_dynamic_weighting_stats()
            except Exception:
                pass
                
        # Add uncertainty handling stats if available
        if self.uncertainty_handler:
            try:
                stats["uncertainty"] = self.uncertainty_handler.get_stats()
            except Exception:
                pass
                
        # Add performance optimization stats if available
        if self.performance_optimizer:
            try:
                stats["performance"] = self.performance_optimizer.get_stats()
            except Exception:
                pass
                
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
    
    def remember_knowledge(self, text: str, source: str = None, confidence: float = None) -> List[int]:
        """
        Store important knowledge in the knowledge store.
        
        Args:
            text: Text to remember
            source: Optional source of the knowledge
            confidence: Optional confidence value for the knowledge
            
        Returns:
            List of IDs for the stored knowledge
        """
        item_ids = self.knowledge_store.add_knowledge(text, source=source, confidence=confidence)
        
        # Check for contradictions if enabled
        if self.uncertainty_handler and self.config.auto_detect_contradictions:
            try:
                # Extract entity IDs from the added items
                entity_ids = []
                if hasattr(self.knowledge_store, 'graph_store'):
                    # Get the entity IDs from the knowledge store's graph store
                    for item_id in item_ids:
                        entity_data = self.knowledge_store.get_knowledge_item(item_id)
                        if entity_data and 'entity_id' in entity_data:
                            entity_ids.append(entity_data['entity_id'])
                
                # Check for contradictions for each entity
                for entity_id in entity_ids:
                    contradictions = self.uncertainty_handler.detect_contradictions(entity_id=entity_id)
                    
                    # Auto-resolve contradictions if found
                    for contradiction in contradictions:
                        logger.info(f"Detected contradiction for entity {contradiction.get('entity')}: "
                                  f"{contradiction.get('target1')} vs {contradiction.get('target2')}")
                        
                        # Resolve using the configured strategy
                        resolution = self.uncertainty_handler.resolve_contradiction(
                            contradiction, 
                            strategy=self.config.default_contradiction_strategy
                        )
                        
                        logger.info(f"Resolved contradiction using {resolution.get('strategy_used')} strategy: "
                                  f"Selected '{resolution.get('resolved_value')}' with confidence {resolution.get('confidence')}")
            except Exception as e:
                logger.error(f"Error detecting contradictions: {e}")
                
        return item_ids
            
    def detect_contradictions(self, entity_id=None, relation_type=None, 
                          max_results=100) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the knowledge graph.
        
        Args:
            entity_id: Optional entity ID to check
            relation_type: Optional relation type to check
            max_results: Maximum number of results to return
            
        Returns:
            List of detected contradictions
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot detect contradictions.")
            return []
            
        return self.uncertainty_handler.detect_contradictions(
            entity_id=entity_id,
            relation_type=relation_type,
            max_results=max_results
        )
        
    def resolve_contradiction(self, contradiction: Dict[str, Any], 
                           strategy: str = None) -> Dict[str, Any]:
        """
        Resolve a contradiction using the specified strategy.
        
        Args:
            contradiction: Contradiction to resolve
            strategy: Resolution strategy (auto, recency, confidence, reliability, or keep_both)
            
        Returns:
            Resolution result
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot resolve contradictions.")
            return {"error": "Uncertainty handling not enabled"}
            
        if strategy is None:
            strategy = self.config.default_contradiction_strategy
            
        return self.uncertainty_handler.resolve_contradiction(contradiction, strategy)
        
    def update_source_reliability(self, source_name: str, reliability_score: float,
                              metadata: Dict[str, Any] = None) -> None:
        """
        Update the reliability score for a knowledge source.
        
        Args:
            source_name: Name of the source
            reliability_score: Reliability score (0.0-1.0)
            metadata: Optional metadata about the source
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot update source reliability.")
            return
            
        self.uncertainty_handler.update_source_reliability(
            source_name=source_name,
            reliability_score=reliability_score,
            metadata=metadata
        )
        
    def get_source_reliability(self, source_name: str) -> float:
        """
        Get the reliability score for a knowledge source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            Reliability score (0.0-1.0)
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot get source reliability.")
            return 0.5  # Default medium reliability
            
        return self.uncertainty_handler.get_source_reliability(source_name)
        
    def add_probability_distribution(self, entity_id: int, relation_id: int,
                                  distribution_type: str, distribution_data: Dict[str, Any]) -> None:
        """
        Add a probability distribution to represent uncertainty about a fact.
        
        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            distribution_type: Type of distribution (discrete, gaussian, etc.)
            distribution_data: Data representing the distribution
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot add probability distribution.")
            return
            
        self.uncertainty_handler.add_probability_distribution(
            entity_id=entity_id,
            relation_id=relation_id,
            distribution_type=distribution_type,
            distribution_data=distribution_data
        )
        
    def get_probability_distribution(self, entity_id: int, relation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the probability distribution for a fact.
        
        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            
        Returns:
            Probability distribution data or None if not found
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot get probability distribution.")
            return None
            
        return self.uncertainty_handler.get_probability_distribution(
            entity_id=entity_id,
            relation_id=relation_id
        )
        
    def reason_with_incomplete_information(self, query: Dict[str, Any],
                                       available_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reason with incomplete information to provide best possible answers.
        
        Args:
            query: The query to answer
            available_knowledge: Available knowledge to reason with
            
        Returns:
            Reasoning result with confidence and explanation
        """
        if not self.uncertainty_handler or not self.config.reason_with_incomplete_info:
            logger.warning("Reasoning with incomplete information is not enabled.")
            return {
                "answer": None,
                "confidence": 0,
                "explanation": ["Reasoning with incomplete information is not enabled"],
                "missing_information": []
            }
            
        return self.uncertainty_handler.reason_with_incomplete_information(
            query=query,
            available_knowledge=available_knowledge
        )
        
    def get_belief_revision_history(self, entity_id: int = None, 
                                 relation_id: int = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the revision history for beliefs about an entity or relation.
        
        Args:
            entity_id: Optional entity ID filter
            relation_id: Optional relation ID filter
            limit: Maximum number of revisions to return
            
        Returns:
            List of belief revisions
        """
        if not self.uncertainty_handler:
            logger.warning("Uncertainty handling is not enabled. Cannot get belief revision history.")
            return []
            
        return self.uncertainty_handler.get_belief_revision_history(
            entity_id=entity_id,
            relation_id=relation_id,
            limit=limit
        )
    
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
        """Close and clean up resources."""
        try:
            if self.memory:
                self.memory.close()
                
            if self.knowledge_store:
                self.knowledge_store.close()
                
            if self.uncertainty_handler:
                self.uncertainty_handler.close()
                
            if self.performance_optimizer:
                self.performance_optimizer.close()
                
            logger.info("CortexFlowManager closed")
        except Exception as e:
            logger.error(f"Error during close: {e}")

    def __del__(self) -> None:
        """Destructor."""
        self.close()

    def get_context(self) -> Dict[str, Any]:
        """Get the current context for model consumption."""
        return self.get_conversation_context()
        
    def clear_context(self) -> None:
        """Clear all context data."""
        self.clear_memory()

    def answer_why_question(self, query: str) -> List[Dict[str, Any]]:
        """
        Answer a why-question using backward chaining logical reasoning.
        
        Args:
            query: The why question to answer
            
        Returns:
            Explanation steps for the answer
        """
        if not self.knowledge_store or not hasattr(self.knowledge_store, "use_inference_engine") or not self.knowledge_store.use_inference_engine:
            return [{"type": "error", "message": "Inference engine is not enabled"}]
        
        try:
            return self.knowledge_store.inference_engine.answer_why_question(query)
        except Exception as e:
            logger.error(f"Error answering why question: {e}")
            return [{"type": "error", "message": f"Error processing question: {str(e)}"}]
    
    def generate_novel_implications(self, iterations: int = None) -> List[Dict[str, Any]]:
        """
        Generate novel implications using forward chaining.
        
        Args:
            iterations: Number of forward chaining iterations (default uses config)
            
        Returns:
            List of newly inferred facts
        """
        if not self.knowledge_store or not hasattr(self.knowledge_store, "use_inference_engine") or not self.knowledge_store.use_inference_engine:
            return []
        
        try:
            if iterations is None:
                iterations = self.config.max_forward_chain_iterations
                
            return self.knowledge_store.inference_engine.forward_chain(iterations=iterations)
        except Exception as e:
            logger.error(f"Error generating implications: {e}")
            return []
    
    def generate_hypotheses(self, observation: str, max_hypotheses: int = None) -> List[Dict[str, Any]]:
        """
        Generate hypotheses to explain an observation using abductive reasoning.
        
        Args:
            observation: The observation to explain
            max_hypotheses: Maximum number of hypotheses to generate (default uses config)
            
        Returns:
            List of hypotheses that could explain the observation
        """
        if not self.knowledge_store or not hasattr(self.knowledge_store, "use_inference_engine") or not self.knowledge_store.use_inference_engine:
            return []
        
        try:
            if max_hypotheses is None:
                max_hypotheses = self.config.max_abductive_hypotheses
                
            return self.knowledge_store.generate_hypotheses(observation, max_hypotheses=max_hypotheses)
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return []
    
    def add_logical_rule(self, name: str, premise_patterns: List[Dict[str, Any]], 
                       conclusion_pattern: Dict[str, Any], confidence: float = 0.8) -> bool:
        """
        Add a logical rule to the inference engine.
        
        Args:
            name: Rule name
            premise_patterns: List of premise patterns that must be satisfied
            conclusion_pattern: The conclusion pattern to infer
            confidence: Rule confidence (0.0-1.0)
            
        Returns:
            Success status
        """
        if not self.knowledge_store or not hasattr(self.knowledge_store, "use_inference_engine") or not self.knowledge_store.use_inference_engine:
            return False
        
        try:
            self.knowledge_store.inference_engine.add_rule(
                name=name,
                premise=premise_patterns,
                conclusion=conclusion_pattern,
                confidence=confidence
            )
            return True
        except Exception as e:
            logger.error(f"Error adding logical rule: {e}")
            return False
    
    def optimize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an optimized query plan for knowledge graph operations.
        
        Args:
            query: Dictionary with query parameters
            
        Returns:
            Optimized query plan
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}
        
        try:
            return self.performance_optimizer.generate_query_plan(query)
        except Exception as e:
            logger.error(f"Error generating optimized query plan: {e}")
            return {"status": "error", "message": str(e)}
    
    def partition_graph(self, method: str = None, partition_count: int = None) -> Dict[str, Any]:
        """
        Partition the knowledge graph for improved performance.
        
        Args:
            method: Partitioning method (louvain, spectral, modularity)
            partition_count: Target number of partitions
            
        Returns:
            Partition statistics
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}
        
        try:
            # Use default method from config if not specified
            if method is None:
                method = self.config.graph_partition_method
                
            # Use default count from config if not specified
            if partition_count is None:
                partition_count = self.config.target_partition_count
                
            return self.performance_optimizer.partition_graph(method, partition_count)
        except Exception as e:
            logger.error(f"Error partitioning graph: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_hop_indexes(self, max_hops: int = None) -> Dict[str, Any]:
        """
        Create indexes for multi-hop queries to speed up traversal.
        
        Args:
            max_hops: Maximum number of hops to index
            
        Returns:
            Indexing statistics
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}
        
        try:
            # Use default from config if not specified
            if max_hops is None:
                max_hops = self.config.max_indexed_hops
                
            return self.performance_optimizer.create_hop_indexes(max_hops)
        except Exception as e:
            logger.error(f"Error creating hop indexes: {e}")
            return {"status": "error", "message": str(e)}
    
    def optimize_path_query(self, start_entity: str, end_entity: str, 
                       max_hops: int = 3, 
                       relation_constraints: List[str] = None) -> Dict[str, Any]:
        """
        Optimize a path query between entities using the query planning system.
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length
            relation_constraints: Optional list of allowed relation types
            
        Returns:
            Optimized query plan
        """
        if not self.performance_optimizer:
            return {"status": "error", "message": "Performance optimizer not enabled"}
        
        try:
            query = {
                "type": "path",
                "start_entity": start_entity,
                "end_entity": end_entity,
                "max_hops": max_hops,
                "relation_constraints": relation_constraints or []
            }
            
            return self.performance_optimizer.optimize_query_execution(query)
        except Exception as e:
            logger.error(f"Error optimizing path query: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the optimizer.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}
        
        try:
            return self.performance_optimizer.get_stats()
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def clear_performance_caches(self) -> Dict[str, Any]:
        """
        Clear all performance optimization caches.
        
        Returns:
            Dictionary with cache clearing statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}
        
        try:
            return self.performance_optimizer.clear_caches()
        except Exception as e:
            logger.error(f"Error clearing performance caches: {e}")
            return {"status": "error", "message": str(e)}
    
    def cache_reasoning_pattern(self, pattern_key: str, pattern_result: Any) -> bool:
        """
        Cache a common reasoning pattern for reuse.
        
        Args:
            pattern_key: Unique identifier for the reasoning pattern
            pattern_result: Result of the reasoning pattern
            
        Returns:
            True if successful, False otherwise
        """
        if not self.performance_optimizer:
            return False
        
        try:
            self.performance_optimizer.cache_reasoning_pattern(pattern_key, pattern_result)
            return True
        except Exception as e:
            logger.error(f"Error caching reasoning pattern: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit rates.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.performance_optimizer:
            return {"status": "disabled", "message": "Performance optimizer not enabled"}
            
        try:
            stats = self.performance_optimizer.get_stats()
            return stats.get("caching", {})
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def multi_hop_query(self, query: str) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning on a query.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with path, entities, score, and other reasoning results
        """
        if not hasattr(self, "knowledge_store") or not hasattr(self.knowledge_store, "graph_store"):
            logger.error("Graph store not available for multi-hop query")
            return {"path": [], "entities": [], "score": 0.0}
            
        result = {
            "path": [],
            "entities": [],
            "score": 0.0,
            "hop_count": 0
        }
        
        try:
            # Use the graph search to find relevant information
            graph_results = self.knowledge_store._graph_search(query, max_results=5)
            
            # Parse entity relations from query
            entity_pair = self._extract_entity_pair(query)
            
            if entity_pair:
                # Try to find paths between the entities
                start_entity, end_entity = entity_pair
                paths = self.knowledge_store.graph_store.path_query(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    max_hops=self.config.max_graph_hops
                )
                
                # If we found paths, format the result
                if paths and len(paths) > 0:
                    best_path = paths[0]  # Use the first path (should be shortest/best)
                    
                    # Format into a linear path
                    formatted_path = []
                    for i, node in enumerate(best_path):
                        # Add the entity
                        formatted_path.append(node.get("entity", f"Entity_{node.get('id')}"))
                        
                        # Add the relation to the next node if not the last node
                        if i < len(best_path) - 1 and "next_relation" in node:
                            formatted_path.append(node["next_relation"].get("type", "related_to"))
                    
                    result["path"] = formatted_path
                    result["entities"] = [node.get("entity", f"Entity_{node.get('id')}") for node in best_path]
                    result["hop_count"] = len(best_path) - 1
                    result["score"] = 0.8  # Default score
            
            # If no direct path found, try to extract paths from graph search results
            if not result["path"] and graph_results:
                for item in graph_results:
                    # Look for graph_path type results
                    if item.get("type") == "graph_path":
                        path_text = item.get("text", "")
                        if path_text:
                            # Parse the path text into components
                            path = [p.strip() for p in path_text.replace("→", "→").split("→")]
                            result["path"] = path
                            result["score"] = item.get("score", 0.5)
                            result["hop_count"] = len(path) - 1 if path else 0
                            break
                    
                    # Collect entities from all results
                    if "entities" in item:
                        result["entities"].extend(item.get("entities", []))
            
            # Ensure entities list is unique
            if result["entities"]:
                result["entities"] = list(set(result["entities"]))
            
            # Log the path if found
            if result["path"]:
                logger.info(f"Found path for query '{query}': {' → '.join(result['path'])}")
            else:
                logger.info(f"No path found for query '{query}'")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in multi_hop_query: {e}")
            return {"path": [], "entities": [], "score": 0.0}
    
    def _extract_entity_pair(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Extract a pair of entities from a query for path finding.
        
        Args:
            query: The query text
            
        Returns:
            Tuple of (start_entity, end_entity) or None if not found
        """
        # Try to match common query patterns like "connection between X and Y"
        connection_pattern = r"(?:connection|relationship|relation)(?:\s+between\s+)([^,]+?)(?:\s+and\s+)([^?\.]+)"
        match = re.search(connection_pattern, query, re.IGNORECASE)
        
        if match:
            start_entity = match.group(1).strip()
            end_entity = match.group(2).strip()
            return (start_entity, end_entity)
        
        # Alternative patterns
        alt_pattern = r"(?:how\s+are|what\s+connects|is\s+there\s+a\s+connection\s+between)\s+([^,]+?)(?:\s+and\s+)([^?\.]+)"
        match = re.search(alt_pattern, query, re.IGNORECASE)
        
        if match:
            start_entity = match.group(1).strip()
            end_entity = match.group(2).strip()
            return (start_entity, end_entity)
            
        return None
        
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        General query interface that routes to specialized query methods.
        
        Args:
            query_text: The query text
            
        Returns:
            Query result
        """
        # For multi-hop reasoning queries, use the multi_hop_query method
        if self.config.enable_multi_hop_queries and self._is_multi_hop_query(query_text):
            logger.info(f"Routing to multi_hop_query: {query_text}")
            return self.multi_hop_query(query_text)
        
        # For standard queries, use knowledge retrieval
        logger.info(f"Performing standard knowledge retrieval: {query_text}")
        knowledge_items = self.knowledge_store.get_relevant_knowledge(query_text)
        
        # Format the result
        result = {
            "items": knowledge_items,
            "answer": self._extract_answer(query_text, knowledge_items),
            "score": max([item.get("score", 0) for item in knowledge_items]) if knowledge_items else 0
        }
        
        return result
    
    def _is_multi_hop_query(self, query: str) -> bool:
        """
        Determine if a query requires multi-hop reasoning.
        
        Args:
            query: The query text
            
        Returns:
            True if the query requires multi-hop reasoning
        """
        # Keywords that suggest multi-hop reasoning is needed
        multi_hop_indicators = [
            r"connection between", r"relationship between", r"related to",
            r"connect", r"path", r"link", r"how are .+ and .+ related",
            r"what is the connection", r"how does .+ relate to",
        ]
        
        # Check if any indicators are present
        for indicator in multi_hop_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True
                
        return False
    
    def _extract_answer(self, query: str, knowledge_items: List[Dict[str, Any]]) -> str:
        """
        Extract an answer from knowledge items.
        
        Args:
            query: The query text
            knowledge_items: List of knowledge items
            
        Returns:
            Extracted answer string
        """
        if not knowledge_items:
            return "No information found."
            
        # Get the highest-scored item
        best_item = max(knowledge_items, key=lambda x: x.get("score", 0))
        
        # Return the text of the best item
        return best_item.get("text", "Information found but text extraction failed.") 