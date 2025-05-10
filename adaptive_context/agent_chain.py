"""
Chain of Agents (CoA) Framework for AdaptiveContext.

This module implements the Chain of Agents approach for complex query processing,
enabling multiple specialized agents to collaborate sequentially on tasks that
require multi-hop reasoning over long contexts.

Based on research from Google's "Chain of Agents: Large Language Models
Collaborating on Long Context Tasks" (2025).
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import requests

from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.knowledge import KnowledgeStore

logger = logging.getLogger('adaptive_context')

class Agent:
    """Base class for specialized agents in the Chain of Agents framework."""
    
    def __init__(
        self, 
        name: str, 
        role: str, 
        config: AdaptiveContextConfig,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        """
        Initialize an agent with a specific role.
        
        Args:
            name: Name identifier for this agent
            role: Description of this agent's specialized role
            config: AdaptiveContext configuration
            knowledge_store: Optional shared knowledge store
        """
        self.name = name
        self.role = role
        self.config = config
        self.knowledge_store = knowledge_store
        self.ollama_host = config.ollama_host
        self.default_model = config.default_model
        
    def process(
        self, 
        query: str,
        context: Dict[str, Any], 
        agent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a query with the agent's specialized capability.
        
        Args:
            query: The original user query
            context: Context information for processing
            agent_history: History of previous agents' processing in the chain
            
        Returns:
            Processing results and any additional context for the next agent
        """
        # Default implementation - override in subclasses
        raise NotImplementedError("Agent subclasses must implement process()")


class ExplorerAgent(Agent):
    """
    Explorer agent that broadly searches the knowledge base to find relevant
    information, without focusing on answering the query directly.
    """
    
    def __init__(self, config: AdaptiveContextConfig, knowledge_store: KnowledgeStore):
        """Initialize the explorer agent."""
        super().__init__(
            name="Explorer",
            role="Explore the knowledge base to find relevant information and topics",
            config=config,
            knowledge_store=knowledge_store
        )
    
    def process(
        self, 
        query: str,
        context: Dict[str, Any], 
        agent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Explore the knowledge base to find relevant information.
        
        This agent focuses on broad exploration rather than answering the query directly.
        It retrieves a diverse set of potentially relevant knowledge.
        """
        if agent_history is None:
            agent_history = []
            
        # Create a prompt that encourages exploration
        prompt = self._create_explorer_prompt(query, context)
        
        # Get relevant knowledge using both semantic and graph-based retrieval
        knowledge_items = []
        
        # Vector-based retrieval
        if self.knowledge_store:
            # Use a higher threshold to get more diverse results
            knowledge_items = self.knowledge_store.get_relevant_knowledge(
                query, 
                max_results=10  # Retrieve more results for exploration
            )
            
            # If graph-based retrieval is enabled, use it for additional exploration
            if hasattr(self.knowledge_store, 'graph_store') and self.knowledge_store.use_graph_rag:
                # Get related entities to expand the search
                if hasattr(self.knowledge_store, '_graph_search'):
                    graph_results = self.knowledge_store._graph_search(query, max_results=5)
                    # Add unique results to knowledge_items
                    existing_texts = {item.get('text', '') for item in knowledge_items}
                    for result in graph_results:
                        if result.get('text', '') not in existing_texts:
                            knowledge_items.append(result)
                            existing_texts.add(result.get('text', ''))
        
        # Format the knowledge as context
        knowledge_context = self._format_knowledge_context(knowledge_items)
        
        # Process with LLM to explore relevant topics
        exploration_results = self._process_with_llm(prompt, knowledge_context)
        
        return {
            "agent": self.name,
            "role": self.role,
            "exploration_results": exploration_results,
            "knowledge_items": knowledge_items
        }
    
    def _create_explorer_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create a prompt that guides the model to explore related topics."""
        return f"""You are the Explorer Agent. Your task is to explore information related to the user's query
without directly answering it. Focus on finding related topics, concepts, and facts that might be useful
for answering the query later. Think broadly and consider multiple perspectives.

USER QUERY: {query}

Your task:
1. Identify key entities and concepts in the query
2. Explore related information that might be helpful for answering the query
3. Do NOT try to answer the query directly
4. Summarize what you've learned from exploring the knowledge base

Output your exploration findings in a clear, structured format.
"""

    def _format_knowledge_context(self, knowledge_items: List[Dict[str, Any]]) -> str:
        """Format knowledge items as context for the LLM."""
        if not knowledge_items:
            return "No relevant knowledge found in the knowledge base."
            
        formatted_items = []
        for i, item in enumerate(knowledge_items):
            formatted_items.append(f"[{i+1}] {item.get('text', '')}")
            
        return "KNOWLEDGE CONTEXT:\n" + "\n".join(formatted_items)
    
    def _process_with_llm(self, prompt: str, knowledge_context: str) -> Dict[str, Any]:
        """Process the prompt and knowledge context with an LLM."""
        # Combine prompt with knowledge context
        full_prompt = f"{prompt}\n\n{knowledge_context}"
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.default_model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                exploration_text = result.get("response", "")
                return {
                    "exploration_text": exploration_text,
                    "status": "success"
                }
            else:
                logger.error(f"Error from LLM: {response.status_code} - {response.text}")
                return {
                    "exploration_text": "Failed to explore knowledge base.",
                    "status": "error",
                    "error": f"LLM API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return {
                "exploration_text": "Failed to process with LLM.",
                "status": "error",
                "error": str(e)
            }


class AnalyzerAgent(Agent):
    """
    Analyzer agent that examines relationships between facts and builds a
    coherent understanding of the information from the Explorer.
    """
    
    def __init__(self, config: AdaptiveContextConfig, knowledge_store: KnowledgeStore):
        """Initialize the analyzer agent."""
        super().__init__(
            name="Analyzer",
            role="Analyze relationships between facts and build coherent understanding",
            config=config,
            knowledge_store=knowledge_store
        )
    
    def process(
        self, 
        query: str,
        context: Dict[str, Any], 
        agent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationships between facts from the Explorer agent.
        
        This agent focuses on understanding connections between pieces of information
        and identifying patterns relevant to the query.
        """
        if agent_history is None or not agent_history:
            return {
                "agent": self.name,
                "role": self.role,
                "analysis_results": {"status": "error", "error": "No previous agent history"},
                "status": "error"
            }
            
        # Get the Explorer agent's results
        explorer_results = agent_history[0]
        
        # Create analyzer prompt
        prompt = self._create_analyzer_prompt(query, explorer_results)
        
        # Process with LLM to analyze relationships
        analysis_results = self._process_with_llm(prompt)
        
        return {
            "agent": self.name,
            "role": self.role,
            "analysis_results": analysis_results
        }
    
    def _create_analyzer_prompt(self, query: str, explorer_results: Dict[str, Any]) -> str:
        """Create a prompt for the analyzer agent."""
        exploration_text = explorer_results.get("exploration_results", {}).get("exploration_text", "No exploration results")
        
        return f"""You are the Analyzer Agent. Your task is to analyze relationships between facts and concepts
discovered by the Explorer Agent. Focus on finding connections, patterns, and logical relationships
that might help answer the user's query.

USER QUERY: {query}

EXPLORER'S FINDINGS:
{exploration_text}

Your task:
1. Identify key relationships between facts and concepts
2. Look for patterns and inconsistencies
3. Organize the information into a coherent structure
4. Highlight the most important connections relevant to the query
5. Do NOT yet try to answer the query directly

Output your analysis in a clear, structured format.
"""

    def _process_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Process the prompt with an LLM."""
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.default_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("response", "")
                return {
                    "analysis_text": analysis_text,
                    "status": "success"
                }
            else:
                logger.error(f"Error from LLM: {response.status_code} - {response.text}")
                return {
                    "analysis_text": "Failed to analyze information.",
                    "status": "error",
                    "error": f"LLM API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return {
                "analysis_text": "Failed to process with LLM.",
                "status": "error",
                "error": str(e)
            }


class SynthesizerAgent(Agent):
    """
    Synthesizer agent that combines information from previous agents to
    generate a comprehensive answer to the original query.
    """
    
    def __init__(self, config: AdaptiveContextConfig, knowledge_store: KnowledgeStore):
        """Initialize the synthesizer agent."""
        super().__init__(
            name="Synthesizer",
            role="Synthesize information and generate comprehensive answer",
            config=config,
            knowledge_store=knowledge_store
        )
    
    def process(
        self, 
        query: str,
        context: Dict[str, Any], 
        agent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize information from previous agents to answer the query.
        
        This agent focuses on generating a comprehensive, coherent answer
        based on the exploration and analysis from previous agents.
        """
        if agent_history is None or len(agent_history) < 2:
            return {
                "agent": self.name,
                "role": self.role,
                "answer": "Insufficient information from previous agents to synthesize an answer.",
                "status": "error"
            }
            
        # Get results from previous agents
        explorer_results = agent_history[0]
        analyzer_results = agent_history[1]
        
        # Create synthesizer prompt
        prompt = self._create_synthesizer_prompt(
            query, 
            explorer_results, 
            analyzer_results
        )
        
        # Process with LLM to synthesize answer
        synthesis_results = self._process_with_llm(prompt)
        
        return {
            "agent": self.name,
            "role": self.role,
            "synthesis_results": synthesis_results,
            "answer": synthesis_results.get("synthesis_text", "Failed to synthesize an answer.")
        }
    
    def _create_synthesizer_prompt(
        self, 
        query: str, 
        explorer_results: Dict[str, Any], 
        analyzer_results: Dict[str, Any]
    ) -> str:
        """Create a prompt for the synthesizer agent."""
        exploration_text = explorer_results.get("exploration_results", {}).get("exploration_text", "No exploration results")
        analysis_text = analyzer_results.get("analysis_results", {}).get("analysis_text", "No analysis results")
        
        return f"""You are the Synthesizer Agent. Your task is to synthesize information from the Explorer
and Analyzer Agents to provide a comprehensive answer to the user's query.

USER QUERY: {query}

EXPLORER'S FINDINGS:
{exploration_text}

ANALYZER'S INSIGHTS:
{analysis_text}

Your task:
1. Synthesize the information from both agents
2. Generate a comprehensive answer to the user's query
3. Ensure your answer is coherent, accurate, and directly addresses the query
4. Include relevant facts and relationships discovered by previous agents
5. Format your answer in a clear, organized manner

Provide your answer directly.
"""

    def _process_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Process the prompt with an LLM."""
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.default_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                synthesis_text = result.get("response", "")
                return {
                    "synthesis_text": synthesis_text,
                    "status": "success"
                }
            else:
                logger.error(f"Error from LLM: {response.status_code} - {response.text}")
                return {
                    "synthesis_text": "Failed to synthesize answer.",
                    "status": "error",
                    "error": f"LLM API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return {
                "synthesis_text": "Failed to process with LLM.",
                "status": "error",
                "error": str(e)
            }


class AgentChainManager:
    """
    Manager class for the Chain of Agents framework.
    Coordinates the sequential processing of agents in the chain.
    """
    
    def __init__(self, config: AdaptiveContextConfig, knowledge_store: KnowledgeStore):
        """
        Initialize the Chain of Agents manager.
        
        Args:
            config: AdaptiveContext configuration
            knowledge_store: Knowledge store for agents to access
        """
        self.config = config
        self.knowledge_store = knowledge_store
        
        # Initialize the agent chain
        self.agents = [
            ExplorerAgent(config, knowledge_store),
            AnalyzerAgent(config, knowledge_store),
            SynthesizerAgent(config, knowledge_store)
        ]
        
        logger.info(f"Initialized Chain of Agents with {len(self.agents)} agents")
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query through the chain of agents.
        
        Args:
            query: User query to process
            context: Additional context information
            
        Returns:
            Results from the full agent chain processing
        """
        if context is None:
            context = {}
            
        start_time = time.time()
        agent_history = []
        
        logger.info(f"Starting Chain of Agents processing for query: {query[:50]}...")
        
        for i, agent in enumerate(self.agents):
            logger.info(f"Running agent {i+1}/{len(self.agents)}: {agent.name}")
            
            # Process with current agent
            agent_start_time = time.time()
            agent_result = agent.process(query, context, agent_history)
            agent_duration = time.time() - agent_start_time
            
            # Add processing duration
            agent_result["processing_time"] = agent_duration
            
            # Add to agent history
            agent_history.append(agent_result)
            
            logger.info(f"Completed agent {agent.name} in {agent_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        
        # Return final result with processing details
        return {
            "query": query,
            "answer": agent_history[-1].get("answer", "No answer generated"),
            "agent_chain": agent_history,
            "total_processing_time": total_duration
        } 