"""
Response orchestration for CortexFlow.

Extracts the response generation logic from CortexFlowManager into a
focused class.  CortexFlowManager delegates to an instance of
ResponseOrchestrator for ``generate_response`` and
``generate_response_stream``.
"""
from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator

logger = logging.getLogger('cortexflow')


class ResponseOrchestrator:
    """Handles LLM response generation, including Chain-of-Agents and
    self-reflection post-processing."""

    def __init__(
        self,
        config,
        llm_client,
        memory,
        knowledge_store,
        agent_chain_manager,
        reflection_engine,
        uncertainty_handler,
        add_message_fn,
        get_conversation_context_fn,
    ):
        """
        Args:
            config: CortexFlowConfig instance.
            llm_client: The LLM client used to generate text.
            memory: ConversationMemory instance.
            knowledge_store: KnowledgeStore instance.
            agent_chain_manager: AgentChainManager or None.
            reflection_engine: ReflectionEngine or None.
            uncertainty_handler: UncertaintyHandler or None.
            add_message_fn: Callable that adds a message to memory
                (the facade's ``add_message`` method so that fact-extraction
                 and dynamic-weighting hooks still fire).
            get_conversation_context_fn: Callable returning the current
                conversation context dict.
        """
        self.config = config
        self.llm_client = llm_client
        self.memory = memory
        self.knowledge_store = knowledge_store
        self.agent_chain_manager = agent_chain_manager
        self.reflection_engine = reflection_engine
        self.uncertainty_handler = uncertainty_handler
        self._add_message = add_message_fn
        self._get_conversation_context = get_conversation_context_fn
        # Companion AI context providers (set by manager after construction)
        self._emotion_tracker = None
        self._persona_manager = None
        self._user_profile_manager = None
        self._relationship_tracker = None
        self._get_current_session_fn = None

    # ------------------------------------------------------------------
    # Companion AI context helpers
    # ------------------------------------------------------------------

    def _build_companion_system_message(self) -> str | None:
        """Build an optional system message from persona + emotion + relationship."""
        parts: list[str] = []

        session = self._get_current_session_fn() if self._get_current_session_fn else None
        user_id = session.user_id if session else None
        persona_id = session.persona_id if session else None

        # Persona-based system prompt
        if self._persona_manager and persona_id:
            user_profile_text = ""
            if self._user_profile_manager and user_id:
                try:
                    user_profile_text = self._user_profile_manager.get_profile_for_prompt(user_id)
                except Exception:  # noqa: S110
                    pass

            emotional_context = ""
            if self._emotion_tracker:
                try:
                    trend = self._emotion_tracker.get_emotional_trend(last_n=5)
                    state = self._emotion_tracker.get_current_state()
                    if state.primary_emotion != "neutral":
                        emotional_context = (
                            f"User's current emotion: {state.primary_emotion} "
                            f"(intensity: {state.intensity:.0%}). "
                            f"Emotional trend: {trend.get('valence_direction', 'stable')}."
                        )
                except Exception:  # noqa: S110
                    pass

            relationship_context = ""
            if self._relationship_tracker and user_id:
                try:
                    relationship_context = self._relationship_tracker.get_relationship_context(
                        user_id, persona_id,
                    )
                except Exception:  # noqa: S110
                    pass

            try:
                full_prompt = self._persona_manager.build_system_prompt(
                    persona_id,
                    user_profile_text=user_profile_text,
                    emotional_context=emotional_context,
                    relationship_context=relationship_context,
                )
                if full_prompt:
                    return full_prompt
            except Exception as e:
                logger.debug("Persona prompt build failed: %s", e)

        # No persona — fall back to individual context blocks
        if self._emotion_tracker:
            try:
                state = self._emotion_tracker.get_current_state()
                if state.primary_emotion != "neutral":
                    parts.append(
                        f"The user seems to be feeling {state.primary_emotion} "
                        f"(intensity: {state.intensity:.0%}). "
                        "Adjust your tone accordingly."
                    )
            except Exception:  # noqa: S110
                pass

        return "\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

            # Use model from config if not specified
            if model is None:
                model = self.config.default_model

            # Initialize variables that both COA and standard paths need
            knowledge = []
            user_messages = []
            query = ""

            # Get conversation context if no prompt provided
            if prompt is None:
                context = self._get_conversation_context()

                # Extract messages
                messages = context["messages"]

                # Add knowledge as system message if available
                knowledge = context.get("knowledge", [])
                if knowledge:
                    knowledge_text = "\n".join(
                        item.get("text", item.get("content", ""))
                        for item in knowledge
                        if item.get("text") or item.get("content")
                    )

                    # Check for contradictions among retrieved knowledge items
                    contradiction_note = ""
                    if self.uncertainty_handler and knowledge_text:
                        try:
                            # Extract entity IDs from knowledge items and check for contradictions
                            seen_entity_ids = set()
                            all_contradictions = []
                            for item in knowledge:
                                entity_id = item.get("entity_id")
                                if entity_id and entity_id not in seen_entity_ids:
                                    seen_entity_ids.add(entity_id)
                                    contradictions = self.uncertainty_handler.detect_contradictions(
                                        entity_id=entity_id, max_results=3
                                    )
                                    all_contradictions.extend(contradictions)

                            if all_contradictions:
                                contradiction_details = []
                                for c in all_contradictions[:3]:  # Limit to top 3
                                    contradiction_details.append(
                                        f"  - {c.get('entity', 'unknown')}: "
                                        f"'{c.get('target1', '?')}' vs '{c.get('target2', '?')}'"
                                    )
                                contradiction_note = (
                                    "\n\nNote: The following knowledge items have detected contradictions. "
                                    "Please acknowledge the uncertainty and present the most reliable information:\n"
                                    + "\n".join(contradiction_details)
                                )
                                logger.info(f"Found {len(all_contradictions)} contradictions in retrieved knowledge")
                        except Exception as e:
                            logger.debug(f"Contradiction detection skipped: {e}")

                    if knowledge_text:
                        # Add knowledge context as a system message
                        system_content = f"Use this knowledge to answer the question:\n{knowledge_text}"
                        if contradiction_note:
                            system_content += contradiction_note
                        messages = [{"role": "system", "content": system_content}] + messages

                # Inject companion AI context (persona + emotion + relationship)
                companion_ctx = self._build_companion_system_message()
                if companion_ctx:
                    messages = [{"role": "system", "content": companion_ctx}] + messages

                # Format as prompt if needed
                if not messages:
                    prompt = "Hello! How can I assist you today?"

                # Extract query from messages -- needed by both COA and standard paths
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                query = user_messages[-1]["content"] if user_messages else ""

                # Enrich context with inference results if available
                inference_context = ""
                if (hasattr(self, 'knowledge_store') and self.knowledge_store and
                        hasattr(self.knowledge_store, 'inference_engine') and
                        self.knowledge_store.inference_engine and query):
                    try:
                        inference_engine = self.knowledge_store.inference_engine
                        # Try backward chaining to find relevant logical derivations
                        fact_pattern = inference_engine._extract_fact_from_question(query)
                        if fact_pattern:
                            success, explanation = inference_engine.backward_chain(fact_pattern)
                            if success and explanation:
                                inference_parts = []
                                for step in explanation[:5]:
                                    fact = step.get('fact', {})
                                    if isinstance(fact, dict):
                                        source = fact.get('source', '')
                                        relation = fact.get('relation', '')
                                        target = fact.get('target', '')
                                        if source and relation and target:
                                            inference_parts.append(f"- {source} {relation} {target}")
                                    elif fact:
                                        inference_parts.append(f"- {fact}")
                                if inference_parts:
                                    inference_context = "Logical inference results:\n" + "\n".join(inference_parts)
                                    logger.info(f"Inference engine provided {len(inference_parts)} derivation steps")
                    except Exception as e:
                        logger.debug(f"Inference enrichment skipped: {e}")

                # Prepend inference context to messages if available
                if inference_context:
                    # Add as a system message before the knowledge system message
                    messages = [{"role": "system", "content": inference_context}] + messages

                # Use Chain of Agents for complex queries if enabled
                if (self.agent_chain_manager is not None and
                    hasattr(self.config, "use_chain_of_agents") and
                    self.config.use_chain_of_agents):

                    try:
                        logger.info(f"Processing query with Chain of Agents: {query[:50]}...")

                        # process_query uses _is_complex_query() internally and
                        # returns skipped=True for simple queries
                        coa_result = self.agent_chain_manager.process_query(
                            query=query,
                            context={"messages": messages, "knowledge": knowledge}
                        )

                        if not coa_result.get("skipped"):
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
                                self._add_message("assistant", generated_text)
                                logger.info(f"Chain of Agents generated response in {coa_result.get('total_processing_time', 0):.2f} seconds")
                                return generated_text
                            # If Chain of Agents produced no answer, fall back to standard processing
                            logger.warning("Chain of Agents failed to generate response, falling back to standard processing")
                        else:
                            logger.info(f"Chain of Agents skipped (simple query): {coa_result.get('reason', '')}")
                    except Exception as e:
                        logger.error(f"Error processing with Chain of Agents: {e}")
                        logger.error(traceback.format_exc())
                        # Continue with standard processing on error
            else:
                messages = [{"role": "user", "content": prompt}]
                # Extract query from the provided prompt for reflection
                user_messages = [{"role": "user", "content": prompt}]
                query = prompt

            logger.debug("Sending request to LLM client")
            generated_text = self.llm_client.generate(messages, model=model)

            if not generated_text or generated_text.startswith("Error:"):
                return f"Error generating response: {generated_text}"

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
            self._add_message("assistant", generated_text)

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def generate_response_stream(self, prompt: str = None, model: str = None) -> Iterator[str]:
        """
        Generate a streaming response using the conversation context.

        Note: Chain of Agents processing is not supported in streaming mode.
        If COA is enabled and the query is complex, this method will run COA
        synchronously first and then stream the resulting text.

        Args:
            prompt: Optional prompt to use instead of the conversation context
            model: Model to use for generation

        Yields:
            Chunks of the generated response
        """
        try:

            # Use model from config if not specified
            if model is None:
                model = self.config.default_model

            # Get conversation context if no prompt provided
            if prompt is None:
                context = self._get_conversation_context()

                # Extract messages
                messages = context["messages"]

                # Add knowledge as system message if available
                knowledge = context.get("knowledge", [])
                if knowledge:
                    knowledge_text = "\n".join(
                        item.get("text", item.get("content", ""))
                        for item in knowledge
                        if item.get("text") or item.get("content")
                    )

                    if knowledge_text:
                        # Add knowledge context as a system message
                        messages = [{"role": "system", "content": f"Use this knowledge to answer the question:\n{knowledge_text}"}] + messages

                # Format as prompt if needed
                if not messages:
                    prompt = "Hello! How can I assist you today?"

                # Check if Chain of Agents should handle this query
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                query = user_messages[-1]["content"] if user_messages else ""

                if (self.agent_chain_manager is not None and
                    hasattr(self.config, "use_chain_of_agents") and
                    self.config.use_chain_of_agents and query):

                    try:
                        logger.info("Streaming mode: running COA synchronously before streaming result")
                        coa_result = self.agent_chain_manager.process_query(
                            query=query,
                            context={"messages": messages, "knowledge": knowledge}
                        )

                        if not coa_result.get("skipped"):
                            generated_text = coa_result.get("answer", "")
                            if generated_text:
                                # Stream the COA result character-by-character in chunks
                                self._add_message("assistant", generated_text)
                                chunk_size = 20
                                for i in range(0, len(generated_text), chunk_size):
                                    yield generated_text[i:i + chunk_size]
                                return
                    except Exception as e:
                        logger.error(f"COA failed in streaming mode, falling back to standard streaming: {e}")
            else:
                messages = [{"role": "user", "content": prompt}]

            logger.debug("Sending streaming request to LLM client")
            full_response = ""
            for chunk in self.llm_client.generate_stream(messages, model=model):
                full_response += chunk
                yield chunk

            # After streaming completes, add the response to memory
            self._add_message("assistant", full_response)

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            logger.error(traceback.format_exc())
            yield f"Error generating streaming response: {str(e)}"
