"""
CortexFlow Manager module.

This module provides the main manager class for the CortexFlow system.
The heavy lifting is delegated to focused domain classes:

- :class:`~cortexflow.response_orchestrator.ResponseOrchestrator`
- :class:`~cortexflow.knowledge_coordinator.KnowledgeCoordinator`
- :class:`~cortexflow.reasoning_facade.ReasoningFacade`

CortexFlowManager acts as a thin **facade** that preserves backward
compatibility -- every public method signature and return type is
unchanged.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from typing import Any

from cortexflow.classifier import ContentClassifier
from cortexflow.interfaces import ContextProvider
from cortexflow.knowledge import KnowledgeStore
from cortexflow.knowledge_coordinator import KnowledgeCoordinator
from cortexflow.llm_client import create_llm_client
from cortexflow.memory import (
    ConversationMemory,
)
from cortexflow.reasoning_facade import ReasoningFacade

# Domain delegates
from cortexflow.response_orchestrator import ResponseOrchestrator

# Add import for Chain of Agents
try:
    from cortexflow.agent_chain import AgentChainManager

    AGENT_CHAIN_ENABLED = True
except ImportError:
    AGENT_CHAIN_ENABLED = False
    logger = logging.getLogger("cortexflow")
    logger.warning(
        "agent_chain module not found. Chain of Agents functionality will be disabled."
    )

# Add import for Self-Reflection
try:
    from cortexflow.reflection import ReflectionEngine

    REFLECTION_ENABLED = True
except ImportError:
    REFLECTION_ENABLED = False
    logger = logging.getLogger("cortexflow")
    logger.warning(
        "reflection module not found. Self-Reflection functionality will be disabled."
    )

# Add import for Dynamic Weighting
try:
    from cortexflow.dynamic_weighting import DynamicWeightingEngine

    DYNAMIC_WEIGHTING_ENABLED = True
except ImportError:
    DYNAMIC_WEIGHTING_ENABLED = False
    logger = logging.getLogger("cortexflow")
    logger.warning(
        "dynamic_weighting module not found. Dynamic Weighting functionality will be disabled."
    )

# Add import for Uncertainty Handling
try:
    from cortexflow.uncertainty_handler import UncertaintyHandler

    UNCERTAINTY_HANDLING_ENABLED = True
except ImportError:
    UNCERTAINTY_HANDLING_ENABLED = False
    logger = logging.getLogger("cortexflow")
    logger.warning(
        "uncertainty_handler module not found. Uncertainty handling functionality will be disabled."
    )

# Add import for Performance Optimization
try:
    from cortexflow.performance_optimizer import PerformanceOptimizer

    PERFORMANCE_OPTIMIZATION_ENABLED = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_ENABLED = False
    logger = logging.getLogger("cortexflow")
    logger.warning(
        "performance_optimizer module not found. Performance optimization functionality will be disabled."
    )

logger = logging.getLogger("cortexflow")


def configure_logging(verbose: bool = False):
    """Configure logging for the cortexflow module.

    Call this from your application if you want CortexFlow log output.
    The library itself never calls ``logging.basicConfig()`` so it will
    not hijack the root logger of importing applications.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    cortex_logger = logging.getLogger("cortexflow")
    cortex_logger.setLevel(level)
    if not cortex_logger.handlers:
        cortex_logger.addHandler(handler)


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
        verbose = (
            self.config.verbose_logging
            if hasattr(self.config, "verbose_logging")
            else False
        )
        configure_logging(verbose)

        # Log initialization
        logger.info(
            f"Initializing CortexFlowManager with {self.config.active_token_limit} active tokens, "
            f"{self.config.working_token_limit} working tokens, {self.config.archive_token_limit} archive tokens"
        )

        try:
            # Initialize LLM client
            self.llm_client = create_llm_client(self.config)

            # Initialize components
            self.knowledge_store = KnowledgeStore(self.config)
            self.memory = ConversationMemory(self.config)

            # Initialize content classifier if enabled
            if (
                hasattr(self.config, "use_ml_classifier")
                and self.config.use_ml_classifier
            ):
                self.classifier = ContentClassifier(self.config)
            else:
                self.classifier = None

            # Initialize Agent Chain Manager if enabled
            self.agent_chain_manager = None
            if (
                hasattr(self.config, "use_chain_of_agents")
                and self.config.use_chain_of_agents
                and AGENT_CHAIN_ENABLED
            ):
                try:
                    self.agent_chain_manager = AgentChainManager(
                        self.config, self.knowledge_store
                    )
                    logger.info("Chain of Agents initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Chain of Agents: {e}")

            # Initialize Reflection Engine if enabled
            self.reflection_engine = None
            if (
                hasattr(self.config, "use_self_reflection")
                and self.config.use_self_reflection
                and REFLECTION_ENABLED
            ):
                try:
                    self.reflection_engine = ReflectionEngine(
                        self.config, self.knowledge_store
                    )
                    logger.info("Self-Reflection Engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Self-Reflection Engine: {e}")

            # Initialize Dynamic Weighting Engine if enabled
            self.weighting_engine = None
            if (
                hasattr(self.config, "use_dynamic_weighting")
                and self.config.use_dynamic_weighting
                and DYNAMIC_WEIGHTING_ENABLED
            ):
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
            if (
                hasattr(self.config, "use_uncertainty_handling")
                and self.config.use_uncertainty_handling
                and UNCERTAINTY_HANDLING_ENABLED
            ):
                try:
                    # Pass the graph store to the uncertainty handler
                    graph_store = (
                        self.knowledge_store.graph_store
                        if hasattr(self.knowledge_store, "graph_store")
                        else None
                    )
                    self.uncertainty_handler = UncertaintyHandler(
                        self.config, graph_store
                    )
                    logger.info("Uncertainty Handler initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Uncertainty Handler: {e}")

            # Initialize Performance Optimizer if enabled
            self.performance_optimizer = None
            if (
                hasattr(self.config, "use_performance_optimization")
                and self.config.use_performance_optimization
                and PERFORMANCE_OPTIMIZATION_ENABLED
            ):
                try:
                    # Pass the graph store to the performance optimizer
                    graph_store = (
                        self.knowledge_store.graph_store
                        if hasattr(self.knowledge_store, "graph_store")
                        else None
                    )
                    self.performance_optimizer = PerformanceOptimizer(
                        self.config, graph_store
                    )
                    logger.info("Performance Optimizer initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Performance Optimizer: {e}")

            # Initialize Personal Fact Detector if enabled
            self.fact_detector = None
            if (
                hasattr(self.config, "use_fact_extraction")
                and self.config.use_fact_extraction
            ):
                try:
                    from cortexflow.fact_detector import PersonalFactDetector

                    self.fact_detector = PersonalFactDetector(use_spacy=False)
                    logger.info("Personal Fact Detector initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Personal Fact Detector: {e}")

            # ---- Companion AI subsystems (Phase 1–3) ----

            # Session management
            self._session_manager = None
            self._current_session = None
            if getattr(self.config, "enable_sessions", False):
                try:
                    from cortexflow.session import SessionManager

                    self._session_manager = SessionManager(
                        db_path=self.config.session.session_db_path,
                        session_ttl=self.config.session.session_ttl,
                        max_sessions_per_user=self.config.session.max_sessions_per_user,
                    )
                    logger.info("Session Manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Session Manager: {e}")

            # Emotion tracking
            self._emotion_tracker = None
            if getattr(self.config, "use_emotion_tracking", False):
                try:
                    from cortexflow.emotion import (
                        EmotionTracker,
                        LLMEmotionDetector,
                        RuleBasedEmotionDetector,
                    )

                    detector_type = self.config.emotion.emotion_detector
                    if detector_type == "llm":
                        detector = LLMEmotionDetector(self.llm_client)
                    else:
                        detector = RuleBasedEmotionDetector()
                    self._emotion_tracker = EmotionTracker(
                        detector,
                        window_size=self.config.emotion.emotion_window_size,
                    )
                    logger.info(
                        "Emotion Tracker initialized (detector=%s)", detector_type
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize Emotion Tracker: {e}")

            # User profile manager
            self._user_profile_manager = None
            if getattr(self.config, "enable_sessions", False):
                try:
                    from cortexflow.user_profile import UserProfileManager

                    self._user_profile_manager = UserProfileManager(
                        db_path=self.config.session.session_db_path,
                    )
                    logger.info("User Profile Manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize User Profile Manager: {e}")

            # Persona management
            self._persona_manager = None
            if getattr(self.config, "use_personas", False):
                try:
                    from cortexflow.persona import PersonaManager

                    self._persona_manager = PersonaManager(
                        db_path=self.config.persona.persona_db_path,
                    )
                    logger.info("Persona Manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Persona Manager: {e}")

            # Relationship tracking
            self._relationship_tracker = None
            if getattr(self.config, "use_relationship_tracking", False):
                try:
                    from cortexflow.relationship import RelationshipTracker

                    self._relationship_tracker = RelationshipTracker(
                        db_path=self.config.relationship.relationship_db_path,
                    )
                    logger.info("Relationship Tracker initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Relationship Tracker: {e}")

            # ---- Event system (gated by config) ----
            self._event_bus = None
            if getattr(self.config, "use_events", False):
                try:
                    from cortexflow.events import EventBus, EventType

                    self._event_bus = EventBus()
                    logger.info("Event Bus initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Event Bus: {e}")

            # ---- Safety pipeline (gated by config) ----
            self._safety_pipeline = None
            if getattr(self.config, "use_safety_pipeline", False):
                try:
                    from cortexflow.safety import SafetyPipeline

                    self._safety_pipeline = SafetyPipeline(
                        enable_pii_detection=self.config.safety.enable_pii_detection,
                        enable_boundary_enforcement=self.config.safety.enable_boundary_enforcement,
                        custom_blocked_patterns=self.config.safety.custom_blocked_patterns,
                    )
                    logger.info("Safety Pipeline initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Safety Pipeline: {e}")

            # ---- Temporal & Episodic subsystems (gated by config) ----
            self._temporal_manager = None
            if getattr(self.config, "use_temporal_facts", False):
                try:
                    from cortexflow.temporal import TemporalManager

                    self._temporal_manager = TemporalManager(
                        db_path=self.config.temporal.temporal_db_path,
                    )
                    logger.info("Temporal Manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Temporal Manager: {e}")

            self._episodic_store = None
            if getattr(self.config, "use_episodic_memory", False):
                try:
                    from cortexflow.episodic_memory import EpisodicMemoryStore

                    self._episodic_store = EpisodicMemoryStore(
                        db_path=self.config.episodic.episodic_db_path,
                    )
                    logger.info("Episodic Memory Store initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Episodic Memory Store: {e}")

            # ---- Instantiate domain delegates ----
            self._response_orchestrator = ResponseOrchestrator(
                config=self.config,
                llm_client=self.llm_client,
                memory=self.memory,
                knowledge_store=self.knowledge_store,
                agent_chain_manager=self.agent_chain_manager,
                reflection_engine=self.reflection_engine,
                uncertainty_handler=self.uncertainty_handler,
                add_message_fn=self.add_message,
                get_conversation_context_fn=self.get_conversation_context,
            )

            # Wire companion AI references into the orchestrator
            self._response_orchestrator._emotion_tracker = self._emotion_tracker
            self._response_orchestrator._persona_manager = self._persona_manager
            self._response_orchestrator._user_profile_manager = (
                self._user_profile_manager
            )
            self._response_orchestrator._relationship_tracker = (
                self._relationship_tracker
            )
            self._response_orchestrator._get_current_session_fn = (
                self.get_current_session
            )

            self._knowledge_coordinator = KnowledgeCoordinator(
                config=self.config,
                knowledge_store=self.knowledge_store,
                uncertainty_handler=self.uncertainty_handler,
            )

            self._reasoning_facade = ReasoningFacade(
                config=self.config,
                performance_optimizer=self.performance_optimizer,
            )

            logger.info("CortexFlowManager initialized")

            if self._event_bus:
                from cortexflow.events import EventType

                self._event_bus.emit_typed(
                    EventType.MANAGER_INITIALIZED, source="manager"
                )

        except Exception as e:
            logger.error(f"Error initializing CortexFlowManager: {e}")
            logger.error(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Event bus accessor
    # ------------------------------------------------------------------

    @property
    def event_bus(self):
        """Access the event bus (None if not enabled)."""
        return self._event_bus

    @property
    def safety_pipeline(self):
        """Access the safety pipeline (None if not enabled)."""
        return self._safety_pipeline

    # ------------------------------------------------------------------
    # Internal helpers (kept on the facade)
    # ------------------------------------------------------------------

    def _update_memory_tier_limits(self, new_limits: dict[str, int]) -> None:
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
                working_limit=new_limits.get(
                    "working", self.config.working_token_limit
                ),
                archive_limit=new_limits.get(
                    "archive", self.config.archive_token_limit
                ),
            )
            logger.info(f"Updated memory tier limits: {new_limits}")
        except Exception as e:
            logger.error(f"Error updating memory tier limits: {e}")

    # ------------------------------------------------------------------
    # Memory-layer methods (stay on the facade)
    # ------------------------------------------------------------------

    def add_message(
        self, role: str, content: str, metadata: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Add a message to the conversation.

        Args:
            role: Message role (e.g., user, assistant, system)
            content: Message content
            metadata: Optional metadata for the message

        Returns:
            Message object that was added
        """
        # Safety check (runs before any other processing)
        if self._safety_pipeline and role == "user":
            from cortexflow.safety import SafetyLevel

            safety_result = self._safety_pipeline.check(content)
            if (
                safety_result.level == SafetyLevel.BLOCKED
                and self.config.safety.block_on_safety_violation
            ):
                logger.warning(
                    f"Message blocked by safety pipeline: {safety_result.reason}"
                )
                return {
                    "role": role,
                    "content": "[Message blocked by safety filter]",
                    "blocked": True,
                    "safety": safety_result.triggered_rules,
                }
            if safety_result.filtered_content:
                content = safety_result.filtered_content
            if metadata is None:
                metadata = {}
            if safety_result.triggered_rules:
                metadata["safety_flags"] = safety_result.triggered_rules

        # Perform classification and set importance metadata before adding to memory
        if role == "user" and self.classifier is not None:
            try:
                result = self.classifier.classify(content)
                metadata = metadata or {}
                metadata["classification"] = result
                # Derive importance from classification confidence (scale 0-1 to 0-10)
                confidence = result.get("confidence", 0.5)
                is_question = result.get("is_question", False)
                # Questions and high-confidence content get higher importance
                importance_score = confidence * 7.0 + (3.0 if is_question else 1.0)
                importance_score = min(10.0, max(0.0, importance_score))
                metadata["importance"] = importance_score
                logger.debug(
                    f"Message classified as: {result}, importance: {importance_score:.1f}"
                )
            except Exception as e:
                logger.error(f"Error classifying message: {e}")
        elif role == "assistant":
            # For assistant messages, use a moderate default importance
            metadata = metadata or {}
            if "importance" not in metadata:
                metadata["importance"] = 5.0

        # Tag message with session context if sessions are active
        if self._current_session is not None:
            metadata = metadata or {}
            metadata["session_id"] = self._current_session.session_id
            metadata["user_id"] = self._current_session.user_id

        message = self.memory.add_message(role, content, metadata)

        # Emotion detection
        if role == "user" and self._emotion_tracker is not None:
            try:
                emotional_state = self._emotion_tracker.process_message(content)
                message["metadata"] = message.get("metadata", {})
                message["metadata"]["emotional_state"] = emotional_state.to_dict()
            except Exception as e:
                logger.debug(f"Emotion detection failed: {e}")
                emotional_state = None
        else:
            emotional_state = None

        # User profile update
        if role == "user" and self._user_profile_manager is not None:
            try:
                user_id = (
                    self._current_session.user_id
                    if self._current_session
                    else self.config.session.default_user_id
                )
                self._user_profile_manager.update_from_message(
                    user_id,
                    content,
                    self.fact_detector,
                )
            except Exception as e:
                logger.debug(f"User profile update failed: {e}")

        # Relationship tracking
        if self._relationship_tracker is not None and self._current_session is not None:
            try:
                persona_id = self._current_session.persona_id or "default"
                self._relationship_tracker.update(
                    user_id=self._current_session.user_id,
                    persona_id=persona_id,
                    message=content,
                    role=role,
                    emotional_state=emotional_state,
                )
            except Exception as e:
                logger.debug(f"Relationship tracking failed: {e}")

        # ---- Temporal fact extraction ----
        if self._temporal_manager and self.fact_detector and role == "user":
            try:
                facts = self.fact_detector.detect_facts(content)
                for fact_dict in facts:
                    from datetime import datetime

                    from cortexflow.temporal import TemporalFact

                    tf = TemporalFact(
                        subject=fact_dict.get("subject", "user"),
                        predicate=fact_dict.get("fact_type", "states"),
                        object=fact_dict.get("value", content[:100]),
                        valid_from=datetime.utcnow().isoformat(),
                        confidence=fact_dict.get("confidence", 0.8),
                        source="conversation",
                    )
                    self._temporal_manager.add_temporal_fact(tf)
                    if self._event_bus:
                        from cortexflow.events import EventType

                        self._event_bus.emit_typed(
                            EventType.TEMPORAL_FACT_ADDED,
                            data={
                                "subject": tf.subject,
                                "predicate": tf.predicate,
                                "object": tf.object,
                            },
                            source="manager",
                        )
            except Exception as e:
                logger.debug(f"Temporal fact extraction failed: {e}")

        # Dual-write: extract personal facts and store in knowledge store
        if role == "user" and self.fact_detector is not None:
            try:
                facts = self.fact_detector.detect_facts(content)
                for fact in facts:
                    self.knowledge_store.add_knowledge(
                        fact["fact_text"],
                        source="conversation_extract",
                        confidence=0.9,
                    )
                    logger.debug(f"Extracted personal fact: {fact['fact_text']}")
            except Exception as e:
                logger.error(f"Error extracting personal facts: {e}")

        # Apply dynamic weighting for user queries
        if (
            role == "user"
            and self.weighting_engine
            and self.config.use_dynamic_weighting
        ):
            try:
                # Get recent context for document type analysis
                context_messages = self.memory.get_context_messages()
                context_content = "\n".join(
                    [m.get("content", "") for m in context_messages[-5:]]
                )

                # Process query and update memory tier allocation
                new_limits = self.weighting_engine.process_query(
                    content, context_content
                )

                # Update memory tier limits
                self._update_memory_tier_limits(new_limits)
            except Exception as e:
                logger.error(f"Error in dynamic weighting: {e}")

        if self._event_bus:
            from cortexflow.events import EventType

            self._event_bus.emit_typed(
                EventType.MESSAGE_ADDED,
                data={"role": role, "content": content},
                source="manager",
            )

        return message

    def get_conversation_context(self, max_tokens: int = None) -> dict[str, Any]:
        """
        Get the full conversation context for generating a response.

        Args:
            max_tokens: Maximum tokens for the context

        Returns:
            Context with messages and knowledge
        """
        context = {"messages": self.memory.get_context_messages(), "knowledge": []}

        # Get the most recent user message for knowledge retrieval
        user_messages = self.memory.get_messages_by_role("user")
        if user_messages:
            last_user_message = user_messages[-1]["content"]

            # Retrieve relevant knowledge for the user's message
            knowledge_items = self.knowledge_store.get_relevant_knowledge(
                last_user_message
            )

            # Apply self-reflection to verify knowledge relevance if enabled
            if (
                self.reflection_engine
                and hasattr(self.config, "use_self_reflection")
                and self.config.use_self_reflection
            ):
                try:
                    knowledge_items = self.reflection_engine.verify_knowledge_relevance(
                        last_user_message, knowledge_items
                    )
                    logger.info(
                        f"Knowledge relevance verification applied: {len(knowledge_items)} items retained"
                    )
                except Exception as e:
                    logger.error(f"Error in knowledge relevance verification: {e}")

            context["knowledge"] = knowledge_items

            if self._event_bus:
                from cortexflow.events import EventType

                self._event_bus.emit_typed(
                    EventType.KNOWLEDGE_RETRIEVED,
                    data={"query": last_user_message},
                    source="manager",
                )

        return context

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear_memory()

        # Reset dynamic weighting to defaults if enabled
        if self.config.use_dynamic_weighting and self.weighting_engine:
            self.reset_dynamic_weighting()

        if self._event_bus:
            from cortexflow.events import EventType

            self._event_bus.emit_typed(EventType.MEMORY_CLEARED, source="manager")

    # ------------------------------------------------------------------
    # Session management (Phase 1A)
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        persona_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Create a new session and make it active."""
        if self._session_manager is None:
            raise RuntimeError(
                "Sessions are not enabled. Use ConfigBuilder.with_sessions() to enable."
            )
        session = self._session_manager.create_session(user_id, persona_id, metadata)
        self._current_session = session
        return session

    def switch_session(self, session_id: str):
        """Switch to an existing session."""
        if self._session_manager is None:
            raise RuntimeError("Sessions are not enabled.")
        session = self._session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        self._current_session = session
        return session

    def get_current_session(self):
        """Return the currently active session context, or ``None``."""
        return self._current_session

    def close_session(self, session_id: str | None = None) -> bool:
        """Close a session (defaults to current)."""
        if self._session_manager is None:
            return False
        sid = session_id or (
            self._current_session.session_id if self._current_session else None
        )
        if sid is None:
            return False

        session = self._current_session

        # Auto-save episode on session close
        if self._episodic_store and session:
            try:
                from datetime import datetime

                from cortexflow.episodic_memory import Episode

                # Gather recent messages from memory
                messages = []
                if hasattr(self.memory, "get_messages"):
                    messages = self.memory.get_messages()
                elif hasattr(self.memory, "messages"):
                    messages = self.memory.messages

                emotions = []
                if self._emotion_tracker:
                    state = self._emotion_tracker.get_current_state()
                    if state and state.primary_emotion:
                        emotions = [state.primary_emotion]

                episode = Episode(
                    session_id=session.session_id
                    if hasattr(session, "session_id")
                    else str(session),
                    user_id=getattr(session, "user_id", None),
                    title="Session conversation",
                    summary=f"Conversation with {len(messages)} messages",
                    messages=[
                        {
                            "role": m.get("role", ""),
                            "content": m.get("content", "")[:200],
                        }
                        for m in messages[-20:]
                    ],  # last 20
                    emotions=emotions,
                    end_time=datetime.utcnow().isoformat(),
                )
                self._episodic_store.save_episode(episode)
                logger.debug("Episode saved for session close")
            except Exception as e:
                logger.debug(f"Episode auto-save failed: {e}")

        result = self._session_manager.close_session(sid)
        if self._current_session and self._current_session.session_id == sid:
            self._current_session = None
        return result

    # ------------------------------------------------------------------
    # Companion AI accessors (Phase 2–3)
    # ------------------------------------------------------------------

    @property
    def emotion_tracker(self):
        """Return the EmotionTracker instance, or ``None``."""
        return self._emotion_tracker

    @property
    def user_profile_manager(self):
        """Return the UserProfileManager instance, or ``None``."""
        return self._user_profile_manager

    @property
    def persona_manager(self):
        """Return the PersonaManager instance, or ``None``."""
        return self._persona_manager

    @property
    def relationship_tracker(self):
        """Return the RelationshipTracker instance, or ``None``."""
        return self._relationship_tracker

    @property
    def temporal_manager(self):
        """Access the temporal fact manager (None if not enabled)."""
        return self._temporal_manager

    @property
    def episodic_store(self):
        """Access the episodic memory store (None if not enabled)."""
        return self._episodic_store

    # ------------------------------------------------------------------
    # Temporal & Episodic convenience methods
    # ------------------------------------------------------------------

    def get_temporal_facts(self, subject: str | None = None) -> list:
        """Get current temporal facts, optionally filtered by subject."""
        if not self._temporal_manager:
            return []
        from datetime import datetime

        return self._temporal_manager.get_facts_at_time(
            datetime.utcnow().isoformat(), subject=subject
        )

    def get_recent_episodes(self, user_id: str | None = None, limit: int = 10) -> list:
        """Get recent episodes from episodic memory."""
        if not self._episodic_store:
            return []
        return self._episodic_store.get_recent_episodes(user_id=user_id, limit=limit)

    def search_episodes(self, query: str, max_results: int = 5) -> list:
        """Search episodic memory by text query."""
        if not self._episodic_store:
            return []
        return self._episodic_store.recall_episodes(query, max_results=max_results)

    # ------------------------------------------------------------------
    # Dynamic weighting (stays on facade -- thin and cross-cutting)
    # ------------------------------------------------------------------

    def get_dynamic_weighting_stats(self) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Stats (cross-cutting, stays on facade)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """
        Get system-wide statistics.

        Returns:
            Dictionary with various statistics
        """
        stats = {
            "memory": {
                "messages": self.memory.get_message_count()
                if hasattr(self.memory, "get_message_count")
                else None,
                "total_tokens": self.memory.get_total_tokens()
                if hasattr(self.memory, "get_total_tokens")
                else None,
                "tiers": self.memory.get_tier_stats()
                if hasattr(self.memory, "get_tier_stats")
                else None,
            },
            "knowledge": {
                "facts": self.knowledge_store.get_fact_count()
                if hasattr(self.knowledge_store, "get_fact_count")
                else None,
                "embeddings": self.knowledge_store.get_embedding_count()
                if hasattr(self.knowledge_store, "get_embedding_count")
                else None,
                "sources": self.knowledge_store.get_source_count()
                if hasattr(self.knowledge_store, "get_source_count")
                else None,
            },
        }

        # Add dynamic weighting stats if available
        if self.weighting_engine:
            try:
                stats["dynamic_weighting"] = self.get_dynamic_weighting_stats()
            except Exception:  # noqa: S110
                pass

        # Add uncertainty handling stats if available
        if self.uncertainty_handler:
            try:
                stats["uncertainty"] = self.uncertainty_handler.get_stats()
            except Exception:  # noqa: S110
                pass

        # Add performance optimization stats if available
        if self.performance_optimizer:
            try:
                stats["performance"] = self.performance_optimizer.get_stats()
            except Exception:  # noqa: S110
                pass

        return stats

    # ------------------------------------------------------------------
    # Response generation (delegated to ResponseOrchestrator)
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
        response = self._response_orchestrator.generate_response(prompt, model)

        if self._event_bus:
            from cortexflow.events import EventType

            self._event_bus.emit_typed(
                EventType.RESPONSE_GENERATED,
                data={"response": response[:200]},
                source="manager",
            )

        return response

    def generate_response_stream(
        self, prompt: str = None, model: str = None
    ) -> Iterator[str]:
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
        return self._response_orchestrator.generate_response_stream(prompt, model)

    # ------------------------------------------------------------------
    # Knowledge operations (delegated to KnowledgeCoordinator)
    # ------------------------------------------------------------------

    def remember_knowledge(
        self, text: str, source: str = None, confidence: float = None
    ) -> list[int]:
        """
        Store important knowledge in the knowledge store.

        DEPRECATED: Use add_knowledge() instead.

        Args:
            text: Text to remember
            source: Optional source of the knowledge
            confidence: Optional confidence value for the knowledge

        Returns:
            List of IDs for the stored knowledge
        """
        return self._knowledge_coordinator.remember_knowledge(text, source, confidence)

    def add_knowledge(
        self, text: str, source: str = None, confidence: float = None
    ) -> list[int]:
        """
        Store important knowledge in the knowledge store.

        Args:
            text: Text to remember
            source: Optional source of the knowledge
            confidence: Optional confidence value for the knowledge

        Returns:
            List of IDs for the stored knowledge
        """
        result = self._knowledge_coordinator.add_knowledge(text, source, confidence)

        if self._event_bus:
            from cortexflow.events import EventType

            self._event_bus.emit_typed(
                EventType.KNOWLEDGE_ADDED,
                data={"text": text, "source": source},
                source="manager",
            )

        return result

    def detect_contradictions(
        self, entity_id=None, relation_type=None, max_results=100
    ) -> list[dict[str, Any]]:
        """
        Detect contradictions in the knowledge graph.

        Args:
            entity_id: Optional entity ID to check
            relation_type: Optional relation type to check
            max_results: Maximum number of results to return

        Returns:
            List of detected contradictions
        """
        return self._knowledge_coordinator.detect_contradictions(
            entity_id, relation_type, max_results
        )

    def resolve_contradiction(
        self, contradiction: dict[str, Any], strategy: str = None
    ) -> dict[str, Any]:
        """
        Resolve a contradiction using the specified strategy.

        Args:
            contradiction: Contradiction to resolve
            strategy: Resolution strategy (auto, recency, confidence, reliability, or keep_both)

        Returns:
            Resolution result
        """
        return self._knowledge_coordinator.resolve_contradiction(
            contradiction, strategy
        )

    def update_source_reliability(
        self,
        source_name: str,
        reliability_score: float,
        metadata: dict[str, Any] = None,
    ) -> None:
        """
        Update the reliability score for a knowledge source.

        Args:
            source_name: Name of the source
            reliability_score: Reliability score (0.0-1.0)
            metadata: Optional metadata about the source
        """
        return self._knowledge_coordinator.update_source_reliability(
            source_name, reliability_score, metadata
        )

    def get_source_reliability(self, source_name: str) -> float:
        """
        Get the reliability score for a knowledge source.

        Args:
            source_name: Name of the source

        Returns:
            Reliability score (0.0-1.0)
        """
        return self._knowledge_coordinator.get_source_reliability(source_name)

    def add_probability_distribution(
        self,
        entity_id: int,
        relation_id: int,
        distribution_type: str,
        distribution_data: dict[str, Any],
    ) -> None:
        """
        Add a probability distribution to represent uncertainty about a fact.

        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            distribution_type: Type of distribution (discrete, gaussian, etc.)
            distribution_data: Data representing the distribution
        """
        return self._knowledge_coordinator.add_probability_distribution(
            entity_id, relation_id, distribution_type, distribution_data
        )

    def get_probability_distribution(
        self, entity_id: int, relation_id: int
    ) -> dict[str, Any] | None:
        """
        Get the probability distribution for a fact.

        Args:
            entity_id: Entity ID
            relation_id: Relation ID

        Returns:
            Probability distribution data or None if not found
        """
        return self._knowledge_coordinator.get_probability_distribution(
            entity_id, relation_id
        )

    def reason_with_incomplete_information(
        self, query: dict[str, Any], available_knowledge: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Reason with incomplete information to provide best possible answers.

        Args:
            query: The query to answer
            available_knowledge: Available knowledge to reason with

        Returns:
            Reasoning result with confidence and explanation
        """
        return self._knowledge_coordinator.reason_with_incomplete_information(
            query, available_knowledge
        )

    def get_belief_revision_history(
        self, entity_id: int = None, relation_id: int = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get the revision history for beliefs about an entity or relation.

        Args:
            entity_id: Optional entity ID filter
            relation_id: Optional relation ID filter
            limit: Maximum number of revisions to return

        Returns:
            List of belief revisions
        """
        return self._knowledge_coordinator.get_belief_revision_history(
            entity_id, relation_id, limit
        )

    def get_knowledge(self, query: str) -> list[dict[str, Any]]:
        """
        Get relevant knowledge for a query.

        Args:
            query: Query text

        Returns:
            List of relevant knowledge items
        """
        return self._knowledge_coordinator.get_knowledge(query)

    def answer_why_question(self, query: str) -> list[dict[str, Any]]:
        """
        Answer a why-question using backward chaining logical reasoning.

        Args:
            query: The why question to answer

        Returns:
            Explanation steps for the answer
        """
        return self._knowledge_coordinator.answer_why_question(query)

    def generate_novel_implications(
        self, iterations: int = None
    ) -> list[dict[str, Any]]:
        """
        Generate novel implications using forward chaining.

        Args:
            iterations: Number of forward chaining iterations (default uses config)

        Returns:
            List of newly inferred facts
        """
        return self._knowledge_coordinator.generate_novel_implications(iterations)

    def generate_hypotheses(
        self, observation: str, max_hypotheses: int = None
    ) -> list[dict[str, Any]]:
        """
        Generate hypotheses to explain an observation using abductive reasoning.

        Args:
            observation: The observation to explain
            max_hypotheses: Maximum number of hypotheses to generate (default uses config)

        Returns:
            List of hypotheses that could explain the observation
        """
        return self._knowledge_coordinator.generate_hypotheses(
            observation, max_hypotheses
        )

    def add_logical_rule(
        self,
        name: str,
        premise_patterns: list[dict[str, Any]],
        conclusion_pattern: dict[str, Any],
        confidence: float = 0.8,
    ) -> bool:
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
        return self._knowledge_coordinator.add_logical_rule(
            name, premise_patterns, conclusion_pattern, confidence
        )

    def multi_hop_query(self, query: str) -> dict[str, Any]:
        """
        Perform multi-hop reasoning on a query.

        Args:
            query: The query text

        Returns:
            Dictionary with path, entities, score, and other reasoning results
        """
        return self._knowledge_coordinator.multi_hop_query(query)

    def query(self, query_text: str) -> dict[str, Any]:
        """
        General query interface that routes to specialized query methods.

        Args:
            query_text: The query text

        Returns:
            Query result
        """
        return self._knowledge_coordinator.query(query_text)

    # ------------------------------------------------------------------
    # Reasoning / performance (delegated to ReasoningFacade)
    # ------------------------------------------------------------------

    def optimize_query(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Generate an optimized query plan for knowledge graph operations.

        Args:
            query: Dictionary with query parameters

        Returns:
            Optimized query plan
        """
        return self._reasoning_facade.optimize_query(query)

    def partition_graph(
        self, method: str = None, partition_count: int = None
    ) -> dict[str, Any]:
        """
        Partition the knowledge graph for improved performance.

        Args:
            method: Partitioning method (louvain, spectral, modularity)
            partition_count: Target number of partitions

        Returns:
            Partition statistics
        """
        return self._reasoning_facade.partition_graph(method, partition_count)

    def create_hop_indexes(self, max_hops: int = None) -> dict[str, Any]:
        """
        Create indexes for multi-hop queries to speed up traversal.

        Args:
            max_hops: Maximum number of hops to index

        Returns:
            Indexing statistics
        """
        return self._reasoning_facade.create_hop_indexes(max_hops)

    def optimize_path_query(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 3,
        relation_constraints: list[str] = None,
    ) -> dict[str, Any]:
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
        return self._reasoning_facade.optimize_path_query(
            start_entity, end_entity, max_hops, relation_constraints
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics from the optimizer.

        Returns:
            Dictionary with performance statistics
        """
        return self._reasoning_facade.get_performance_stats()

    def clear_performance_caches(self) -> dict[str, Any]:
        """
        Clear all performance optimization caches.

        Returns:
            Dictionary with cache clearing statistics
        """
        return self._reasoning_facade.clear_performance_caches()

    def cache_reasoning_pattern(self, pattern_key: str, pattern_result: Any) -> bool:
        """
        Cache a common reasoning pattern for reuse.

        Args:
            pattern_key: Unique identifier for the reasoning pattern
            pattern_result: Result of the reasoning pattern

        Returns:
            True if successful, False otherwise
        """
        return self._reasoning_facade.cache_reasoning_pattern(
            pattern_key, pattern_result
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics including hit rates.

        Returns:
            Dictionary with cache statistics
        """
        return self._reasoning_facade.get_cache_stats()

    # ------------------------------------------------------------------
    # Lifecycle & ContextProvider interface (stay on facade)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close and clean up resources."""
        if self._event_bus:
            from cortexflow.events import EventType

            self._event_bus.emit_typed(EventType.MANAGER_CLOSING, source="manager")

        for component_name in (
            "memory",
            "knowledge_store",
            "uncertainty_handler",
            "performance_optimizer",
            "ontology",
            "graph_store",
            "_session_manager",
            "_user_profile_manager",
            "_persona_manager",
            "_relationship_tracker",
            "_temporal_manager",
            "_episodic_store",
        ):
            component = getattr(self, component_name, None)
            if component and hasattr(component, "close"):
                try:
                    # Avoid double-closing graph_store if it's owned by knowledge_store
                    if component_name == "graph_store":
                        ks = getattr(self, "knowledge_store", None)
                        if (
                            ks
                            and hasattr(ks, "graph_store")
                            and component is ks.graph_store
                        ):
                            continue
                    component.close()
                except Exception as e:
                    logger.error(f"Error closing {component_name}: {e}")

        logger.info("CortexFlowManager closed")

    def __enter__(self) -> CortexFlowManager:
        """Support ``with CortexFlowManager() as mgr:`` usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure resources are released when exiting a with-block."""
        self.close()
        return None

    def __del__(self) -> None:
        """Destructor -- best-effort fallback; prefer using a context manager."""
        try:
            self.close()
        except Exception:  # noqa: S110
            pass

    def get_context(self) -> dict[str, Any]:
        """Get the current context for model consumption."""
        return self.get_conversation_context()

    def clear_context(self) -> None:
        """Clear all context data."""
        self.clear_memory()
