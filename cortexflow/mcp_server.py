"""
MCP (Model Context Protocol) server for CortexFlow.

Exposes CortexFlow memory, knowledge, and companion AI capabilities
as MCP tools over stdio transport.  Configuration is driven by
``CORTEXFLOW_*`` environment variables.

Run directly::

    python -m cortexflow.mcp_server

Or via the console-script entry point::

    cortexflow-mcp
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from cortexflow.config import ConfigBuilder
from cortexflow.manager import CortexFlowManager

logger = logging.getLogger("cortexflow.mcp")


# ---------------------------------------------------------------------------
# Lifespan: build a CortexFlowManager from env vars
# ---------------------------------------------------------------------------

@dataclass
class AppContext:
    """Typed lifespan context holding the initialised manager."""

    manager: CortexFlowManager


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialise CortexFlowManager from environment variables."""
    builder = ConfigBuilder()

    db_path = os.environ.get("CORTEXFLOW_DB_PATH")
    if db_path:
        builder.with_knowledge_store(knowledge_store_path=db_path)

    verbose = os.environ.get("CORTEXFLOW_VERBOSE", "").lower() in ("1", "true", "yes")
    builder.with_debug(verbose_logging=verbose)

    # Companion AI features (opt-in via env vars)
    if os.environ.get("CORTEXFLOW_SESSIONS", "").lower() in ("1", "true", "yes"):
        builder.with_sessions()
    if os.environ.get("CORTEXFLOW_EMOTIONS", "").lower() in ("1", "true", "yes"):
        builder.with_emotions()
    if os.environ.get("CORTEXFLOW_PERSONAS", "").lower() in ("1", "true", "yes"):
        builder.with_persona()
    if os.environ.get("CORTEXFLOW_RELATIONSHIPS", "").lower() in ("1", "true", "yes"):
        builder.with_relationship()
    if os.environ.get("CORTEXFLOW_FACT_EXTRACTION", "").lower() in ("1", "true", "yes"):
        builder.with_fact_extraction()

    config = builder.build()
    manager = CortexFlowManager(config)
    try:
        yield AppContext(manager=manager)
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("cortexflow", lifespan=lifespan)


def _get_manager(ctx: Context[ServerSession, AppContext]) -> CortexFlowManager:
    """Convenience accessor for the CortexFlowManager from tool context."""
    return ctx.request_context.lifespan_context.manager


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def add_memory(
    role: str,
    content: str,
    ctx: Context[ServerSession, AppContext],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Add a message to conversation memory."""
    manager = _get_manager(ctx)
    msg = manager.add_message(role, content, metadata)
    return json.dumps(msg, default=str)


@mcp.tool()
async def search_memory(
    query: str,
    ctx: Context[ServerSession, AppContext],
    max_results: int = 5,
) -> str:
    """Search the knowledge store for relevant information."""
    manager = _get_manager(ctx)
    results = manager.knowledge_store.retrieve(query, max_results=max_results)
    return json.dumps(results, default=str)


@mcp.tool()
async def add_knowledge(
    text: str,
    ctx: Context[ServerSession, AppContext],
    source: str | None = None,
    confidence: float | None = None,
) -> str:
    """Add knowledge to the store."""
    manager = _get_manager(ctx)
    ids = manager.add_knowledge(text, source=source, confidence=confidence)
    return json.dumps({"stored_ids": ids})


@mcp.tool()
async def get_conversation_context(
    ctx: Context[ServerSession, AppContext],
    max_tokens: int | None = None,
) -> str:
    """Get current conversation context."""
    manager = _get_manager(ctx)
    context = manager.get_conversation_context(max_tokens=max_tokens)
    return json.dumps(context, default=str)


@mcp.tool()
async def get_user_profile(
    user_id: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get user profile data."""
    manager = _get_manager(ctx)
    pm = manager.user_profile_manager
    if pm is None:
        return json.dumps({"error": "User profile tracking is not enabled."})
    profile = pm.get_profile(user_id)
    return json.dumps(profile.to_dict(), default=str)


@mcp.tool()
async def get_emotional_state(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get current emotional state."""
    manager = _get_manager(ctx)
    tracker = manager.emotion_tracker
    if tracker is None:
        return json.dumps({"error": "Emotion tracking is not enabled."})
    state = tracker.get_current_state()
    return json.dumps(state.to_dict(), default=str)


@mcp.tool()
async def get_relationship_state(
    user_id: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get relationship state for a user."""
    manager = _get_manager(ctx)
    tracker = manager.relationship_tracker
    if tracker is None:
        return json.dumps({"error": "Relationship tracking is not enabled."})
    session = manager.get_current_session()
    persona_id = session.persona_id if session and session.persona_id else "default"
    state = tracker.get_state(user_id, persona_id)
    return json.dumps(state.to_dict(), default=str)


@mcp.tool()
async def manage_persona(
    action: str,
    ctx: Context[ServerSession, AppContext],
    persona_id: str | None = None,
) -> str:
    """List, get, or set the active persona.

    Args:
        action: One of "list", "get", or "set".
        persona_id: Required for "get" and "set" actions.
    """
    manager = _get_manager(ctx)
    pm = manager.persona_manager
    if pm is None:
        return json.dumps({"error": "Persona management is not enabled."})

    if action == "list":
        personas = pm.list_personas()
        return json.dumps([p.to_dict() for p in personas], default=str)
    elif action == "get":
        if not persona_id:
            return json.dumps({"error": "persona_id is required for 'get' action."})
        persona = pm.get_persona(persona_id)
        if persona is None:
            return json.dumps({"error": f"Persona '{persona_id}' not found."})
        return json.dumps(persona.to_dict(), default=str)
    elif action == "set":
        if not persona_id:
            return json.dumps({"error": "persona_id is required for 'set' action."})
        persona = pm.get_persona(persona_id)
        if persona is None:
            return json.dumps({"error": f"Persona '{persona_id}' not found."})
        session = manager.get_current_session()
        if session:
            session.persona_id = persona_id
        return json.dumps({"active_persona": persona_id})
    else:
        return json.dumps({"error": f"Unknown action '{action}'. Use 'list', 'get', or 'set'."})


@mcp.tool()
async def create_session(
    ctx: Context[ServerSession, AppContext],
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a new conversation session."""
    manager = _get_manager(ctx)
    uid = user_id or "default"
    try:
        session = manager.create_session(uid, metadata=metadata)
        return json.dumps(session.to_dict(), default=str)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console-script entry point for ``cortexflow-mcp``."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
