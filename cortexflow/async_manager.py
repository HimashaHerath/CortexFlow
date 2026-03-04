"""
Async wrapper for CortexFlowManager.

Provides ``AsyncCortexFlowManager`` which wraps the synchronous
``CortexFlowManager`` using ``asyncio.to_thread()`` for CPU/IO-bound
operations, while leveraging native async for LLM calls where supported.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from cortexflow.config import CortexFlowConfig
from cortexflow.manager import CortexFlowManager

logger = logging.getLogger("cortexflow")


class AsyncCortexFlowManager:
    """Async facade over ``CortexFlowManager``.

    Every method delegates to the underlying sync manager via
    ``asyncio.to_thread`` unless a native async path is available
    (e.g. ``llm_client.agenerate``).

    Usage::

        async with AsyncCortexFlowManager(config) as mgr:
            await mgr.add_message("user", "Hello")
            response = await mgr.generate_response()
    """

    def __init__(self, config: CortexFlowConfig | None = None):
        self._sync = CortexFlowManager(config)
        self.config = self._sync.config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def __aenter__(self) -> AsyncCortexFlowManager:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def add_message(self, role: str, content: str,
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.add_message, role, content, metadata)

    async def get_conversation_context(self, max_tokens: int | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_conversation_context, max_tokens)

    async def clear_memory(self) -> None:
        await asyncio.to_thread(self._sync.clear_memory)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def generate_response(self, prompt: str | None = None,
                                model: str | None = None) -> str:
        """Generate a response, preferring native async LLM call."""
        llm = self._sync.llm_client
        if hasattr(llm, "agenerate") and prompt is not None:
            # Direct async LLM path (skips COA/reflection — simple prompt mode)
            messages = [{"role": "user", "content": prompt}]
            result = await llm.agenerate(messages, model=model)
            await self.add_message("assistant", result)
            return result
        # Full orchestration path (sync, off-loaded to thread)
        return await asyncio.to_thread(self._sync.generate_response, prompt, model)

    async def generate_response_stream(self, prompt: str | None = None,
                                       model: str | None = None) -> AsyncIterator[str]:
        """Yield response chunks asynchronously."""
        # Off-load the sync iterator to a background thread and bridge
        # via an asyncio.Queue.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _producer() -> None:
            try:
                for chunk in self._sync.generate_response_stream(prompt, model):
                    queue.put_nowait(chunk)
            finally:
                queue.put_nowait(None)  # sentinel

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _producer)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        await task  # propagate exceptions

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    async def add_knowledge(self, text: str, source: str | None = None,
                            confidence: float | None = None) -> list[int]:
        return await asyncio.to_thread(self._sync.add_knowledge, text, source, confidence)

    async def get_knowledge(self, query: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync.get_knowledge, query)

    async def multi_hop_query(self, query: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.multi_hop_query, query)

    async def query(self, query_text: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.query, query_text)

    # ------------------------------------------------------------------
    # Session management (pass-through when sessions enabled)
    # ------------------------------------------------------------------

    async def create_session(self, user_id: str, persona_id: str | None = None,
                             metadata: dict[str, Any] | None = None):
        return await asyncio.to_thread(self._sync.create_session, user_id, persona_id, metadata)

    async def switch_session(self, session_id: str):
        return await asyncio.to_thread(self._sync.switch_session, session_id)

    async def get_current_session(self):
        return self._sync.get_current_session()

    # ------------------------------------------------------------------
    # Stats & misc
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_stats)

    async def get_context(self) -> dict[str, Any]:
        return await self.get_conversation_context()

    async def clear_context(self) -> None:
        await self.clear_memory()
