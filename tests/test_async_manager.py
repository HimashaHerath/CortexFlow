"""Tests for cortexflow.async_manager."""
import asyncio
import pytest

from cortexflow.config import CortexFlowConfig
from cortexflow.async_manager import AsyncCortexFlowManager


def _run(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def config():
    return CortexFlowConfig()


class TestAsyncCortexFlowManager:
    def test_create_and_close(self, config):
        async def _test():
            mgr = AsyncCortexFlowManager(config)
            await mgr.close()
        _run(_test())

    def test_context_manager(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                assert mgr.config is not None
        _run(_test())

    def test_add_message(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                msg = await mgr.add_message("user", "Hello!")
                assert msg["role"] == "user"
                assert msg["content"] == "Hello!"
        _run(_test())

    def test_get_conversation_context(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                await mgr.add_message("system", "You are helpful.")
                ctx = await mgr.get_conversation_context()
                assert "messages" in ctx
        _run(_test())

    def test_clear_memory(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                await mgr.add_message("user", "Test")
                await mgr.clear_memory()
                ctx = await mgr.get_conversation_context()
                assert "messages" in ctx
        _run(_test())

    def test_get_stats(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                stats = await mgr.get_stats()
                assert "memory" in stats
                assert "knowledge" in stats
        _run(_test())

    def test_add_knowledge(self, config):
        async def _test():
            async with AsyncCortexFlowManager(config) as mgr:
                ids = await mgr.add_knowledge("The sky is blue", source="test")
                assert isinstance(ids, list)
        _run(_test())
