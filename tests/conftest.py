"""Shared pytest fixtures for CortexFlow tests."""
from __future__ import annotations

import pytest

from cortexflow.config import (
    CortexFlowConfig,
    KnowledgeStoreConfig,
    MemoryConfig,
)


@pytest.fixture
def default_config():
    """Return a default CortexFlowConfig."""
    return CortexFlowConfig()


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary database path inside pytest's tmp_path."""
    return str(tmp_path / "test_cortexflow.db")


@pytest.fixture
def config_with_tmp_db(tmp_db_path):
    """Return a CortexFlowConfig with a temporary database path."""
    return CortexFlowConfig(
        knowledge_store=KnowledgeStoreConfig(knowledge_store_path=tmp_db_path),
    )


@pytest.fixture
def memory_config():
    """Return a MemoryConfig with small limits for fast tests."""
    return MemoryConfig(
        active_token_limit=512,
        working_token_limit=1024,
        archive_token_limit=2048,
    )


@pytest.fixture
def small_config(tmp_db_path, memory_config):
    """Return a config with small memory limits and a temp DB — ideal for unit tests."""
    return CortexFlowConfig(
        memory=memory_config,
        knowledge_store=KnowledgeStoreConfig(knowledge_store_path=tmp_db_path),
    )
