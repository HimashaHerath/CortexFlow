"""CortexFlow framework integrations.

Lazy-loaded adapters for popular LLM frameworks.  Import only the
integration you need to avoid pulling in heavy optional dependencies.
"""


def get_langchain_integration():
    """Return LangChain integration classes."""
    from cortexflow.integrations.langchain import (
        CortexFlowChatMessageHistory,
        CortexFlowRetriever,
        CortexFlowMemory,
    )
    return CortexFlowChatMessageHistory, CortexFlowRetriever, CortexFlowMemory


def get_crewai_integration():
    """Return CrewAI integration classes."""
    from cortexflow.integrations.crewai import CortexFlowCrewStorage
    return (CortexFlowCrewStorage,)
