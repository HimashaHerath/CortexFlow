"""LangChain integration quickstart — use CortexFlow as LangChain memory + retriever.

Requires: pip install cortexflow-llm[langchain]
"""
from cortexflow import CortexFlowManager, ConfigBuilder

# Build config with sessions enabled
config = ConfigBuilder().with_sessions().build()
manager = CortexFlowManager(config)

# --- Chat Message History ---
from cortexflow.integrations.langchain import (
    CortexFlowChatMessageHistory,
    CortexFlowRetriever,
    CortexFlowMemory,
)

history = CortexFlowChatMessageHistory(manager)
history.add_user_message("My name is Alice and I love hiking.")
history.add_ai_message("Nice to meet you, Alice! Hiking is wonderful.")

print("Messages in history:", len(history.messages))
for msg in history.messages:
    print(f"  [{msg.type}] {msg.content[:60]}")

# --- Retriever ---
retriever = CortexFlowRetriever(manager, max_results=3)
# After adding knowledge, the retriever can search it
manager.add_knowledge("CortexFlow supports multi-tier memory.")

# --- Convenience wrapper ---
memory = CortexFlowMemory(manager)
print("\nMemory messages:", len(memory.messages))

manager.close()
print("\nDone!")
