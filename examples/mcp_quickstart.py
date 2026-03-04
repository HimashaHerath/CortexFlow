"""MCP Server quickstart — expose CortexFlow as an MCP tool server.

Run:
    cortexflow-mcp                    # uses default config
    CORTEXFLOW_DB_PATH=my.db cortexflow-mcp   # custom DB path

Or programmatically:
    python examples/mcp_quickstart.py
"""

import os

# Configure via environment variables before importing
os.environ.setdefault("CORTEXFLOW_VERBOSE", "1")

from cortexflow.mcp_server import main

if __name__ == "__main__":
    print("Starting CortexFlow MCP server (stdio transport)...")
    print("Available tools: add_memory, search_memory, add_knowledge,")
    print("  get_conversation_context, get_user_profile, get_emotional_state,")
    print("  get_relationship_state, manage_persona, create_session")
    main()
