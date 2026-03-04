"""CrewAI integration quickstart — use CortexFlow as CrewAI storage backend.

Requires: pip install cortexflow-llm[crewai]
"""

from cortexflow import ConfigBuilder, CortexFlowManager
from cortexflow.integrations.crewai import CortexFlowCrewStorage

# Create manager
config = ConfigBuilder().build()
manager = CortexFlowManager(config)

# Create CrewAI-compatible storage
storage = CortexFlowCrewStorage(manager)

# Save knowledge through CrewAI's interface
storage.save(
    key="project-info", value="CortexFlow uses cognitive-inspired architecture."
)
storage.save(key="team-info", value="The team uses Python 3.10+ and pytest.")

# Search
results = storage.search("architecture", limit=5)
print(f"Found {len(results)} results for 'architecture':")
for r in results:
    print(f"  - {r.get('context', r.get('text', ''))[:80]}")

# Reset clears all stored knowledge
# storage.reset()

manager.close()
print("\nDone!")
