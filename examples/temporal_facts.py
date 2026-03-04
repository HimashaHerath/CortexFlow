"""Temporal facts + episodic memory quickstart.

Demonstrates time-aware fact management and session-based episode storage.
"""

from cortexflow.episodic_memory import Episode, EpisodicMemoryStore
from cortexflow.temporal import TemporalFact, TemporalManager

# --- Temporal Facts (standalone) ---
tm = TemporalManager()

# Alice lives in NYC from 2020
fact1_id = tm.add_temporal_fact(
    TemporalFact(
        subject="Alice",
        predicate="lives_in",
        object="New York",
        valid_from="2020-01-01",
        confidence=0.9,
    )
)

# Alice moves to SF in 2024 — supersedes the NYC fact
fact2_id = tm.supersede_fact(
    fact1_id,
    TemporalFact(
        subject="Alice",
        predicate="lives_in",
        object="San Francisco",
        valid_from="2024-06-01",
        confidence=0.95,
    ),
)

# Query at different times
facts_2022 = tm.get_facts_at_time("2022-06-01", subject="Alice")
facts_2025 = tm.get_facts_at_time("2025-01-01", subject="Alice")
print(f"Alice in 2022: {facts_2022[0].object}")  # New York
print(f"Alice in 2025: {facts_2025[0].object}")  # San Francisco

# Timeline
timeline = tm.get_fact_timeline("Alice", "lives_in")
print(f"\nTimeline ({len(timeline)} entries):")
for f in timeline:
    print(f"  {f.valid_from or '?'} -> {f.valid_until or 'present'}: {f.object}")

tm.close()

# --- Episodic Memory (standalone) ---
em = EpisodicMemoryStore()

em.save_episode(
    Episode(
        session_id="s1",
        user_id="alice",
        title="Travel planning",
        summary="Discussed hiking trails in Yosemite and camping gear.",
        topics=["hiking", "yosemite", "camping"],
        emotions=["excited"],
    )
)

em.save_episode(
    Episode(
        session_id="s1",
        user_id="alice",
        title="Gear review",
        summary="Reviewed waterproof jackets and trail shoes.",
        topics=["gear", "shoes", "jackets"],
    )
)

# Search by text
results = em.recall_episodes("hiking")
print(f"\nEpisodes about hiking: {len(results)}")
for ep in results:
    print(f"  - {ep.title}: {ep.summary[:60]}")

# Session summary
summary = em.summarize_session("s1")
print(f"\nSession summary: {summary.summary}")

em.close()
print("\nDone!")
