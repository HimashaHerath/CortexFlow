"""Tests for cortexflow.temporal — TemporalManager and TemporalFact."""

from cortexflow.temporal import TemporalFact, TemporalManager

# ──────────────────────────────────────────────────────────────
# TemporalManager
# ──────────────────────────────────────────────────────────────


class TestTemporalManager:
    def _make_manager(self) -> TemporalManager:
        return TemporalManager(db_path=":memory:")

    def test_add_temporal_fact(self):
        mgr = self._make_manager()
        fact = TemporalFact(
            subject="alice",
            predicate="lives_in",
            object="New York",
            valid_from="2020-01-01T00:00:00",
            confidence=0.9,
            source="conversation",
        )
        fact_id = mgr.add_temporal_fact(fact)
        assert fact_id is not None
        assert isinstance(fact_id, int)
        assert fact_id >= 1
        mgr.close()

    def test_update_fact_validity(self):
        mgr = self._make_manager()
        fact = TemporalFact(
            subject="alice",
            predicate="lives_in",
            object="New York",
            valid_from="2020-01-01T00:00:00",
        )
        fact_id = mgr.add_temporal_fact(fact)
        updated = mgr.update_fact_validity(fact_id, "2023-06-01T00:00:00")
        assert updated is True
        # Verify the update took effect
        timeline = mgr.get_fact_timeline("alice", "lives_in")
        assert len(timeline) == 1
        assert timeline[0].valid_until == "2023-06-01T00:00:00"

        # Updating a non-existent ID returns False
        assert mgr.update_fact_validity(9999, "2025-01-01T00:00:00") is False
        mgr.close()

    def test_supersede_fact(self):
        mgr = self._make_manager()
        old_fact = TemporalFact(
            subject="alice",
            predicate="lives_in",
            object="New York",
            valid_from="2020-01-01T00:00:00",
        )
        old_id = mgr.add_temporal_fact(old_fact)

        new_fact = TemporalFact(
            subject="alice",
            predicate="lives_in",
            object="San Francisco",
            valid_from="2023-06-01T00:00:00",
        )
        new_id = mgr.supersede_fact(old_id, new_fact)
        assert new_id != old_id

        # Old fact should now have valid_until set and superseded_by pointing to new
        timeline = mgr.get_fact_timeline("alice", "lives_in")
        assert len(timeline) == 2
        old_entry = [f for f in timeline if f.id == old_id][0]
        assert old_entry.valid_until is not None
        assert old_entry.superseded_by == new_id
        mgr.close()

    def test_get_facts_at_time(self):
        mgr = self._make_manager()
        # Fact valid 2020-2023
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="New York",
                valid_from="2020-01-01T00:00:00",
                valid_until="2023-01-01T00:00:00",
            )
        )
        # Fact valid 2023-ongoing
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="San Francisco",
                valid_from="2023-01-01T00:00:00",
            )
        )
        # Another subject
        mgr.add_temporal_fact(
            TemporalFact(
                subject="bob",
                predicate="works_at",
                object="Acme Corp",
                valid_from="2021-01-01T00:00:00",
            )
        )

        # Query at 2021: should get alice=NY and bob=Acme
        facts_2021 = mgr.get_facts_at_time("2021-06-01T00:00:00")
        subjects = {f.subject for f in facts_2021}
        assert "alice" in subjects
        assert "bob" in subjects
        alice_2021 = [f for f in facts_2021 if f.subject == "alice"][0]
        assert alice_2021.object == "New York"

        # Query at 2024: should get alice=SF and bob=Acme
        facts_2024 = mgr.get_facts_at_time("2024-06-01T00:00:00")
        alice_2024 = [f for f in facts_2024 if f.subject == "alice"][0]
        assert alice_2024.object == "San Francisco"

        # Filter by subject
        alice_only = mgr.get_facts_at_time("2024-06-01T00:00:00", subject="alice")
        assert len(alice_only) == 1
        assert alice_only[0].subject == "alice"
        mgr.close()

    def test_get_fact_timeline(self):
        mgr = self._make_manager()
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="Boston",
                valid_from="2015-01-01T00:00:00",
                valid_until="2020-01-01T00:00:00",
                created_at="2015-01-01T00:00:00",
            )
        )
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="New York",
                valid_from="2020-01-01T00:00:00",
                created_at="2020-01-01T00:00:00",
            )
        )
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="works_at",
                object="Google",
                valid_from="2018-01-01T00:00:00",
                created_at="2018-01-01T00:00:00",
            )
        )

        # Full timeline for alice
        full_timeline = mgr.get_fact_timeline("alice")
        assert len(full_timeline) == 3

        # Filtered by predicate
        lives_timeline = mgr.get_fact_timeline("alice", predicate="lives_in")
        assert len(lives_timeline) == 2
        # Should be ordered by created_at ASC
        assert lives_timeline[0].object == "Boston"
        assert lives_timeline[1].object == "New York"
        mgr.close()

    def test_detect_temporal_conflicts_overlapping(self):
        mgr = self._make_manager()
        # Two overlapping facts for same subject+predicate
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="New York",
                valid_from="2020-01-01T00:00:00",
                valid_until="2024-01-01T00:00:00",
            )
        )
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="San Francisco",
                valid_from="2023-01-01T00:00:00",
                valid_until="2025-01-01T00:00:00",
            )
        )
        conflicts = mgr.detect_temporal_conflicts()
        assert len(conflicts) == 1
        f1, f2 = conflicts[0]
        assert {f1.object, f2.object} == {"New York", "San Francisco"}
        mgr.close()

    def test_detect_temporal_conflicts_non_overlapping(self):
        mgr = self._make_manager()
        # Two non-overlapping facts
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="New York",
                valid_from="2020-01-01T00:00:00",
                valid_until="2023-01-01T00:00:00",
            )
        )
        mgr.add_temporal_fact(
            TemporalFact(
                subject="alice",
                predicate="lives_in",
                object="San Francisco",
                valid_from="2023-01-01T00:00:00",
                valid_until="2025-01-01T00:00:00",
            )
        )
        conflicts = mgr.detect_temporal_conflicts()
        assert len(conflicts) == 0
        mgr.close()
