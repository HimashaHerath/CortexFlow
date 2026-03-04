"""Tests for cortexflow.episodic_memory — EpisodicMemoryStore and Episode."""

from cortexflow.episodic_memory import Episode, EpisodicMemoryStore

# ──────────────────────────────────────────────────────────────
# EpisodicMemoryStore
# ──────────────────────────────────────────────────────────────


class TestEpisodicMemoryStore:
    def _make_store(self) -> EpisodicMemoryStore:
        return EpisodicMemoryStore(db_path=":memory:")

    def test_save_and_retrieve_episode(self):
        store = self._make_store()
        ep = Episode(
            session_id="s1",
            user_id="u1",
            title="First chat",
            summary="We discussed Python testing",
            messages=[{"role": "user", "content": "Hello"}],
            emotions=["happy"],
            topics=["python", "testing"],
            importance=0.8,
        )
        ep_id = store.save_episode(ep)
        assert ep_id is not None
        assert isinstance(ep_id, int)

        recent = store.get_recent_episodes()
        assert len(recent) == 1
        assert recent[0].id == ep_id
        assert recent[0].title == "First chat"
        assert recent[0].summary == "We discussed Python testing"
        assert recent[0].emotions == ["happy"]
        assert recent[0].topics == ["python", "testing"]
        assert recent[0].importance == 0.8
        store.close()

    def test_recall_episodes_fts(self):
        store = self._make_store()
        store.save_episode(
            Episode(
                session_id="s1",
                user_id="u1",
                title="Machine learning discussion",
                summary="Talked about neural networks and deep learning",
                topics=["ml", "neural networks"],
            )
        )
        store.save_episode(
            Episode(
                session_id="s2",
                user_id="u1",
                title="Cooking recipe",
                summary="Shared a pasta recipe with tomato sauce",
                topics=["cooking", "pasta"],
            )
        )

        # FTS search for "neural"
        results = store.recall_episodes("neural")
        assert len(results) >= 1
        assert any(
            "neural" in r.summary.lower() or "neural" in r.title.lower()
            for r in results
        )

        # FTS search for "pasta"
        results = store.recall_episodes("pasta")
        assert len(results) >= 1
        assert any(
            "pasta" in r.summary.lower() or "pasta" in " ".join(r.topics).lower()
            for r in results
        )
        store.close()

    def test_get_recent_episodes_with_user_filter(self):
        store = self._make_store()
        store.save_episode(
            Episode(
                session_id="s1",
                user_id="u1",
                title="User 1 chat",
                summary="Chat by user 1",
            )
        )
        store.save_episode(
            Episode(
                session_id="s2",
                user_id="u2",
                title="User 2 chat",
                summary="Chat by user 2",
            )
        )
        store.save_episode(
            Episode(
                session_id="s3",
                user_id="u1",
                title="Another user 1 chat",
                summary="Another chat by user 1",
            )
        )

        # All episodes
        all_eps = store.get_recent_episodes()
        assert len(all_eps) == 3

        # Filtered by user
        u1_eps = store.get_recent_episodes(user_id="u1")
        assert len(u1_eps) == 2
        assert all(ep.user_id == "u1" for ep in u1_eps)

        u2_eps = store.get_recent_episodes(user_id="u2")
        assert len(u2_eps) == 1
        assert u2_eps[0].user_id == "u2"
        store.close()

    def test_summarize_session(self):
        store = self._make_store()
        store.save_episode(
            Episode(
                session_id="s1",
                user_id="u1",
                title="Episode 1",
                summary="First part of the conversation",
                messages=[{"role": "user", "content": "Hi"}],
                emotions=["neutral"],
                topics=["greeting"],
                importance=0.3,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T10:05:00",
            )
        )
        store.save_episode(
            Episode(
                session_id="s1",
                user_id="u1",
                title="Episode 2",
                summary="Second part of the conversation",
                messages=[{"role": "user", "content": "Tell me about Python"}],
                emotions=["curious"],
                topics=["python"],
                importance=0.7,
                start_time="2024-01-01T10:05:00",
                end_time="2024-01-01T10:15:00",
            )
        )

        summary = store.summarize_session("s1")
        assert summary is not None
        assert summary.session_id == "s1"
        assert summary.user_id == "u1"
        assert "2 episodes" in summary.summary
        assert len(summary.messages) == 2
        assert set(summary.emotions) == {"neutral", "curious"}
        assert set(summary.topics) == {"greeting", "python"}
        assert summary.importance == 0.7  # max of 0.3 and 0.7
        assert summary.start_time == "2024-01-01T10:00:00"
        assert summary.end_time == "2024-01-01T10:15:00"
        store.close()

    def test_summarize_session_nonexistent(self):
        store = self._make_store()
        summary = store.summarize_session("nonexistent")
        assert summary is None
        store.close()

    def test_empty_store(self):
        store = self._make_store()
        recent = store.get_recent_episodes()
        assert recent == []
        results = store.recall_episodes("anything")
        assert results == []
        store.close()
