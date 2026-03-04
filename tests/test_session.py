"""Tests for cortexflow.session and cortexflow.user_store."""
import time

from cortexflow.session import SessionContext, SessionManager
from cortexflow.user_store import UserStore

# ──────────────────────────────────────────────────────────────
# SessionContext dataclass
# ──────────────────────────────────────────────────────────────

class TestSessionContext:
    def test_create_default(self):
        ctx = SessionContext(session_id="s1", user_id="u1")
        assert ctx.session_id == "s1"
        assert ctx.user_id == "u1"
        assert ctx.persona_id is None
        assert ctx.is_active is True
        assert ctx.metadata == {}

    def test_touch_updates_timestamp(self):
        ctx = SessionContext(session_id="s1", user_id="u1")
        old_ts = ctx.last_active_at
        time.sleep(0.01)
        ctx.touch()
        assert ctx.last_active_at > old_ts

    def test_roundtrip_dict(self):
        ctx = SessionContext(session_id="s1", user_id="u1", persona_id="p1",
                            metadata={"key": "val"})
        d = ctx.to_dict()
        ctx2 = SessionContext.from_dict(d)
        assert ctx2.session_id == "s1"
        assert ctx2.persona_id == "p1"
        assert ctx2.metadata == {"key": "val"}


# ──────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────

class TestSessionManager:
    def setup_method(self):
        self.mgr = SessionManager(db_path=":memory:", session_ttl=3600,
                                  max_sessions_per_user=3)

    def teardown_method(self):
        self.mgr.close()

    def test_create_and_get(self):
        sess = self.mgr.create_session("user1")
        assert sess.user_id == "user1"
        assert sess.is_active

        fetched = self.mgr.get_session(sess.session_id)
        assert fetched is not None
        assert fetched.session_id == sess.session_id

    def test_list_sessions(self):
        self.mgr.create_session("user1")
        self.mgr.create_session("user1")
        self.mgr.create_session("user2")

        u1_sessions = self.mgr.list_sessions("user1")
        assert len(u1_sessions) == 2
        u2_sessions = self.mgr.list_sessions("user2")
        assert len(u2_sessions) == 1

    def test_close_session(self):
        sess = self.mgr.create_session("user1")
        assert self.mgr.close_session(sess.session_id)

        # Should not appear in active list
        active = self.mgr.list_sessions("user1", active_only=True)
        assert len(active) == 0

        # Should appear in all list
        all_sessions = self.mgr.list_sessions("user1", active_only=False)
        assert len(all_sessions) == 1

    def test_session_limit_enforcement(self):
        # max_sessions_per_user=3
        s1 = self.mgr.create_session("user1")
        self.mgr.create_session("user1")
        self.mgr.create_session("user1")
        # 4th session should auto-close the oldest
        self.mgr.create_session("user1")

        active = self.mgr.list_sessions("user1", active_only=True)
        assert len(active) == 3
        # s1 (oldest) should have been closed
        assert self.mgr.get_session(s1.session_id).is_active is False

    def test_touch_session(self):
        sess = self.mgr.create_session("user1")
        old_ts = sess.last_active_at
        time.sleep(0.01)
        self.mgr.touch_session(sess.session_id)
        updated = self.mgr.get_session(sess.session_id)
        assert updated.last_active_at > old_ts

    def test_cleanup_expired(self):
        mgr = SessionManager(db_path=":memory:", session_ttl=0)  # 0 = immediate expiry
        mgr.create_session("user1")
        time.sleep(0.01)
        closed = mgr.cleanup_expired()
        assert closed == 1
        mgr.close()

    def test_get_nonexistent_returns_none(self):
        assert self.mgr.get_session("nonexistent") is None

    def test_persona_id_stored(self):
        sess = self.mgr.create_session("user1", persona_id="alice")
        assert sess.persona_id == "alice"
        fetched = self.mgr.get_session(sess.session_id)
        assert fetched.persona_id == "alice"


# ──────────────────────────────────────────────────────────────
# UserStore
# ──────────────────────────────────────────────────────────────

class TestUserStore:
    def setup_method(self):
        self.store = UserStore(db_path=":memory:")

    def teardown_method(self):
        self.store.close()

    def test_create_and_get(self):
        user = self.store.create_user("u1", display_name="Alice",
                                       metadata={"tier": "premium"})
        assert user["user_id"] == "u1"
        assert user["display_name"] == "Alice"
        assert user["metadata"]["tier"] == "premium"

    def test_get_nonexistent(self):
        assert self.store.get_user("nope") is None

    def test_update_metadata(self):
        self.store.create_user("u1", metadata={"a": 1})
        self.store.update_user_metadata("u1", {"b": 2})
        user = self.store.get_user("u1")
        assert user["metadata"] == {"a": 1, "b": 2}

    def test_update_nonexistent_returns_false(self):
        assert self.store.update_user_metadata("nope", {"x": 1}) is False

    def test_delete_user(self):
        self.store.create_user("u1")
        assert self.store.delete_user("u1")
        assert self.store.get_user("u1") is None
        assert self.store.delete_user("u1") is False

    def test_list_users(self):
        self.store.create_user("u1")
        self.store.create_user("u2")
        users = self.store.list_users()
        assert len(users) == 2

    def test_upsert(self):
        self.store.create_user("u1", display_name="Alice")
        self.store.create_user("u1", display_name="Bob")
        user = self.store.get_user("u1")
        assert user["display_name"] == "Bob"
