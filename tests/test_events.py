"""Tests for cortexflow.events — EventBus, Event, EventType."""
import threading
import time

import pytest

from cortexflow.events import Event, EventBus, EventHandler, EventType


# ──────────────────────────────────────────────────────────────
# Event dataclass
# ──────────────────────────────────────────────────────────────

class TestEventDataclass:
    def test_event_fields(self):
        event = Event(type=EventType.MESSAGE_ADDED, data={"role": "user"}, source="test")
        assert event.type == EventType.MESSAGE_ADDED
        assert event.data == {"role": "user"}
        assert event.source == "test"
        assert isinstance(event.timestamp, str)
        assert len(event.timestamp) > 0

    def test_event_defaults(self):
        event = Event(type=EventType.MEMORY_CLEARED)
        assert event.data == {}
        assert event.source == ""
        assert event.timestamp  # non-empty


# ──────────────────────────────────────────────────────────────
# EventBus — on / emit
# ──────────────────────────────────────────────────────────────

class TestOnAndEmit:
    def test_on_and_emit(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.on(EventType.MESSAGE_ADDED, handler)
        bus.emit_typed(EventType.MESSAGE_ADDED, data={"role": "user", "content": "hi"})

        assert len(received) == 1
        assert received[0].type == EventType.MESSAGE_ADDED
        assert received[0].data["role"] == "user"

    def test_on_returns_handler(self):
        bus = EventBus()
        handler = lambda e: None
        result = bus.on(EventType.MEMORY_CLEARED, handler)
        assert result is handler


# ──────────────────────────────────────────────────────────────
# EventBus — on_all
# ──────────────────────────────────────────────────────────────

class TestOnAll:
    def test_on_all_receives_all_types(self):
        bus = EventBus()
        received = []

        bus.on_all(lambda e: received.append(e.type))

        bus.emit_typed(EventType.MESSAGE_ADDED)
        bus.emit_typed(EventType.MEMORY_CLEARED)
        bus.emit_typed(EventType.KNOWLEDGE_ADDED)

        assert received == [
            EventType.MESSAGE_ADDED,
            EventType.MEMORY_CLEARED,
            EventType.KNOWLEDGE_ADDED,
        ]


# ──────────────────────────────────────────────────────────────
# EventBus — off / off_all
# ──────────────────────────────────────────────────────────────

class TestOff:
    def test_off_removes_handler(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)

        bus.on(EventType.MESSAGE_ADDED, handler)
        assert bus.off(EventType.MESSAGE_ADDED, handler) is True

        bus.emit_typed(EventType.MESSAGE_ADDED)
        assert len(received) == 0

    def test_off_returns_false_if_not_found(self):
        bus = EventBus()
        assert bus.off(EventType.MESSAGE_ADDED, lambda e: None) is False

    def test_off_all_removes_global_handler(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)

        bus.on_all(handler)
        assert bus.off_all(handler) is True

        bus.emit_typed(EventType.MESSAGE_ADDED)
        assert len(received) == 0

    def test_off_all_returns_false_if_not_found(self):
        bus = EventBus()
        assert bus.off_all(lambda e: None) is False


# ──────────────────────────────────────────────────────────────
# EventBus — emit_typed
# ──────────────────────────────────────────────────────────────

class TestEmitTyped:
    def test_emit_typed_creates_and_emits(self):
        bus = EventBus()
        received = []
        bus.on(EventType.KNOWLEDGE_ADDED, lambda e: received.append(e))

        event = bus.emit_typed(EventType.KNOWLEDGE_ADDED, data={"text": "fact"}, source="test")

        assert isinstance(event, Event)
        assert event.type == EventType.KNOWLEDGE_ADDED
        assert event.data == {"text": "fact"}
        assert event.source == "test"
        assert len(received) == 1
        assert received[0] is event

    def test_emit_typed_default_data(self):
        bus = EventBus()
        event = bus.emit_typed(EventType.MANAGER_INITIALIZED)
        assert event.data == {}
        assert event.source == ""


# ──────────────────────────────────────────────────────────────
# Handler exception isolation
# ──────────────────────────────────────────────────────────────

class TestHandlerExceptionCaught:
    def test_bad_handler_does_not_break_others(self):
        bus = EventBus()
        results = []

        def good_handler_before(e: Event):
            results.append("before")

        def bad_handler(e: Event):
            raise ValueError("boom")

        def good_handler_after(e: Event):
            results.append("after")

        bus.on(EventType.MESSAGE_ADDED, good_handler_before)
        bus.on(EventType.MESSAGE_ADDED, bad_handler)
        bus.on(EventType.MESSAGE_ADDED, good_handler_after)

        bus.emit_typed(EventType.MESSAGE_ADDED)

        assert results == ["before", "after"]


# ──────────────────────────────────────────────────────────────
# handler_count
# ──────────────────────────────────────────────────────────────

class TestHandlerCount:
    def test_handler_count_specific(self):
        bus = EventBus()
        bus.on(EventType.MESSAGE_ADDED, lambda e: None)
        bus.on(EventType.MESSAGE_ADDED, lambda e: None)
        bus.on(EventType.MEMORY_CLEARED, lambda e: None)

        assert bus.handler_count(EventType.MESSAGE_ADDED) == 2
        assert bus.handler_count(EventType.MEMORY_CLEARED) == 1
        assert bus.handler_count(EventType.KNOWLEDGE_ADDED) == 0

    def test_handler_count_total(self):
        bus = EventBus()
        bus.on(EventType.MESSAGE_ADDED, lambda e: None)
        bus.on_all(lambda e: None)

        assert bus.handler_count() == 2

    def test_handler_count_empty(self):
        bus = EventBus()
        assert bus.handler_count() == 0


# ──────────────────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_register_and_emit(self):
        bus = EventBus()
        results = []
        lock = threading.Lock()

        def handler(e: Event):
            with lock:
                results.append(e.type)

        errors = []

        def register_and_emit(event_type: EventType):
            try:
                bus.on(event_type, handler)
                for _ in range(10):
                    bus.emit_typed(event_type)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_and_emit, args=(EventType.MESSAGE_ADDED,)),
            threading.Thread(target=register_and_emit, args=(EventType.MEMORY_CLEARED,)),
            threading.Thread(target=register_and_emit, args=(EventType.KNOWLEDGE_ADDED,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        # Each thread emits 10 events, each heard by its handler at minimum
        assert len(results) >= 30
