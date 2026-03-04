"""Event system for CortexFlow — pub/sub middleware with thread-safe event bus."""
from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("cortexflow")


class EventType(enum.Enum):
    """All CortexFlow event types."""
    MESSAGE_ADDED = "message_added"
    MEMORY_CLEARED = "memory_cleared"
    KNOWLEDGE_ADDED = "knowledge_added"
    KNOWLEDGE_RETRIEVED = "knowledge_retrieved"
    SESSION_CREATED = "session_created"
    SESSION_CLOSED = "session_closed"
    RESPONSE_GENERATED = "response_generated"
    EMOTION_DETECTED = "emotion_detected"
    RELATIONSHIP_STAGE_CHANGED = "relationship_stage_changed"
    TEMPORAL_FACT_ADDED = "temporal_fact_added"
    MANAGER_INITIALIZED = "manager_initialized"
    MANAGER_CLOSING = "manager_closing"


@dataclass
class Event:
    """An event emitted by the CortexFlow system."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = ""


# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe event bus."""

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._lock = threading.RLock()

    def on(self, event_type: EventType, handler: EventHandler) -> EventHandler:
        """Register a handler for a specific event type. Returns the handler for use as decorator."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
        return handler

    def on_all(self, handler: EventHandler) -> EventHandler:
        """Register a handler for ALL event types. Returns the handler for use as decorator."""
        with self._lock:
            self._global_handlers.append(handler)
        return handler

    def off(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unregister a handler. Returns True if handler was found and removed."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
                return True
            except ValueError:
                return False

    def off_all(self, handler: EventHandler) -> bool:
        """Unregister a global handler. Returns True if found and removed."""
        with self._lock:
            try:
                self._global_handlers.remove(handler)
                return True
            except ValueError:
                return False

    def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers. Handler exceptions are caught and logged."""
        with self._lock:
            specific = list(self._handlers.get(event.type, []))
            global_h = list(self._global_handlers)

        for handler in specific + global_h:
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler %s raised an exception for %s", handler, event.type)

    def emit_typed(self, event_type: EventType, data: dict[str, Any] | None = None,
                   source: str = "") -> Event:
        """Convenience: create an Event and emit it. Returns the emitted Event."""
        event = Event(type=event_type, data=data or {}, source=source)
        self.emit(event)
        return event

    def handler_count(self, event_type: EventType | None = None) -> int:
        """Return the number of registered handlers (specific type or total)."""
        with self._lock:
            if event_type is not None:
                return len(self._handlers.get(event_type, []))
            total = sum(len(h) for h in self._handlers.values())
            return total + len(self._global_handlers)
