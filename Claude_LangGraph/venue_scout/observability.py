"""Lightweight observability helpers for Venue Scout.

Tracks in-memory counters and recent failures, and emits structured logs.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from threading import Lock
from zoneinfo import ZoneInfo

_COUNTERS: dict[str, int] = {}
_RECENT_FAILURES: deque[dict] = deque(maxlen=200)
_RECENT_EVENTS: deque[dict] = deque(maxlen=200)
_LOCK = Lock()


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def increment(metric: str, value: int = 1) -> None:
    """Increment a named counter."""
    with _LOCK:
        _COUNTERS[metric] = _COUNTERS.get(metric, 0) + value


def log_event(kind: str, **fields) -> None:
    """Emit a structured event line and keep a small recent buffer."""
    payload = {
        "ts": _timestamp(),
        "kind": kind,
        **fields,
    }
    with _LOCK:
        _RECENT_EVENTS.append(payload)
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def record_failure(component: str, reason: str, **fields) -> None:
    """Record failure metadata for diagnostics."""
    payload = {
        "ts": _timestamp(),
        "component": component,
        "reason": reason,
        **fields,
    }
    with _LOCK:
        _RECENT_FAILURES.append(payload)
        _COUNTERS[f"{component}.failures"] = _COUNTERS.get(f"{component}.failures", 0) + 1
    print(json.dumps({"kind": "failure", **payload}, ensure_ascii=True), flush=True)


def snapshot() -> dict:
    """Get current counters and recent failure/event buffers."""
    with _LOCK:
        return {
            "ts": _timestamp(),
            "uptime_seconds": int(time.monotonic()),
            "counters": dict(_COUNTERS),
            "recent_failures": list(_RECENT_FAILURES),
            "recent_events": list(_RECENT_EVENTS),
        }

