"""Opt-in execution tracing for diagnostics.

The benchmark and application stay quiet.  A caller such as
``eval.debug_question`` enables a collector for one run; low-level boundaries
(LLM, embedding and Couchbase) then append complete, ordered events without
threading a logger through every public API.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time


_collector = ContextVar("prism_trace_collector", default=None)


def _safe(value):
    """Keep useful parameters while avoiding megabytes of embedding output."""
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 100 and all(isinstance(v, (int, float)) for v in value[:10]):
            return f"<vector: {len(value)} dimensions>"
        return [_safe(v) for v in value]
    return value


@contextmanager
def capture():
    events = []
    token = _collector.set(events)
    started = time.perf_counter()
    try:
        yield events
    finally:
        events.append({"type": "trace_summary",
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        _collector.reset(token)


def add(event_type: str, **fields) -> None:
    events = _collector.get()
    if events is not None:
        events.append({"type": event_type, **_safe(fields)})

