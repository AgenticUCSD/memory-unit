"""Safe, optional DeepEval / Confident AI tracing for memory-unit.

Phase 5 (steps 1-2): memory-unit had no tracing. This adds it so the memory
service's LLM/retrieval work shows up under the *same* ``thread_id`` as the
provider's spans (the provider forwards ``X-Thread-Id`` on ``/resolve``).

Design contract — tracing must never break a request:
- deepeval is imported behind ``try/except``; if it is not installed, or tracing
  is disabled, ``tracing_callbacks()`` returns ``[]`` and ``traced`` is an identity
  decorator. memory-unit's existing behaviour (and its offline test suite) is
  unchanged whether or not deepeval is present.
- Tracing is OFF unless a real ``CONFIDENT_API_KEY`` is set (and not explicitly
  disabled), so offline/CI runs never flush or hit the network.
- We use a **custom span type**, never the built-in ``retriever`` type, which
  requires an ``embedder`` field and errors at flush when omitted.
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, List, Optional

try:  # optional dependency; keep memory-unit importable without it
    from deepeval.integrations.langchain import CallbackHandler as _CallbackHandler
    from deepeval.tracing import observe as _observe
    from deepeval.tracing import update_current_trace as _update_current_trace
    _HAS_DEEPEVAL = True
except Exception:  # pragma: no cover - import guard
    _CallbackHandler = None
    _observe = None
    _update_current_trace = None
    _HAS_DEEPEVAL = False


def tracing_enabled() -> bool:
    """True only when deepeval is available, a Confident key is set, and tracing
    is not explicitly disabled — mirrors ``auth.validation_enabled()`` style.

    Gating on the key presence means offline/CI runs (no key) never trace or flush.
    """
    if not _HAS_DEEPEVAL:
        return False
    if not os.getenv("CONFIDENT_API_KEY"):
        return False
    return os.getenv("MEMORY_TRACING", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def tracing_callbacks(thread_id: Optional[str] = None) -> List[Any]:
    """LangChain callbacks to attach to an agent invoke — ``[CallbackHandler(...)]``
    when tracing is enabled, else ``[]``. Never raises.

    ``thread_id`` is passed to the CallbackHandler *constructor* — deepeval groups
    spans by that, not by LangChain's ``configurable.thread_id``."""
    if not tracing_enabled():
        return []
    try:
        return [_CallbackHandler(thread_id=thread_id)] if thread_id else [_CallbackHandler()]
    except Exception:  # pragma: no cover - defensive: never fail on tracing
        return []


def tag_current_trace_thread(thread_id: Optional[str]) -> None:
    """Best-effort: tag the currently-active DeepEval trace with ``thread_id`` so a
    manual span (e.g. the deterministic resolve path, which has no LangChain agent)
    lands on the same thread as the rest of the pipeline. No-op / never raises when
    tracing is off, there is no active trace, or deepeval is absent."""
    if not thread_id or not tracing_enabled() or _update_current_trace is None:
        return
    try:
        _update_current_trace(thread_id=thread_id)
    except Exception:  # pragma: no cover - defensive: never fail on tracing
        return


def traced(name: str) -> Callable:
    """Return a decorator emitting a custom-type span named ``name`` when tracing
    is enabled, and calling the plain function otherwise.

    The observed wrapper is built once at decoration time (no network/flush until
    the function is called); the enabled check is per call, so offline runs (no
    key) never emit or flush a span.
    """
    def decorator(fn: Callable) -> Callable:
        if not _HAS_DEEPEVAL:
            return fn
        try:
            observed = _observe(name=name, type="custom")(fn)
        except Exception:  # pragma: no cover - defensive
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not tracing_enabled():
                return fn(*args, **kwargs)
            try:
                return observed(*args, **kwargs)
            except Exception:
                # Tracing must never break the call. deepeval's span machinery can
                # raise (e.g. "a span must have a valid trace" when there is no
                # active parent trace, or a stale trace context left by a prior
                # LangChain CallbackHandler run). Fall back to the untraced call.
                # The functions traced here are idempotent reads, so the fallback
                # is safe even if `observed` failed after invoking `fn`.
                return fn(*args, **kwargs)

        return wrapper

    return decorator
