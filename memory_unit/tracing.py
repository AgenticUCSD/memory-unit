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
from typing import Any, Callable, List

try:  # optional dependency; keep memory-unit importable without it
    from deepeval.integrations.langchain import CallbackHandler as _CallbackHandler
    from deepeval.tracing import observe as _observe
    _HAS_DEEPEVAL = True
except Exception:  # pragma: no cover - import guard
    _CallbackHandler = None
    _observe = None
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


def tracing_callbacks() -> List[Any]:
    """LangChain callbacks to attach to an agent invoke — ``[CallbackHandler()]``
    when tracing is enabled, else ``[]``. Never raises."""
    if not tracing_enabled():
        return []
    try:
        return [_CallbackHandler()]
    except Exception:  # pragma: no cover - defensive: never fail on tracing
        return []


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
            if tracing_enabled():
                return observed(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
