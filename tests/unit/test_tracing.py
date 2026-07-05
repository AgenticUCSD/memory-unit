"""Phase 5 (steps 1-2): memory-unit tracing must be safe and correctly gated.

Tracing is optional and must never affect behavior: disabled → identity decorator,
empty callbacks, and gated off whenever there is no Confident key or it is
explicitly turned off.
"""
from memory_unit import tracing


def test_tracing_disabled_without_key(monkeypatch):
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_TRACING", raising=False)
    assert tracing.tracing_enabled() is False
    assert tracing.tracing_callbacks() == []


def test_tracing_disabled_by_flag_even_with_key(monkeypatch):
    monkeypatch.setenv("CONFIDENT_API_KEY", "fake-key")
    monkeypatch.setenv("MEMORY_TRACING", "false")
    assert tracing.tracing_enabled() is False
    assert tracing.tracing_callbacks() == []


def test_traced_preserves_behavior_when_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_TRACING", "false")
    calls = []

    @tracing.traced(name="test.span")
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(2, 3) == 5
    assert calls == [(2, 3)]


def test_traced_falls_back_when_span_raises(monkeypatch):
    """A tracing failure (e.g. deepeval 'span must have a valid trace') must never
    propagate — the wrapped function still runs and returns normally."""
    monkeypatch.setenv("CONFIDENT_API_KEY", "fake-key")
    monkeypatch.setenv("MEMORY_TRACING", "true")
    monkeypatch.setattr(tracing, "_HAS_DEEPEVAL", True)

    def boom_observe(name=None, type=None):
        def deco(fn):
            def obs(*a, **k):
                raise ValueError("A span must have a valid trace.")
            return obs
        return deco

    monkeypatch.setattr(tracing, "_observe", boom_observe)

    calls = []

    @tracing.traced(name="retrieval.test")
    def fetch(x):
        calls.append(x)
        return x * 3

    assert fetch(7) == 21
    assert calls == [7]
