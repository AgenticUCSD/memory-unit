"""Unit tests for the per-user MemoryUnit registry (api._get_unit_for + LRU cap)
and the namespacing helper (core._namespace).

Offline: MemoryUnit is replaced with a light stand-in so no Chroma/Drive is built.
"""

import pytest

import api as api_module
from memory_unit.core import _namespace


class _Stub:
    def __init__(self, persist_dir=None, model_name="gpt-4o", user_id=None):
        self.user_id = user_id
        self.persist_dir = persist_dir


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    monkeypatch.setattr(api_module, "MemoryUnit", _Stub)
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


# ── _namespace ─────────────────────────────────────────────────────

def test_namespace_keeps_alnum_id_verbatim():
    # A numeric Google `sub` is a valid Chroma name — kept readable.
    assert _namespace("108234510293847561234"[:40]) == "108234510293847561234"[:40]
    assert _namespace("abc123") == "abc123"


def test_namespace_hashes_unsafe_ids_without_collision():
    a = _namespace("user@example.com")
    b = _namespace("user_example.com")
    assert a != b  # distinct ids never collide
    assert a.startswith("u") and a[1:].isalnum()  # valid collection suffix


def test_namespace_empty_for_falsy():
    assert _namespace(None) == ""
    assert _namespace("") == ""


# ── _get_unit_for ──────────────────────────────────────────────────

def test_get_without_create_returns_none():
    assert api_module._get_unit_for("nobody", create=False) is None


def test_create_registers_and_binds_user_id():
    unit = api_module._get_unit_for("u1", create=True)
    assert unit is not None
    assert unit.user_id == "u1"
    assert api_module._memory_units["u1"] is unit
    # Idempotent: same user returns the same instance (no duplicate build).
    assert api_module._get_unit_for("u1", create=True) is unit


def test_lru_eviction_past_cap(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_USERS", "2")
    a = api_module._get_unit_for("a", create=True)
    b = api_module._get_unit_for("b", create=True)
    # Touch "a" so "b" becomes least-recently-used.
    assert api_module._get_unit_for("a", create=False) is a
    c = api_module._get_unit_for("c", create=True)  # over cap -> evict LRU ("b")

    assert set(api_module._memory_units) == {"a", "c"}
    assert "b" not in api_module._memory_units
    assert c.user_id == "c"


def test_access_marks_most_recently_used(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_USERS", "2")
    api_module._get_unit_for("a", create=True)
    api_module._get_unit_for("b", create=True)
    api_module._get_unit_for("a", create=True)  # a is now MRU
    api_module._get_unit_for("c", create=True)  # evicts LRU (b)
    assert "a" in api_module._memory_units
    assert "b" not in api_module._memory_units


def test_persist_dir_falls_back_to_env(monkeypatch):
    # Containers set MEMORY_PERSIST_DIR to a writable path (Cloud Run: /tmp); a unit
    # created without an explicit persist_dir must pick it up.
    monkeypatch.setenv("MEMORY_PERSIST_DIR", "/tmp/memory_data")
    unit = api_module._get_unit_for("u1", create=True)
    assert unit.persist_dir == "/tmp/memory_data"


def test_explicit_persist_dir_wins_over_env(monkeypatch):
    monkeypatch.setenv("MEMORY_PERSIST_DIR", "/tmp/memory_data")
    unit = api_module._get_unit_for("u1", create=True, persist_dir="/data/explicit")
    assert unit.persist_dir == "/data/explicit"


def test_no_persist_dir_when_env_unset(monkeypatch):
    monkeypatch.delenv("MEMORY_PERSIST_DIR", raising=False)
    unit = api_module._get_unit_for("u1", create=True)
    assert unit.persist_dir is None  # in-package default used downstream


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
