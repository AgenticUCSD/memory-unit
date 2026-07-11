"""End-to-end read-isolation across users — the core of the multi-tenancy fix.

Uses REAL MemoryUnits (not fakes) seeded via /learn (write-back needs no Drive), so
it exercises the actual per-user Chroma collections + BM25 indices. Proves that two
users hydrated into the SAME process cannot read each other's context — the gap the
old single-occupant lockout only papered over. Offline: token validation off, no
Drive; embeddings fall back to Chroma's default fn when OPENAI_API_KEY is unset.
"""

import os

os.environ["MEMORY_VALIDATE_TOKEN"] = "false"

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app

_AUTH = {"Authorization": "Bearer ya29.fake"}


@pytest.fixture(autouse=True)
def reset_registry(tmp_path, monkeypatch):
    # Real per-user MemoryUnits, but rooted under a throwaway dir so each user's
    # Chroma/JSONL lives on disk under tmp (and is namespaced per user inside).
    base = str(tmp_path)
    from memory_unit.core import MemoryUnit as RealMemoryUnit

    def _factory(persist_dir=None, model_name="gpt-4o", user_id=None):
        return RealMemoryUnit(persist_dir=base, model_name=model_name, user_id=user_id)

    monkeypatch.setattr(api_module, "MemoryUnit", _factory)
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _learn(client, user_id, text):
    r = client.post(
        "/learn", json={"items": [{"text": text}]},
        headers={**_AUTH, "X-User-Id": user_id},
    )
    assert r.status_code == 200, r.text


def _resolve(client, user_id, field):
    r = client.post(
        "/resolve", json={"fields": [field]}, headers={"X-User-Id": user_id}
    )
    assert r.status_code == 200, r.text
    return r.json()["slots"][0]


def test_resolve_is_isolated_between_users(client):
    # Two users write distinct facts for the SAME field name.
    _learn(client, "alice", "The recipient is alice-contact@example.com.")
    _learn(client, "bob", "The recipient is bob-contact@example.com.")

    alice = _resolve(client, "alice", "recipient")
    bob = _resolve(client, "bob", "recipient")

    # Each user resolves ONLY their own value — no cross-user bleed.
    assert alice["value"] == "alice-contact@example.com"
    assert bob["value"] == "bob-contact@example.com"


def test_second_user_does_not_evict_first(client):
    _learn(client, "alice", "The recipient is alice-contact@example.com.")
    # A second user hydrating/learning must NOT wipe alice's data (the old model
    # cleared a shared store on every hydrate).
    _learn(client, "bob", "The recipient is bob-contact@example.com.")

    # Alice still resolves her own value after bob wrote his.
    assert _resolve(client, "alice", "recipient")["value"] == "alice-contact@example.com"
    # Both units are resident and distinct.
    assert api_module._memory_units["alice"] is not api_module._memory_units["bob"]


def test_user_without_data_sees_nothing_from_others(client):
    _learn(client, "alice", "The recipient is alice-contact@example.com.")
    # carol has a unit but no matching fact -> missing, never alice's value.
    _learn(client, "carol", "Unrelated note about weather.")
    carol = _resolve(client, "carol", "recipient")
    assert carol["value"] != "alice-contact@example.com"


def test_distinct_users_get_distinct_collections(client):
    _learn(client, "alice", "hello")
    _learn(client, "bob", "hello")
    a = api_module._memory_units["alice"]
    b = api_module._memory_units["bob"]
    # Physical partition: different Chroma collection names per user.
    assert a.vector_store.collection.name != b.vector_store.collection.name


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
