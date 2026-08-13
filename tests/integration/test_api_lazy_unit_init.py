"""API tests for feat/lazy-unit-init.

``require_user_unit`` used to 503 until the caller ran /hydrate for that user on
*this* process. It now lazily creates a real (initially empty) MemoryUnit via
``_get_unit_for(create=True)``, which also re-ingests that user's durable "learned"
(write-back) blocks at creation time — so a cold instance can serve resolve()/query()
from what the user already taught it, with zero Drive I/O.

Fully offline: real MemoryUnit + tmp_path Chroma/JSONL storage, no Drive or OpenAI
key needed (mirrors tests/unit/test_learn.py's and
test_api_learn.py::test_learn_lazy_inits_and_serves_resolve's pattern).
"""

import os

os.environ["MEMORY_VALIDATE_TOKEN"] = "false"

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api as api_module
from api import app
from memory_unit.core import MemoryUnit as RealMemoryUnit


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ── no unit yet -> 200, not 503 ───────────────────────────────────────

def test_resolve_for_never_seen_user_is_200_all_missing(client):
    resp = client.post(
        "/resolve",
        json={"fields": ["recipient", "duration"]},
        headers={"X-User-Id": "brand-new-user"},
    )
    assert resp.status_code == 200, resp.text
    slots = resp.json()["slots"]
    assert [s["status"] for s in slots] == ["missing", "missing"]
    assert "brand-new-user" in api_module._memory_units  # lazily created & registered


# ── cold registry still serves durable learned context ────────────────

def test_cold_registry_resolves_previously_learned_block_without_hydrate(
    client, tmp_path, monkeypatch
):
    # Seed the durable learned store the way a prior /learn call would have,
    # bypassing the API/registry entirely (this stands in for e.g. a row already
    # sitting in Postgres `planner.context_blocks` from before this process started).
    seed = RealMemoryUnit(persist_dir=str(tmp_path), user_id="u-cold")
    assert seed.learn([{"text": "Default recipient for updates is dana@example.com."}]) == 1

    # Cold registry: nothing resident for this user. Route MemoryUnit construction
    # at the same persist_dir/user namespace the seed used, mirroring how a real
    # deploy's MEMORY_PERSIST_DIR is stable across process restarts.
    assert "u-cold" not in api_module._memory_units
    monkeypatch.setattr(
        api_module,
        "MemoryUnit",
        lambda *a, **k: RealMemoryUnit(persist_dir=str(tmp_path), user_id=k.get("user_id")),
    )

    resp = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "u-cold"}
    )
    assert resp.status_code == 200, resp.text
    slot = resp.json()["slots"][0]
    assert slot["status"] == "present"
    assert slot["value"] == "dana@example.com"
    # Resolved without ever calling /hydrate -> the lazy unit is still un-hydrated.
    assert api_module._memory_units["u-cold"].is_hydrated is False


# ── learned-store failure degrades, never 500s ─────────────────────────

def test_learned_store_failure_at_creation_degrades_to_200_not_500(client, monkeypatch):
    def boom(self):
        raise RuntimeError("learned store unreachable")

    monkeypatch.setattr(RealMemoryUnit, "_reload_learned", boom)

    resp = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "u-broken"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["slots"][0]["status"] == "missing"
    assert "u-broken" in api_module._memory_units  # unit still created & usable


# ── auth is unchanged by the lazy-create switch ─────────────────────────

def test_auth_missing_token_is_still_401(client, monkeypatch):
    # Read endpoints skip the bearer check when validation is off (module default
    # here); turn validation back on to prove a missing token still 401s before any
    # unit is created — the lazy-create change must not bypass auth.
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    resp = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "u1"}
    )
    assert resp.status_code == 401
    assert "u1" not in api_module._memory_units


def test_auth_mismatched_user_id_is_still_401(client, monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")

    def mismatched(token, x_user_id):
        raise HTTPException(status_code=401, detail="User ID does not match token")

    monkeypatch.setattr(api_module, "verify_google_token", mismatched)

    resp = client.post(
        "/resolve",
        json={"fields": ["recipient"]},
        headers={"Authorization": "Bearer ya29.fake", "X-User-Id": "u1"},
    )
    assert resp.status_code == 401
    assert "u1" not in api_module._memory_units


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


# ── re-creating a unit must not accumulate duplicates on disk ─────────

def test_repeated_unit_creation_does_not_grow_the_vector_store(tmp_path):
    """The regression that makes read-path lazy creation safe to run repeatedly.

    _reload_learned() now runs on *every* construction, not only after hydrate's
    clear(). Chroma is a PersistentClient on disk, so without a deterministic
    doc_id each rebuild would re-add the same learned blocks under a fresh uuid4
    and the collection would grow without bound across LRU evict/recreate cycles.
    On Cloud Run persist_dir is memory-backed /tmp, so that is an instance-memory
    leak, not just wasted disk.
    """
    base = str(tmp_path / "store")
    user = "dup-check-user"

    first = RealMemoryUnit(persist_dir=base, user_id=user)
    first.learn([{"text": "My preferred meeting duration is 45 minutes."}])
    after_learn = first.vector_store.collection.count()
    assert after_learn > 0, "learn() should have indexed at least one block"

    # Simulate: evicted from the registry, then lazily re-created by a later read
    # over the same on-disk directory. Each pass re-runs _reload_learned().
    for _ in range(3):
        rebuilt = RealMemoryUnit(persist_dir=base, user_id=user)
        rebuilt._reload_learned()
        assert rebuilt.vector_store.collection.count() == after_learn

    # And the block is still actually resolvable after the rebuilds.
    final = RealMemoryUnit(persist_dir=base, user_id=user)
    final._reload_learned()
    slots = final.resolve(["duration"], user_id=user)
    assert slots[0]["status"] == "present"
    assert "45" in str(slots[0]["value"])
