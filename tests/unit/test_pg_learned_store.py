"""Parity + tenancy tests for PGLearnedStore against a real Postgres.

Exercises the pg backend for memory-unit's durable write-back store
(planner.context_blocks). GUARDED: skipped unless TEST_PLANNER_DATABASE_URL points at a
throwaway Postgres with the `planner` schema applied — so the offline suite stays green
and these never touch prod.

The default JSONL store's behavior is guarded separately by tests/unit/test_learn.py.
This file focuses on what's new: per-user dedup, durability, and cross-user isolation.
"""

import hashlib
import os

import pytest

TEST_DB_URL = os.getenv("TEST_PLANNER_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set TEST_PLANNER_DATABASE_URL (throwaway pg with the planner schema) to run",
)

from memory_unit.storage.learned_store import PGLearnedStore, _TABLE  # noqa: E402


def _rec(text, category=None, task_id=None, scope=None):
    return {
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "category": category,
        "task_id": task_id,
        "scope": scope,
    }


@pytest.fixture
def store():
    s = PGLearnedStore(TEST_DB_URL)
    with s._connect() as conn:
        conn.execute(f"TRUNCATE {_TABLE}")
    return s


def test_dedup_per_user(store):
    r = _rec("Default recipient is dana@example.com.")
    store.append([r], user_id="u1")
    store.append([r], user_id="u1")  # duplicate — ON CONFLICT DO NOTHING
    assert store.hashes("u1") == {r["hash"]}
    assert len(store.load("u1")) == 1


def test_records_scoped_by_user(store):
    ra, rb = _rec("alpha fact"), _rec("beta fact")
    store.append([ra], user_id="u1")
    store.append([rb], user_id="u2")
    assert store.hashes("u1") == {ra["hash"]}
    assert store.hashes("u2") == {rb["hash"]}
    assert [x["text"] for x in store.load("u1")] == ["alpha fact"]
    assert [x["text"] for x in store.load("u2")] == ["beta fact"]


def test_cross_user_isolation(store):
    store.append([_rec("secret for u1")], user_id="u1")
    # u2 must see nothing of u1's — the whole point of scoping the shared table.
    assert store.hashes("u2") == set()
    assert store.load("u2") == []


def test_durability_across_store_instances(store):
    r = _rec("Preferred meeting duration is 45 minutes.")
    store.append([r], user_id="u1")
    fresh = PGLearnedStore(TEST_DB_URL)  # simulates a new instance / re-hydrate
    got = fresh.load("u1")
    assert [x["text"] for x in got] == ["Preferred meeting duration is 45 minutes."]


def test_null_user_dedup_is_null_safe(store):
    r = _rec("global fact")
    store.append([r], user_id=None)
    store.append([r], user_id=None)  # COALESCE(user_id,'') unique index dedups
    assert len(store.load(None)) == 1


def test_record_fields_roundtrip(store):
    r = _rec("proj number is 42", category="task_patterns", task_id="t9", scope="user:u1")
    store.append([r], user_id="u1")
    got = store.load("u1")[0]
    assert got["text"] == "proj number is 42"
    assert got["category"] == "task_patterns"
    assert got["task_id"] == "t9"
    assert got["scope"] == "user:u1"
