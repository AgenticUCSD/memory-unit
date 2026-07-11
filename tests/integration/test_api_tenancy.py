"""API-level tests for the tenancy slice: CORS allow-list + per-user routing.

Run fully offline: per-user fake units are registered directly, so no ChromaDB /
Drive / OpenAI key is needed. Token validation is disabled here (these test routing
+ isolation, not auth — auth is covered in test_api_token_validation.py).
"""

import os
from types import SimpleNamespace

os.environ["MEMORY_VALIDATE_TOKEN"] = "false"

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    """Stand-in exposing just what /query and /stats touch, tagged by user."""

    def __init__(self, user_id, answer="ok"):
        self.user_id = user_id
        self._answer = answer
        self.documents = []

    def query(self, query_text, thread_id=None, **_):
        return SimpleNamespace(
            answer=self._answer,
            sources=[],
            context_for_extension="",
            context_for_task_identifier="",
            context_for_workflow_builder="",
            user_preferences=[],
            task_patterns=[],
            workflow_trends=[],
        )

    def get_stats(self):
        return {
            "is_hydrated": True,
            "total_documents": 0,
            "vector_store_count": 0,
            "keyword_index_size": 0,
        }


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _register(user_id, answer="ok"):
    api_module._memory_units[user_id] = FakeMemoryUnit(user_id, answer=answer)


# ── per-user routing guard ────────────────────────────────────────

def test_hydrate_requires_user_id(client):
    # Auth present, but no X-User-Id -> 400 (and not the 401 auth path).
    resp = client.post(
        "/hydrate",
        json={"root_folder_id": "root123"},
        headers={"Authorization": "Bearer ya29.fake"},
    )
    assert resp.status_code == 400
    assert "X-User-Id" in resp.json()["detail"]


def test_query_without_user_id_is_rejected(client):
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 400


def test_query_for_user_without_a_unit_is_503(client):
    # A user who has not hydrated has no unit yet -> 503 (not 403; there is no
    # cross-user lockout anymore).
    _register("user-1")
    resp = client.post(
        "/query", json={"query": "hi"}, headers={"X-User-Id": "user-2"}
    )
    assert resp.status_code == 503


def test_query_routes_to_callers_own_unit(client):
    resp = client.post(
        "/query", json={"query": "hi"}, headers={"X-User-Id": "user-1"}
    )
    # 503 before hydrate...
    assert resp.status_code == 503
    _register("user-1")
    resp = client.post(
        "/query", json={"query": "hi"}, headers={"X-User-Id": "user-1"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] == "ok"


def test_query_is_isolated_between_users(client):
    # Two users each have their own unit; each /query sees only its own.
    _register("user-1", answer="one")
    _register("user-2", answer="two")
    r1 = client.post("/query", json={"query": "hi"}, headers={"X-User-Id": "user-1"})
    r2 = client.post("/query", json={"query": "hi"}, headers={"X-User-Id": "user-2"})
    assert r1.json()["answer"] == "one"
    assert r2.json()["answer"] == "two"


def test_stats_routes_per_user(client):
    assert client.get("/stats").status_code == 400  # no X-User-Id
    assert client.get("/stats", headers={"X-User-Id": "user-1"}).status_code == 503
    _register("user-1")
    assert client.get("/stats", headers={"X-User-Id": "user-1"}).status_code == 200


# ── CORS allow-list ────────────────────────────────────────────

def test_cors_allows_configured_origin(client):
    # /health is open; an allowed origin is echoed back in ACAO.
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_blocks_unconfigured_origin(client):
    resp = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in resp.headers


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
