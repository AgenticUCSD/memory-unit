"""API-level tests for the tenancy slice: CORS allow-list + single-tenant owner guard.

Run fully offline: the global memory unit and owner are set directly, and the memory
unit is a fake, so no ChromaDB / Drive / OpenAI key is needed.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    """Stand-in exposing just what /query and /stats touch."""

    def query(self, query_text):
        return SimpleNamespace(
            answer="ok",
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
def reset_global():
    api_module._memory_unit = None
    api_module._owner_user_id = None
    yield
    api_module._memory_unit = None
    api_module._owner_user_id = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def hydrated():
    """Simulate a unit hydrated by user-1 without going through Drive."""
    api_module._memory_unit = FakeMemoryUnit()
    api_module._owner_user_id = "user-1"


# ── owner guard ────────────────────────────────────────────────

def test_hydrate_requires_user_id(client):
    # Auth present, but no X-User-Id -> 400 (and not the 401 auth path).
    resp = client.post(
        "/hydrate",
        json={"root_folder_id": "root123"},
        headers={"Authorization": "Bearer ya29.fake"},
    )
    assert resp.status_code == 400
    assert "X-User-Id" in resp.json()["detail"]


def test_query_without_user_id_is_rejected(client, hydrated):
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 400


def test_query_with_wrong_user_is_forbidden(client, hydrated):
    resp = client.post(
        "/query", json={"query": "hi"}, headers={"X-User-Id": "user-2"}
    )
    assert resp.status_code == 403


def test_query_with_owner_succeeds(client, hydrated):
    resp = client.post(
        "/query", json={"query": "hi"}, headers={"X-User-Id": "user-1"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] == "ok"


def test_stats_is_guarded(client, hydrated):
    assert client.get("/stats").status_code == 400
    assert client.get("/stats", headers={"X-User-Id": "user-2"}).status_code == 403
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
