"""API tests for /resolve — structured slot resolution with per-user routing.

Fully offline: per-user fake units are registered directly and token validation is
off (routing/shape, not auth), so no ChromaDB / Drive / OpenAI key is exercised.
"""

import os

os.environ["MEMORY_VALIDATE_TOKEN"] = "false"

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    def __init__(self, user_id):
        self.user_id = user_id
        self.documents = []

    def resolve(self, fields, user_id=None, scope=None, min_score=0.0, thread_id=None, **_):
        # Echo the user_id it was called with so the test can assert per-user scoping.
        return [
            {
                "field": f,
                "value": f"val-{f}-{user_id}",
                "source": "context",
                "confidence": 0.5,
                "status": "present",
            }
            for f in fields
        ]


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _register(user_id):
    api_module._memory_units[user_id] = FakeMemoryUnit(user_id)


def test_resolve_requires_user_id(client):
    _register("user-1")
    resp = client.post("/resolve", json={"fields": ["recipient"]})
    assert resp.status_code == 400


def test_resolve_unknown_user_is_503(client):
    _register("user-1")
    resp = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "user-2"}
    )
    assert resp.status_code == 503


def test_resolve_routes_to_callers_unit(client):
    _register("user-1")
    resp = client.post(
        "/resolve",
        json={"fields": ["recipient", "duration"]},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    slots = resp.json()["slots"]
    assert [s["field"] for s in slots] == ["recipient", "duration"]
    # Value carries the caller's own user_id -> resolve was scoped to their unit.
    assert slots[0]["value"] == "val-recipient-user-1"
    assert slots[0]["source"] == "context"
    assert slots[0]["status"] == "present"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
