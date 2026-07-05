"""API tests for /resolve — structured slot resolution behind the owner guard.

Fully offline: the global memory unit is a fake and the owner is set directly,
so no ChromaDB / Drive / OpenAI key is exercised.
"""

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    def resolve(self, fields, user_id=None, scope=None, min_score=0.0, thread_id=None, **_):
        return [
            {
                "field": f,
                "value": f"val-{f}",
                "source": "context",
                "confidence": 0.5,
                "status": "present",
            }
            for f in fields
        ]


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
    api_module._memory_unit = FakeMemoryUnit()
    api_module._owner_user_id = "user-1"


def test_resolve_requires_user_id(client, hydrated):
    resp = client.post("/resolve", json={"fields": ["recipient"]})
    assert resp.status_code == 400


def test_resolve_wrong_user_is_forbidden(client, hydrated):
    resp = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "user-2"}
    )
    assert resp.status_code == 403


def test_resolve_owner_succeeds(client, hydrated):
    resp = client.post(
        "/resolve",
        json={"fields": ["recipient", "duration"]},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    slots = resp.json()["slots"]
    assert [s["field"] for s in slots] == ["recipient", "duration"]
    assert slots[0]["value"] == "val-recipient"
    assert slots[0]["source"] == "context"
    assert slots[0]["status"] == "present"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
