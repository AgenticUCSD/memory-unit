"""API tests for /learn (write-back) — behind the owner guard.

Offline: the memory unit is a fake; core write-back logic is unit-tested in
tests/unit/test_learn.py.
"""

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    def __init__(self):
        self.learned = []

    def learn(self, items):
        self.learned.extend(items)
        return len(items)


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


def test_learn_requires_user_id(client, hydrated):
    resp = client.post("/learn", json={"items": [{"text": "x"}]})
    assert resp.status_code == 400


def test_learn_wrong_user_is_forbidden(client, hydrated):
    resp = client.post(
        "/learn", json={"items": [{"text": "x"}]}, headers={"X-User-Id": "user-2"}
    )
    assert resp.status_code == 403


def test_learn_owner_succeeds(client, hydrated):
    resp = client.post(
        "/learn",
        json={"items": [{"text": "Default recipient is a@b.com"}, {"text": "y"}]},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 2
    assert len(api_module._memory_unit.learned) == 2
