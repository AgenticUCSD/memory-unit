"""API tests for /learn (write-back) — behind the owner guard.

Offline: the memory unit is a fake; core write-back logic is unit-tested in
tests/unit/test_learn.py.
"""

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app
from memory_unit.core import MemoryUnit as RealMemoryUnit


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


@pytest.fixture
def no_token_validation(monkeypatch):
    """/learn requires a bearer token; these tests use a fake one, so turn off the
    Google tokeninfo check (validated separately in test_api_token_validation)."""
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "false")


_AUTH = {"Authorization": "Bearer ya29.fake"}


def test_learn_requires_user_id(client, hydrated):
    resp = client.post("/learn", json={"items": [{"text": "x"}]}, headers=_AUTH)
    assert resp.status_code == 400


def test_learn_wrong_user_is_forbidden(client, hydrated):
    resp = client.post(
        "/learn",
        json={"items": [{"text": "x"}]},
        headers={**_AUTH, "X-User-Id": "user-2"},
    )
    assert resp.status_code == 403


def test_learn_without_token_is_401(client, hydrated):
    resp = client.post(
        "/learn", json={"items": [{"text": "x"}]}, headers={"X-User-Id": "user-1"}
    )
    assert resp.status_code == 401


def test_learn_owner_succeeds(client, hydrated, no_token_validation):
    resp = client.post(
        "/learn",
        json={"items": [{"text": "Default recipient is a@b.com"}, {"text": "y"}]},
        headers={**_AUTH, "X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 2
    assert len(api_module._memory_unit.learned) == 2


def test_learn_lazy_inits_binds_owner_and_serves_resolve(
    client, tmp_path, monkeypatch, no_token_validation
):
    # Fresh server: no _memory_unit, no owner. /learn should init the unit, claim
    # ownership, and make the fact resolvable — all without a Drive hydrate.
    monkeypatch.setattr(
        api_module, "MemoryUnit", lambda *a, **k: RealMemoryUnit(persist_dir=str(tmp_path))
    )

    r1 = client.post(
        "/learn",
        json={"items": [{"text": "Default recipient is zoe@example.com."}]},
        headers={**_AUTH, "X-User-Id": "u9"},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["learned"] == 1
    assert api_module._owner_user_id == "u9"  # first writer claimed the unit

    r2 = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "u9"}
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["slots"][0]["value"] == "zoe@example.com"
