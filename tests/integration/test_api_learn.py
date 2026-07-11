"""API tests for /learn (write-back) — per-user routing.

Offline: the memory unit is a fake; core write-back logic is unit-tested in
tests/unit/test_learn.py.
"""

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app
from memory_unit.core import MemoryUnit as RealMemoryUnit


class FakeMemoryUnit:
    def __init__(self, persist_dir=None, model_name="gpt-4o", user_id=None):
        self.user_id = user_id
        self.documents = []
        self.learned = []

    def learn(self, items, thread_id=None, **_):
        self.learned.extend(items)
        return len(items)


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def hydrated():
    api_module._memory_units["user-1"] = FakeMemoryUnit(user_id="user-1")


@pytest.fixture
def no_token_validation(monkeypatch):
    """/learn requires a bearer token; these tests use a fake one, so turn off the
    Google tokeninfo check (validated separately in test_api_token_validation)."""
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "false")


_AUTH = {"Authorization": "Bearer ya29.fake"}


def test_learn_requires_user_id(client, hydrated):
    resp = client.post("/learn", json={"items": [{"text": "x"}]}, headers=_AUTH)
    assert resp.status_code == 400


def test_learn_without_token_is_401(client, hydrated):
    resp = client.post(
        "/learn", json={"items": [{"text": "x"}]}, headers={"X-User-Id": "user-1"}
    )
    assert resp.status_code == 401


def test_learn_succeeds(client, hydrated, no_token_validation):
    resp = client.post(
        "/learn",
        json={"items": [{"text": "Default recipient is a@b.com"}, {"text": "y"}]},
        headers={**_AUTH, "X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 2
    assert len(api_module._memory_units["user-1"].learned) == 2


def test_learn_is_isolated_per_user(client, monkeypatch, no_token_validation):
    # Each user's /learn lands in that user's own unit; users never share one.
    monkeypatch.setattr(api_module, "MemoryUnit", FakeMemoryUnit)
    client.post(
        "/learn", json={"items": [{"text": "u1 fact"}]},
        headers={**_AUTH, "X-User-Id": "user-1"},
    )
    client.post(
        "/learn", json={"items": [{"text": "u2 a"}, {"text": "u2 b"}]},
        headers={**_AUTH, "X-User-Id": "user-2"},
    )
    assert [i["text"] for i in api_module._memory_units["user-1"].learned] == ["u1 fact"]
    assert [i["text"] for i in api_module._memory_units["user-2"].learned] == ["u2 a", "u2 b"]
    assert api_module._memory_units["user-1"] is not api_module._memory_units["user-2"]


def test_learn_lazy_inits_and_serves_resolve(
    client, tmp_path, monkeypatch, no_token_validation
):
    # Fresh server: no unit for u9. /learn should create u9's unit and make the fact
    # resolvable — all without a Drive hydrate.
    monkeypatch.setattr(
        api_module,
        "MemoryUnit",
        lambda *a, **k: RealMemoryUnit(persist_dir=str(tmp_path), user_id=k.get("user_id")),
    )

    r1 = client.post(
        "/learn",
        json={"items": [{"text": "Default recipient is zoe@example.com."}]},
        headers={**_AUTH, "X-User-Id": "u9"},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["learned"] == 1
    assert "u9" in api_module._memory_units  # unit lazily created for the writer

    r2 = client.post(
        "/resolve", json={"fields": ["recipient"]}, headers={"X-User-Id": "u9"}
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["slots"][0]["value"] == "zoe@example.com"
