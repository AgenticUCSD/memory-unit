"""
API-level tests for the /hydrate and /refresh endpoints.

These are a regression guard for the api.py <-> core.py signature mismatch that
made /hydrate and /refresh raise TypeError (api.py called
MemoryUnit(auth_token=...) and hydrate_from_drive() with the wrong arity).

They run fully offline: the core MemoryUnit is replaced with a fake, so no
ChromaDB, no Google Drive, and no OpenAI key are required.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    """Minimal stand-in for core.MemoryUnit that records how it was called."""

    last_init_kwargs = None

    def __init__(self, persist_dir=None, model_name="gpt-4o"):
        # Record kwargs so the test can assert auth_token is NOT passed here.
        FakeMemoryUnit.last_init_kwargs = {
            "persist_dir": persist_dir,
            "model_name": model_name,
        }
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.auth_token = None
        self.root_folder_id = None
        self.folder_config = None
        self.hydrate_calls = []

    def hydrate_from_drive(self, root_folder_id, auth_token, thread_id=None, **_):
        # Mirror the real signature + side effects relied on by /refresh.
        self.hydrate_calls.append((root_folder_id, auth_token))
        self.auth_token = auth_token
        self.root_folder_id = root_folder_id
        return {
            "status": "success",
            "documents_indexed": 3,
            "preference_summaries": 1,  # extra key — must be tolerated
            "folder_structure": {
                "root": "Workspace",
                "user_provided": "My Knowledge",
                "machine_generated": "Generated Knowledge",
            },
            "stats": {"total_chunks": 3},
        }


@pytest.fixture(autouse=True)
def reset_global():
    """Each test starts with no hydrated memory unit and no owner."""
    api_module._memory_unit = None
    api_module._owner_user_id = None
    yield
    api_module._memory_unit = None
    api_module._owner_user_id = None


@pytest.fixture(autouse=True)
def disable_token_validation(monkeypatch):
    """These tests exercise hydrate/refresh *wiring* with fake tokens, not token
    validity — so turn off the Google tokeninfo check (covered separately in
    test_api_token_validation.py)."""
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "false")


@pytest.fixture
def client():
    return TestClient(app)


def test_hydrate_requires_auth(client):
    resp = client.post("/hydrate", json={"root_folder_id": "root123"})
    assert resp.status_code == 401


def test_hydrate_succeeds_and_wires_args_correctly(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        resp = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.fake", "X-User-Id": "user-1"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["documents_indexed"] == 3

    # The bug was passing auth_token to the constructor — assert we don't.
    assert "auth_token" not in FakeMemoryUnit.last_init_kwargs

    # hydrate_from_drive must receive BOTH the folder id and the token.
    assert api_module._memory_unit.hydrate_calls == [("root123", "ya29.fake")]


def test_refresh_before_hydrate_returns_503(client):
    # No hydration yet -> dependency reports the unit is uninitialized.
    resp = client.post("/refresh", headers={"Authorization": "Bearer ya29.fake"})
    assert resp.status_code == 503


def test_refresh_reuses_stored_root_folder(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.first", "X-User-Id": "user-1"},
        )
        resp = client.post(
            "/refresh",
            headers={"Authorization": "Bearer ya29.second", "X-User-Id": "user-1"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "refreshed"
    # Refresh re-hydrates the same folder with the new token, no folder id needed.
    assert api_module._memory_unit.hydrate_calls[-1] == ("root123", "ya29.second")


# ── single-tenant owner guard on /hydrate (takeover prevention) ──

def test_hydrate_by_different_user_is_forbidden(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        r1 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a", "X-User-Id": "user-1"},
        )
        assert r1.status_code == 200, r1.text
        first_unit = api_module._memory_unit

        # A different valid-token user must NOT be able to take over / wipe the unit.
        r2 = client.post(
            "/hydrate",
            json={"root_folder_id": "rootZZZ"},
            headers={"Authorization": "Bearer ya29.b", "X-User-Id": "user-2"},
        )

    assert r2.status_code == 403
    # Incumbent owner + unit are untouched (no takeover, no re-create/wipe).
    assert api_module._owner_user_id == "user-1"
    assert api_module._memory_unit is first_unit


def test_same_user_can_rehydrate(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        r1 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a", "X-User-Id": "user-1"},
        )
        r2 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a2", "X-User-Id": "user-1"},
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text  # the owner may re-hydrate
    assert api_module._owner_user_id == "user-1"


def test_refresh_with_expired_stored_token_returns_401(client, monkeypatch):
    # A unit hydrated earlier, whose stored token has since expired.
    unit = FakeMemoryUnit()
    unit.auth_token = "ya29.stored-expired"
    unit.root_folder_id = "root123"
    api_module._memory_unit = unit
    api_module._owner_user_id = "user-1"

    def expired(token, x_user_id):
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

    # Override the (env-disabled) no-op so the expired stored token is actually checked.
    monkeypatch.setattr(api_module, "verify_google_token", expired)

    # No Authorization header -> falls back to the stored (now expired) token.
    resp = client.post("/refresh", headers={"X-User-Id": "user-1"})

    # The fix: a clean 401 ("re-authenticate"), not a 500 from a failed Drive call.
    assert resp.status_code == 401, resp.text
