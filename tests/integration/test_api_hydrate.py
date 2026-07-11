"""
API-level tests for the /hydrate and /refresh endpoints.

Originally a regression guard for the api.py <-> core.py signature mismatch that
made /hydrate and /refresh raise TypeError. Now also covers per-user routing:
/hydrate builds/refreshes the *calling user's* own unit and never touches another
user's (the old single-occupant 403 lockout is gone — see the isolation tests).

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

    def __init__(self, persist_dir=None, model_name="gpt-4o", user_id=None):
        # Record kwargs so the test can assert auth_token is NOT passed here and
        # that user_id IS (per-user namespacing).
        FakeMemoryUnit.last_init_kwargs = {
            "persist_dir": persist_dir,
            "model_name": model_name,
            "user_id": user_id,
        }
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.user_id = user_id
        self.auth_token = None
        self.root_folder_id = None
        self.folder_config = None
        self.documents = []
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
def reset_registry():
    """Each test starts with an empty per-user registry."""
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


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

    # The bug was passing auth_token to the constructor — assert we don't; and that
    # the per-user id IS threaded in for namespacing.
    assert "auth_token" not in FakeMemoryUnit.last_init_kwargs
    assert FakeMemoryUnit.last_init_kwargs["user_id"] == "user-1"

    # hydrate_from_drive must receive BOTH the folder id and the token.
    assert api_module._memory_units["user-1"].hydrate_calls == [("root123", "ya29.fake")]


def test_refresh_before_hydrate_returns_503(client):
    # No unit for this user yet -> 503 (uninitialized), even with a valid header.
    resp = client.post(
        "/refresh",
        headers={"Authorization": "Bearer ya29.fake", "X-User-Id": "user-1"},
    )
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
    assert api_module._memory_units["user-1"].hydrate_calls[-1] == ("root123", "ya29.second")


# ── per-user isolation on /hydrate (the old 403 lockout is replaced) ──

def test_hydrate_by_different_user_is_isolated(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        r1 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a", "X-User-Id": "user-1"},
        )
        assert r1.status_code == 200, r1.text
        first_unit = api_module._memory_units["user-1"]

        # A different user now gets their OWN unit (no 403 lockout) and does not
        # touch user-1's data.
        r2 = client.post(
            "/hydrate",
            json={"root_folder_id": "rootZZZ"},
            headers={"Authorization": "Bearer ya29.b", "X-User-Id": "user-2"},
        )

    assert r2.status_code == 200, r2.text
    # Both users are resident, each with their own distinct unit.
    assert api_module._memory_units["user-1"] is first_unit  # untouched
    assert api_module._memory_units["user-2"] is not first_unit
    # user-1's unit only ever hydrated its own folder.
    assert first_unit.hydrate_calls == [("root123", "ya29.a")]
    assert api_module._memory_units["user-2"].hydrate_calls == [("rootZZZ", "ya29.b")]


def test_same_user_rehydrate_reuses_their_unit(client):
    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        r1 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a", "X-User-Id": "user-1"},
        )
        unit = api_module._memory_units["user-1"]
        r2 = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer ya29.a2", "X-User-Id": "user-1"},
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text  # the same user may re-hydrate
    # Re-hydrate reuses the same unit object and re-runs hydrate on it.
    assert api_module._memory_units["user-1"] is unit
    assert unit.hydrate_calls == [("root123", "ya29.a"), ("root123", "ya29.a2")]


def test_refresh_with_expired_stored_token_returns_401(client, monkeypatch):
    # A unit hydrated earlier, whose stored token has since expired.
    unit = FakeMemoryUnit(user_id="user-1")
    unit.auth_token = "ya29.stored-expired"
    unit.root_folder_id = "root123"
    api_module._memory_units["user-1"] = unit

    def expired(token, x_user_id):
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

    # Override the (env-disabled) no-op so the expired stored token is actually checked.
    monkeypatch.setattr(api_module, "verify_google_token", expired)

    # No Authorization header -> falls back to the stored (now expired) token.
    resp = client.post("/refresh", headers={"X-User-Id": "user-1"})

    # The fix: a clean 401 ("re-authenticate"), not a 500 from a failed Drive call.
    assert resp.status_code == 401, resp.text


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
