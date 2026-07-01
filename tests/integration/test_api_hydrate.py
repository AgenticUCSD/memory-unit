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

    def hydrate_from_drive(self, root_folder_id, auth_token):
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
