"""API-level tests that /hydrate enforces token validation.

Validation itself is unit-tested in tests/unit/test_auth.py; here we only prove
the endpoint wires it in — a rejected token yields 401 and the memory unit is not
bound to an owner.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api as api_module
from api import app


class FakeMemoryUnit:
    def __init__(self, persist_dir=None, model_name="gpt-4o", user_id=None):
        self.user_id = user_id
        self.auth_token = None
        self.root_folder_id = None
        self.folder_config = None
        self.documents = []

    def hydrate_from_drive(self, root_folder_id, auth_token, thread_id=None, **_):
        return {
            "status": "success",
            "documents_indexed": 1,
            "folder_structure": {},
            "stats": {},
        }


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_hydrate_rejects_invalid_token(client, monkeypatch):
    def reject(token, x_user_id):
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

    monkeypatch.setattr(api_module, "verify_google_token", reject)

    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        resp = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer bad-token", "X-User-Id": "user-1"},
        )

    assert resp.status_code == 401
    # A rejected token must not create a unit for that user.
    assert "user-1" not in api_module._memory_units


def test_hydrate_accepts_valid_token(client, monkeypatch):
    monkeypatch.setattr(api_module, "verify_google_token", lambda token, x_user_id: None)

    with patch.object(api_module, "MemoryUnit", FakeMemoryUnit):
        resp = client.post(
            "/hydrate",
            json={"root_folder_id": "root123"},
            headers={"Authorization": "Bearer good-token", "X-User-Id": "user-1"},
        )

    assert resp.status_code == 200, resp.text
    assert "user-1" in api_module._memory_units


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
