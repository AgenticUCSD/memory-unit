"""Phase 5 (steps 1-2): the X-Thread-Id header must reach the core methods.

Previously the header was read on /hydrate but dropped everywhere; the provider
forwards it on /resolve for trace correlation. These tests confirm it is now
threaded into resolve()/query() (fully offline via a recording fake).
"""

import os

os.environ["MEMORY_VALIDATE_TOKEN"] = "false"

import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

import api as api_module
from api import app


class RecordingMemoryUnit:
    def __init__(self, user_id="user-1"):
        self.user_id = user_id
        self.documents = []
        self.seen = {}

    def resolve(self, fields, user_id=None, scope=None, min_score=0.0, thread_id=None, **_):
        self.seen["resolve_thread_id"] = thread_id
        return [
            {"field": f, "value": None, "source": None, "confidence": 0.0, "status": "missing"}
            for f in fields
        ]

    def query(self, query_text, thread_id=None, **_):
        self.seen["query_thread_id"] = thread_id
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


@pytest.fixture(autouse=True)
def reset_registry():
    api_module._memory_units.clear()
    yield
    api_module._memory_units.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake():
    unit = RecordingMemoryUnit(user_id="user-1")
    api_module._memory_units["user-1"] = unit
    return unit


def test_resolve_forwards_thread_id(client, fake):
    resp = client.post(
        "/resolve",
        json={"fields": ["recipient"]},
        headers={"X-User-Id": "user-1", "X-Thread-Id": "T-abc"},
    )
    assert resp.status_code == 200
    assert fake.seen["resolve_thread_id"] == "T-abc"


def test_resolve_without_thread_id_is_none(client, fake):
    resp = client.post(
        "/resolve",
        json={"fields": ["recipient"]},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200
    assert fake.seen["resolve_thread_id"] is None


def test_query_forwards_thread_id(client, fake):
    resp = client.post(
        "/query",
        json={"query": "anything"},
        headers={"X-User-Id": "user-1", "X-Thread-Id": "T-xyz"},
    )
    assert resp.status_code == 200
    assert fake.seen["query_thread_id"] == "T-xyz"
