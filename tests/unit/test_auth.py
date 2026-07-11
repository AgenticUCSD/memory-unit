"""Unit tests for server-side Google token validation (memory_unit.auth).

The network (Google tokeninfo) is mocked, so these run fully offline.
"""

import urllib.error

import pytest
from fastapi import HTTPException

from memory_unit import auth


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def clear_cache():
    auth._TOKEN_CACHE.clear()
    yield
    auth._TOKEN_CACHE.clear()


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "false")

    def boom(*a, **k):
        raise AssertionError("network must not be called when validation is disabled")

    monkeypatch.setattr(auth.urllib.request, "urlopen", boom)
    auth.verify_google_token("anything", "user-1")  # no exception, no network


def test_valid_token_sub_match(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResp(200, '{"sub": "user-1", "email": "a@b.com"}'),
    )
    auth.verify_google_token("tok", "user-1")  # no raise


def test_sub_mismatch_is_401(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResp(200, '{"sub": "someone-else"}'),
    )
    with pytest.raises(HTTPException) as ei:
        auth.verify_google_token("tok", "user-1")
    assert ei.value.status_code == 401


def test_invalid_token_is_401(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")

    def raise_http_error(*a, **k):
        raise urllib.error.HTTPError("url", 400, "Bad Request", {}, None)

    monkeypatch.setattr(auth.urllib.request, "urlopen", raise_http_error)
    with pytest.raises(HTTPException) as ei:
        auth.verify_google_token("bad", "user-1")
    assert ei.value.status_code == 401


def test_google_unreachable_is_503(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")

    def raise_urlerror(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(auth.urllib.request, "urlopen", raise_urlerror)
    with pytest.raises(HTTPException) as ei:
        auth.verify_google_token("tok", "user-1")
    assert ei.value.status_code == 503


def test_cache_avoids_second_network_call(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    calls = {"n": 0}

    def once(url, timeout=None):
        calls["n"] += 1
        return _FakeResp(200, '{"sub": "user-1"}')

    monkeypatch.setattr(auth.urllib.request, "urlopen", once)
    auth.verify_google_token("tok", "user-1")
    auth.verify_google_token("tok", "user-1")
    assert calls["n"] == 1  # second call served from cache


# ── MEMORY_REQUIRE_SUB (strict-sub) flag ───────────────────────────

def _subless(url, timeout=None):
    # An API-scope token (gmail/calendar, no openid) returns no `sub`.
    return _FakeResp(200, '{"email": "a@b.com"}')


def test_subless_token_passes_by_default(monkeypatch):
    # Default (flag off) is lenient — matches the executor and valid API-scope tokens.
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.delenv("MEMORY_REQUIRE_SUB", raising=False)
    monkeypatch.setattr(auth.urllib.request, "urlopen", _subless)
    auth.verify_google_token("tok", "user-1")  # no raise


def test_subless_token_rejected_when_require_sub(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.setenv("MEMORY_REQUIRE_SUB", "true")
    monkeypatch.setattr(auth.urllib.request, "urlopen", _subless)
    with pytest.raises(HTTPException) as ei:
        auth.verify_google_token("tok", "user-1")
    assert ei.value.status_code == 401


def test_sub_token_still_passes_when_require_sub(monkeypatch):
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.setenv("MEMORY_REQUIRE_SUB", "true")
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResp(200, '{"sub": "user-1"}'),
    )
    auth.verify_google_token("tok", "user-1")  # no raise


def test_require_sub_enforced_on_cached_path(monkeypatch):
    # A sub-less token cached while lenient must still be rejected once strict mode
    # flips on (the cache stores sub=None and _check_sub re-evaluates the flag).
    monkeypatch.setenv("MEMORY_VALIDATE_TOKEN", "true")
    monkeypatch.setattr(auth.urllib.request, "urlopen", _subless)
    auth.verify_google_token("tok", "user-1")  # caches sub=None (lenient)
    monkeypatch.setenv("MEMORY_REQUIRE_SUB", "true")
    with pytest.raises(HTTPException) as ei:
        auth.verify_google_token("tok", "user-1")  # cache hit, now strict
    assert ei.value.status_code == 401
