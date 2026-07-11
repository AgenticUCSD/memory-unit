"""Server-side verification of the caller's Google OAuth access token.

memory-unit consumes the caller's Google token to read their Drive and binds the
unit's owner on hydrate, so before trusting it we verify it against Google's
tokeninfo endpoint and cross-check the returned ``sub`` against the caller's
``X-User-Id`` — the same approach the executor uses (``services/auth.py``). Until
now the bearer was only *extracted*, never *verified*.

Validation is ON by default; set ``MEMORY_VALIDATE_TOKEN=false`` to skip it for
local/offline development (or tests that exercise wiring, not auth). Uses only the
standard library so it adds no dependency.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from fastapi import HTTPException

GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
TOKENINFO_TIMEOUT_SECONDS = 5.0
TOKEN_CACHE_TTL_SECONDS = 300  # 5 minutes, matching the executor

# token -> (sub_or_None, expires_at_monotonic)
_TOKEN_CACHE: dict = {}


def validation_enabled() -> bool:
    """Token validation is on unless explicitly disabled."""
    return os.getenv("MEMORY_VALIDATE_TOKEN", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def require_sub_enabled() -> bool:
    """Strict mode: reject a validated token that carries no ``sub``.

    OFF by default (only meaningful when :func:`validation_enabled`). A Google
    access token with only API scopes (gmail/calendar, no openid) returns no
    ``sub``, so the default trusts ``X-User-Id`` for those — matching the
    executor. But once the store is per-user, an unauthenticated ``X-User-Id``
    is the way to read another user's data, so enabling this (``MEMORY_REQUIRE_SUB``)
    binds every request to a real Google identity. Turn on only once callers are
    known to send an ``openid`` token, or those valid API-scope tokens will 401.
    """
    return os.getenv("MEMORY_REQUIRE_SUB", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _check_sub(sub: Optional[str], x_user_id: str) -> None:
    """Enforce the sub/X-User-Id relationship for a validated token."""
    if sub and sub != x_user_id:
        raise HTTPException(status_code=401, detail="User ID does not match token")
    if require_sub_enabled() and not sub:
        raise HTTPException(
            status_code=401,
            detail="Token has no subject; an openid-scoped token is required",
        )


def _get_cached(token: str):
    """Return a 1-tuple ``(sub,)`` on a live cache hit, else None (a miss)."""
    entry = _TOKEN_CACHE.get(token)
    if entry is None:
        return None
    sub, expires_at = entry
    if time.monotonic() >= expires_at:
        _TOKEN_CACHE.pop(token, None)
        return None
    return (sub,)


def _set_cached(token: str, sub: Optional[str]) -> None:
    _TOKEN_CACHE[token] = (sub, time.monotonic() + TOKEN_CACHE_TTL_SECONDS)


def verify_google_token(token: str, x_user_id: str) -> None:
    """Validate a Google access token; cross-check its ``sub`` against ``x_user_id``.

    No-op when validation is disabled. Raises ``HTTPException(401)`` on an
    invalid/expired token or a sub mismatch, and ``503`` if Google is unreachable
    (so the caller can distinguish "your token is bad" from "we couldn't check").
    """
    if not validation_enabled():
        return

    cached = _get_cached(token)
    if cached is not None:
        _check_sub(cached[0], x_user_id)
        return

    url = GOOGLE_TOKENINFO_URL + "?" + urllib.parse.urlencode({"access_token": token})
    try:
        with urllib.request.urlopen(url, timeout=TOKENINFO_TIMEOUT_SECONDS) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # tokeninfo answers 400 for a bad/expired token.
        if exc.code in (400, 401):
            raise HTTPException(status_code=401, detail="Invalid or expired access token")
        raise HTTPException(status_code=503, detail="Token validation failed upstream")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not reach Google tokeninfo: {exc}"
        )

    if status != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

    try:
        info = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token info response")

    # Cross-check `sub` when present. Lenient when absent by default: a Google
    # access token carrying only API scopes (e.g. gmail/calendar, no openid) does
    # not return `sub` from tokeninfo, so we then trust X-User-Id — matching the
    # executor's `services/auth.py`. Set MEMORY_REQUIRE_SUB=on (see require_sub_enabled)
    # to reject those sub-less tokens once per-user isolation makes an unauthenticated
    # X-User-Id a cross-user read vector.
    sub = info.get("sub")
    _check_sub(sub, x_user_id)

    _set_cached(token, sub)
