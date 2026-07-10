"""Unit tests for MemoryUnit.resolve() — structured slot resolution.

Runs offline: only the deterministic BM25 keyword index is populated (no Drive,
no OpenAI key needed). The vector store is left empty so the vector fallback
returns nothing, keeping "missing" fields genuinely missing.
"""

from memory_unit.core import MemoryUnit


def _hydrated_unit(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    mu.keyword_searcher.index_documents(
        [
            "Preferred meeting duration is 30 minutes for standups.",
            "Default recipient for status updates is alice@example.com.",
        ],
        [{"filename": "prefs.txt"}, {"filename": "contacts.txt"}],
    )
    mu.is_hydrated = True
    return mu


def test_resolve_fills_known_field(tmp_path):
    mu = _hydrated_unit(tmp_path)
    out = mu.resolve(["recipient"])

    assert len(out) == 1
    slot = out[0]
    assert slot["field"] == "recipient"
    assert slot["status"] == "present"
    # Value is extracted to just the email, not the whole sentence.
    assert slot["value"] == "alice@example.com"
    # The originating snippet is preserved as evidence for transparency.
    assert slot["evidence"] and "Default recipient" in slot["evidence"]
    assert slot["source"] == "context"
    assert 0.0 < slot["confidence"] <= 1.0


def test_resolve_extracts_duration_number(tmp_path):
    mu = _hydrated_unit(tmp_path)
    slot = mu.resolve(["meeting_duration"])[0]
    assert slot["status"] == "present"
    assert slot["value"] == "30 minutes"


def test_extract_value_prefers_number_with_unit():
    # Regression: the "1" in "1:1s" must not beat the real "30 minutes".
    mu = MemoryUnit.__new__(MemoryUnit)
    got = mu._extract_value(
        "meeting_duration", "Preferred meeting duration for 1:1s is 30 minutes."
    )
    assert got == "30 minutes"


def test_extract_value_head_is_word_bounded():
    # "cat" must match the standalone word, not the "cat" inside "vacation".
    mu = MemoryUnit.__new__(MemoryUnit)
    got = mu._extract_value("cat", "vacation: beach. The cat is fluffy")
    assert got == "fluffy"


def test_extract_value_falls_back_to_clause():
    # No email/number type match -> clause after the connector, trimmed.
    mu = MemoryUnit.__new__(MemoryUnit)  # no __init__ needed for the pure helper
    got = mu._extract_value("topic", "The topic is Q3 planning and budget review. More text.")
    assert got == "Q3 planning and budget review"


def test_resolve_unknown_field_is_missing(tmp_path):
    mu = _hydrated_unit(tmp_path)
    out = mu.resolve(["nonexistent_zzz_slot"])

    assert out[0]["status"] == "missing"
    assert out[0]["value"] is None
    assert out[0]["confidence"] == 0.0


def test_resolve_not_hydrated_all_missing(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    out = mu.resolve(["recipient", "duration"])

    assert [s["status"] for s in out] == ["missing", "missing"]
    assert all(s["value"] is None for s in out)


def test_resolve_preserves_field_order(tmp_path):
    mu = _hydrated_unit(tmp_path)
    fields = ["duration", "recipient", "unknown_zzz"]
    out = mu.resolve(fields)
    assert [s["field"] for s in out] == fields


def test_resolve_serves_after_learn_without_hydrate(tmp_path):
    # Hydrate-gap fix: a unit seeded via learn() is queryable without a Drive hydrate.
    mu = MemoryUnit(persist_dir=str(tmp_path))
    assert mu.is_hydrated is False
    mu.learn([{"text": "Default recipient is eve@example.com."}])
    slot = mu.resolve(["recipient"])[0]
    assert slot["status"] == "present"
    assert slot["value"] == "eve@example.com"


def test_resolve_prefers_more_specific_scope(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    mu.learn([
        {"text": "The billing project code is PROJ-GLOBAL-1.", "scope": "global"},
        {"text": "The billing project code is PROJ-USER-9.", "scope": "user"},
    ])
    # "user" is more specific than "global" -> the user-scoped value wins.
    slot = mu.resolve(["project_code"], scope=["user", "global"])[0]
    assert slot["status"] == "present"
    assert "PROJ-USER-9" in slot["value"]
    assert slot["scope"] == "user"


# ── LLM value extraction (opt-in via MEMORY_RESOLVE_LLM; deterministic fallback) ──


def test_resolve_llm_disabled_is_deterministic(tmp_path, monkeypatch):
    # Flag OFF (default): even if the LLM extractor *would* return something else,
    # resolve() uses the deterministic value.
    monkeypatch.delenv("MEMORY_RESOLVE_LLM", raising=False)
    mu = _hydrated_unit(tmp_path)
    monkeypatch.setattr(mu, "_extract_value_llm", lambda field, text: "LLM-SHOULD-NOT-RUN")
    slot = mu.resolve(["recipient"])[0]
    assert slot["value"] == "alice@example.com"  # deterministic, LLM never consulted


def test_resolve_value_uses_llm_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_RESOLVE_LLM", "true")
    mu = _hydrated_unit(tmp_path)
    monkeypatch.setattr(mu, "_extract_value_llm", lambda field, text: "alice (from LLM)")
    slot = mu.resolve(["recipient"])[0]
    assert slot["value"] == "alice (from LLM)"
    # Evidence is still the original snippet, not the LLM output.
    assert "Default recipient" in slot["evidence"]


def test_resolve_value_falls_back_when_llm_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_RESOLVE_LLM", "true")
    mu = _hydrated_unit(tmp_path)
    monkeypatch.setattr(mu, "_extract_value_llm", lambda field, text: None)
    slot = mu.resolve(["recipient"])[0]
    assert slot["value"] == "alice@example.com"  # deterministic fallback


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Records the prompt and returns a canned content (or raises)."""

    def __init__(self, content=None, raises=False):
        self._content = content
        self._raises = raises

    def invoke(self, prompt):
        if self._raises:
            raise RuntimeError("boom")
        return _FakeMsg(self._content)


def _bare_unit():
    mu = MemoryUnit.__new__(MemoryUnit)  # no __init__ / no key needed
    mu.model_name = "gpt-4o"
    return mu


def test_extract_value_llm_parses_content():
    mu = _bare_unit()
    mu._extractor_llm = _FakeLLM(content='  "bob@x.com" ')
    assert mu._extract_value_llm("recipient", "email bob@x.com somewhere") == "bob@x.com"


def test_extract_value_llm_none_reply_is_none():
    mu = _bare_unit()
    mu._extractor_llm = _FakeLLM(content="NONE")
    assert mu._extract_value_llm("recipient", "no email here") is None


def test_extract_value_llm_error_is_none():
    mu = _bare_unit()
    mu._extractor_llm = _FakeLLM(raises=True)
    assert mu._extract_value_llm("recipient", "anything") is None


def test_extract_value_llm_empty_text_is_none():
    mu = _bare_unit()
    mu._extractor_llm = _FakeLLM(content="should-not-matter")
    assert mu._extract_value_llm("recipient", "   ") is None
