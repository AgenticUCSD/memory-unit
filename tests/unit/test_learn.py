"""Unit tests for write-back (MemoryUnit.learn) — self-learning ingest.

Offline: exercises the deterministic keyword path (vector add is best-effort and
guarded), so no OpenAI key is required.
"""

from memory_unit.core import MemoryUnit


def test_learn_makes_value_resolvable(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    mu.is_hydrated = True

    n = mu.learn([{"text": "Default recipient for updates is dana@example.com."}])
    assert n == 1

    slot = mu.resolve(["recipient"])[0]
    assert slot["status"] == "present"
    assert slot["value"] == "dana@example.com"


def test_learn_dedups_identical_text(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    mu.is_hydrated = True

    assert mu.learn([{"text": "Fact A about widgets."}]) == 1
    assert mu.learn([{"text": "Fact A about widgets."}]) == 0  # duplicate skipped


def test_learn_empty_is_zero(tmp_path):
    mu = MemoryUnit(persist_dir=str(tmp_path))
    mu.is_hydrated = True
    assert mu.learn([]) == 0
    assert mu.learn([{"text": "   "}]) == 0  # blank text ignored


def test_learned_context_persists_across_units(tmp_path):
    # Learn on one unit, then reload into a fresh unit on the same persist dir —
    # this is what a re-hydrate does, so learning must survive it.
    mu1 = MemoryUnit(persist_dir=str(tmp_path))
    mu1.is_hydrated = True
    mu1.learn([{"text": "Preferred meeting duration is 45 minutes."}])

    mu2 = MemoryUnit(persist_dir=str(tmp_path))
    mu2._reload_learned()
    mu2.is_hydrated = True

    slot = mu2.resolve(["meeting_duration"])[0]
    assert slot["status"] == "present"
    assert slot["value"] == "45 minutes"
