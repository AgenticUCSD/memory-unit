"""Phase 1 — pluggable durable store for write-back ("learned") context.

Backs ``MemoryUnit.learn()`` / ``_reload_learned()``. Two interchangeable backends,
selected by ``STORE_BACKEND`` (default ``jsonl``):

- ``JSONLLearnedStore`` — the historical behavior: a per-instance
  ``{persist_dir}/learned_context.jsonl``. Single-tenant, so it ignores ``user_id``.
- ``PGLearnedStore`` — the shared ``planner.context_blocks`` table (``STORE_BACKEND=pg``).
  Because the table is shared across users/instances, it **scopes every row by
  ``user_id``** (per-user dedup + filtered reads) so no user sees another's learned facts.

Uniform interface:
  - ``hashes(user_id) -> set[str]``     — existing content hashes (for dedup)
  - ``append(records, user_id) -> None`` — persist new records (already deduped by caller)
  - ``load(user_id) -> list[dict]``     — records to re-index after a hydrate

Each record is ``{hash, text, category, task_id, scope}`` — the JSONL format, unchanged.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set


def _store_backend() -> str:
    return os.getenv("STORE_BACKEND", "jsonl").strip().lower()


def make_learned_store(persist_dir: Optional[str]):
    """Return the configured learned-context store. ``STORE_BACKEND=pg`` → Postgres
    (``planner.context_blocks``); anything else → the default per-instance JSONL."""
    if _store_backend() == "pg":
        return PGLearnedStore(os.getenv("PLANNER_DATABASE_URL", ""))
    return JSONLLearnedStore(persist_dir)


class JSONLLearnedStore:
    """Per-instance JSONL durable store — the historical default. Single-tenant, so
    ``user_id`` is accepted for interface symmetry but ignored."""

    def __init__(self, persist_dir: Optional[str]):
        self._path = (
            os.path.join(persist_dir, "learned_context.jsonl") if persist_dir else None
        )

    def hashes(self, user_id: Optional[str] = None) -> Set[str]:
        out: Set[str] = set()
        if not self._path or not os.path.exists(self._path):
            return out
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # skip a corrupt/partial line, keep the rest
                    if rec.get("hash"):
                        out.add(rec["hash"])
        except OSError:
            return out
        return out

    def append(self, records: List[Dict[str, Any]], user_id: Optional[str] = None) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to persist learned context: {e}")

    def load(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._path or not os.path.exists(self._path):
            return out
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # skip a corrupt line, don't drop the rest
                out.append(rec)
        return out


_TABLE = "planner.context_blocks"


class PGLearnedStore:
    """Shared Postgres durable store (``planner.context_blocks``), scoped by ``user_id``."""

    def __init__(self, database_url: str):
        if not database_url:
            raise RuntimeError(
                "STORE_BACKEND=pg but PLANNER_DATABASE_URL is not set. Provide the "
                "planner_app connection string, or unset STORE_BACKEND to use the JSONL store."
            )
        import psycopg
        from psycopg.rows import dict_row

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._url = database_url

    def _connect(self):
        return self._psycopg.connect(
            self._url, autocommit=True, row_factory=self._dict_row
        )

    def hashes(self, user_id: Optional[str] = None) -> Set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT hash FROM {_TABLE} "
                "WHERE COALESCE(user_id, '') = COALESCE(%s, '')",
                (user_id,),
            ).fetchall()
        return {r["hash"] for r in rows}

    def append(self, records: List[Dict[str, Any]], user_id: Optional[str] = None) -> None:
        if not records:
            return
        with self._connect() as conn:
            for rec in records:
                conn.execute(
                    f"""
                    INSERT INTO {_TABLE} (user_id, hash, text, category, task_id, scope)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (COALESCE(user_id, ''), hash) DO NOTHING
                    """,
                    (
                        user_id,
                        rec.get("hash"),
                        rec.get("text"),
                        rec.get("category"),
                        rec.get("task_id"),
                        rec.get("scope"),
                    ),
                )

    def load(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT hash, text, category, task_id, scope FROM {_TABLE} "
                "WHERE COALESCE(user_id, '') = COALESCE(%s, '') "
                "ORDER BY created_at, id",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
