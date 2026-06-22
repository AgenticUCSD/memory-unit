# Drive Connection — state, gotchas, and how to test it

This note documents how memory-unit connects to Google Drive, what was fixed, what
still needs credentials to validate, and the concrete gotchas that will bite during
end-to-end testing. Written for M (and anyone testing the Drive-dependent paths).

## The intended loop

```
Chrome extension  ──creates──▶  Drive: "Agent Workspace" (root)
                                  ├── "My Knowledge"         (user-provided docs)
                                  └── "Generated Knowledge"  (machine-generated context)

workflow-provider AnalyzerAgent ──distills traces──▶ user_preferences.txt /
                                                      task_patterns.txt /
                                                      workflow_trends.txt

memory-unit  ──POST /hydrate {root_folder_id} + Bearer token──▶ reads BOTH subfolders,
             indexes everything into one Chroma + BM25 index, serves /query and
             /context/* to the extension / task-identifier / workflow-builder.
```

## What was fixed (this change)

- `POST /hydrate` and `POST /refresh` **previously threw `TypeError`** (api.py called
  `MemoryUnit(auth_token=...)` and `hydrate_from_drive()` with the wrong arity). They now
  match `core.py`: `MemoryUnit(persist_dir, model_name)` + `hydrate_from_drive(root_folder_id,
  auth_token)`. The Drive **read** path is now reachable.
- `hydrate_from_drive` no longer hard-requires `OPENAI_API_KEY` — the query agent is only
  needed at `/query` time, so hydration/indexing works without an OpenAI key (indexing falls
  back to Chroma's default local embeddings).
- `/refresh` now reuses the root folder captured at the last `/hydrate` (stored on
  `MemoryUnit.root_folder_id`) and reports `"refreshed"` correctly.
- Regression test: `tests/integration/test_api_hydrate.py` (offline, mocked).

## Verified folder-name matching

memory-unit's matcher (`memory_unit/drive/client.py:109`) is case-insensitive substring:
- user-provided folder ← name contains `user`, `provided`, or `my knowledge`
- machine-generated folder ← name contains `machine` or `generated`

The extension's `My Knowledge` and `Generated Knowledge` names **do** resolve correctly. The
`root_folder_id` you pass to `/hydrate` must be the **"Agent Workspace"** folder id.

## ⚠ Gotchas that will bite end-to-end testing

1. **Machine-generated folder is read shallow + text-only.** `core.hydrate_from_drive`
   processes only **top-level** files whose mime type is `text/plain` or Google Doc, and
   **skips subfolders** (`core.py:131-174`). Consequences:
   - The analyzer's `user_preferences.txt` / `task_patterns.txt` / `workflow_trends.txt` must
     sit at the **root** of "Generated Knowledge" (as plain text or Google Docs) to be ingested.
   - The extension currently writes thread logs as **JSON** into a **"Thread Logs" subfolder**
     under "Generated Knowledge" → **not ingested** (wrong location *and* wrong type).
2. **User-provided folder skips Drive shortcuts.** `DocumentProcessor.process_drive_files`
   skips `application/vnd.google-apps.shortcut`. The extension adds Drive-hosted files to
   "My Knowledge" as **shortcuts** (only non-Drive HTTPS files are uploaded as real copies).
   So Drive-hosted docs added via the extension **won't be ingested** as written today.

## What is NOT built (deliberately deferred — safe choice)

The **write-back** side (analyzer → Drive) is not implemented. We are **not** giving
workflow-provider a Drive client or passing user OAuth tokens to `/analyze_traces`, because
that broadens the provider's attack surface and can't be tested without credentials. Distilled
context also carries a prompt-injection risk (a hostile email could poison learned context),
so any automated write-back must pass a safety gate first.

**Recommended owner: the extension** — it already holds the user's OAuth token and Drive write
scope, and already builds the folder structure. Flow: provider returns knowledge text → the
extension writes it to the (gated) "Generated Knowledge" folder **at the root, as text/plain**.

## How to test

### Now, offline (no credentials)
```
cd memory-unit && ./venv/bin/python -m pytest tests/ -v          # 44 tests, all mocked
```

### Live hydrate against real Drive (needs a Google OAuth token)
1. In Drive, create `Agent Workspace` with subfolders `My Knowledge` and `Generated Knowledge`
   (or reuse the folder the extension created). Put a couple of `.txt`/PDF files in
   `My Knowledge` (as real files, not shortcuts) and the 3 knowledge `.txt` files at the **root**
   of `Generated Knowledge`.
2. Get an OAuth access token with Drive read scope (e.g. from the extension via
   `chrome.identity.getAuthToken`, or the OAuth playground).
3. Start the service and hydrate:
   ```
   cd memory-unit && OPENAI_API_KEY=... ./venv/bin/python api.py    # serves on :8000
   curl -X POST localhost:8000/hydrate \
     -H "Authorization: Bearer <google_access_token>" \
     -H "Content-Type: application/json" \
     -d '{"root_folder_id":"<Agent Workspace folder id>"}'
   curl localhost:8000/stats
   curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"query":"what are my meeting preferences?"}'
   ```
   `OPENAI_API_KEY` is only needed for `/query` (the agent) and for OpenAI embeddings; hydrate
   alone works without it.
```
