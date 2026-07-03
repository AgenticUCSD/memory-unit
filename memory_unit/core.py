"""
Core MemoryUnit implementation.
"""

import hashlib
import json
import os
import re
from typing import List, Dict, Any, Optional

# Deterministic value extraction for common slot kinds.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|%|percent|dollars?|usd)?\b",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"(?:\bis\b|\bare\b|:|=)\s*(.+)", re.IGNORECASE)

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from memory_unit.models.documents import ContextDocument, FolderSummary
from memory_unit.models.query import ContextQueryResult, DriveFolderConfig
from memory_unit.storage.vector_store import VectorStore
from memory_unit.storage.bm25_search import BM25Searcher
from memory_unit.processing.document_processor import DocumentProcessor
from memory_unit.processing.preference_analyzer import PreferenceAnalyzer
from memory_unit.drive.client import GoogleDriveClient
from memory_unit.agents.tools import MemoryTools, QueryResult


class MemoryUnit:
    """
    Agentic RAG Memory Unit.

    Provides:
    - Google Drive integration with 2 subfolders
    - Vector store (Chroma) + keyword search (BM25)
    - Agentic RAG with hybrid search
    - Context injection for Extension, Task Identifier, Workflow Builder

    Usage:
        # Create without auth (persistent storage)
        memory = MemoryUnit(persist_dir="./data")

        # Later, hydrate with ephemeral token from extension
        result = memory.hydrate_from_drive(
            root_folder_id="folder123",
            auth_token="oauth_token_from_extension"
        )
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        model_name: str = "gpt-4o"
    ):
        self.auth_token: Optional[str] = None
        self.root_folder_id: Optional[str] = None
        self.persist_dir = persist_dir
        self.model_name = model_name
        # Durable store for write-back ("learned") context, re-applied on hydrate.
        self._learned_path = (
            os.path.join(persist_dir, "learned_context.jsonl") if persist_dir else None
        )

        # Initialize persistent components that don't need auth
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.keyword_searcher = BM25Searcher()
        self.memory_tools = MemoryTools(self.vector_store, self.keyword_searcher)

        # Document processors
        self.document_processor = DocumentProcessor()
        self.preference_analyzer = PreferenceAnalyzer(llm_model=model_name)

        # Components requiring auth (initialized on hydrate)
        self.drive_client: Optional[GoogleDriveClient] = None
        self.query_agent: Optional[Any] = None

        # State
        self.documents: List[ContextDocument] = []
        self.is_hydrated: bool = False

        # Try to initialize agent (may fail if no OpenAI key, that's ok)
        try:
            self._initialize_agent()
        except Exception:
            self.query_agent = None

    def hydrate_from_drive(
        self,
        root_folder_id: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Hydrate the memory unit from Google Drive.

        This is called by the extension with an ephemeral auth token.
        Combines both folders into unified index:
        - User Provided Context: Diverse documents -> chunked
        - Machine Generated Context: Preferences/trends -> chunked
        - All content indexed together in vector store + keyword search

        Args:
            root_folder_id: Drive folder ID containing User and Machine subfolders
            auth_token: OAuth token from extension (ephemeral)

        Returns:
            Dict with status, document counts, and folder structure info
        """
        # Store the ephemeral token and the root folder so /refresh can re-hydrate
        # without the caller re-supplying the folder id.
        self.auth_token = auth_token
        self.root_folder_id = root_folder_id

        # Create drive client with ephemeral token
        self.drive_client = GoogleDriveClient(auth_token)

        # Re-initialize agent if needed (now that we may have context). The agent is
        # only required at query time, so a missing OpenAI key must not break hydration
        # — mirror the tolerant behavior of __init__.
        if not self.query_agent:
            try:
                self._initialize_agent()
            except Exception:
                self.query_agent = None

        # Clear existing data before re-hydrating
        self.clear()

        # Get folder structure
        structure = self.drive_client.get_folder_structure(root_folder_id)

        all_documents = []
        preference_summaries = []
        stats = {
            "user_provided_count": 0,
            "machine_generated_count": 0,
            "total_chunks": 0,
            "preference_files": 0
        }

        # Process user-provided folder
        if structure.get("user_provided"):
            user_docs = self.document_processor.process_drive_files(
                self.drive_client,
                structure["user_provided"]["id"],
                "user_provided"
            )
            all_documents.extend(user_docs)
            stats["user_provided_count"] = len(user_docs)

        # Process machine-generated folder
        if structure.get("machine_generated"):
            machine_folder_id = structure["machine_generated"]["id"]
            files = self.drive_client.list_files_in_folder(machine_folder_id)

            for file_info in files:
                mime_type = file_info.get("mimeType", "")

                if mime_type == "application/vnd.google-apps.folder":
                    continue
                if mime_type not in ["text/plain", "application/vnd.google-apps.document"]:
                    continue

                try:
                    content = self.drive_client.get_file_content(file_info["id"], mime_type)
                    if not content or not content.strip():
                        continue

                    # Analyze as preference file
                    summary = self.preference_analyzer.analyze_preference_file(
                        filename=file_info["name"],
                        content=content,
                        last_modified=file_info.get("modifiedTime", "")
                    )
                    preference_summaries.append(summary)

                    # Chunk and add to unified index
                    chunks = self.document_processor.chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            doc = ContextDocument(
                                content=chunk,
                                source=file_info.get("webViewLink", ""),
                                filename=file_info["name"],
                                doc_type=".txt",
                                folder="machine_generated",
                                chunk_index=i
                            )
                            all_documents.append(doc)

                except Exception as e:
                    print(f"Error processing {file_info['name']}: {e}")
                    continue

            stats["machine_generated_count"] = len(all_documents) - stats["user_provided_count"]
            stats["preference_files"] = len(preference_summaries)

        # Index preference summaries for metadata-enriched retrieval
        if preference_summaries:
            self.preference_analyzer.index_summaries(preference_summaries)

        # Store and index all documents together
        self.documents = all_documents
        stats["total_chunks"] = len(all_documents)

        if all_documents:
            self.vector_store.clear()
            self.vector_store.add_documents(all_documents)

            # Index everything for keyword search
            texts = [doc.content for doc in all_documents]
            metadatas = [
                {
                    "source": doc.source,
                    "filename": doc.filename,
                    "folder": doc.folder,
                    "is_preference": doc.folder == "machine_generated"
                }
                for doc in all_documents
            ]
            self.keyword_searcher.index_documents(texts, metadatas)

        self.is_hydrated = True

        # Re-apply previously learned (write-back) context so it survives the
        # clear()+rebuild that hydrate does. Best-effort: must not break hydrate.
        try:
            self._reload_learned()
        except Exception as e:
            print(f"Failed to reload learned context: {e}")

        return {
            "status": "success",
            "documents_indexed": len(all_documents),
            "preference_summaries": len(preference_summaries),
            "folder_structure": {
                "root": structure.get("root", {}).get("name") if structure.get("root") else None,
                "user_provided": structure.get("user_provided", {}).get("name") if structure.get("user_provided") else None,
                "machine_generated": structure.get("machine_generated", {}).get("name") if structure.get("machine_generated") else None
            },
            "stats": stats
        }

    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent for query routing."""
        model = ChatOpenAI(model=self.model_name, temperature=0)

        self.query_agent = create_agent(
            model=model,
            response_format=ToolStrategy(QueryResult),
            system_prompt=(
                "You are an intelligent Memory Unit assistant that helps retrieve relevant context from a user's documents.\n\n"
                "You have access to these tools:\n"
                "- vector_search: Find documents about concepts, ideas, topics (semantic search)\n"
                "- keyword_search: Find specific words, names, exact phrases (literal search)\n"
                "- get_document_count: Check how many documents are indexed\n\n"
                "Follow this EXACT workflow:\n"
                "1. Call BOTH vector_search AND keyword_search with the initial query\n"
                "2. Review the results - if you have enough information, provide your answer\n"
                "3. Only search again if the first search returned no relevant results\n\n"
                "DO NOT make more than 2 rounds of searches. After reviewing results, you MUST answer.\n\n"
                "After providing your answer, you MUST populate the sources_cited field with the filenames of documents you used.\n"
                "Also populate these fields with appropriate context:\n"
                "- context_for_extension: Brief context summary for the Extension component\n"
                "- context_for_task_identifier: Specific context for Task Identifier about task patterns\n"
                "- context_for_workflow_builder: Relevant workflow patterns for Workflow Builder"
            ),
            tools=self.memory_tools.get_tools()
        )

    def query(self, query_text: str) -> ContextQueryResult:
        """
        Query the memory unit using agentic RAG.

        Uses unified search over all content (user docs + preferences):
        - Vector/keyword search treats all documents equally
        - Preference metadata enriches component-specific context
        - Agent routes to best sources regardless of folder
        """
        if not self.is_hydrated:
            return ContextQueryResult(
                answer="Memory unit not yet hydrated. Please call hydrate_from_drive() first.",
                sources=[],
                context_for_extension="",
                context_for_task_identifier="",
                context_for_workflow_builder="",
                user_preferences=[],
                task_patterns=[],
                workflow_trends=[]
            )

        if not self.query_agent:
            raise ValueError("Query agent not initialized")

        # Run unified agentic search
        content = f"Retrieve relevant context for: {query_text}\n\n"

        chat = [{"role": "user", "content": content}]

        result = self.query_agent.invoke(
            {"messages": chat},
            config={"recursion_limit": 12}
        )

        # Extract result
        query_result = self._extract_query_result(result)

        # Get relevant preferences by category
        user_prefs = self.preference_analyzer.get_relevant_preferences(
            query_text, category="user_preferences", top_k=2
        )
        task_patterns = self.preference_analyzer.get_relevant_preferences(
            query_text, category="task_patterns", top_k=2
        )
        workflow_trends = self.preference_analyzer.get_relevant_preferences(
            query_text, category="workflow_trends", top_k=2
        )

        # Build component-specific context
        context_for_task_id = self._build_task_identifier_context(
            query_result.get("context_for_task_identifier", ""),
            user_prefs,
            task_patterns
        )
        context_for_workflow = self._build_workflow_builder_context(
            query_result.get("context_for_workflow_builder", ""),
            workflow_trends,
            user_prefs
        )

        return ContextQueryResult(
            answer=query_result.get("answer", ""),
            sources=[{"filename": s} for s in query_result.get("sources_cited", [])],
            context_for_extension=query_result.get("context_for_extension", ""),
            context_for_task_identifier=context_for_task_id,
            context_for_workflow_builder=context_for_workflow,
            user_preferences=[p.raw_content[:500] for p in user_prefs],
            task_patterns=[p.raw_content[:500] for p in task_patterns],
            workflow_trends=[p.raw_content[:500] for p in workflow_trends]
        )

    def resolve(
        self,
        fields: List[str],
        user_id: Optional[str] = None,
        scope: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Resolve task parameter *slots* to concrete values from indexed context.

        Unlike ``query()`` (which returns prose for injection), this returns a
        structured ``field -> value`` list so the planner can pre-fill task
        parameters from the user's own context before falling back to a human
        (HITL). Each result carries ``source`` and a bounded ``confidence`` so the
        caller can decide whether to trust the value or still ask.

        Deliberately deterministic — keyword/BM25 over the unified index, no LLM —
        so it is cheap and testable offline. LLM-based value *extraction* from the
        matched evidence is a follow-up; today ``value`` is the best matching
        context snippet.

        Args:
            fields: slot names to resolve (e.g. ["recipient", "meeting_duration"]).
            user_id: reserved for per-user isolation (single-tenant today).
            scope: optional **ordered** list of preferred scopes, most-specific
                first (e.g. ["thread:T1", "user:U1", "global"]). When given,
                evidence tagged with a more-specific scope wins over a less-specific
                or unscoped one; unscoped evidence is always an allowed fallback.
            min_score: minimum BM25 score before a field counts as resolved.

        Returns:
            One dict per input field: ``{field, value, evidence, source, confidence,
            scope, status}``. Unresolved fields come back ``status="missing"``.
        """
        resolved: List[Dict[str, Any]] = []
        for field in fields:
            item: Dict[str, Any] = {
                "field": field,
                "value": None,
                "evidence": None,
                "source": None,
                "confidence": 0.0,
                "scope": None,
                "status": "missing",
            }

            if self._servable() and field and field.strip():
                best = self._best_evidence_for(field, min_score=min_score, scope=scope)
                if best is not None:
                    item.update(best)
                    item["status"] = "present"

            resolved.append(item)

        return resolved

    def _servable(self) -> bool:
        """Whether the unit can serve resolve() — after a Drive hydrate OR once
        seeded via learn() (write-back). Lets /learn make a fresh unit queryable
        without a Drive round-trip."""
        if self.is_hydrated or self.documents:
            return True
        try:
            return self.vector_store.count() > 0
        except Exception:
            return False

    @staticmethod
    def _scope_rank(meta_scope: Optional[str], scope: List[str]) -> int:
        """Preference rank of a hit's scope: lower = more preferred. Scopes listed
        earlier (more specific) rank lower; an unlisted/unscoped hit ranks last but
        is still allowed as a fallback."""
        if meta_scope in scope:
            return scope.index(meta_scope)
        return len(scope)

    def _best_evidence_for(
        self, field: str, min_score: float = 0.0, scope: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Best supporting snippet for a slot name, or None.

        Keyword/BM25 first (deterministic, no embeddings); vector search is a
        best-effort fallback when the keyword index yields nothing. When ``scope``
        is given, a more-specific-scoped hit beats a higher-relevance but
        less-specific one.
        """
        # Slot names are typically snake_case; the BM25 tokenizer drops "_"-joined
        # words, so search on a space-normalized query (e.g. "meeting_duration" ->
        # "meeting duration"). The original `field` still drives value extraction.
        query = field.replace("_", " ").strip() or field

        hits = self.keyword_searcher.search(query, top_k=5 if scope else 1)
        candidates = [h for h in hits if float(h.get("score", 0.0)) > min_score]
        if candidates:
            if scope:
                # Prefer scope specificity first, then relevance.
                chosen = min(
                    candidates,
                    key=lambda h: (
                        self._scope_rank((h.get("metadata") or {}).get("scope"), scope),
                        -float(h.get("score", 0.0)),
                    ),
                )
            else:
                chosen = candidates[0]
            score = float(chosen["score"])
            snippet = chosen["content"].strip()
            return {
                "value": self._extract_value(field, snippet),
                "evidence": snippet[:500],
                "source": "context",
                "scope": (chosen.get("metadata") or {}).get("scope"),
                # Map an unbounded BM25 score monotonically into (0, 1).
                "confidence": round(score / (score + 1.0), 3),
            }

        # Vector fallback — embeddings may be unavailable offline, so guard it.
        try:
            results = self.vector_store.query(query, n_results=1)
            docs = results.get("documents") or [[]]
            dists = results.get("distances") or [[]]
            if docs and docs[0]:
                distance = float(dists[0][0]) if dists and dists[0] else 1.0
                snippet = docs[0][0].strip()
                # Chroma distances are >= 0 (smaller = closer). Convert to (0, 1].
                return {
                    "value": self._extract_value(field, snippet),
                    "evidence": snippet[:500],
                    "source": "context",
                    "scope": None,
                    "confidence": round(1.0 / (1.0 + distance), 3),
                }
        except Exception:
            pass

        return None

    def _extract_value(self, field: str, text: str) -> str:
        """Best-effort concise value for ``field`` from an evidence snippet.

        Deterministic and conservative: type-aware regexes for common slot kinds
        (email, number/duration), else the clause following the field mention,
        else a trimmed first sentence. Always returns a non-empty string (falls
        back to the snippet) so a resolved slot never has an empty value. The full
        snippet is preserved separately as ``evidence`` for transparency.
        """
        text = " ".join(text.split())  # normalize whitespace
        if not text:
            return text
        field_l = field.lower()

        if any(k in field_l for k in ("email", "recipient", "sender", "contact", "address")):
            m = _EMAIL_RE.search(text)
            if m:
                return m.group(0)

        if any(
            k in field_l
            for k in ("duration", "length", "minutes", "time", "number", "count", "amount", "size", "budget")
        ):
            num = self._pick_number(text)
            if num:
                return num

        # "<field> ... is/:/= <value>" — the clause after a connector following
        # the field mention.
        head = field_l.split("_")[0]
        idx = text.lower().find(head) if head else -1
        if idx != -1:
            m = _CONNECTOR_RE.search(text[idx:])
            if m:
                clause = re.split(r"[.;\n]", m.group(1))[0].strip()
                if clause:
                    return clause[:200]

        # Fallback: first sentence / trimmed snippet.
        first = re.split(r"[.;\n]", text)[0].strip()
        return (first or text)[:200]

    def _pick_number(self, text: str) -> Optional[str]:
        """Pick the most meaningful number in ``text``.

        Prefers a number carrying a unit ("30 minutes", "20%") over a bare number,
        so incidental figures like the "1" in "1:1s" don't win over the real
        value. Falls back to the first bare number if none carry a unit.
        """
        fallback = None
        for m in _NUMBER_RE.finditer(text):
            tok = m.group(0).strip()
            if not tok:
                continue
            if re.search(r"[A-Za-z%]", tok):  # unit-bearing -> take immediately
                return tok
            if fallback is None:
                fallback = tok
        return fallback

    def learn(self, items: List[Dict[str, Any]]) -> int:
        """Write-back: ingest distilled 'learned' context so future resolve()/query()
        calls benefit (self-learning).

        Each item is ``{text, category?, task_id?}``. Blocks are added to the
        unified index (vector + keyword) and appended to a durable JSONL in
        ``persist_dir`` so they survive the clear()+rebuild of a re-hydrate.
        Duplicate text (same content hash) is skipped. Returns the number of new
        blocks learned.

        Note: true cross-restart durability on ephemeral hosts still depends on
        the Phase-1 shared store (or the extension writing back to Drive); this
        persists to the same ``persist_dir`` as the rest of the index.
        """
        seen = self._learned_hashes()
        records = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            records.append(
                {
                    "hash": h,
                    "text": text,
                    "category": item.get("category"),
                    "task_id": item.get("task_id"),
                    "scope": item.get("scope"),
                }
            )

        if not records:
            return 0

        docs = [self._learned_doc(r) for r in records]
        scopes = [r.get("scope") for r in records]
        self._index_documents(docs, scopes)
        self.documents.extend(docs)
        self._persist_learned(records)
        return len(records)

    def _learned_doc(self, record: Dict[str, Any]) -> ContextDocument:
        name = record.get("task_id") or record["hash"][:8]
        return ContextDocument(
            content=record["text"],
            source="write-back",
            filename=f"learned/{name}.txt",
            doc_type=".txt",
            folder="machine_generated",
            chunk_index=0,
        )

    def _index_documents(
        self, docs: List[ContextDocument], scopes: Optional[List[Optional[str]]] = None
    ) -> None:
        """Add docs to the vector store (best-effort) and rebuild the keyword index.

        ``scopes`` (parallel to ``docs``) tags each doc's keyword metadata with a
        scope label used by resolve()'s hierarchical scope preference.
        """
        if not docs:
            return
        if scopes is None:
            scopes = [None] * len(docs)
        # Vector add can fail without a usable embeddings backend; the keyword
        # (BM25) path is the deterministic one resolve() uses first, so guard it.
        try:
            self.vector_store.add_documents(docs)
        except Exception as e:
            print(f"Vector add failed for learned docs: {e}")

        texts = list(self.keyword_searcher.documents) + [d.content for d in docs]
        metas = list(self.keyword_searcher.metadatas) + [
            {
                "source": d.source,
                "filename": d.filename,
                "folder": d.folder,
                "is_preference": d.folder == "machine_generated",
                "scope": scopes[i],
            }
            for i, d in enumerate(docs)
        ]
        self.keyword_searcher.index_documents(texts, metas)

    def _learned_hashes(self) -> set:
        if not self._learned_path or not os.path.exists(self._learned_path):
            return set()
        hashes = set()
        try:
            with open(self._learned_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("hash"):
                        hashes.add(rec["hash"])
        except Exception:
            return hashes
        return hashes

    def _persist_learned(self, records: List[Dict[str, Any]]) -> None:
        if not self._learned_path:
            return
        try:
            with open(self._learned_path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"Failed to persist learned context: {e}")

    def _reload_learned(self) -> None:
        """Re-ingest persisted learned blocks into the (freshly rebuilt) index."""
        if not self._learned_path or not os.path.exists(self._learned_path):
            return
        docs, scopes = [], []
        with open(self._learned_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (rec.get("text") or "").strip():
                    docs.append(self._learned_doc(rec))
                    scopes.append(rec.get("scope"))
        if docs:
            self._index_documents(docs, scopes)
            self.documents.extend(docs)

    def _build_task_identifier_context(
        self,
        base_context: str,
        user_prefs: List[FolderSummary],
        task_patterns: List[FolderSummary]
    ) -> str:
        """Build context for Task Identifier component."""
        parts = [base_context] if base_context else []

        if user_prefs:
            parts.append("\n## User Preferences\n")
            for pref in user_prefs:
                parts.append(f"From {pref.filename}:")
                parts.extend([f"- {kp}" for kp in pref.key_points[:5]])

        if task_patterns:
            parts.append("\n## Historical Task Patterns\n")
            for pattern in task_patterns:
                parts.append(f"From {pattern.filename}:")
                parts.extend([f"- {kp}" for kp in pattern.key_points[:5]])

        return "\n".join(parts)

    def _build_workflow_builder_context(
        self,
        base_context: str,
        workflow_trends: List[FolderSummary],
        user_prefs: List[FolderSummary]
    ) -> str:
        """Build context for Workflow Builder component."""
        parts = [base_context] if base_context else []

        if workflow_trends:
            parts.append("\n## Workflow Trends & Patterns\n")
            for trend in workflow_trends:
                parts.append(f"From {trend.filename}:")
                parts.extend([f"- {kp}" for kp in trend.key_points[:5]])

        if user_prefs:
            parts.append("\n## User Style Preferences\n")
            for pref in user_prefs:
                parts.append(f"From {pref.filename}:")
                parts.extend([f"- {kp}" for kp in pref.key_points[:3]])

        return "\n".join(parts)

    def _extract_query_result(self, result: Any) -> Dict[str, Any]:
        """Extract QueryResult from agent output."""
        if isinstance(result, dict):
            structured = result.get("structured_response") or result.get("output")
            if structured:
                if isinstance(structured, QueryResult):
                    return {
                        "answer": structured.answer,
                        "sources_cited": structured.sources_cited,
                        "context_for_extension": structured.context_for_extension,
                        "context_for_task_identifier": structured.context_for_task_identifier,
                        "context_for_workflow_builder": structured.context_for_workflow_builder
                    }
                return structured
            return result

        if hasattr(result, "structured_response"):
            sr = result.structured_response
            if isinstance(sr, QueryResult):
                return {
                    "answer": sr.answer,
                    "sources_cited": sr.sources_cited,
                    "context_for_extension": sr.context_for_extension,
                    "context_for_task_identifier": sr.context_for_task_identifier,
                    "context_for_workflow_builder": sr.context_for_workflow_builder
                }
            return sr

        if hasattr(result, "output"):
            return {"answer": str(result.output)}

        return {"answer": str(result) if result else "No response generated"}

    # Context Injection Methods (for system diagram integration)

    def get_context_for_extension(self, query: str) -> str:
        """Get additional context for Extension component."""
        result = self.query(query)

        parts = []
        if result.context_for_extension:
            parts.append(result.context_for_extension)

        if result.user_preferences:
            parts.append("\n## User Preferences\n")
            parts.extend([f"- {p}" for p in result.user_preferences[:3]])

        return "\n".join(parts) if parts else result.answer

    def get_context_for_task_identifier(self, task_description: str) -> str:
        """Get additional context for Task Identifier component."""
        task_patterns = self.preference_analyzer.get_relevant_preferences(
            task_description, category="task_patterns", top_k=3
        )
        user_prefs = self.preference_analyzer.get_relevant_preferences(
            task_description, category="user_preferences", top_k=2
        )

        parts = [f"Context for task identification: {task_description}"]

        if task_patterns:
            parts.append("\n## Similar Historical Task Patterns")
            for pattern in task_patterns:
                parts.append(f"\nFrom {pattern.filename}:")
                parts.extend([f"- {kp}" for kp in pattern.key_points[:5]])

        if user_prefs:
            parts.append("\n## User's Task Handling Preferences")
            for pref in user_prefs:
                parts.append(f"\nFrom {pref.filename}:")
                parts.extend([f"- {kp}" for kp in pref.key_points[:3]])

        query = f"Find similar tasks and patterns related to: {task_description}"
        result = self.query(query)

        if result.context_for_task_identifier:
            parts.append("\n## Relevant Document Context")
            parts.append(result.context_for_task_identifier)

        return "\n".join(parts)

    def get_context_for_workflow_builder(self, task_description: str) -> str:
        """Get additional context for Workflow Builder component."""
        workflow_trends = self.preference_analyzer.get_relevant_preferences(
            task_description, category="workflow_trends", top_k=3
        )
        user_prefs = self.preference_analyzer.get_relevant_preferences(
            task_description, category="user_preferences", top_k=2
        )

        parts = [f"Context for workflow building: {task_description}"]

        if workflow_trends:
            parts.append("\n## Workflow Patterns & Trends")
            for trend in workflow_trends:
                parts.append(f"\nFrom {trend.filename}:")
                parts.extend([f"- {kp}" for kp in trend.key_points[:5]])

        if user_prefs:
            parts.append("\n## User's Workflow Style Preferences")
            for pref in user_prefs:
                parts.append(f"\nFrom {pref.filename}:")
                parts.extend([f"- {kp}" for kp in pref.key_points[:3]])

        query = f"Find relevant workflow patterns for: {task_description}"
        result = self.query(query)

        if result.context_for_workflow_builder:
            parts.append("\n## Relevant Document Context")
            parts.append(result.context_for_workflow_builder)

        return "\n".join(parts)

    def get_direct_preferences(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Get raw preferences by category."""
        if category:
            summaries = self.preference_analyzer.category_index.get(category, [])
            return {
                category: [item for s in summaries for item in s.key_points]
            }

        return {
            cat: [item for s in summaries for item in s.key_points]
            for cat, summaries in self.preference_analyzer.category_index.items()
        }

    # Utility Methods

    def get_stats(self) -> Dict[str, Any]:
        """Get memory unit statistics."""
        return {
            "is_hydrated": self.is_hydrated,
            "total_documents": len(self.documents),
            "vector_store_count": self.vector_store.count() if self.vector_store else 0,
            "keyword_index_size": len(self.keyword_searcher.documents) if self.keyword_searcher else 0
        }

    def clear(self) -> None:
        """Clear all indexed data."""
        if self.vector_store:
            self.vector_store.clear()
        if self.keyword_searcher:
            self.keyword_searcher = BM25Searcher()
        self.documents = []
        self.is_hydrated = False


def create_memory_unit(
    persist_dir: Optional[str] = None,
    model_name: str = "gpt-4o"
) -> MemoryUnit:
    """
    Factory function to create a MemoryUnit ready for hydration.

    The auth token is NOT provided here - it comes from the extension
    when hydrate_from_drive() is called.

    Args:
        persist_dir: Directory for Chroma persistence
        model_name: OpenAI model for the agent

    Returns:
        MemoryUnit ready to be hydrated with ephemeral token
    """
    return MemoryUnit(
        persist_dir=persist_dir,
        model_name=model_name
    )
