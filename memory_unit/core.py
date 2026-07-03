"""
Core MemoryUnit implementation.
"""

from typing import List, Dict, Any, Optional

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
        scope: Optional[str] = None,
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
            scope: reserved for hierarchical scope (global|org|role|user|thread).
            min_score: minimum BM25 score before a field counts as resolved.

        Returns:
            One dict per input field: ``{field, value, source, confidence, status}``.
            Unresolved fields come back with ``value=None`` and ``status="missing"``.
        """
        resolved: List[Dict[str, Any]] = []
        for field in fields:
            item: Dict[str, Any] = {
                "field": field,
                "value": None,
                "source": None,
                "confidence": 0.0,
                "status": "missing",
            }

            if self.is_hydrated and field and field.strip():
                best = self._best_evidence_for(field, min_score=min_score)
                if best is not None:
                    item.update(best)
                    item["status"] = "present"

            resolved.append(item)

        return resolved

    def _best_evidence_for(
        self, field: str, min_score: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """Best supporting snippet for a slot name, or None.

        Keyword/BM25 first (deterministic, no embeddings); vector search is a
        best-effort fallback when the keyword index yields nothing.
        """
        hits = self.keyword_searcher.search(field, top_k=1)
        if hits and float(hits[0].get("score", 0.0)) > min_score:
            score = float(hits[0]["score"])
            return {
                "value": hits[0]["content"].strip()[:500],
                "source": "context",
                # Map an unbounded BM25 score monotonically into (0, 1).
                "confidence": round(score / (score + 1.0), 3),
            }

        # Vector fallback — embeddings may be unavailable offline, so guard it.
        try:
            results = self.vector_store.query(field, n_results=1)
            docs = results.get("documents") or [[]]
            dists = results.get("distances") or [[]]
            if docs and docs[0]:
                distance = float(dists[0][0]) if dists and dists[0] else 1.0
                # Chroma distances are >= 0 (smaller = closer). Convert to (0, 1].
                return {
                    "value": docs[0][0].strip()[:500],
                    "source": "context",
                    "confidence": round(1.0 / (1.0 + distance), 3),
                }
        except Exception:
            pass

        return None

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
