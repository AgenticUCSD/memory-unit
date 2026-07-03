"""
FastAPI endpoints for the Memory Unit.

Provides HTTP interface for:
- Hydrating memory from Google Drive
- Querying for context
- Context injection endpoints for Extension, Task Identifier, Workflow Builder

Auth token flow (from Chrome extension):
- Extension gets token via chrome.identity.getAuthToken()
- Sends in Authorization: Bearer <token> header
- Also sends X-User-Id and X-Thread-Id headers
"""

from typing import Optional, Dict, Any, List
import os
import logging

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load OPENAI_API_KEY / CONFIDENT_API_KEY / PORT / HOST from .env (memory-unit had no
# config module that did this, so the .env file was previously inert).
load_dotenv()

from memory_unit import MemoryUnit, ContextQueryResult, DriveFolderConfig
from memory_unit.auth import verify_google_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Memory Unit API",
    description="Memory unit with Google Drive integration and agentic RAG",
    version="1.0.0"
)

# CORS middleware. Origins come from the ALLOWED_ORIGINS env var (comma-separated);
# we no longer ship `allow_origins=["*"]` — that is both a tenancy hole and an invalid
# combination with allow_credentials=True (browsers reject a credentialed wildcard).
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory unit instance
_memory_unit: Optional[MemoryUnit] = None

# The user_id (Google `sub`) that hydrated the current memory. Until per-user data
# isolation lands, the unit holds one user's documents at a time, so we refuse to serve
# a *different* user from the hydrating user's data. Set on /hydrate.
_owner_user_id: Optional[str] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class HydrateRequest(BaseModel):
    root_folder_id: str = Field(..., description="Root Drive folder ID with 2 subfolders")
    persist_dir: Optional[str] = Field(None, description="Chroma persistence directory")
    model_name: str = Field("gpt-4o", description="OpenAI model for agent")


class HydrateResponse(BaseModel):
    status: str
    documents_indexed: int
    folder_structure: Dict[str, Any]
    stats: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text to retrieve context")
    n_results: int = Field(5, description="Number of results to retrieve")


class ContextResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_for_extension: str
    context_for_task_identifier: str
    context_for_workflow_builder: str

    # Machine-generated preference data
    user_preferences: List[str] = []
    task_patterns: List[str] = []
    workflow_trends: List[str] = []


class ResolveRequest(BaseModel):
    fields: List[str] = Field(..., description="Slot/parameter names to resolve to values")
    scope: Optional[str] = Field(None, description="Reserved: global|org|role|user|thread")
    min_score: float = Field(0.0, description="Minimum BM25 score before a field counts as resolved")


class ResolvedSlot(BaseModel):
    field: str
    value: Optional[str] = None
    evidence: Optional[str] = None  # the snippet `value` was extracted from
    source: Optional[str] = None
    confidence: float = 0.0
    status: str = "missing"


class ResolveResponse(BaseModel):
    slots: List[ResolvedSlot]


class LearnItem(BaseModel):
    text: str = Field(..., description="Distilled fact to remember (write-back)")
    category: Optional[str] = Field(
        None, description="user_preferences|task_patterns|workflow_trends"
    )
    task_id: Optional[str] = Field(None, description="Originating task, for provenance")


class LearnRequest(BaseModel):
    items: List[LearnItem]


class LearnResponse(BaseModel):
    learned: int


class ExtensionContextRequest(BaseModel):
    query: str


class TaskIdentifierContextRequest(BaseModel):
    task_description: str


class WorkflowBuilderContextRequest(BaseModel):
    task_description: str


class PreferencesResponse(BaseModel):
    user_preferences: List[str]
    task_patterns: List[str]
    workflow_trends: List[str]


class StatsResponse(BaseModel):
    is_hydrated: bool
    total_documents: int
    vector_store_count: int
    keyword_index_size: int


class HealthResponse(BaseModel):
    status: str
    memory_unit_initialized: bool
    documents_indexed: int


# =============================================================================
# Dependencies
# =============================================================================

def get_memory_unit() -> MemoryUnit:
    """Dependency to get initialized memory unit."""
    if _memory_unit is None:
        raise HTTPException(status_code=503, detail="Memory unit not initialized. Call /hydrate first.")
    return _memory_unit


def require_owner(x_user_id: Optional[str] = Header(None)) -> str:
    """Tenancy guard: every data endpoint must carry X-User-Id, and (once the unit has
    been hydrated) that user must match the user who hydrated it. Prevents serving one
    user's memory to another while the store is still single-tenant."""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    if _owner_user_id is not None and x_user_id != _owner_user_id:
        raise HTTPException(
            status_code=403,
            detail="This memory unit was hydrated for a different user.",
        )
    return x_user_id


# =============================================================================
# Health & Status
# =============================================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    global _memory_unit
    return HealthResponse(
        status="healthy",
        memory_unit_initialized=_memory_unit is not None,
        documents_indexed=_memory_unit.vector_store.count() if _memory_unit and _memory_unit.vector_store else 0
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats(memory: MemoryUnit = Depends(get_memory_unit), _: str = Depends(require_owner)):
    """Get memory unit statistics."""
    return StatsResponse(**memory.get_stats())


# =============================================================================
# Hydration
# =============================================================================

def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract token from Authorization: Bearer <token> header."""
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


@app.post("/hydrate", response_model=HydrateResponse)
def hydrate_memory(
    request: HydrateRequest,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None)
):
    """
    Hydrate the memory unit from Google Drive.

    Auth token is passed in Authorization: Bearer <token> header (from extension).
    This fetches documents from the 2 subfolders and indexes them
    in the vector store + keyword search database.
    """
    global _memory_unit, _owner_user_id

    auth_token = extract_bearer_token(authorization)
    if not auth_token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header required")

    # Verify the Google token (and that its `sub` matches X-User-Id) before we
    # trust it to read Drive and bind this unit's owner. 401 on a bad token,
    # 503 if Google is unreachable.
    verify_google_token(auth_token, x_user_id)

    try:
        logger.info(f"Initializing memory unit with folder: {request.root_folder_id}")

        # Create or re-initialize memory unit. The ephemeral auth token is NOT a
        # constructor argument — it is supplied to hydrate_from_drive() at call time.
        _memory_unit = MemoryUnit(
            persist_dir=request.persist_dir,
            model_name=request.model_name
        )

        _memory_unit.folder_config = DriveFolderConfig(
            root_folder_id=request.root_folder_id,
            user_provided_folder_id="",
            machine_generated_folder_id=""
        )

        # Hydrate from Drive (root folder id + ephemeral token).
        result = _memory_unit.hydrate_from_drive(request.root_folder_id, auth_token)

        # Claim ownership only after a successful hydrate, so a failed hydrate doesn't
        # lock the unit to a user with no data.
        _owner_user_id = x_user_id

        logger.info(f"Hydrated {result['documents_indexed']} documents")

        return HydrateResponse(**result)

    except Exception as e:
        logger.error(f"Hydration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/query", response_model=ContextResponse)
def query_memory(request: QueryRequest, memory: MemoryUnit = Depends(get_memory_unit), _: str = Depends(require_owner)):
    """
    Query the memory unit using agentic RAG.

    Combines hybrid search over diverse user documents with targeted
    retrieval from machine-generated preference/trend files.
    """
    try:
        result = memory.query(request.query)
        return ContextResponse(
            answer=result.answer,
            sources=result.sources,
            context_for_extension=result.context_for_extension,
            context_for_task_identifier=result.context_for_task_identifier,
            context_for_workflow_builder=result.context_for_workflow_builder,
            user_preferences=result.user_preferences,
            task_patterns=result.task_patterns,
            workflow_trends=result.workflow_trends
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resolve", response_model=ResolveResponse)
def resolve_slots(
    request: ResolveRequest,
    x_user_id: str = Depends(require_owner),
    memory: MemoryUnit = Depends(get_memory_unit),
):
    """Resolve task parameter slots to concrete values (structured field->value).

    This is the parameter-resolution surface the planner calls to pre-fill task
    slots from user context before falling back to HITL. Unlike /query it returns
    typed slots with source + confidence, not prose. Unresolved fields come back
    with status="missing" so the caller knows to ask the human.
    """
    try:
        results = memory.resolve(
            request.fields,
            user_id=x_user_id,
            scope=request.scope,
            min_score=request.min_score,
        )
        return ResolveResponse(slots=[ResolvedSlot(**r) for r in results])
    except Exception as e:
        logger.error(f"Resolve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn", response_model=LearnResponse)
def learn_context(
    request: LearnRequest,
    x_user_id: str = Depends(require_owner),
    memory: MemoryUnit = Depends(get_memory_unit),
):
    """Write-back: ingest distilled context learned from completed tasks so future
    resolve()/query() calls benefit. In-repo self-learning; durable Drive
    persistence is a follow-up (extension-owned)."""
    try:
        count = memory.learn([item.model_dump() for item in request.items])
        return LearnResponse(learned=count)
    except Exception as e:
        logger.error(f"Learn failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector-search")
def vector_search(
    query: str,
    n_results: int = 5,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """Direct vector search (semantic similarity)."""
    try:
        results = memory.vector_store.query(query, n_results=n_results)
        return {
            "query": query,
            "results": [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/keyword-search")
def keyword_search(
    query: str,
    top_k: int = 5,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """Direct keyword search (BM25)."""
    try:
        results = memory.keyword_searcher.search(query, top_k=top_k)
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Context Injection Endpoints (for System Diagram integration)
# =============================================================================

@app.post("/context/extension")
def get_extension_context(
    request: ExtensionContextRequest,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """
    Get additional context for Extension component.

    From system diagram:
    [Memory Unit] --> [Additional ctxt for Task/Workflow] --> [Extension]
    """
    try:
        context = memory.get_context_for_extension(request.query)
        return {
            "context": context,
            "target": "extension"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/task-identifier")
def get_task_identifier_context(
    request: TaskIdentifierContextRequest,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """
    Get additional context for Task Identifier component.

    From system diagram:
    [Memory Unit] --> [Additional ctxt for Task/Workflow] --> [Task Identifier]
    """
    try:
        context = memory.get_context_for_task_identifier(request.task_description)
        return {
            "context": context,
            "target": "task_identifier"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/workflow-builder")
def get_workflow_builder_context(
    request: WorkflowBuilderContextRequest,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """
    Get additional context for Workflow Builder component.

    From system diagram:
    [Memory Unit] --> [Additional ctxt for Workflow] --> [Workflow Builder]
    """
    try:
        context = memory.get_context_for_workflow_builder(request.task_description)
        return {
            "context": context,
            "target": "workflow_builder"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preferences", response_model=PreferencesResponse)
def get_preferences(
    category: Optional[str] = None,
    memory: MemoryUnit = Depends(get_memory_unit),
    _: str = Depends(require_owner)
):
    """
    Get raw machine-generated preferences by category.

    Categories:
    - user_preferences: User's style, habits, likes/dislikes
    - task_patterns: Common task types, frequencies, patterns
    - workflow_trends: Successful workflows, optimization opportunities
    """
    try:
        prefs = memory.get_direct_preferences(category)
        return PreferencesResponse(
            user_preferences=prefs.get("user_preferences", []),
            task_patterns=prefs.get("task_patterns", []),
            workflow_trends=prefs.get("workflow_trends", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Management
# =============================================================================

@app.post("/clear")
def clear_memory(memory: MemoryUnit = Depends(get_memory_unit), _: str = Depends(require_owner)):
    """Clear all indexed documents."""
    try:
        memory.clear()
        return {"status": "cleared", "message": "All documents removed from memory"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh")
def refresh_memory(
    authorization: Optional[str] = Header(None),
    memory: MemoryUnit = Depends(get_memory_unit),
    x_user_id: str = Depends(require_owner)
):
    """Refresh memory by re-hydrating from Drive."""
    try:
        supplied_token = extract_bearer_token(authorization)

        if supplied_token:
            # A freshly-supplied token must be verified (and belong to the owner).
            verify_google_token(supplied_token, x_user_id)
            auth_token = supplied_token
        else:
            # Fall back to the token already verified at hydrate time.
            auth_token = memory.auth_token

        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")

        # Resolve the root folder id captured at the last hydrate. Without it there
        # is nothing to refresh — the caller must hydrate first.
        root_folder_id = getattr(memory, "root_folder_id", None)
        if not root_folder_id and memory.folder_config:
            root_folder_id = memory.folder_config.root_folder_id
        if not root_folder_id:
            raise HTTPException(
                status_code=400,
                detail="Nothing to refresh; call /hydrate first."
            )

        # Re-hydrate (clears old data internally) with the root folder + token.
        result = memory.hydrate_from_drive(root_folder_id, auth_token)

        # Spread result first so its "status": "success" does not clobber "refreshed".
        return {
            **result,
            "status": "refreshed"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
