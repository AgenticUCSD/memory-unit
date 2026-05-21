"""
Agentic RAG Memory Unit

A modular memory system that:
1. Connects to Google Drive using an auth token
2. Accesses folders for user-provided and machine-generated context
3. Hydrates a vector store (Chroma) + keyword search DB from Drive documents
4. Provides agentic RAG capabilities for context injection
"""

# Data Models
from memory_unit.models.documents import ContextDocument, FolderSummary
from memory_unit.models.query import ContextQueryResult, DriveFolderConfig

# Storage Components
from memory_unit.storage.vector_store import VectorStore
from memory_unit.storage.bm25_search import BM25Searcher

# Processing Components
from memory_unit.processing.document_processor import DocumentProcessor
from memory_unit.processing.preference_analyzer import PreferenceAnalyzer

# Drive Integration
from memory_unit.drive.client import GoogleDriveClient

# Main Memory Unit
from memory_unit.core import MemoryUnit, create_memory_unit

# Query Models
from memory_unit.agents.tools import MemoryTools, QueryResult

__all__ = [
    # Data Models
    "ContextDocument",
    "FolderSummary",
    "ContextQueryResult",
    "DriveFolderConfig",
    # Storage
    "VectorStore",
    "BM25Searcher",
    # Processing
    "DocumentProcessor",
    "PreferenceAnalyzer",
    # Drive
    "GoogleDriveClient",
    # Main
    "MemoryUnit",
    "create_memory_unit",
    # Agents
    "MemoryTools",
    "QueryResult",
]
