"""
Shared test fixtures and configuration.
"""

import os

# Never emit or flush real traces during tests. deepeval loads the repo `.env`
# (which may carry a CONFIDENT_API_KEY) on import, which would otherwise make
# tracing_enabled() true; force it off for the whole test session. setdefault
# keeps an explicit MEMORY_TRACING from the caller respected.
os.environ.setdefault("MEMORY_TRACING", "false")

import pytest
from unittest.mock import Mock
from memory_unit.models.documents import ContextDocument, FolderSummary


@pytest.fixture
def sample_documents():
    """Sample context documents for testing."""
    return [
        ContextDocument(
            content="Python is a versatile programming language.",
            source="https://example.com/doc1",
            filename="python_guide.txt",
            doc_type=".txt",
            folder="user_provided",
            chunk_index=0,
            doc_id="doc_001"
        ),
        ContextDocument(
            content="Machine learning requires large datasets.",
            source="https://example.com/doc2",
            filename="ml_basics.pdf",
            doc_type=".pdf",
            folder="user_provided",
            chunk_index=0,
            doc_id="doc_002"
        ),
        ContextDocument(
            content="User prefers morning meetings.",
            source="https://example.com/prefs",
            filename="user_preferences.txt",
            doc_type=".txt",
            folder="machine_generated",
            chunk_index=0,
            doc_id="doc_003"
        ),
    ]


@pytest.fixture
def sample_preferences():
    """Sample preference summaries."""
    return [
        FolderSummary(
            filename="prefs.txt",
            category="user_preferences",
            key_points=["prefers async", "likes detail"],
            last_updated="2024-01-01",
            raw_content="User preferences"
        ),
        FolderSummary(
            filename="tasks.txt",
            category="task_patterns",
            key_points=["daily standups", "weekly reviews"],
            last_updated="2024-01-02",
            raw_content="Task patterns"
        )
    ]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client fixture."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client, mock_collection
