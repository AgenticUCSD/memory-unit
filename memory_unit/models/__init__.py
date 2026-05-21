"""
Data models for documents and folder summaries.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ContextDocument:
    """A document from the memory store with content and metadata."""
    content: str
    source: str
    filename: str
    doc_type: str = "text"
    folder: str = ""  # Which subfolder it came from
    chunk_index: int = 0
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class FolderSummary:
    """Summary of machine-generated preference/trend files."""
    filename: str
    category: str  # e.g., "user_preferences", "task_patterns", "workflow_trends"
    key_points: List[str]
    last_updated: str
    raw_content: str
