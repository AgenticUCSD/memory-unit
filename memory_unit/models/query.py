"""
Data models for query results and configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ContextQueryResult:
    """Result from querying the memory unit."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)

    # Context for different components
    context_for_extension: str = ""
    context_for_task_identifier: str = ""
    context_for_workflow_builder: str = ""

    # Specialized context from machine-generated folder
    user_preferences: List[str] = field(default_factory=list)
    task_patterns: List[str] = field(default_factory=list)
    workflow_trends: List[str] = field(default_factory=list)


@dataclass
class DriveFolderConfig:
    """Configuration for Drive folder structure."""
    root_folder_id: str
    user_provided_folder_id: str
    machine_generated_folder_id: str
