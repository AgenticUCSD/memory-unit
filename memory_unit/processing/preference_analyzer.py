"""
Preference analyzer for machine-generated context files.
"""

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

from memory_unit.models.documents import FolderSummary


class PreferenceAnalyzer:
    """
    Specialized analyzer for machine-generated preference/trend files.

    These files contain natural language summaries of:
    - User preferences (communication style, scheduling habits, etc.)
    - Task patterns (common task types, workflows that worked well)
    - Workflow trends (successful patterns, failed approaches)

    Unlike diverse user documents, these are structured text files that
    benefit from categorical extraction and targeted retrieval.
    """

    # Categories for organizing preference content
    CATEGORIES = {
        "user_preferences": [
            "communication_style", "meeting_preferences", "deadline_handling",
            "notification_preferences", "delegation_style", "detail_level"
        ],
        "task_patterns": [
            "common_task_types", "task_frequency", "task_complexity",
            "preferred_tools", "automation_candidates"
        ],
        "workflow_trends": [
            "successful_workflows", "failed_approaches", "optimization_opportunities",
            "common_steps", "preferred_tools"
        ]
    }

    def __init__(self, llm_model: Optional[str] = None):
        self.model_name = llm_model or "gpt-4o-mini"
        self.summaries: List[FolderSummary] = []
        self.category_index: Dict[str, List[FolderSummary]] = defaultdict(list)

    def analyze_preference_file(
        self,
        filename: str,
        content: str,
        last_modified: str
    ) -> FolderSummary:
        """
        Analyze a preference/trend file and extract structured summary.

        Uses LLM to categorize and extract key points from natural language summaries.
        """
        # Simple heuristic categorization based on filename
        category = self._categorize_by_filename(filename)

        # Extract key points (can use LLM for more sophisticated extraction)
        key_points = self._extract_key_points(content)

        summary = FolderSummary(
            filename=filename,
            category=category,
            key_points=key_points,
            last_updated=last_modified,
            raw_content=content[:5000]  # Keep first 5000 chars for reference
        )

        return summary

    def _categorize_by_filename(self, filename: str) -> str:
        """Categorize file based on its name."""
        name_lower = filename.lower()

        if any(word in name_lower for word in ["preference", "style", "habit", "like"]):
            return "user_preferences"
        elif any(word in name_lower for word in ["task", "work", "job", "activity"]):
            return "task_patterns"
        elif any(word in name_lower for word in ["workflow", "process", "automation", "flow"]):
            return "workflow_trends"
        else:
            return "general"

    def _extract_key_points(self, content: str) -> List[str]:
        """
        Extract key bullet points from preference text.

        Simple extraction - can be enhanced with LLM for complex parsing.
        """
        key_points = []

        # Look for bullet points
        bullet_pattern = r'^[\s]*[-•*][\s]+(.+)$'
        for line in content.split('\n'):
            match = re.match(bullet_pattern, line, re.MULTILINE)
            if match:
                point = match.group(1).strip()
                if len(point) > 10:  # Filter out very short bullets
                    key_points.append(point)

        # Also look for "Key:" or "Preference:" patterns
        key_colon_pattern = r'(?:^|\n)(?:Key|Preference|Pattern|Trend)s?:\s*(.+?)(?=\n|$)'
        matches = re.findall(key_colon_pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 10:
                key_points.append(match.strip())

        return key_points[:20]  # Limit to top 20 points

    def index_summaries(self, summaries: List[FolderSummary]) -> None:
        """Index summaries for quick category-based retrieval."""
        self.summaries = summaries
        self.category_index = defaultdict(list)

        for summary in summaries:
            self.category_index[summary.category].append(summary)

    def get_relevant_preferences(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3
    ) -> List[FolderSummary]:
        """
        Get relevant preference summaries for a query.

        Uses simple keyword matching - can be enhanced with embeddings.
        """
        query_terms = set(self._tokenize(query.lower()))

        candidates = []
        if category and category in self.category_index:
            candidates = self.category_index[category]
        else:
            candidates = self.summaries

        scored = []
        for summary in candidates:
            score = self._score_relevance(summary, query_terms)
            if score > 0:
                scored.append((score, summary))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:top_k]]

    def _score_relevance(self, summary: FolderSummary, query_terms: set) -> float:
        """Score how relevant a summary is to query terms."""
        score = 0.0

        # Check key points
        content = ' '.join(summary.key_points).lower()
        content_terms = set(self._tokenize(content))

        # Simple overlap scoring
        overlap = query_terms & content_terms
        if overlap:
            score += len(overlap) / len(query_terms)

        # Boost exact category matches
        for cat, keywords in self.CATEGORIES.items():
            if any(kw in ' '.join(summary.key_points).lower() for kw in keywords):
                score += 0.5

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def synthesize_preferences(
        self,
        category: str,
        context: Optional[str] = None
    ) -> str:
        """
        Synthesize a natural language summary of preferences in a category.

        Useful for injecting into prompts for Task Identifier and Workflow Builder.
        """
        summaries = self.category_index.get(category, [])

        if not summaries:
            return ""

        parts = [f"# {category.replace('_', ' ').title()}\n"]

        for summary in summaries:
            parts.append(f"## From {summary.filename}\n")
            for point in summary.key_points[:5]:  # Top 5 points per file
                parts.append(f"- {point}")
            parts.append("")

        if context:
            parts.append(f"\nRelevant to: {context}")

        return '\n'.join(parts)

    def get_all_preferences_text(self) -> str:
        """Get all preferences as a single text block for vector indexing."""
        parts = []
        for summary in self.summaries:
            parts.append(f"=== {summary.filename} ({summary.category}) ===")
            parts.append(summary.raw_content)
            parts.append("")
        return '\n'.join(parts)
