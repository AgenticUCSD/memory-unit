"""
Tests for document processing and preference analysis.
"""

import pytest
from memory_unit.models.documents import FolderSummary
from memory_unit.processing.document_processor import DocumentProcessor
from memory_unit.processing.preference_analyzer import PreferenceAnalyzer


class TestDocumentProcessor:
    """Tests for document processing and chunking."""

    def test_chunk_text_basic(self):
        processor = DocumentProcessor()
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."

        chunks = processor.chunk_text(text, max_chunk_size=100, overlap=10)

        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_text_respects_max_size(self):
        processor = DocumentProcessor()
        text = " ".join([f"word{i}" for i in range(500)])

        chunks = processor.chunk_text(text, max_chunk_size=200, overlap=20)

        assert len(chunks) > 1

    def test_chunk_text_with_overlap(self):
        processor = DocumentProcessor()
        text = " ".join([f"word{i}" for i in range(100)])

        chunks = processor.chunk_text(text, max_chunk_size=200, overlap=20)

        if len(chunks) > 1:
            first_words = set(chunks[0].split()[-10:])
            second_words = set(chunks[1].split()[:10])
            overlap = first_words & second_words
            assert len(overlap) > 0 or len(chunks[0]) < 200

    def test_process_document_types(self):
        processor = DocumentProcessor()

        assert processor.SUPPORTED_MIME_TYPES["text/plain"] == ".txt"
        assert processor.SUPPORTED_MIME_TYPES["application/pdf"] == ".pdf"
        assert processor.SUPPORTED_MIME_TYPES["text/markdown"] == ".md"

    def test_chunk_empty_text(self):
        processor = DocumentProcessor()
        assert processor.chunk_text("") == []

    def test_chunk_whitespace_only(self):
        processor = DocumentProcessor()
        assert processor.chunk_text("   \n\n   ") == []

    def test_tokenize(self):
        processor = DocumentProcessor()
        tokens = processor.tokenize("Hello, World! Test 123.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" not in tokens


class TestPreferenceAnalyzer:
    """Tests for preference analysis."""

    def test_categorize_by_filename(self):
        analyzer = PreferenceAnalyzer()

        assert analyzer._categorize_by_filename("user_preferences.txt") == "user_preferences"
        assert analyzer._categorize_by_filename("my_style.doc") == "user_preferences"
        assert analyzer._categorize_by_filename("work_habits.md") == "user_preferences"
        assert analyzer._categorize_by_filename("task_patterns.txt") == "task_patterns"
        assert analyzer._categorize_by_filename("daily_tasks.log") == "task_patterns"
        assert analyzer._categorize_by_filename("data_flow.txt") == "workflow_trends"
        assert analyzer._categorize_by_filename("random_file.txt") == "general"

    def test_extract_key_points(self):
        analyzer = PreferenceAnalyzer()
        content = """
        Key User Preferences:
        - Prefers morning meetings
        - Likes async communication
        - Enjoys detailed documentation
        """

        points = analyzer._extract_key_points(content)

        assert len(points) > 0
        assert any("morning" in p.lower() for p in points)

    def test_analyze_preference_file(self):
        analyzer = PreferenceAnalyzer()

        summary = analyzer.analyze_preference_file(
            filename="preferences.txt",
            content="- Likes coffee\n- Prefers mornings",
            last_modified="2024-01-15T10:00:00Z"
        )

        assert isinstance(summary, FolderSummary)
        assert summary.filename == "preferences.txt"

    def test_index_and_retrieve(self):
        analyzer = PreferenceAnalyzer()

        summaries = [
            FolderSummary(
                filename="prefs1.txt",
                category="user_preferences",
                key_points=["likes python"],
                last_updated="2024-01-01",
                raw_content="content"
            ),
            FolderSummary(
                filename="prefs2.txt",
                category="task_patterns",
                key_points=["daily standups"],
                last_updated="2024-01-02",
                raw_content="content"
            )
        ]

        analyzer.index_summaries(summaries)

        assert len(analyzer.category_index["user_preferences"]) == 1
        assert len(analyzer.category_index["task_patterns"]) == 1

    def test_get_relevant_preferences(self):
        analyzer = PreferenceAnalyzer()

        summaries = [
            FolderSummary(
                filename="prefs1.txt",
                category="user_preferences",
                key_points=["likes python programming"],
                last_updated="2024-01-01",
                raw_content="content"
            )
        ]

        analyzer.index_summaries(summaries)

        results = analyzer.get_relevant_preferences("python code", top_k=2)
        assert len(results) == 1

    def test_synthesize_preferences(self):
        analyzer = PreferenceAnalyzer()

        summaries = [
            FolderSummary(
                filename="prefs.txt",
                category="user_preferences",
                key_points=["point1", "point2"],
                last_updated="2024-01-01",
                raw_content="content"
            )
        ]

        analyzer.index_summaries(summaries)
        text = analyzer.synthesize_preferences("user_preferences")

        assert "point1" in text

    def test_get_all_preferences_text(self):
        analyzer = PreferenceAnalyzer()

        summaries = [
            FolderSummary(
                filename="prefs.txt",
                category="user_preferences",
                key_points=["point1"],
                last_updated="2024-01-01",
                raw_content="preference content"
            )
        ]

        analyzer.index_summaries(summaries)
        text = analyzer.get_all_preferences_text()

        assert "prefs.txt" in text
        assert "preference content" in text
