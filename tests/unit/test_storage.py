"""
Tests for storage backends (vector store and BM25 search).
"""

import pytest
from unittest.mock import Mock, patch

from memory_unit.models.documents import ContextDocument
from memory_unit.storage.vector_store import VectorStore
from memory_unit.storage.bm25_search import BM25Searcher


class TestBM25Searcher:
    """Tests for BM25 keyword search."""

    def test_basic_search(self):
        searcher = BM25Searcher()

        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A brown dog sleeps in the sun",
            "The fox runs quickly through the forest"
        ]

        searcher.index_documents(documents)
        results = searcher.search("brown fox", top_k=2)

        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_search_ranking(self):
        searcher = BM25Searcher()

        documents = [
            "Python programming language is great for data science",
            "JavaScript is used for web development",
            "Python is also used for machine learning and AI"
        ]

        searcher.index_documents(documents)
        results = searcher.search("python machine learning", top_k=2)

        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]
        assert "python" in results[0]["content"].lower()

    def test_search_no_matches(self):
        searcher = BM25Searcher()
        searcher.index_documents(["hello world"])
        assert searcher.search("xyz abc") == []

    def test_empty_index(self):
        searcher = BM25Searcher()
        assert searcher.search("test") == []

    def test_metadata_preservation(self):
        searcher = BM25Searcher()

        documents = ["Python code example"]
        metadatas = [{"filename": "test.py", "folder": "src"}]

        searcher.index_documents(documents, metadatas)
        results = searcher.search("python")

        assert len(results) == 1
        assert results[0]["metadata"]["filename"] == "test.py"

    def test_tokenization(self):
        searcher = BM25Searcher()
        tokens = searcher.tokenize("Hello, World! Test 123.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" not in tokens


class TestVectorStore:
    """Tests for ChromaDB vector store with mocked Chroma client."""

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_add_and_count_documents(self, mock_chroma):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        store = VectorStore(persist_dir="/tmp/test", collection_name="test")
        store.collection = mock_collection
        store.client = mock_client

        docs = [
            ContextDocument(content="doc1", source="s1", filename="f1", doc_id="d1"),
            ContextDocument(content="doc2", source="s2", filename="f2", doc_id="d2")
        ]

        store.add_documents(docs)
        mock_collection.add.assert_called()

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_query_documents(self, mock_chroma):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        store = VectorStore(persist_dir="/tmp/test", collection_name="test")
        store.collection = mock_collection
        store.client = mock_client

        mock_collection.query.return_value = {
            'documents': [['doc1 text', 'doc2 text']],
            'metadatas': [[{'filename': 'f1.txt'}, {'filename': 'f2.txt'}]],
            'distances': [[0.1, 0.2]]
        }

        results = store.query("test query", n_results=2)

        assert "documents" in results
        mock_collection.query.assert_called_once()

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_clear_collection(self, mock_chroma):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        store = VectorStore(persist_dir="/tmp/test", collection_name="test")
        store.collection = mock_collection
        store.client = mock_client

        store.clear()

        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called()

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_count_documents(self, mock_chroma):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        store = VectorStore(persist_dir="/tmp/test", collection_name="test")
        store.collection = mock_collection
        store.client = mock_client

        mock_collection.count.return_value = 5
        assert store.count() == 5
