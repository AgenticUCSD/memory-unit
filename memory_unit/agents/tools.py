"""
Tools for the agentic RAG query agent.
"""

from typing import List

from langchain.tools import tool
from pydantic import BaseModel

from memory_unit.storage.vector_store import VectorStore
from memory_unit.storage.bm25_search import BM25Searcher


class QueryResult(BaseModel):
    """Result from the agentic memory query."""
    answer: str
    sources_cited: List[str] = []
    context_for_extension: str = ""
    context_for_task_identifier: str = ""
    context_for_workflow_builder: str = ""


class MemoryTools:
    """Tools for the agentic RAG query agent."""

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_searcher: BM25Searcher
    ):
        self.vector_store = vector_store
        self.keyword_searcher = keyword_searcher
        self._register_tools()

    def _register_tools(self):
        """Register tools for LangChain agent."""
        self.vector_search_tool = tool(self._vector_search)
        self.keyword_search_tool = tool(self._keyword_search)
        self.get_document_count_tool = tool(self._get_document_count)

    def _vector_search(self, query: str, n_results: int = 5) -> str:
        """
        Search documents using vector/semantic similarity.
        Use for: concepts, ideas, topics, semantic meaning.
        """
        results = self.vector_store.query(query, n_results=n_results)

        if not results or not results["documents"] or not results["documents"][0]:
            return "No results found."

        output = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            filename = metadata.get("filename", "unknown") if metadata else "unknown"
            folder = metadata.get("folder", "") if metadata else ""
            score = 1 - distance if distance else 0
            output.append(
                f"[Result {i+1}] Source: {filename} (folder: {folder}, similarity: {score:.3f})\n{doc}\n"
            )

        return "\n".join(output)

    def _keyword_search(self, query: str, top_k: int = 5) -> str:
        """
        Search documents using keyword/BM25 matching.
        Use for: specific words, names, exact phrases, identifiers.
        """
        if not self.keyword_searcher.documents:
            return "No documents indexed for keyword search."

        results = self.keyword_searcher.search(query, top_k=top_k)

        if not results:
            return "No keyword matches found."

        output = []
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            folder = metadata.get("folder", "")
            output.append(
                f"[Result {i+1}] Source: {filename} (folder: {folder}, score: {result['score']:.3f})\n{result['content']}\n"
            )

        return "\n".join(output)

    def _get_document_count(self) -> str:
        """Get the number of documents indexed."""
        count = self.vector_store.count()
        return f"There are {count} document chunks currently indexed."

    def get_tools(self):
        """Return list of tools for the agent."""
        return [
            self.vector_search_tool,
            self.keyword_search_tool,
            self.get_document_count_tool
        ]
