"""
ChromaDB wrapper for document storage and retrieval.
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from memory_unit.models.documents import ContextDocument


class VectorStore:
    """ChromaDB wrapper for document storage and retrieval."""

    def __init__(
        self,
        collection_name: str = "memory_documents",
        persist_dir: Optional[str] = None
    ):
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_data")

        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        # Use OpenAI embeddings if key available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, documents: List[ContextDocument]) -> None:
        """Add context documents to the collection."""
        if not documents:
            return

        texts = [doc.content for doc in documents]
        metadatas = [
            {
                "source": doc.source,
                "filename": doc.filename,
                "folder": doc.folder,
                "doc_type": doc.doc_type,
                "chunk_index": doc.chunk_index
            }
            for doc in documents
        ]
        ids = [doc.doc_id for doc in documents]

        # Batch add
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the collection for similar documents."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_dict
        )

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn
        )
