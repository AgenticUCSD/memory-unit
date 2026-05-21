"""
BM25-style keyword search for documents.
"""

import re
from collections import defaultdict
from typing import Dict, List, Any, Optional


class BM25Searcher:
    """BM25-style keyword search for documents."""

    def __init__(self):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: defaultdict = defaultdict(int)
        self.avg_doc_len: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Index documents for BM25 search."""
        self.documents = documents
        self.metadatas = metadatas or []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        total_len = 0

        for doc in documents:
            tokens = self.tokenize(doc)
            total_len += len(tokens)

            tf = defaultdict(int)
            seen_terms = set()
            for token in tokens:
                tf[token] += 1
                if token not in seen_terms:
                    seen_terms.add(token)
                    self.doc_freqs[token] += 1

            self.term_freqs.append(dict(tf))

        self.avg_doc_len = total_len / len(documents) if documents else 0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 search. Returns top k matching documents with scores."""
        if not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * len(self.documents)
        k1 = 1.5
        b = 0.75

        for doc_idx, tf in enumerate(self.term_freqs):
            doc_len = sum(tf.values())
            score = 0.0

            for term in query_tokens:
                if term not in tf:
                    continue

                df = self.doc_freqs.get(term, 0)
                idf = (len(self.documents) - df + 0.5) / (df + 0.5)
                idf = max(0.1, idf)

                freq = tf[term]
                tf_component = (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                )

                score += idf * tf_component

            scores[doc_idx] = score

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = {
                    "content": self.documents[idx],
                    "score": scores[idx],
                    "index": idx
                }
                if idx < len(self.metadatas):
                    result["metadata"] = self.metadatas[idx]
                results.append(result)

        return results
