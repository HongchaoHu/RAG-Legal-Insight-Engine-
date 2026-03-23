from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from langchain_core.documents import Document

from summarizer import LegalDocumentStore

logger = logging.getLogger(__name__)


class LegalVectorStore:
    """Stores and queries legal summaries in a FAISS index."""

    def __init__(
        self,
        index_path: str | Path = "index.faiss",
        mapping_path: str | Path = "index_mapping.json",
        document_store: LegalDocumentStore | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.document_store = document_store or LegalDocumentStore()
        self.index: faiss.Index | None = None
        self.mapping: dict[str, dict[str, Any]] = {}

    def create_vectorstore(self, docs: Iterable[Document]) -> tuple[faiss.Index, dict[str, dict[str, Any]]]:
        docs_list = list(docs)
        if not docs_list:
            raise ValueError("No documents provided to create_vectorstore")

        vectors: list[list[float]] = []
        mapping: dict[str, dict[str, Any]] = {}

        for idx, doc in enumerate(docs_list):
            summary = self.document_store.generate_summary(doc)
            embedding = self.document_store.embeddings.embed_documents([summary])[0]
            vectors.append(embedding)
            mapping[str(idx)] = {
                "summary": summary,
                "source_path": doc.metadata.get("source_path") or doc.metadata.get("source", "unknown"),
                "metadata": dict(doc.metadata),
            }

        vectors_array = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(vectors_array.shape[1])
        index.add(vectors_array)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        self.index = index
        self.mapping = mapping
        logger.info("Built and persisted FAISS index with %s vectors", len(vectors))
        return index, mapping

    def load_index(self) -> tuple[faiss.Index, dict[str, dict[str, Any]]]:
        if not self.index_path.exists() or not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Missing index artifacts: index={self.index_path.exists()}, mapping={self.mapping_path.exists()}"
            )

        index = faiss.read_index(str(self.index_path))
        mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        self.index = index
        self.mapping = mapping
        return index, mapping

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if self.index is None or not self.mapping:
            self.load_index()

        assert self.index is not None
        query_vector = self.document_store.embeddings.embed_query(question)
        query_array = np.asarray([query_vector], dtype="float32")
        distances, indices = self.index.search(query_array, top_k)

        results: list[dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            mapped = self.mapping.get(str(int(idx)), {})
            relevance_score = float(1.0 / (1.0 + max(float(distance), 0.0)))
            results.append(
                {
                    "summary": mapped.get("summary", ""),
                    "source_path": mapped.get("source_path", "unknown"),
                    "relevance_score": round(relevance_score, 4),
                    "metadata": mapped.get("metadata", {}),
                }
            )

        return results
