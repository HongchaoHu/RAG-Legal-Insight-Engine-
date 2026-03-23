from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


class LegalDocumentStore:
    """Holds LLM and embedding models for legal document processing."""

    def __init__(self, llm: ChatOpenAI | None = None, embeddings: OpenAIEmbeddings | None = None) -> None:
        self._llm = llm
        self._embeddings = embeddings

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return self._llm

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return self._embeddings

    def generate_summary(self, doc: Document) -> str:
        """Generate a concise legal summary with a hard cap of 100 words."""
        prompt = (
            "You are a legal assistant. Summarize the following legal text in no more than "
            "100 words. Focus on obligations, rights, parties, penalties, and key dates. "
            "Return plain text only.\n\n"
            f"Text:\n{doc.page_content}"
        )

        try:
            response = self.llm.invoke(prompt)
            raw_summary = response.content if hasattr(response, "content") else str(response)
            normalized = " ".join(raw_summary.split())
            words = normalized.split(" ") if normalized else []
            return " ".join(words[:100]).strip()
        except Exception as exc:
            message = f"Summary generation failed for {doc.metadata.get('source_path', 'unknown')}: {exc}"
            logger.error(message)
            fallback_words = " ".join(doc.page_content.split()).split(" ")[:100]
            return " ".join(fallback_words).strip()
