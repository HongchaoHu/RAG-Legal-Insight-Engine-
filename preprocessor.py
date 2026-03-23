from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class LegalTextPreprocessor:
    """Clean legal text while preserving document metadata."""

    collapse_whitespace_pattern: re.Pattern[str] = re.compile(r"\s+")
    control_chars_pattern: re.Pattern[str] = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")

    def clean_document(self, document: Document) -> Document:
        cleaned_text = document.page_content
        cleaned_text = cleaned_text.replace("“", '"').replace("”", '"')
        cleaned_text = cleaned_text.replace("‘", "'").replace("’", "'")
        cleaned_text = self.control_chars_pattern.sub(" ", cleaned_text)
        cleaned_text = self.collapse_whitespace_pattern.sub(" ", cleaned_text).strip()

        return Document(page_content=cleaned_text, metadata=dict(document.metadata))
