from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _format_creation_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def load_documents(root_directory: str | Path) -> List[Document]:
    """Load all .txt documents recursively and enrich metadata.

    Args:
        root_directory: Root directory containing legal text files.

    Returns:
        A list of LangChain Document objects with enriched metadata.
    """
    root_path = Path(root_directory)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    expected_files = sorted(root_path.rglob("*.txt"))
    logger.info("Discovered %s text files under %s", len(expected_files), root_path)

    loader = DirectoryLoader(
        str(root_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=True,
        silent_errors=True,
    )

    try:
        loaded_documents = loader.load()
    except Exception as exc:
        logger.exception("Document loading failed: %s", exc)
        return []

    file_to_metadata: dict[str, tuple[str, str]] = {}
    for file_path in expected_files:
        try:
            source_path = str(file_path.resolve())
            creation_date = _format_creation_time(file_path.stat().st_ctime)
            file_to_metadata[source_path] = (source_path, creation_date)
        except Exception as exc:
            logger.warning("Failed to read metadata for %s: %s", file_path, exc)

    enriched_documents: list[Document] = []
    for document in loaded_documents:
        source = document.metadata.get("source", "")
        resolved_source = str(Path(source).resolve()) if source else source
        source_path, creation_date = file_to_metadata.get(
            resolved_source, (resolved_source, datetime.now(timezone.utc).isoformat())
        )

        metadata = dict(document.metadata)
        metadata["source_path"] = source_path
        metadata["creation_date"] = creation_date
        enriched_documents.append(Document(page_content=document.page_content, metadata=metadata))

    loaded_count = len(enriched_documents)
    logger.info("Loaded %s/%s files successfully", loaded_count, len(expected_files))
    if loaded_count != len(expected_files):
        logger.warning(
            "Some files could not be loaded. expected=%s loaded=%s",
            len(expected_files),
            loaded_count,
        )

    return enriched_documents
