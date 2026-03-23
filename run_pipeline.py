from __future__ import annotations

import argparse
import logging
from pathlib import Path

from document_loader import load_documents
from preprocessor import LegalTextPreprocessor
from vector_store import LegalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(data_dir: str | Path, artifacts_dir: str | Path = "artifacts") -> None:
    documents = load_documents(data_dir)
    if not documents:
        logger.warning("No documents loaded. Exiting pipeline.")
        return

    preprocessor = LegalTextPreprocessor()
    cleaned_documents = [preprocessor.clean_document(doc) for doc in documents]

    artifacts_path = Path(artifacts_dir)
    vector_store = LegalVectorStore(
        index_path=artifacts_path / "index.faiss",
        mapping_path=artifacts_path / "index_mapping.json",
    )
    vector_store.create_vectorstore(cleaned_documents)
    logger.info("Pipeline complete: %s documents indexed", len(cleaned_documents))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RAG artifacts from legal .txt files")
    parser.add_argument("--data-dir", required=True, help="Root folder containing legal .txt files")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Folder where index.faiss and index_mapping.json are stored",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(args.data_dir, args.artifacts_dir)
