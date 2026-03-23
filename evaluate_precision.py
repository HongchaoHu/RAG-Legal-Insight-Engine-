from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vector_store import LegalVectorStore


def evaluate_precision(
    vector_store: LegalVectorStore,
    labeled_queries: list[dict[str, Any]],
    top_k: int = 3,
) -> float:
    if not labeled_queries:
        raise ValueError("No labeled queries supplied")

    hits = 0
    for item in labeled_queries:
        question = item["question"]
        expected_token = item["expected_source_contains"]
        results = vector_store.search(question, top_k=top_k)
        if any(expected_token.lower() in result.get("source_path", "").lower() for result in results):
            hits += 1

    return hits / len(labeled_queries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval precision for labeled legal queries")
    parser.add_argument("--queries-file", required=True, help="Path to JSON file with labeled queries")
    parser.add_argument("--index-path", default="artifacts/index.faiss", help="Path to FAISS index")
    parser.add_argument(
        "--mapping-path", default="artifacts/index_mapping.json", help="Path to index mapping JSON"
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieval window")
    args = parser.parse_args()

    queries_path = Path(args.queries_file)
    labeled_queries = json.loads(queries_path.read_text(encoding="utf-8"))

    store = LegalVectorStore(index_path=args.index_path, mapping_path=args.mapping_path)
    store.load_index()

    precision = evaluate_precision(store, labeled_queries, top_k=args.top_k)
    print(f"Precision@{args.top_k}: {precision:.2%}")


if __name__ == "__main__":
    main()
