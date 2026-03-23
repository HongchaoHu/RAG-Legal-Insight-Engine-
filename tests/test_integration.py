from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from document_loader import load_documents
from preprocessor import LegalTextPreprocessor
from vector_store import LegalVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "contract" in lower else 0.0,
                    1.0 if "termination" in lower else 0.0,
                    float(len(text) % 7),
                ]
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "contract" in lower else 0.0,
            1.0 if "termination" in lower else 0.0,
            float(len(text) % 7),
        ]


class FakeDocumentStore:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()

    def generate_summary(self, doc):
        return doc.page_content[:100]


def test_end_to_end_query_returns_non_empty_answer(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()

    for idx in range(20):
        content = (
            "This contract includes a termination clause and obligations for both parties."
            if idx % 2 == 0
            else "This legal notice describes payment deadlines and compliance requirements."
        )
        (data_dir / f"doc_{idx}.txt").write_text(content, encoding="utf-8")

    docs = load_documents(data_dir)
    preprocessor = LegalTextPreprocessor()
    cleaned_docs = [preprocessor.clean_document(doc) for doc in docs]

    store = LegalVectorStore(
        index_path=tmp_path / "artifacts" / "index.faiss",
        mapping_path=tmp_path / "artifacts" / "index_mapping.json",
        document_store=FakeDocumentStore(),
    )
    store.create_vectorstore(cleaned_docs)

    app = create_app(vector_store=store)
    client = TestClient(app)

    response = client.post("/query", json={"question": "What are termination obligations in a contract?", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answers"]
    assert payload["answers"][0]["summary"]
