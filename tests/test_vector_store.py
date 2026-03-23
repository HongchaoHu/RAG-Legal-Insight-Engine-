from pathlib import Path

from langchain_core.documents import Document

from vector_store import LegalVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class FakeDocumentStore:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()

    def generate_summary(self, doc: Document) -> str:
        return doc.page_content[:50]


def test_create_load_and_search_vectorstore(tmp_path: Path) -> None:
    docs = [
        Document(page_content="Lease agreement for residential property.", metadata={"source_path": "a.txt"}),
        Document(page_content="Employment contract termination clause.", metadata={"source_path": "b.txt"}),
    ]

    store = LegalVectorStore(
        index_path=tmp_path / "index.faiss",
        mapping_path=tmp_path / "index_mapping.json",
        document_store=FakeDocumentStore(),
    )

    store.create_vectorstore(docs)
    store.load_index()
    results = store.search("termination clause", top_k=2)

    assert len(results) >= 1
    assert all(result["summary"] for result in results)
    assert all("source_path" in result for result in results)
