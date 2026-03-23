from langchain_core.documents import Document

from summarizer import LegalDocumentStore


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def invoke(self, prompt: str) -> FakeResponse:
        return FakeResponse("This is a short legal summary about obligations and penalties.")


class FailingLLM:
    def invoke(self, prompt: str) -> FakeResponse:
        raise RuntimeError("test failure")


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_generate_summary_happy_path() -> None:
    store = LegalDocumentStore(llm=FakeLLM(), embeddings=FakeEmbeddings())
    doc = Document(page_content="Long legal contract text.", metadata={"source_path": "doc.txt"})

    summary = store.generate_summary(doc)

    assert summary
    assert len(summary.split()) <= 100


def test_generate_summary_fallback_on_error() -> None:
    store = LegalDocumentStore(llm=FailingLLM(), embeddings=FakeEmbeddings())
    doc = Document(page_content="Fallback text should be used for summary.", metadata={"source_path": "doc.txt"})

    summary = store.generate_summary(doc)

    assert "Fallback text" in summary
