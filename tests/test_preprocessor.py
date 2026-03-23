from langchain_core.documents import Document

from preprocessor import LegalTextPreprocessor


def test_clean_document_normalizes_whitespace_and_quotes() -> None:
    preprocessor = LegalTextPreprocessor()
    source = Document(
        page_content='  “Party A”\n\nshall   pay  ‘Party B’.\x0b ',
        metadata={"source_path": "sample.txt"},
    )

    cleaned = preprocessor.clean_document(source)

    assert cleaned.page_content == '"Party A" shall pay \'Party B\'.'
    assert cleaned.metadata == source.metadata
