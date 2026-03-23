from pathlib import Path

from document_loader import load_documents


def test_load_documents_counts_txt_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Contract clause one.", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("Contract clause two.", encoding="utf-8")
    (nested / "ignore.md").write_text("Not part of corpus.", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    assert all("source_path" in doc.metadata for doc in docs)
    assert all("creation_date" in doc.metadata for doc in docs)
