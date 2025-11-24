from __future__ import annotations

import pytest
from annas.store import (
    DocumentMetadata,
    get_or_create_chroma,
    load_elements,
    query_chunks,
)
from unstructured.documents.elements import Element, ElementMetadata, Text, Title
from pathlib import Path


def test_chromadb_ingest_and_query(annas_tmp: Path) -> None:
    pytest.importorskip("chromadb")
    work_dir = annas_tmp
    collection = "test-chroma"
    md5 = "a" * 32
    elements: list[Element] = [
        Title(
            "COPYRIGHT",
            metadata=ElementMetadata(page_number=1, category_depth=1),
        ),
        Text(
            "Content line",
            metadata=ElementMetadata(page_number=1),
        ),
    ]
    document_metadata = DocumentMetadata(
        author="Test Author",
        title="Sample Book",
        filename="test_author__sample_book.epub",
        extras=["Tag One", "Tag Two"],
        format="epub",
        size_bytes=1024,
    )
    client = get_or_create_chroma(work_dir)
    load_elements(
        work_dir, md5, elements, collection, document_metadata, client=client
    )
    load_elements(
        work_dir, md5, elements, collection, document_metadata, client=client
    )
    stored = client.get_or_create_collection(collection).get(where={"md5": md5})
    ids = stored["ids"]
    assert ids and all(identifier.startswith(f"{md5}:") for identifier in ids)
    # ensure dedupe removed duplicates
    assert len(ids) == len(set(ids))
    metadatas = stored["metadatas"]
    assert metadatas is not None
    metadata = metadatas[0]
    assert metadata["md5"] == md5
    assert metadata["filename"] == document_metadata.filename
    assert metadata["title"] == document_metadata.title
    assert metadata["author"] == document_metadata.author
    assert metadata["tags"] == "Tag One|Tag Two"
    results = query_chunks(client, collection, "Content", n=1, md5s=None)
    assert results and "Content" in results[0].text
