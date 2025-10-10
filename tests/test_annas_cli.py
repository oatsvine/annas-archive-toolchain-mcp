from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from unstructured.documents.elements import ElementMetadata

from annas import ANNAS_BASE_URL, Annas


def _metadata(
    *,
    page: int | None = None,
    depth: int | None = None,
    html: str | None = None,
) -> ElementMetadata:
    meta = ElementMetadata(page_number=page)
    if depth is not None:
        meta.category_depth = depth
    if html is not None:
        meta.text_as_html = html
    return meta


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def test_search_returns_results(workdir: Path) -> None:
    annas = Annas(work_path=workdir)
    try:
        results = annas.search("plato", limit=5)
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"Search unavailable: {exc!r}")

    assert isinstance(results, list)
    assert results, "Expected at least one search result for 'plato'"
    assert len(results) <= 5
    for entry in results:
        assert set(entry.keys()) == {"md5", "title", "url"}
        assert len(entry["md5"]) == 32
        assert entry["md5"].islower()
        assert entry["md5"] in entry["url"]
        assert entry["url"].startswith(f"{ANNAS_BASE_URL}/md5/")
        assert entry["title"] == entry["title"].strip()


def test_fetch_real_download_when_secret_key_present(workdir: Path) -> None:
    secret = os.environ.get("ANNAS_SECRET_KEY")
    if not secret:
        pytest.skip("ANNAS_SECRET_KEY not configured; skipping live download test")

    annas = Annas(work_path=workdir, secret_key=secret)
    results = annas.search("plato")
    assert results, "Expected non-empty search results"
    md5 = results[0]["md5"]
    download_path = annas.fetch(md5)
    assert download_path.exists()


def test_search_text_reads_markdown_context(workdir: Path) -> None:
    annas = Annas(work_path=workdir)
    md5 = "a" * 32
    markdown_dir = workdir / md5
    markdown_dir.mkdir(parents=True)
    markdown_path = markdown_dir / "sample.md"
    markdown_path.write_text("line1\nmatch here\nline3\nline4\n", encoding="utf-8")

    snippet = annas.search_text(md5, "match", before=1, after=1, limit=1)
    assert "line1" in snippet
    assert "line3" in snippet


def test_sanitize_filename_truncates_and_preserves_extension(workdir: Path) -> None:
    annas = Annas(work_path=workdir)
    long_name = "very long " * 40 + ".pdf"
    safe = annas._sanitize_filename(long_name, "a" * 32)
    assert len(safe) <= 120
    assert safe.endswith(".pdf")


def test_elements_to_markdown_formats_titles_and_lists(workdir: Path) -> None:
    annas = Annas(work_path=workdir)
    elements = [
        SimpleNamespace(
            text="Introduction", category="Title", metadata=_metadata(page=1, depth=1)
        ),
        SimpleNamespace(
            text="Item one", category="ListItem", metadata=_metadata(page=1)
        ),
        SimpleNamespace(
            text="Item two", category="ListItem", metadata=_metadata(page=1)
        ),
        SimpleNamespace(
            text="More info", category="NarrativeText", metadata=_metadata(page=1)
        ),
        SimpleNamespace(
            text="Next Section", category="Title", metadata=_metadata(page=2, depth=2)
        ),
        SimpleNamespace(
            text="Final text", category="NarrativeText", metadata=_metadata(page=2)
        ),
    ]

    markdown = annas._elements_to_markdown(elements)
    assert "# Introduction" in markdown
    assert "- Item one" in markdown
    assert "## Next Section" in markdown
    assert "<!-- page 2 -->" in markdown


def test_chromadb_ingest_and_query(workdir: Path) -> None:
    pytest.importorskip("chromadb")
    annas = Annas(work_path=workdir)
    collection = "test-chroma"
    md5 = "a" * 32
    elements = [
        SimpleNamespace(
            text="Intro section", category="Title", metadata=_metadata(page=1, depth=1)
        ),
        SimpleNamespace(
            text="Content line", category="NarrativeText", metadata=_metadata(page=1)
        ),
    ]
    text = annas._elements_to_markdown(elements)
    annas._load_elements(md5, elements, collection, text)
    annas._load_elements(md5, elements, collection, text)
    stored = annas._collection(collection).get(where={"md5": md5})
    ids = stored["ids"]
    assert ids and all(identifier.startswith(f"{md5}:") for identifier in ids)
    # ensure dedupe removed duplicates
    assert len(ids) == len(set(ids))
    metadata = stored["metadatas"][0]
    assert metadata["md5"] == md5
    results = annas.query_text(collection, "Intro", 1)
    assert results and "Intro" in results[0]
