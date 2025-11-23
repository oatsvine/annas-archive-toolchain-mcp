from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest
from annas.cli import (
    _compute_pdf_ocr_ratio,
    _count_text_characters,
    _detect_extension,
    _document_metadata_from_path,
    _elements_to_markdown,
    _sanitize_filename,
    download,
    search_catalog,
    search_downloaded_text,
)
from annas.state import AnnasState, configure_state
from annas.store import _looks_like_chapter
from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    ListItem,
    Text,
    Title,
)


class _StubElement:
    def __init__(self, text: str, category: str = "") -> None:
        self.text = text
        self.category = category
        self.metadata = ElementMetadata()


def test_download_artifact_real_download_when_secret_key_present(
    tmp_path: Path,
) -> None:
    secret = os.environ.get("ANNAS_SECRET_KEY")
    if not secret:
        pytest.skip("ANNAS_SECRET_KEY not configured; skipping live download test")

    state = configure_state(tmp_path, secret)
    preferred_formats = {"epub", "mobi", "azw", "azw3", "txt", "html"}
    results = search_catalog("plato", limit=60)
    assert results, "Expected non-empty search results"
    candidate = next(
        (
            entry
            for entry in results
            if entry.file_format and entry.file_format.lower() in preferred_formats
        ),
        None,
    )
    if candidate is None:
        pytest.skip("No supported search results available for download verification")
    md5 = candidate.md5
    download_path = download(md5)
    assert download_path.exists()


def test_search_downloaded_text_reads_markdown_context(annas_tmp: AnnasState) -> None:
    state = annas_tmp
    md5 = "a" * 32
    markdown_dir = state.work_path / md5
    markdown_dir.mkdir(parents=True)
    markdown_path = markdown_dir / "sample.md"
    markdown_path.write_text("line1\nmatch here\nline3\nline4\n", encoding="utf-8")

    snippet = search_downloaded_text(md5, "match", before=1, after=1, limit=1)
    assert "line1" in snippet
    assert "line3" in snippet


def test_sanitize_filename_truncates_and_preserves_extension(
    annas_tmp: AnnasState,
) -> None:
    long_name = "very long " * 40 + ".pdf"
    safe = _sanitize_filename(long_name, "a" * 32)
    assert len(safe) <= 120
    assert safe.endswith(".pdf")


def test_elements_to_markdown_formats_titles_and_lists(annas_tmp: AnnasState) -> None:
    elements: list[Element] = [
        Title(
            "Introduction",
            metadata=ElementMetadata(page_number=1, category_depth=1),
        ),
        ListItem(
            "Item one",
            metadata=ElementMetadata(page_number=1),
        ),
        ListItem(
            "Item two",
            metadata=ElementMetadata(page_number=1),
        ),
        Text(
            "More info",
            metadata=ElementMetadata(page_number=1),
        ),
        Title(
            "Next Section",
            metadata=ElementMetadata(page_number=2, category_depth=2),
        ),
        Text(
            "Final text",
            metadata=ElementMetadata(page_number=2),
        ),
    ]
    markdown = _elements_to_markdown(elements)
    assert "# Introduction" in markdown
    assert "- Item one" in markdown
    assert "## Next Section" in markdown
    assert "<!-- page 2 -->" in markdown


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Chapter 3 – Methods", True),
        ("CHAP. IX.", True),
        ("1. Background", True),
        ("3) Appendix", True),
        ("Section IV", True),
        ("Foreword", False),
        ("Appendix", False),
    ],
)
def test_looks_like_chapter_uses_upstream_patterns(
    heading: str, expected: bool
) -> None:
    assert _looks_like_chapter(heading) is expected


def test_detect_extension_falls_back_to_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.custom"
    path.write_bytes(b"hello world")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("libmagic missing")

    monkeypatch.setattr("annas.cli.detect_filetype", _raise)
    assert _detect_extension(path) == "custom"


def test_document_metadata_from_path_parses_segments(tmp_path: Path) -> None:
    path = tmp_path / "author__title__extra-info.epub"
    path.write_bytes(b"data")
    metadata = _document_metadata_from_path(path)
    assert metadata.author == "Author"
    assert metadata.title == "Title"
    assert metadata.extras == ["Extra-Info"]
    assert metadata.format == "epub"


def test_corpus_files_under_size_limit(corpus_files: list[Path]) -> None:
    for path in corpus_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        assert size_mb <= 10.0, f"Corpus file {path.name} exceeds 10 MB"


def test_partition_fast_on_corpus_pdfs(corpus_paths: dict[str, list[Path]]) -> None:
    from unstructured.partition.auto import partition

    pdfs = corpus_paths.get("pdf", [])
    assert pdfs, "Expected at least one PDF sample in corpus"
    for pdf in pdfs:
        elements = partition(
            filename=str(pdf),
            strategy="fast",
            metadata_filename=pdf.name,
        )
        assert any((getattr(el, "text", "") or "").strip() for el in elements)


def test_partition_auto_on_corpus_epubs(
    partition_elements: Callable[[Path], list[Element]],
    corpus_paths: dict[str, list[Path]],
) -> None:
    epubs = corpus_paths.get("epub", [])
    assert epubs, "Expected at least one EPUB sample in corpus"
    for epub in epubs:
        elements = partition_elements(epub)
        assert elements, f"Partition produced no elements for {epub.name}"


def test_count_text_characters_strips_whitespace() -> None:
    elements = [
        _StubElement("  Hello  "),
        _StubElement("", ""),
        _StubElement(" \n "),
        _StubElement("World"),
    ]
    total = _count_text_characters(elements)
    assert total == 10


def test_compute_pdf_ocr_ratio_uses_fast_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"dummy")
    hi_elements = [_StubElement("HelloWorld")]
    fast_calls: list[str] = []

    def _fake_partition(
        *, filename: str, strategy: str, metadata_filename: str, **_: object
    ):
        assert filename == str(path)
        assert metadata_filename == "sample.pdf"
        fast_calls.append(strategy)
        assert strategy == "fast"
        return [_StubElement("Hello")]

    monkeypatch.setattr("annas.cli.partition", _fake_partition)
    ratio = _compute_pdf_ocr_ratio(path, "sample.pdf", hi_elements)
    assert fast_calls == ["fast"]
    assert ratio == pytest.approx((10 - 5) / 10)


def test_compute_pdf_ocr_ratio_no_text_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"dummy")
    hi_elements = [_StubElement("   ")]
    called = False

    def _fake_partition(**_: object):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("annas.cli.partition", _fake_partition)
    ratio = _compute_pdf_ocr_ratio(path, "sample.pdf", hi_elements)
    assert ratio == 1.0
    assert called is False


def test_compute_pdf_ocr_ratio_non_pdf_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "sample.epub"
    path.write_bytes(b"epub")
    ratio = _compute_pdf_ocr_ratio(path, "sample.epub", [])
    assert ratio is None
