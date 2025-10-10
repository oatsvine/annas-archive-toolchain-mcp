import re
from pathlib import Path
from typing import List
from urllib.parse import quote

import pytest

from annas import ANNAS_BASE_URL, Annas

QUERIES: List[str] = [
    "philosophy",
    "science",
    "history",
    "poetry",
    "mathematics",
    "technology",
    "art",
    "economics",
    "fiction",
    "biography",
]


def _synthetic_annas_filename(title: str, author: str, md5: str, extension: str) -> str:
    """Approximate the raw fast-download naming style visible in existing payloads."""
    parts = [segment for segment in [title, author, md5, "Anna's Archive"] if segment]
    base = " -- ".join(parts)
    encoded = quote(base, safe="-_'", encoding="utf-8", errors="strict").replace(
        "%", "_"
    )
    return f"{encoded}.{(extension or 'bin').lower()}"


def test_search_result_invariants(tmp_path: Path) -> None:
    workdir = tmp_path / "annas"
    annas = Annas(work_path=workdir)

    try:
        sample = annas.search("plato", limit=100)
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"Search unavailable: {exc!r}")

    if len(sample) < 100:
        pytest.skip("Insufficient search results for invariant checks")

    md5s = [entry["md5"] for entry in sample]
    urls = [entry["url"] for entry in sample]
    titles = [entry["title"] for entry in sample]

    assert len(sample) == len(set(md5s)), "duplicate md5 values detected"
    assert len(sample) == len(set(urls)), "duplicate urls detected"
    assert len(sample) == len(set(titles)), "duplicate titles detected"

    for entry in sample:
        assert set(entry.keys()) == {"md5", "title", "url"}
        assert re.fullmatch(r"[0-9a-f]{32}", entry["md5"])
        assert entry["md5"] in entry["url"]
        assert entry["url"].startswith(f"{ANNAS_BASE_URL}/md5/")
        assert entry["title"] == entry["title"].strip()


def test_document_metadata_from_sanitized_filename(tmp_path: Path) -> None:
    md5 = "28de8eb69b63bc7b8a7ccc72c354782b"
    annas = Annas(work_path=tmp_path / "annas")
    raw_name = _synthetic_annas_filename(
        title="The Republic",
        author="Plato",
        md5=md5,
        extension="epub",
    )
    sanitized = annas._sanitize_filename(raw_name, md5)
    metadata = annas._document_metadata_from_path(Path(tmp_path / sanitized))
    assert metadata["title"] == "The Republic"
    assert metadata["author"] == "Plato"


def test_extract_chapter_heading() -> None:
    chunk = "# Chapter 5\nSome discussion about forms."
    chapter = Annas._extract_chapter(chunk)
    assert chapter == "Chapter 5"


def test_detect_extension_pdf(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"%PDF-1.7\nrest")
    annas = Annas(work_path=tmp_path / "annas")
    detected = annas._detect_extension(path)
    assert detected == "pdf"
