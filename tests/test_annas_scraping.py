from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pytest

from annas.state import configure_state
from annas.cli import download
from annas.scrape import ANNAS_BASE_URL, SearchResult, scrape_search_results


def _collect_live_results(query: str, limit: int) -> List[SearchResult]:
    try:
        results = scrape_search_results(query, limit=limit)
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"Anna's Archive search unavailable: {exc!r}")
    if not results:
        pytest.skip(f"No search results for query={query!r}")
    return results


def _assert_metadata_richness(results: Iterable[SearchResult]) -> None:
    formats = {entry.file_format for entry in results if entry.file_format}
    languages = {entry.language for entry in results if entry.language}
    categories = {entry.category for entry in results if entry.category}
    assert formats, "Expected at least one result with file_format metadata"
    assert languages, "Expected at least one result with language metadata"
    assert categories, "Expected at least one result with category metadata"


def _parse_size_label(label: str) -> Optional[int]:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?B)", label.upper())
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }
    return int(number * multipliers[unit])


def test_scrape_search_results_invariants() -> None:
    results = _collect_live_results("philosophy", limit=25)
    if len(results) < 10:
        pytest.skip("Insufficient results returned for invariants test")

    md5s = [entry.md5 for entry in results]
    titles = [entry.title for entry in results]
    urls = [entry.url for entry in results]

    assert len(md5s) == len(set(md5s)), "Duplicate md5s detected"
    assert len(titles) == len(set(title.casefold() for title in titles)), "Duplicate titles detected"
    assert len(urls) == len(set(urls)), "Duplicate URLs detected"

    _assert_metadata_richness(results)

    for entry in results:
        assert isinstance(entry, SearchResult)
        assert entry.md5 in entry.url
        assert entry.url.startswith(f"{ANNAS_BASE_URL}/md5/")
        assert entry.title == entry.title.strip()
        if entry.file_size_bytes is not None:
            assert entry.file_size_bytes > 0
        if entry.file_size_label is not None:
            parsed = _parse_size_label(entry.file_size_label)
            assert parsed == entry.file_size_bytes
        if entry.download_count is not None:
            assert entry.download_count >= 0
        if entry.language_code is not None:
            assert entry.language_code == entry.language_code.strip()


def test_scrape_search_results_paginates_across_pages() -> None:
    results = _collect_live_results("science", limit=70)
    if len(results) < 40:
        pytest.skip("Search returned fewer than 40 results; cannot confirm pagination")

    assert len(results) == 70, "Expected limit to be honored when pagination succeeds"
    assert len(set(entry.md5 for entry in results)) == len(results)
    _assert_metadata_richness(results)


def test_scraped_metadata_matches_download(tmp_path: Path) -> None:
    secret = os.environ.get("ANNAS_SECRET_KEY")
    if not secret:
        pytest.skip("ANNAS_SECRET_KEY not configured; skipping download verification")

    preferred_formats = {"epub", "fb2", "djvu", "doc", "docx", "txt"}
    conversion_formats = {"mobi", "azw", "azw3"}
    candidate: Optional[SearchResult] = None
    candidate_requires_conversion = False
    fallback_candidate: Optional[SearchResult] = None

    queries = ["epub", "fiction", "mathematics"]
    for query in queries:
        results = _collect_live_results(query, limit=40)
        candidate = next(
            (
                entry
                for entry in results
                if entry.file_size_bytes
                and entry.file_format
                and entry.file_format.lower() in preferred_formats
            ),
            None,
        )
        if candidate:
            candidate_requires_conversion = False
            break

        if fallback_candidate is None:
            fallback_candidate = next(
                (
                    entry
                    for entry in results
                    if entry.file_size_bytes
                    and entry.file_format
                    and entry.file_format.lower() in conversion_formats
                ),
                None,
            )

    if candidate is None and fallback_candidate is not None:
        candidate = fallback_candidate
        candidate_requires_conversion = True

    if candidate is None:
        pytest.skip("No suitable search result with metadata available for download comparison")

    configure_state(tmp_path / "annas", secret)
    download_path = download(candidate.md5)
    assert download_path.exists(), "Expected downloaded artifact to exist"

    actual_size = download_path.stat().st_size
    expected_size = candidate.file_size_bytes
    assert expected_size is not None

    assert candidate.file_format is not None
    expected_format = candidate.file_format.lower()
    actual_format = download_path.suffix.lstrip(".").lower()

    if candidate_requires_conversion:
        assert expected_format in conversion_formats
        assert actual_format == "html"
        assert actual_size > 0
        return

    tolerance = max(int(expected_size * 0.25), 524_288)
    assert abs(actual_size - expected_size) <= tolerance, (
        "Downloaded file size differs significantly from scraped metadata",
        actual_size,
        expected_size,
    )

    assert actual_format == expected_format, (
        "Downloaded file extension does not match scraped metadata",
        actual_format,
        expected_format,
    )
