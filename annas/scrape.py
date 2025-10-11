"""Scraping helpers for Anna's Archive search pages."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode, urljoin

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from playwright.sync_api import (
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

ANNAS_BASE_URL = "https://annas-archive.org"
SEARCH_ENDPOINT = f"{ANNAS_BASE_URL}/search"


class SearchResult(BaseModel):
    """Structured representation of a single Anna's Archive search card."""

    model_config = ConfigDict(extra="forbid")

    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    is_verified: Optional[bool] = None
    language: Optional[str] = None
    language_code: Optional[str] = None
    file_format: Optional[str] = None
    file_size_bytes: Optional[int] = Field(default=None, ge=1)
    file_size_label: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=0)
    category: Optional[str] = None
    source: Optional[str] = None
    source_path: Optional[str] = None
    description: Optional[str] = None
    cover_url: Optional[str] = None
    download_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("md5")
    @staticmethod
    def _lowercase_md5(value: str) -> str:
        return value.lower()

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized.startswith(f"{ANNAS_BASE_URL}/md5/"):
            raise ValueError("url must point to Anna's Archive md5 endpoint")
        md5 = info.data.get("md5") if info.data else None
        if md5 and md5 not in normalized:
            raise ValueError("url must include md5")
        return normalized


def scrape_search_results(
    query: str, *, limit: int, max_pages: int = 20
) -> List[SearchResult]:
    """Scrape Anna's Archive search pages with Playwright and return structured results."""
    assert query, "query must be non-empty"
    assert limit >= 1, "limit must be >= 1"
    assert max_pages >= 1, "max_pages must be >= 1"

    seen_md5: Set[str] = set()
    seen_titles: Set[str] = set()
    collected: List[SearchResult] = []
    page_number = 1

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            browser_page = browser.new_page()
            browser_page.set_default_timeout(15000)
            while len(collected) < limit and page_number <= max_pages:
                logger.info(
                    "Collecting search results: page={} collected={}/{}",
                    page_number,
                    len(collected),
                    limit,
                )
                page_results = _collect_page_results(
                    browser_page, query, page_number, seen_md5, seen_titles
                )
                if not page_results:
                    break
                collected.extend(page_results)
                page_number += 1
        finally:
            browser.close()

    logger.info("Collected {} search results limit={}", len(collected), limit)
    return collected[:limit]


def _collect_page_results(
    browser_page: Page,
    query: str,
    page_number: int,
    seen_md5: Set[str],
    seen_titles: Set[str],
) -> List[SearchResult]:
    params: Dict[str, str] = {"q": query}
    if page_number > 1:
        params["page"] = str(page_number)
    url = f"{SEARCH_ENDPOINT}?{urlencode(params)}"

    try:
        browser_page.goto(url, wait_until="networkidle")
    except PlaywrightTimeoutError:
        browser_page.wait_for_load_state("domcontentloaded")

    browser_page.wait_for_timeout(400)
    cards = browser_page.locator(
        "div.flex.border-b",
        has=browser_page.locator("a.js-vim-focus"),
    )
    card_count = cards.count()
    if card_count == 0:
        if _page_has_no_results(browser_page):
            return []
        try:
            browser_page.wait_for_selector(
                "div.flex.border-b a.js-vim-focus",
                timeout=3000,
            )
        except PlaywrightTimeoutError:
            return []
        cards = browser_page.locator(
            "div.flex.border-b",
            has=browser_page.locator("a.js-vim-focus"),
        )
        card_count = cards.count()
        if card_count == 0:
            return []

    page_results: List[SearchResult] = []
    for index in range(card_count):
        result = _parse_result_card(
            cards.nth(index),
            seen_md5,
            seen_titles,
        )
        if result is not None:
            page_results.append(result)

    return page_results


def _parse_result_card(
    card: Locator,
    seen_md5: Set[str],
    seen_titles: Set[str],
) -> Optional[SearchResult]:
    title_locator = card.locator("a.js-vim-focus").first
    href = title_locator.get_attribute("href") or ""
    md5 = _extract_md5(href)
    if not md5 or md5 in seen_md5:
        return None

    title_text_raw = _locator_inner_text(title_locator)
    if not title_text_raw:
        return None
    title = _clean_whitespace(title_text_raw)
    if not title:
        return None
    title_key = title.casefold()
    if title_key in seen_titles:
        return None

    detail_text = _locator_inner_text(card.locator("div.text-gray-800").first)
    detail_metadata = _parse_detail_text(detail_text)

    description = _locator_inner_text(card.locator("div.text-gray-600").first)
    source_path = _locator_inner_text(card.locator("div.font-mono").first)

    cover_url = None
    cover_candidates = card.locator("img")
    if cover_candidates.count():
        cover_url = cover_candidates.first.get_attribute("src")

    seen_md5.add(md5)
    seen_titles.add(title_key)

    return SearchResult(
        md5=md5,
        title=title,
        url=urljoin(ANNAS_BASE_URL, href),
        description=description,
        source_path=source_path,
        cover_url=cover_url,
        **detail_metadata,
    )


def _locator_inner_text(locator: Locator) -> Optional[str]:
    if locator.count() == 0:
        return None
    value = locator.inner_text().strip()
    return value or None


def _parse_detail_text(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    normalized = _clean_whitespace(raw.replace("\xa0", " "))
    if not normalized:
        return {}

    segments = [segment.strip() for segment in normalized.split("·")]
    metadata: Dict[str, Any] = {}
    after_save = False
    for segment in segments:
        cleaned = segment.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered == "save":
            after_save = True
            continue
        if after_save:
            count = _parse_integer(cleaned)
            if count is not None and "download_count" not in metadata:
                metadata["download_count"] = count
            continue

        if cleaned.startswith("✅"):
            metadata["is_verified"] = True
            cleaned = cleaned.lstrip("✅").strip()

        if "language" not in metadata:
            language_match = re.match(r"(.+?)\s*\[([^\]]+)\]$", cleaned)
            if language_match:
                metadata["language"] = _clean_whitespace(language_match.group(1))
                metadata["language_code"] = language_match.group(2).strip()
                continue

        if "file_format" not in metadata and _looks_like_file_format(cleaned):
            metadata["file_format"] = cleaned.replace(" ", "")
            continue

        if "file_size_label" not in metadata:
            size_data = _parse_size_label(cleaned)
            if size_data is not None:
                label, size_bytes = size_data
                metadata["file_size_label"] = label
                metadata["file_size_bytes"] = size_bytes
                continue

        if "year" not in metadata and re.fullmatch(r"\d{4}", cleaned):
            metadata["year"] = int(cleaned)
            continue

        if "source" not in metadata and cleaned.startswith("🚀"):
            metadata["source"] = cleaned.lstrip("🚀").strip()
            continue

        if "category" not in metadata and _looks_like_category(cleaned):
            metadata["category"] = _clean_whitespace(_strip_leading_emoji(cleaned))
            continue

    return metadata


def _clean_whitespace(value: str) -> str:
    return " ".join(value.split())


def _extract_md5(href: str) -> str:
    path = (href or "").split("?")[0].rstrip("/")
    candidate = path.split("/")[-1]
    if re.fullmatch(r"[0-9a-fA-F]{32}", candidate):
        return candidate.lower()
    return ""


def _parse_integer(value: str) -> Optional[int]:
    cleaned = re.sub(r"[^\d]", "", value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _parse_size_label(value: str) -> Optional[Tuple[str, int]]:
    match = re.fullmatch(
        r"(?P<number>[0-9]+(?:[.,][0-9]+)?)\s*(?P<unit>[KMGTP]?B)",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    number = float(match.group("number").replace(",", "."))
    unit = match.group("unit").upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    bytes_size = int(number * multiplier)
    return (value.replace(" ", ""), bytes_size)


def _strip_leading_emoji(value: str) -> str:
    if value and not value[0].isascii():
        return value[1:].strip()
    return value


def _looks_like_file_format(value: str) -> bool:
    if not value or value.endswith("B"):
        return False
    if any(char.islower() for char in value):
        return False
    tokens = value.replace("/", " ").replace("+", " ").split()
    return all(token.isalnum() for token in tokens) and any(
        token.isalpha() for token in tokens
    )


def _looks_like_category(value: str) -> bool:
    return bool(value) and not value[0].isascii() and not value.startswith("🚀")


def _page_has_no_results(browser_page: Page) -> bool:
    return browser_page.locator("text=No files found.").count() > 0
