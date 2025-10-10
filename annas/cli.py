"""Utility helpers for interacting with Anna's Archive content.

The module exposes a Fire-compatible CLI that can search Anna's Archive,
download and normalize artifacts, derive markdown snapshots, and push document
chunks into a Chroma vector store. All helpers follow the in-house convention
of typed, fail-fast surface areas.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple
from urllib.parse import unquote, urljoin, urlsplit, urlencode

import backoff
import certifi
import chromadb
import fire
import mobi
import requests
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult, Where
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from playwright.sync_api import (
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)
from unstructured.documents.elements import ElementMetadata
from unstructured.partition.auto import partition

ANNAS_BASE_URL = "https://annas-archive.org"
SEARCH_ENDPOINT = f"{ANNAS_BASE_URL}/search"
FAST_DOWNLOAD_ENDPOINT = f"{ANNAS_BASE_URL}/dyn/api/fast_download.json"
MD5_PATTERN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
ANNA_BRAND_PATTERN = re.compile(r"anna[’']?s archive", re.IGNORECASE)


class ElementLike(Protocol):
    """Protocol capturing the subset of Element attributes relied on by helpers."""

    text: str
    category: str
    metadata: ElementMetadata


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


class Annas:
    def __init__(
        self,
        work_path: Optional[Path | str] = None,
        secret_key: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        resolved_work = work_path or os.environ.get("ANNAS_DOWNLOAD_PATH")
        assert resolved_work, "ANNAS_DOWNLOAD_PATH must be provided"

        logger.remove()
        logger.add(sys.stderr, level=log_level.upper(), serialize=False)

        self.work_path = Path(resolved_work).expanduser()
        self.work_path.mkdir(parents=True, exist_ok=True)

        env_secret = secret_key or os.environ.get("ANNAS_SECRET_KEY")
        self.secret_key = env_secret
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "audio-agents-annas/0.1 (+https://annas-archive.org)",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._chroma_client: Optional[ClientAPI] = None

    # ─────────────────────────────── HTTP helpers ───────────────────────────────
    @staticmethod
    @backoff.on_exception(backoff.expo, requests.RequestException, max_time=60)
    def _retrying_get(
        session: requests.Session,
        url: str,
        *,
        params: Optional[Dict[str, str]] = None,
        stream: bool = False,
        timeout: int = 30,
    ) -> requests.Response:
        response = session.get(
            url,
            params=params,
            stream=stream,
            timeout=timeout,
            verify=certifi.where(),
            allow_redirects=True,
        )
        response.raise_for_status()
        return response

    # ────────────────────────────────── Public API ──────────────────────────────────
    # TODO: Never return anything other than pydantic models or primitive types (no dynamic dicts).
    def search(self, query: str, limit: int = 200) -> List[SearchResult]:
        """Search Anna's Archive cards and return structured result dictionaries.

        The query is stripped and validated before iterating through up to twenty
        result pages (Anna's Archive paginates search responses). Each unique md5
        hash is wrapped in a `SearchResult` model to ensure URL integrity and capture
        the published metadata (language, format, size, provenance) before serializing
        into a plain dictionary suitable for CLI output.
        """
        # DONE: document search flow and validation expectations.

        normalized = query.strip()
        assert normalized, "query must be non-empty"
        assert limit >= 1, "limit must be >= 1"

        capped_limit = min(limit, 200)
        seen_md5: Set[str] = set()
        seen_titles: Set[str] = set()
        collected: List[SearchResult] = []
        page_number = 1

        # TODO: Move scraping and parsing logic to `scrape.py` keep this class focused on workflow using clean models.
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                browser_page = browser.new_page()
                browser_page.set_default_timeout(15000)
                while len(collected) < capped_limit and page_number <= 20:
                    logger.info(
                        "Searching Anna's Archive",
                        query=normalized,
                        page=page_number,
                        current=len(collected),
                        target=capped_limit,
                    )
                    page_results = self._collect_page_results(
                        browser_page, normalized, page_number, seen_md5, seen_titles
                    )
                    if not page_results:
                        break
                    collected.extend(page_results)
                    page_number += 1
            finally:
                browser.close()

        logger.info(
            "Collected {} search results limit={}", len(collected), capped_limit
        )
        return collected[:capped_limit]

    def fetch(self, md5: str, collection: Optional[str] = None) -> Path:
        """Download a file by md5, normalize artifacts, and optionally ingest chunks.

        The method validates the md5, invokes the fast download endpoint with the
        configured secret key, streams the payload into the work directory, and
        extracts auxiliary markdown. When `collection` is provided, the derived
        markdown is chunked and upserted into the named Chroma collection.
        """
        # NOTE: Keep fetching logic in this module.
        md5 = self._validate_md5(md5)
        assert (
            self.secret_key
        ), "ANNAS_SECRET_KEY environment variable is required for downloads"

        logger.info("Fetching md5={md5}", md5=md5)
        params = {"md5": md5, "key": self.secret_key}
        response = self._retrying_get(
            self._session, FAST_DOWNLOAD_ENDPOINT, params=params
        )

        payload = response.json()
        download_url = payload.get("download_url")
        assert download_url, payload.get("error") or "fast download API returned no URL"

        download_response = self._retrying_get(
            self._session,
            download_url,
            stream=True,
            timeout=300,
        )

        download_path = self._write_stream(md5, download_response)
        detected_extension = self._detect_extension(download_path)
        assert detected_extension is not None, "Unable to detect downloaded file format"
        document_metadata = self._document_metadata_from_path(download_path)
        strategy = "hi_res" if download_path.suffix.lower() == ".pdf" else "fast"
        elements = partition(filename=str(download_path), strategy=strategy)
        markdown = self._elements_to_markdown(elements)
        markdown_path = self._markdown_path_for_md5(md5)
        markdown_path.write_text(markdown, encoding="utf-8")
        if collection:
            self._load_elements(
                md5,
                elements,
                collection,
                markdown,
                document_metadata,
            )
        logger.info("Download complete", md5=md5, path=str(download_path))
        return download_path

    def search_text(
        self, md5: str, needle: str, before: int = 2, after: int = 2, limit: int = 3
    ) -> str:
        """Return markdown snippets around a needle for an existing md5 artifact.

        The method loads the cached markdown (if present), scans for case-insensitive
        occurrences of `needle`, and returns up to `limit` snippets. Each snippet
        preserves a window of `before` and `after` lines from the hit, matching the
        CLI's expectation for compact context previews.
        """
        # NOTE: Keep pre-fetch artifact search in this module.
        md5 = self._validate_md5(md5)
        normalized = needle.strip()
        assert normalized, "needle must be non-empty"
        assert before >= 0, "before must be non-negative"
        assert after >= 0, "after must be non-negative"
        assert limit >= 1, "limit must be >= 1"

        markdown_path = self._markdown_path_for_md5(md5)
        # TODO: Remove this and irradicate this pattern from codebase! Error swallowing is never acceptable. Review `AGENTS.md` in ensure all rules are applied with common sense.
        try:
            content = markdown_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

        lines = content.splitlines()
        if not lines:
            return ""

        lower_needle = normalized.lower()
        snippets: List[str] = []
        for index, line in enumerate(lines):
            if lower_needle not in line.lower():
                continue
            start = max(0, index - before)
            end = min(len(lines), index + after + 1)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                snippets.append(snippet)
            if len(snippets) >= limit:
                break

        return "\n\n".join(snippets)

    def query_text(
        self, collection: str, query: str, n: int, md5s: Optional[str] = None
    ) -> List[str]:
        """Query a Chroma collection and format the best-matching document chunks.

        Results are constrained by `md5s` when provided, otherwise the full
        collection is searched. The raw Chroma payload is validated and rendered
        into human-readable snippets that include chunk order, inferred headings,
        and the originating md5 hash.
        """
        # NOTE: Keep pre-fetch artifact search in this module.
        # TODO: Make sure this is well tested in the test suite.
        assert query.strip(), "query must be non-empty"
        assert n >= 1, "limit must be >= 1"
        md5_filter: Optional[Where] = None
        if md5s:
            md5_list = [self._validate_md5(m) for m in md5s.split(",") if m.strip()]
            if md5_list:
                md5_filter = {"md5": {"$in": md5_list}}
        # Research (2025-09-28): Chroma query returns documents as list per query index;
        # see https://docs.trychroma.com/docs/querying-collections/query-and-get
        collection_client = self._collection(collection)
        # NOTE: Ok to use this non-pydantic structure as long as it's only internal and direct fit for purpose.
        query_result: QueryResult = collection_client.query(
            query_texts=[query],
            n_results=n,
            where=md5_filter,
            include=["documents", "metadatas"],
        )
        docs = query_result["documents"]
        assert docs is not None, "Chroma query must return documents"
        assert docs, "Chroma query returned no document rows"
        metadatas = query_result["metadatas"]
        primary_docs: List[str] = docs[0]
        assert metadatas and len(metadatas[0]) == len(
            primary_docs
        ), "Metadata count mismatch"
        assert primary_docs, "Chroma query returned empty document set"
        assembled: List[str] = []
        for i, doc in enumerate(primary_docs):
            # Make beautiful heading
            meta = metadatas[0][i] or {}
            base_title = meta.get("title") or "Untitled"
            chapter = meta.get("chapter")
            heading = base_title if not chapter else f"{base_title} — {chapter}"
            snippet = (
                f"─────────────────── Chunk {i+1} - {heading} ({meta.get('md5', 'unknown')}) ───────────────────\n\n"
                f"{doc.strip()}\n"
            )
            assembled.append(snippet)
        return assembled

    # ─────────────────────────────── Internal helpers ───────────────────────────────
    @staticmethod
    def _clean_whitespace(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _extract_md5(href: str) -> str:
        path = (href or "").split("?")[0].rstrip("/")
        candidate = path.split("/")[-1]
        if re.fullmatch(r"[0-9a-fA-F]{32}", candidate):
            return candidate.lower()
        return ""

    def _collect_page_results(
        self,
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
            if self._page_has_no_results(browser_page):
                return []
            # If results failed to hydrate, give the page a brief chance and retry once.
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
            result = self._parse_result_card(
                cards.nth(index),
                seen_md5,
                seen_titles,
            )
            if result is not None:
                page_results.append(result)

        return page_results

    def _parse_result_card(
        self,
        card: Locator,
        seen_md5: Set[str],
        seen_titles: Set[str],
    ) -> Optional[SearchResult]:
        title_locator = card.locator("a.js-vim-focus").first
        href = title_locator.get_attribute("href") or ""
        md5 = self._extract_md5(href)
        if not md5 or md5 in seen_md5:
            return None

        title_text_raw = self._locator_inner_text(title_locator)
        if not title_text_raw:
            return None
        title = self._clean_whitespace(title_text_raw)
        if not title:
            return None
        title_key = title.casefold()
        if title_key in seen_titles:
            return None

        detail_text = self._locator_inner_text(card.locator("div.text-gray-800").first)
        detail_metadata = self._parse_detail_text(detail_text)

        description = self._locator_inner_text(card.locator("div.text-gray-600").first)
        source_path = self._locator_inner_text(card.locator("div.font-mono").first)

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

    @staticmethod
    def _locator_inner_text(locator: Locator) -> Optional[str]:
        if locator.count() == 0:
            return None
        value = locator.inner_text().strip()
        return value or None

    def _parse_detail_text(self, raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        normalized = self._clean_whitespace(raw.replace("\xa0", " "))
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
                count = self._parse_integer(cleaned)
                if count is not None and "download_count" not in metadata:
                    metadata["download_count"] = count
                continue

            if cleaned.startswith("✅"):
                metadata["is_verified"] = True
                cleaned = cleaned.lstrip("✅").strip()

            if "language" not in metadata:
                language_match = re.match(r"(.+?)\s*\[([^\]]+)\]$", cleaned)
                if language_match:
                    metadata["language"] = self._clean_whitespace(
                        language_match.group(1)
                    )
                    metadata["language_code"] = language_match.group(2).strip()
                    continue

            if "file_format" not in metadata and self._looks_like_file_format(cleaned):
                metadata["file_format"] = cleaned.replace(" ", "")
                continue

            if "file_size_label" not in metadata:
                size_data = self._parse_size_label(cleaned)
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

            if "category" not in metadata and self._looks_like_category(cleaned):
                metadata["category"] = self._clean_whitespace(
                    self._strip_leading_emoji(cleaned)
                )
                continue

        return metadata

    @staticmethod
    def _parse_integer(value: str) -> Optional[int]:
        cleaned = re.sub(r"[^\d]", "", value)
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None

    @staticmethod
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

    @staticmethod
    def _strip_leading_emoji(value: str) -> str:
        # Most icons are single code points; fallback to original string when ascii.
        if value and not value[0].isascii():
            return value[1:].strip()
        return value

    @staticmethod
    def _looks_like_file_format(value: str) -> bool:
        if not value or value.endswith("B"):
            return False
        if any(char.islower() for char in value):
            return False
        tokens = value.replace("/", " ").replace("+", " ").split()
        return all(token.isalnum() for token in tokens) and any(
            token.isalpha() for token in tokens
        )

    @staticmethod
    def _looks_like_category(value: str) -> bool:
        return bool(value) and not value[0].isascii() and not value.startswith("🚀")

    @staticmethod
    def _page_has_no_results(browser_page: Page) -> bool:
        return browser_page.locator("text=No files found.").count() > 0

    def _elements_to_markdown(self, elements: Sequence[ElementLike]) -> str:
        lines: List[str] = []
        list_buffer: List[str] = []
        last_page: Optional[int] = None

        def flush_list() -> None:
            if list_buffer:
                lines.extend(list_buffer)
                list_buffer.clear()
                lines.append("")

        for element in elements:
            # Research (2025-09-28): unstructured Elements expose `.text`, `.category`, and
            # `.metadata` (ElementMetadata) attributes deterministically. Docs:
            # https://docs.unstructured.io/open-source/how-to/get-elements
            text = element.text
            if not text or not text.strip():
                continue
            category = element.category.lower()
            metadata = element.metadata
            assert isinstance(
                metadata, ElementMetadata
            ), "element.metadata must be ElementMetadata"
            page = metadata.page_number
            if page is not None and page != last_page:
                flush_list()
                lines.append(f"<!-- page {page} -->")
                lines.append("")
                last_page = page

            stripped = text.strip()
            if category == "title":
                flush_list()
                # Research (2025-09-28): `category_depth` encodes heading hierarchy; see
                # https://docs.unstructured.io/platform/document-elements
                depth = metadata.category_depth
                if depth is None:
                    # FIXME(2025-09-28): Some partitioners omit `category_depth` when the
                    # source lacks hierarchy (same doc as above). Default to level 1 but
                    # consider deriving depth from element order.
                    depth = 1
                depth = max(1, min(int(depth), 6))
                lines.append(f"{'#' * depth} {stripped}")
                lines.append("")
                continue

            if category in {"list_item", "listitem"}:
                list_buffer.append(f"- {stripped}")
                continue

            flush_list()
            if category in {"table", "figure"}:
                # Research (2025-09-28): Table metadata exposes `text_as_html`; see
                # https://docs.unstructured.io/platform/document-elements#table-specific-metadata
                html_text = metadata.text_as_html
                assert html_text is not None, "Table elements require HTML text"
                lines.append(html_text)
            else:
                lines.append(stripped)
            lines.append("")

        flush_list()
        while lines and lines[-1] == "":
            lines.pop()
        lines.append("")
        return "\n".join(lines)

    def _load_elements(
        self,
        md5: str,
        elements: Sequence[ElementLike],
        collection: str,
        text: str,
        document_metadata: Optional[Dict[str, Optional[List[str] | str]]] = None,
    ) -> None:
        document_metadata = document_metadata or {}
        coll = self._collection(collection)
        existing = coll.get(where={"md5": md5})
        ids = existing["ids"]
        if ids:
            coll.delete(ids=ids)
        # Research (2025-09-28): Element metadata always available as ElementMetadata with
        # optional page_number/title attributes; see https://docs.unstructured.io/platform/document-elements
        page_numbers = set()
        doc_title = (document_metadata.get("title") or "").strip()
        author_name = (document_metadata.get("author") or "").strip()
        tag_segments = document_metadata.get("extras") or []
        format_label = (document_metadata.get("format") or "").strip().upper()
        size_bytes = document_metadata.get("size_bytes")
        title = doc_title
        title_found = False
        for element in elements:
            metadata = element.metadata
            assert isinstance(
                metadata, ElementMetadata
            ), "element.metadata must be ElementMetadata"
            if metadata.page_number is not None:
                page_numbers.add(metadata.page_number)
            if (
                not title_found
                and element.category.lower() == "title"
                and element.text.strip()
            ):
                candidate = element.text.strip()
                if not self._is_generic_heading(
                    candidate
                ) and not self._looks_like_chapter(candidate):
                    title = candidate
                    title_found = True
        # Research (2025-09-28): Chroma recall improves with consistent chunk sizes/overlaps; see
        # https://research.trychroma.com/evaluating-chunking
        # Research (2025-09-28): RecursiveCharacterTextSplitter preserves structure for generic text;
        # see https://python.langchain.com/docs/how_to/recursive_text_splitter/
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " "],
        )
        raw_chunks = splitter.split_text(text)
        assert raw_chunks, "No ingestable chunks produced for Chroma ingestion"

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for index, chunk in enumerate(raw_chunks, start=1):
            page_matches = [
                int(match) for match in re.findall(r"<!-- page (\d+) -->", chunk)
            ]
            clean_chunk = re.sub(r"<!-- page \d+ -->\s*", "", chunk).strip()
            if not clean_chunk:
                continue
            documents.append(clean_chunk)
            metadata: Dict[str, str] = {
                "md5": md5,
                "chunk_index": str(index),
                "title": title or doc_title or "Untitled",
            }
            if author_name:
                metadata["author"] = author_name
            if tag_segments:
                metadata["tags"] = "|".join(tag_segments)
            if format_label:
                metadata["format"] = format_label
            if isinstance(size_bytes, int):
                metadata["size_bytes"] = str(size_bytes)
            if page_matches:
                first = min(page_matches)
                last = max(page_matches)
                metadata["page_start"] = str(first)
                metadata["page_end"] = str(last)
            if page_numbers:
                metadata["pages_total"] = str(len(page_numbers))
            chapter = self._extract_chapter(clean_chunk)
            if chapter and chapter.lower() != metadata["title"].lower():
                metadata["chapter"] = chapter
            metadatas.append(metadata)

        assert documents, "All ingestable chunks were empty after cleaning page markers"
        chunk_ids = [f"{md5}:{index:04d}" for index in range(1, len(documents) + 1)]
        coll.upsert(ids=chunk_ids, documents=documents, metadatas=metadatas)

    def _collection(self, name: str) -> Collection:
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.work_path / "chroma")
            )
        return self._chroma_client.get_or_create_collection(name)

    @staticmethod
    def _validate_md5(md5: str) -> str:
        candidate = md5.strip().lower()
        assert re.fullmatch(r"[0-9a-f]{32}", candidate), f"Invalid md5 hash: {md5}"
        return candidate

    def _artifact_dir(self, md5: str) -> Path:
        path = self.work_path / md5
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_stream(self, md5: str, response: requests.Response) -> Path:
        filename = self._filename_from_response(md5, response)
        target_dir = self._artifact_dir(md5)
        target_file = target_dir / filename
        logger.info("Writing download to {path}", path=str(target_file))
        with target_file.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    fh.write(chunk)
        if target_file.suffix.lower() in [".mobi", ".azw", ".azw3"]:
            try:
                tempdir, name = mobi.extract(str(target_file))
                tmp_file = Path(tempdir) / name
                logger.info(f"Extracted {target_file.name} => {tmp_file.name}")
                assert tmp_file.exists(), f"Extracted file missing: {tmp_file}"
                sanitized_name = self._sanitize_filename(name, md5)
                converted_path = target_file.with_name(sanitized_name)
                shutil.move(str(tmp_file), str(converted_path))
                target_file.unlink(missing_ok=True)
                target_file = converted_path
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("MOBI conversion failed: {}", exc)
        detected_extension = self._detect_extension(target_file)
        expected_extension = target_file.suffix.lstrip(".").lower()
        assert (
            expected_extension
        ), "Downloaded file is missing an extension in the Content-Disposition header"
        if detected_extension is not None:
            assert detected_extension.lower() == expected_extension, (
                "Downloaded file extension mismatch",
                expected_extension,
                detected_extension,
            )
        return target_file

    def _filename_from_response(self, md5: str, response: requests.Response) -> str:
        disposition = response.headers.get("content-disposition", "")
        filename = ""
        if "filename*=" in disposition:
            match = re.search(r"filename\*=([^']*)'[^']*'([^;]+)", disposition)
            if match:
                filename = unquote(match.group(2))
        if not filename and "filename=" in disposition:
            raw = disposition.split("filename=")[-1].split(";")[0].strip()
            stripped = raw.strip().strip('"').strip("'")
            filename = unquote(stripped)
        if not filename:
            path = urlsplit(response.url).path
            filename = path.split("/")[-1] if path else ""
        filename = filename or "download.bin"
        return self._sanitize_filename(filename, md5)

    def _sanitize_filename(self, filename: str, md5: str) -> str:
        stem, extension = self._split_extension(filename)
        decoded = self._decode_underscore_hex(stem)
        decoded = unquote(decoded)
        decoded = unicodedata.normalize("NFKC", decoded)
        decoded = decoded.replace("\\", "_").replace("/", "_")

        trimmed = self._strip_branding(decoded, md5)
        segments = self._extract_segments(trimmed)

        if not segments:
            segments = [trimmed]

        title_segment = segments[0]
        author_segment = segments[1] if len(segments) > 1 else "Unknown"
        extra_segments = segments[2:]

        isbn_tokens = self._extract_isbns(trimmed)

        slug_components = [
            self._slug(author_segment) or "unknown",
            self._slug(title_segment) or "untitled",
        ]
        slug_components.extend(self._slug(segment) for segment in extra_segments)
        slug_components.extend(isbn_tokens)

        filtered = [component for component in slug_components if component]
        slug_body = "__".join(filtered) if filtered else "document"

        max_len = 120
        allowed = max(8, max_len - (len(extension) + 1 if extension else 0))
        truncated = slug_body[:allowed]
        return f"{truncated}.{extension}" if extension else truncated

    @staticmethod
    def _decode_underscore_hex(value: str) -> str:
        pattern = re.compile(r"(?:_[0-9A-Fa-f]{2})+")
        cursor = 0
        segments: List[str] = []
        for match in pattern.finditer(value):
            start, end = match.span()
            segments.append(value[cursor:start])
            encoded = match.group().replace("_", "%")
            decoded = unquote(encoded)
            segments.append(decoded)
            cursor = end
        segments.append(value[cursor:])
        return "".join(segments)

    @staticmethod
    def _split_extension(name: str) -> tuple[str, str]:
        if "." in name:
            stem, ext = name.rsplit(".", 1)
            return stem, ext.lower()
        return name, ""

    @staticmethod
    def _strip_branding(value: str, md5: str) -> str:
        without_md5 = MD5_PATTERN.sub("", value)
        without_target_md5 = without_md5.replace(md5, "")
        return ANNA_BRAND_PATTERN.sub("", without_target_md5)

    @staticmethod
    def _extract_segments(value: str) -> List[str]:
        normalized = re.sub(r"[\t\r\n]+", " ", value)
        normalized = re.sub(r"\s+[-–—]{2,}\s+", "|", normalized)
        normalized = re.sub(r"\s+-\s+", "|", normalized)
        segments = [segment.strip(" _-") for segment in normalized.split("|")]
        cleaned: List[str] = []
        for segment in segments:
            if not segment:
                continue
            if MD5_PATTERN.search(segment):
                continue
            if ANNA_BRAND_PATTERN.search(segment):
                continue
            cleaned.append(segment)
        return cleaned

    @staticmethod
    def _slug(value: str) -> str:
        normal = unicodedata.normalize("NFKD", value)
        stripped = "".join(ch for ch in normal if not unicodedata.combining(ch))
        lowered = stripped.lower()
        lowered = lowered.replace("'", "_")
        slug = re.sub(r"[^0-9a-z]+", "_", lowered)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    @staticmethod
    def _extract_isbns(value: str) -> List[str]:
        candidates = re.findall(
            r"\b(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]{1,7}[- ]?[0-9]{1,7}[- ]?[0-9Xx]\b",
            value,
        )
        cleaned: List[str] = []
        for candidate in candidates:
            digits = re.sub(r"[^0-9Xx]", "", candidate)
            if len(digits) not in {10, 13}:
                continue
            cleaned.append(digits.lower())
        return cleaned

    @staticmethod
    def _detect_extension(path: Path) -> Optional[str]:
        try:
            with path.open("rb") as fh:
                header = fh.read(4096)
        except FileNotFoundError:
            return None
        if header.startswith(b"%PDF"):
            return "pdf"
        if header.startswith(b"PK\x03\x04"):
            try:
                import zipfile

                with zipfile.ZipFile(path) as zf:
                    if "mimetype" in zf.namelist():
                        mime = zf.read("mimetype").decode("utf-8", "ignore").strip()
                        if mime == "application/epub+zip":
                            return "epub"
                    if any(name.endswith(".rels") for name in zf.namelist()):
                        return "docx"
            except Exception:  # pylint: disable=broad-except
                return "zip"
            return "zip"
        if b"BOOKMOBI" in header[:4096]:
            return "mobi"
        if header.startswith(b"\x7fELF"):
            return "bin"
        if header[:4] == b"Rar!":
            return "rar"
        if header[:2] == b"PK":
            return "zip"
        suffix = path.suffix.lower().lstrip(".")
        return suffix or None

    @staticmethod
    def _deslug(value: str) -> str:
        words = value.replace("_", " ").strip()
        return words.title() if words else ""

    def _document_metadata_from_path(
        self, path: Path
    ) -> Dict[str, Optional[List[str] | str]]:
        stem, _ = self._split_extension(path.name)
        components = [part for part in stem.split("__") if part]
        size_bytes = path.stat().st_size if path.exists() else None
        if not components:
            return {
                "author": None,
                "title": path.stem,
                "extras": [],
                "format": self._split_extension(path.name)[1],
                "size_bytes": size_bytes,
            }
        author_slug = components[0]
        title_slug = components[1] if len(components) > 1 else components[0]
        extras = components[2:]
        return {
            "author": self._deslug(author_slug) or None,
            "title": self._deslug(title_slug) or path.stem,
            "extras": [self._deslug(part) for part in extras if part],
            "format": self._split_extension(path.name)[1],
            "size_bytes": size_bytes,
        }

    @staticmethod
    def _extract_chapter(chunk: str) -> Optional[str]:
        match = re.search(r"^#{1,6}\s+(.+)", chunk, re.MULTILINE)
        if not match:
            return None
        heading = match.group(1).strip()
        if Annas._is_generic_heading(heading):
            return None
        return heading

    @staticmethod
    def _is_generic_heading(value: str) -> bool:
        lowered = value.strip().lower()
        generics = {
            "table of contents",
            "contents",
            "preface",
            "foreword",
            "introduction",
        }
        return lowered in generics

    @staticmethod
    def _looks_like_chapter(value: str) -> bool:
        lowered = value.strip().lower()
        return bool(re.match(r"chapter\s+\d+", lowered))

    def _markdown_path_for_md5(self, md5: str) -> Path:
        target_dir = self._artifact_dir(md5)
        candidates = sorted(target_dir.glob("*.md"))
        if not candidates:
            return target_dir / "converted.md"
        return candidates[0]


if __name__ == "__main__":
    fire.Fire(Annas)


# • Lessons Learned

#   - Lean on Typed Element APIs Instead of getattr
#     Feedback: Falling back to getattr on Unstructured elements diluted type safety and contradicted the “strongly typed, deterministic” convention.
#     Lesson: When a library documents stable attributes (e.g., Element.text, Element.metadata.page_number), access them directly and fail fast if they are missing; don’t normalize away type errors with permissive lookups. This keeps contracts explicit and surfaces upstream extraction issues immediately.
#   - Chunk Text with Proven Splitters Before Vector Ingestion
#     Feedback: Loading entire markdown files into Chroma ignored guidance to manage chunk size/overlap internally.
#     Lesson: Always split long documents with a domain-appropriate strategy (e.g., RecursiveCharacterTextSplitter) and store overlapping chunks before embedding; this preserves semantic context and aligns with our retrieval-augmented workflows.
#   - Use Asserts for Deterministic Preconditions, Reserve try/except for External Failures
#     Feedback: Wrapping imports and core invariants in try/except created hidden fallbacks even though those dependencies are fully deterministic and under our control.
#     Lesson: Guard predictable preconditions with single asserts and let violations crash loudly; only catch exceptions when interfacing with non-deterministic, third-party steps (e.g., mobi.extract) where failures are expected and we still want to preserve already-downloaded artifacts.
#   - Decode and Normalize Filenames Before Sanitizing
#     Feedback: Leaving percent-style sequences (e.g., `_20`) in output paths produced unreadable artifacts and broke downstream tooling.
#     Lesson: Expand encoded segments to their Unicode characters before applying the usual filename sanitizer so derived artifacts stay human-readable without weakening the safety checks.
