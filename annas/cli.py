"""Utility helpers for interacting with Anna's Archive content.

The module exposes a Fire-compatible CLI that can search Anna's Archive,
download and normalize artifacts, derive markdown snapshots, and push document
chunks into a Chroma vector store. All helpers follow the in-house convention
of typed, fail-fast surface areas.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from rich.pretty import pretty_repr
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, cast
from urllib.parse import unquote, urlsplit

import backoff
import certifi
import chromadb
import fire
import mobi
import requests
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, QueryResult, Where
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from annas.scrape import ANNAS_BASE_URL, SearchResult, scrape_search_results
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Element, ElementMetadata, ListItem, Title
from unstructured.file_utils.filetype import detect_filetype
from unstructured.partition.auto import partition
from unstructured.staging.base import element_to_md
from unstructured.nlp.patterns import ENUMERATED_BULLETS_RE, NUMBERED_LIST_RE

# from unstructured.metrics.evaluate import evaluate_extraction

FAST_DOWNLOAD_ENDPOINT = f"{ANNAS_BASE_URL}/dyn/api/fast_download.json"
MD5_PATTERN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
ANNA_BRAND_PATTERN = re.compile(r"anna[’']?s archive", re.IGNORECASE)


class ElementLike(Protocol):
    """Protocol capturing the subset of Element attributes relied on by helpers."""

    text: str
    category: str
    metadata: ElementMetadata


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from sanitized filenames."""

    model_config = ConfigDict(extra="forbid")

    author: Optional[str] = Field(default=None, description="Primary author name")
    title: str = Field(min_length=1, description="Normalized title")
    filename: str = Field(min_length=1, description="Sanitized filename")
    extras: List[str] = Field(
        default_factory=list, description="Additional metadata segments"
    )
    format: str = Field(min_length=1, description="Lowercase file extension")
    size_bytes: Optional[int] = Field(
        default=None, ge=0, description="File size in bytes when available"
    )

    @field_validator("author", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("title", "format", "filename", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("must be non-empty")
        return text

    @field_validator("extras", mode="before")
    @classmethod
    def _normalize_extras(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            cleaned: List[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        raise TypeError("extras must be a sequence of strings")

    @field_validator("size_bytes")
    @classmethod
    def _validate_size(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 0:
            raise ValueError("size_bytes must be non-negative")
        return value


class ArtifactValidationError(RuntimeError):
    """Raised when a downloaded artifact fails structural validation."""


class Annas:
    def __init__(
        self,
        work_path: Path | str = os.environ.get("ANNAS_DOWNLOAD_PATH", "/tmp/annas"),
        secret_key: Optional[str] = os.environ.get("ANNAS_SECRET_KEY"),
    ) -> None:

        self.work_path = Path(work_path).resolve()
        self.work_path.mkdir(parents=True, exist_ok=True)

        self.secret_key = secret_key
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

    def _search_catalog(self, query: str, limit: int = 20) -> List[SearchResult]:

        normalized = query.strip()
        assert normalized, "query must be non-empty"
        assert limit >= 1, "limit must be >= 1"
        logger.info("Searching '{}'... limit={}", normalized, limit)

        capped_limit = min(limit, 200)

        results = scrape_search_results(normalized, limit=capped_limit)
        return results

    def search_catalog(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search the live Anna's Archive catalog and return structured results.

        The query is stripped and validated before iterating through up to twenty
        result pages (Anna's Archive paginates search responses). Each unique md5
        hash is wrapped in a `SearchResult` model to ensure URL integrity and capture
        the published metadata (language, format, size, provenance).
        """
        return self._search_catalog(query, limit=limit)

    def download(
        self,
        md5: str,
        collection: Optional[str] = None,
        *,
        ocr_limit: Optional[float] = None,
    ) -> Path:
        """Download a file by md5, normalize artifacts, and optionally ingest chunks.

        The method validates the md5, invokes the fast download endpoint with the
        configured secret key, streams the payload into the work directory, and
        extracts auxiliary markdown. When `collection` is provided, the derived
        markdown is chunked and upserted into the named Chroma collection.

        Parameters
        ----------
        md5:
            Anna's Archive identifier for the artifact.
        collection:
            Optional Chroma collection name for ingestion.
        ocr_limit:
            Optional float between 0.0 and 1.0 bounding how much of a PDF's text can
            originate from OCR before aborting the download. This guards against
            image-only scans that produce poor downstream quality.
        """

        if ocr_limit is not None:
            assert 0.0 <= ocr_limit <= 1.0, "ocr_limit must be between 0.0 and 1.0"

        md5 = self._validate_md5(md5)
        assert (
            self.secret_key
        ), "ANNAS_SECRET_KEY environment variable is required for downloads"

        logger.info("Fetching md5={md5}", md5=md5)
        last_error: Optional[Exception] = None
        download_path: Optional[Path] = None
        path_candidates: List[Optional[int]] = [None, 1, 2]
        domain_candidates: List[Optional[int]] = [None, 1, 2, 3]
        for path_index in path_candidates:
            for domain_index in domain_candidates:
                params = {"md5": md5, "key": self.secret_key}
                if path_index is not None:
                    params["path_index"] = str(path_index)
                if domain_index is not None:
                    params["domain_index"] = str(domain_index)
                try:
                    response = self._retrying_get(
                        self._session, FAST_DOWNLOAD_ENDPOINT, params=params
                    )
                except requests.HTTPError as exc:
                    logger.warning(
                        "Fast download request failed for path {} domain {}: {}",
                        path_index or 0,
                        domain_index or 0,
                        exc,
                    )
                    last_error = exc
                    continue

                payload = response.json()
                download_url = payload.get("download_url")
                assert download_url, (
                    payload.get("error") or "fast download API returned no URL"
                )

                try:
                    download_response = self._retrying_get(
                        self._session,
                        download_url,
                        stream=True,
                        timeout=300,
                    )
                    download_path = self._write_stream(md5, download_response)
                    break
                except ArtifactValidationError as exc:
                    logger.warning(
                        "Path {} domain {} returned invalid artifact: {}",
                        path_index or 0,
                        domain_index or 0,
                        exc,
                    )
                    last_error = exc
                    continue
            if download_path is not None:
                break

        assert (
            download_path is not None
        ), f"Failed to download valid artifact: {last_error}"
        detected_extension = self._detect_extension(download_path)
        assert detected_extension is not None, "Unable to detect downloaded file format"
        document_metadata = self._document_metadata_from_path(download_path)
        strategy = "hi_res" if download_path.suffix.lower() == ".pdf" else "fast"
        # Research (2025-10-11): Passing `metadata_filename` mirrors the coverage in
        # `test_unstructured/partition/common/test_metadata.py::it_uses_metadata_filename_arg_value_when_present`,
        # ensuring downstream chunk metadata retains the sanitized filename without extra plumbing.
        elements = partition(
            filename=str(download_path),
            strategy=strategy,
            metadata_filename=document_metadata.filename,
        )

        if strategy == "hi_res" and ocr_limit is not None:
            ocr_ratio = self._compute_pdf_ocr_ratio(
                download_path,
                document_metadata.filename,
                elements,
            )
            if ocr_ratio is not None and ocr_ratio > ocr_limit:
                raise ArtifactValidationError(
                    (
                        "PDF relies on OCR for {:.1%} of extracted text, exceeding limit {:.1%}"
                    ).format(ocr_ratio, ocr_limit)
                )
            logger.debug(
                "Computed OCR ratio {:.3f} for {filename}",
                ocr_ratio or 0.0,
                filename=document_metadata.filename,
            )
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

    @staticmethod
    def _count_text_characters(elements: Sequence[ElementLike]) -> int:
        total = 0
        for element in elements:
            text = getattr(element, "text", "")
            if not text:
                continue
            stripped = str(text).strip()
            if stripped:
                total += len(stripped)
        return total

    def _compute_pdf_ocr_ratio(
        self,
        path: Path,
        metadata_filename: str,
        hi_res_elements: Sequence[ElementLike],
    ) -> Optional[float]:
        if path.suffix.lower() != ".pdf":
            return None

        hi_chars = self._count_text_characters(hi_res_elements)
        if hi_chars <= 0:
            return 1.0

        try:
            fast_elements = partition(
                filename=str(path),
                strategy="fast",
                metadata_filename=metadata_filename,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Unable to compute fast-text baseline: {}", exc)
            return 1.0

        fast_chars = self._count_text_characters(cast(Sequence[ElementLike], fast_elements))
        if fast_chars <= 0:
            return 1.0
        if fast_chars >= hi_chars:
            return 0.0
        return (hi_chars - fast_chars) / hi_chars

    def download_artifact(self, md5: str, collection: Optional[str] = None) -> Path:
        # NOTE(compat): Retain the legacy method name used by MCP tooling and tests.
        return self.download(md5, collection=collection)

    def search_downloaded_text(
        self, md5: str, needle: str, before: int = 2, after: int = 2, limit: int = 3
    ) -> str:
        """Return markdown snippets around a needle for an existing md5 artifact.

        The method loads the cached markdown (if present), scans for case-insensitive
        occurrences of `needle`, and returns up to `limit` snippets. Each snippet
        preserves a window of `before` and `after` lines from the hit, matching the
        CLI's expectation for compact context previews.
        """

        md5 = self._validate_md5(md5)
        normalized = needle.strip()
        assert normalized, "needle must be non-empty"
        assert before >= 0, "before must be non-negative"
        assert after >= 0, "after must be non-negative"
        assert limit >= 1, "limit must be >= 1"

        markdown_path = self._markdown_path_for_md5(md5)

        if not markdown_path.exists():
            return ""
        content = markdown_path.read_text(encoding="utf-8")

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

    def metadata(
        self, collection: str, n: int = 10, md5s: Optional[str] = None
    ) -> List[str]:
        # TODO: Make private collection search helper to avoid duplication (see query_collection)
        md5_filter: Optional[Where] = None
        if md5s:
            md5_list = [self._validate_md5(m) for m in md5s.split(",") if m.strip()]
            if md5_list:
                md5_filter = cast(Where, {"md5": {"$in": md5_list}})
        collection_client = self._collection(collection)
        query_result: QueryResult = collection_client.query(
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
            assembled.append(json.dumps(meta))
        return assembled

    def query_collection(
        self, collection: str, query: str, n: int = 10, md5s: Optional[str] = None
    ) -> List[str]:
        """Query a Chroma collection and format the best-matching document chunks.

        Results are constrained by `md5s` when provided, otherwise the full
        collection is searched. The raw Chroma payload is validated and rendered
        into human-readable snippets that include chunk order, inferred headings,
        and the originating md5 hash.
        """

        assert query.strip(), "query must be non-empty"
        assert n >= 1, "limit must be >= 1"
        md5_filter: Optional[Where] = None
        if md5s:
            md5_list = [self._validate_md5(m) for m in md5s.split(",") if m.strip()]
            if md5_list:
                md5_filter = cast(Where, {"md5": {"$in": md5_list}})
        # Research (2025-09-28): Chroma query returns documents as list per query index;
        # see https://docs.trychroma.com/docs/querying-collections/query-and-get
        collection_client = self._collection(collection)

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
        print(pretty_repr(metadatas))
        primary_docs: List[str] = docs[0]
        assert metadatas and len(metadatas[0]) == len(
            primary_docs
        ), "Metadata count mismatch"
        assert primary_docs, "Chroma query returned empty document set"
        assembled: List[str] = []
        for i, doc in enumerate(primary_docs):
            # Make beautiful heading
            meta = metadatas[0][i] or {}
            doc_id = query_result["ids"][0][i]
            base_title = meta.get("title") or "Untitled"
            chapter = meta.get("chapter")
            heading = base_title if not chapter else f"{base_title} — {chapter}"
            snippet = (
                f"─────────────────── Chunk {i+1} - {heading} ({doc_id}) ───────────────────\n\n"
                f"{doc.strip()}\n"
            )
            assembled.append(snippet)
        return assembled

    def _elements_to_markdown(self, elements: Sequence[ElementLike]) -> str:
        lines: List[str] = []
        last_page: Optional[int] = None

        for raw_element in elements:
            raw_text = getattr(raw_element, "text", "")
            if not raw_text or not str(raw_text).strip():
                continue

            metadata = raw_element.metadata
            assert isinstance(
                metadata, ElementMetadata
            ), "element.metadata must be ElementMetadata"

            page = metadata.page_number
            if page is not None and page != last_page:
                lines.append(f"<!-- page {page} -->")
                lines.append("")
                last_page = page

            category = getattr(raw_element, "category", "").lower()
            element = cast(Element, raw_element)
            # Research (2025-10-11): `element_to_md` mirrors the staging helpers exercised in
            # `test_unstructured/staging/test_base.py::test_element_to_md_conversion`, letting us
            # lean on unstructured's markdown serialization instead of maintaining bespoke logic
            # for tables, images, and other element variants.
            if category == "title" or isinstance(element, Title):
                depth = metadata.category_depth or 1
                depth = max(1, min(int(depth), 6))
                heading = str(raw_text).strip()
                if heading:
                    lines.append(f"{'#' * depth} {heading}")
                    lines.append("")
                continue

            rendered: str
            if isinstance(element, Element):
                rendered = element_to_md(element).strip()
            else:
                rendered = str(raw_text).strip()
            if not rendered:
                continue
            if (category in {"list_item", "listitem"} or isinstance(element, ListItem)) and not rendered.startswith("- "):
                rendered = f"- {rendered}"
            lines.append(rendered)
            lines.append("")

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
        document_metadata: Optional[DocumentMetadata] = None,
    ) -> None:
        coll = self._collection(collection)
        existing = coll.get(where={"md5": md5})
        ids = existing["ids"]
        if ids:
            coll.delete(ids=ids)

        typed_elements = [cast(Element, element) for element in elements]
        if not typed_elements:
            return

        document_page_numbers: set[int] = set()
        raw_title = document_metadata.title if document_metadata else None
        doc_title = raw_title.strip() if isinstance(raw_title, str) else ""
        raw_author = document_metadata.author if document_metadata else None
        author_name = raw_author.strip() if isinstance(raw_author, str) else ""
        raw_extras = document_metadata.extras if document_metadata else []
        tag_segments = list(raw_extras)
        raw_format = document_metadata.format if document_metadata else ""
        format_label = raw_format.strip().upper() if isinstance(raw_format, str) else ""
        fallback_suffix = (raw_format or "bin").strip().lower() or "bin"
        filename_value = (
            document_metadata.filename
            if document_metadata and isinstance(document_metadata.filename, str)
            else f"{md5}.{fallback_suffix}"
        )
        size_value = document_metadata.size_bytes if document_metadata else None
        size_bytes = size_value if isinstance(size_value, int) else None
        title = doc_title
        title_found = False
        for element in typed_elements:
            metadata = element.metadata
            assert isinstance(metadata, ElementMetadata), "element.metadata must be ElementMetadata"
            if metadata.page_number is not None:
                document_page_numbers.add(metadata.page_number)
            if not title_found and isinstance(element, Title) and element.text.strip():
                candidate = element.text.strip()
                if not self._is_generic_heading(candidate) and not self._looks_like_chapter(candidate):
                    title = candidate
                    title_found = True

        # Research (2025-10-11): `chunk_elements` mirrors the behaviors covered in
        # `test_unstructured/chunking/test_basic.py`, giving us overlap-aware text windows without
        # hand-rolled splitters.
        chunks = chunk_elements(
            typed_elements,
            include_orig_elements=True,
            max_characters=1200,
            overlap=200,
        )
        assert chunks, "No ingestable chunks produced for Chroma ingestion"

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for chunk in chunks:
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            orig_elements = list(chunk.metadata.orig_elements or [])
            chunk_pages = {
                el.metadata.page_number
                for el in orig_elements
                if getattr(el, "metadata", None) is not None
                and isinstance(el.metadata, ElementMetadata)
                and el.metadata.page_number is not None
            }

            chunk_index = len(documents) + 1
            documents.append(chunk_text)
            chunk_metadata: Dict[str, str] = {
                "md5": md5,
                "chunk_index": f"{chunk_index}",
                "title": title or doc_title or "Untitled",
                "filename": filename_value,
            }
            if author_name:
                chunk_metadata["author"] = author_name
            if tag_segments:
                chunk_metadata["tags"] = "|".join(tag_segments)
            if format_label:
                chunk_metadata["format"] = format_label
            if isinstance(size_bytes, int):
                chunk_metadata["size_bytes"] = str(size_bytes)
            if chunk_pages:
                chunk_metadata["page_start"] = str(min(chunk_pages))
                chunk_metadata["page_end"] = str(max(chunk_pages))
            if document_page_numbers:
                chunk_metadata["pages_total"] = str(len(document_page_numbers))

            chapter = next(
                (
                    el.text.strip()
                    for el in orig_elements
                    if isinstance(el, Title)
                    and el.text.strip()
                    and not self._is_generic_heading(el.text)
                ),
                None,
            )
            if not chapter:
                chapter = self._extract_chapter(chunk_text)
            if chapter and chapter.lower() != chunk_metadata["title"].lower():
                chunk_metadata["chapter"] = chapter

            metadatas.append(chunk_metadata)

        assert documents, "All ingestable chunks were empty after processing"
        chunk_ids = [f"{md5}:{index:04d}" for index in range(1, len(documents) + 1)]
        coll.upsert(
            ids=chunk_ids,
            documents=documents,
            metadatas=cast(List[Metadata], metadatas),
        )

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
                extracted_suffix = tmp_file.suffix or ".html"
                converted_path = target_file.with_suffix(extracted_suffix)
                if converted_path.exists():
                    converted_path.unlink()
                shutil.move(str(tmp_file), str(converted_path))
                # Preserve associated assets if the extractor produced directories.
                for item in Path(tempdir).iterdir():
                    if item == tmp_file or not item.exists():
                        continue
                    destination = converted_path.parent / item.name
                    if destination.exists():
                        continue
                    shutil.move(str(item), str(destination))
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
            assert self._extensions_match(expected_extension, detected_extension), (
                "Downloaded file extension mismatch",
                expected_extension,
                detected_extension,
            )
        self._assert_readable_container(target_file, expected_extension)
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
    def _extensions_match(expected: str, detected: str) -> bool:
        expected_lower = expected.lower()
        detected_lower = detected.lower()
        if detected_lower == expected_lower:
            return True
        # EPUB/DOCX ship as ZIP containers; some uploads miss proper metadata.
        if detected_lower == "zip" and expected_lower in {"epub", "docx"}:
            return True
        return False

    @staticmethod
    def _assert_readable_container(path: Path, extension: str) -> None:
        lowered = extension.lower()
        if lowered in {"epub", "docx"}:
            import zipfile

            if not zipfile.is_zipfile(path):
                raise ArtifactValidationError(
                    f"Corrupt {lowered} container: {path.name}"
                )

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
            if segment.strip().lower() in {"null", "unknown", "n/a", "na"}:
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
            file_type = detect_filetype(str(path))
        except Exception:  # pylint: disable=broad-except
            file_type = None

        if file_type is not None:
            # Research (2025-10-11): `detect_filetype` is validated across archive/media variants
            # in `test_unstructured/file_utils/test_filetype.py`; using its canonical extension
            # avoids our bespoke header sniffing.
            primary_ext = next(iter(getattr(file_type, "_extensions", [])), None)
            if primary_ext:
                return primary_ext.lstrip(".").lower()

        suffix = path.suffix.lower().lstrip(".")
        return suffix or None

    @staticmethod
    def _deslug(value: str) -> str:
        words = value.replace("_", " ").strip()
        return words.title() if words else ""

    def _document_metadata_from_path(self, path: Path) -> DocumentMetadata:

        stem, extension = self._split_extension(path.name)
        components = [part for part in stem.split("__") if part]
        size_bytes = path.stat().st_size if path.exists() else None
        format_label = extension or path.suffix.lstrip(".").lower() or "bin"
        if not components:
            return DocumentMetadata(
                author=None,
                title=path.stem,
                filename=path.name,
                extras=[],
                format=format_label,
                size_bytes=size_bytes,
            )
        author_slug = components[0]
        title_slug = components[1] if len(components) > 1 else components[0]
        extras = components[2:]
        normalized_author = self._deslug(author_slug) or None
        normalized_title = self._deslug(title_slug) or path.stem
        normalized_extras = [self._deslug(part) for part in extras if part]

        return DocumentMetadata(
            author=normalized_author,
            title=normalized_title,
            filename=path.name,
            extras=normalized_extras,
            format=format_label,
            size_bytes=size_bytes,
        )

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
            "copyright",
            "copyright notice",
            "all rights reserved",
            "license",
            "project gutenberg",
            "cover",
            "title page",
            "credits",
            "about this ebook",
        }
        return lowered in generics

    @staticmethod
    def _looks_like_chapter(value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if lowered.startswith(("chapter", "chap.")):
            return True
        # Research (2025-10-11): `ENUMERATED_BULLETS_RE` and `NUMBERED_LIST_RE` are the same
        # markers the core partitioners rely on for section detection (see
        # `unstructured/partition/common/common.py`), so piggybacking on them keeps our heuristics
        # aligned with upstream chapter parsing.
        if NUMBERED_LIST_RE.match(stripped):
            return True
        if ENUMERATED_BULLETS_RE.match(stripped):
            return True
        return bool(re.match(r"(?i)(?:book|part|section)\s+[ivxlcdm\d]+", stripped))

    def _markdown_path_for_md5(self, md5: str) -> Path:
        target_dir = self._artifact_dir(md5)
        candidates = sorted(target_dir.glob("*.md"))
        if not candidates:
            return target_dir / "converted.md"
        return candidates[0]


if __name__ == "__main__":
    fire.Fire(Annas)
