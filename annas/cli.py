"""Utility helpers and Typer CLI for interacting with Anna's Archive content.

The module exposes a Typer-based CLI that can search Anna's Archive, download and
normalize artifacts, derive markdown snapshots, and push document chunks into a
Chroma vector store. All helpers follow the in-house convention of typed,
fail-fast surface areas.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence, cast
from urllib.parse import unquote, urlsplit

import backoff
import certifi
import mobi
import requests
import typer
from loguru import logger
from rich.pretty import pretty_repr
from unstructured.documents.elements import Element, ElementMetadata, ListItem, Title
from unstructured.file_utils.filetype import detect_filetype
from unstructured.partition.auto import partition
from unstructured.staging.base import element_to_md

from annas.common import work_dir_callback
from annas.scrape import ANNAS_BASE_URL, SearchResult, scrape_search_results
from annas.store import (
    DocumentMetadata,
    ElementLike,
    load_elements,
    store_app,
)

FAST_DOWNLOAD_ENDPOINT = f"{ANNAS_BASE_URL}/dyn/api/fast_download.json"
MD5_PATTERN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
ANNA_BRAND_PATTERN = re.compile(r"anna[’']?s archive", re.IGNORECASE)


class ArtifactValidationError(RuntimeError):
    """Raised when a downloaded artifact fails structural validation."""


app = typer.Typer(help="CLI for Anna's Archive helpers.")
app.add_typer(store_app, name="store")


def _build_session() -> requests.Session:
    """Construct a requests session with repo-standard headers."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "audio-agents-annas/0.1 (+https://annas-archive.org)",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


@app.command()
def search_catalog(
    query: Annotated[str, typer.Argument(help="Query string to search Anna's Archive")],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=200,
            help="Maximum number of results to return (capped at 200)",
        ),
    ] = 20,
) -> List[SearchResult]:
    """Search the live Anna's Archive catalog and return structured results."""

    normalized = query.strip()
    assert normalized, "query must be non-empty"
    assert limit >= 1, "limit must be >= 1"
    logger.info("Searching '{}'... limit={}", normalized, limit)

    capped_limit = min(limit, 200)
    results = scrape_search_results(normalized, limit=capped_limit)
    typer.echo(pretty_repr(results))
    return results


@app.command()
def download(
    md5: Annotated[str, typer.Argument(help="Anna's Archive md5 identifier")],
    work_dir: Annotated[
        Path,
        typer.Option(
            envvar="ANNAS_DOWNLOAD_PATH",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
            callback=work_dir_callback,
            help="Directory for downloads and derived artifacts",
        ),
    ],
    secret_key: Annotated[
        str,
        typer.Option(
            envvar="ANNAS_SECRET_KEY",
            help="Secret key for fast download API",
            show_default=False,
        ),
    ],
    collection: Annotated[
        Optional[str],
        typer.Option(help="Optional Chroma collection name for ingestion"),
    ] = None,
    ocr_limit: Annotated[
        Optional[float],
        typer.Option(
            help="Abort PDFs when OCR text exceeds this fraction",
            min=0.0,
            max=1.0,
        ),
    ] = None,
) -> Path:
    """Download a file by md5, normalize artifacts, and optionally ingest chunks."""

    if ocr_limit is not None:
        assert 0.0 <= ocr_limit <= 1.0, "ocr_limit must be between 0.0 and 1.0"

    md5_value = _validate_md5(md5)

    session = _build_session()

    logger.info("Fetching md5={md5}", md5=md5_value)
    last_error: Optional[Exception] = None
    download_path: Optional[Path] = None
    path_candidates: List[Optional[int]] = [None, 1, 2]
    domain_candidates: List[Optional[int]] = [None, 1, 2, 3]
    for path_index in path_candidates:
        for domain_index in domain_candidates:
            params = {"md5": md5_value, "key": secret_key}
            if path_index is not None:
                params["path_index"] = str(path_index)
            if domain_index is not None:
                params["domain_index"] = str(domain_index)
            try:
                response = _retrying_get(session, FAST_DOWNLOAD_ENDPOINT, params=params)
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
                download_response = _retrying_get(
                    session,
                    download_url,
                    stream=True,
                    timeout=300,
                )
                artifact_dir = work_dir / md5_value
                artifact_dir.mkdir(parents=True, exist_ok=True)
                download_path = _write_stream(
                    md5_value, download_response, artifact_dir
                )
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

    assert download_path is not None, f"Failed to download valid artifact: {last_error}"
    detected_extension = _detect_extension(download_path)
    assert detected_extension is not None, "Unable to detect downloaded file format"
    document_metadata = _document_metadata_from_path(download_path)
    strategy = "hi_res" if download_path.suffix.lower() == ".pdf" else "fast"
    elements = partition(
        filename=str(download_path),
        strategy=strategy,
        metadata_filename=document_metadata.filename,
    )

    if strategy == "hi_res" and ocr_limit is not None:
        ocr_ratio = _compute_pdf_ocr_ratio(
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
    markdown = _elements_to_markdown(elements)
    markdown_path = _first_markdown_path(work_dir, md5_value)
    markdown_path.write_text(markdown, encoding="utf-8")
    if collection:
        load_elements(
            work_dir,
            md5_value,
            elements,
            collection,
            markdown,
            document_metadata,
        )
    logger.info("Download complete", md5=md5_value, path=str(download_path))
    typer.echo(str(download_path))
    return download_path


@app.command()
def search_downloaded_text(
    md5: Annotated[str, typer.Argument(help="md5 to search within")],
    needle: Annotated[str, typer.Argument(help="Case-insensitive text to find")],
    work_dir: Annotated[
        Path,
        typer.Option(
            envvar="ANNAS_DOWNLOAD_PATH",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
            callback=work_dir_callback,
            help="Directory containing downloaded artifacts",
        ),
    ],
    before: Annotated[
        int,
        typer.Option("--before", min=0, help="Lines to include before each match"),
    ] = 2,
    after: Annotated[
        int,
        typer.Option("--after", min=0, help="Lines to include after each match"),
    ] = 2,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum snippets to return"),
    ] = 3,
) -> str:
    """Return markdown snippets around a needle for an existing md5 artifact."""

    md5_value = _validate_md5(md5)
    normalized = needle.strip()
    assert normalized, "needle must be non-empty"
    assert before >= 0, "before must be non-negative"
    assert after >= 0, "after must be non-negative"
    assert limit >= 1, "limit must be >= 1"

    markdown_path = _first_markdown_path(work_dir, md5_value)

    if not markdown_path.exists():
        typer.echo("")
        return ""
    content = markdown_path.read_text(encoding="utf-8")

    lines = content.splitlines()
    if not lines:
        typer.echo("")
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

    rendered = "\n\n".join(snippets)
    typer.echo(rendered)
    return rendered


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
    path: Path,
    metadata_filename: str,
    hi_res_elements: Sequence[ElementLike],
) -> Optional[float]:
    if path.suffix.lower() != ".pdf":
        return None

    hi_chars = _count_text_characters(hi_res_elements)
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

    fast_chars = _count_text_characters(cast(Sequence[ElementLike], fast_elements))
    if fast_chars <= 0:
        return 1.0
    if fast_chars >= hi_chars:
        return 0.0
    return (hi_chars - fast_chars) / hi_chars


def _elements_to_markdown(elements: Sequence[ElementLike]) -> str:
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
        if (
            category in {"list_item", "listitem"} or isinstance(element, ListItem)
        ) and not rendered.startswith("- "):
            rendered = f"- {rendered}"
        lines.append(rendered)
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    return "\n".join(lines)


def _validate_md5(md5: str) -> str:
    candidate = md5.strip().lower()
    assert re.fullmatch(r"[0-9a-f]{32}", candidate), f"Invalid md5 hash: {md5}"
    return candidate


def _artifact_dir(work_dir: Path, md5: str) -> Path:
    path = Path(work_dir) / md5
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_stream(md5: str, response: requests.Response, target_dir: Path) -> Path:
    filename = _filename_from_response(md5, response)
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
    detected_extension = _detect_extension(target_file)
    expected_extension = target_file.suffix.lstrip(".").lower()
    assert (
        expected_extension
    ), "Downloaded file is missing an extension in the Content-Disposition header"
    if detected_extension is not None:
        assert _extensions_match(expected_extension, detected_extension), (
            "Downloaded file extension mismatch",
            expected_extension,
            detected_extension,
        )
    _assert_readable_container(target_file, expected_extension)
    return target_file


def _filename_from_response(md5: str, response: requests.Response) -> str:
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
    return _sanitize_filename(filename, md5)


def _sanitize_filename(filename: str, md5: str) -> str:
    stem, extension = _split_extension(filename)
    decoded = _decode_underscore_hex(stem)
    decoded = unquote(decoded)
    decoded = unicodedata.normalize("NFKC", decoded)
    decoded = decoded.replace("\\", "_").replace("/", "_")

    trimmed = _strip_branding(decoded, md5)
    segments = _extract_segments(trimmed)

    if not segments:
        segments = [trimmed]

    title_segment = segments[0]
    author_segment = segments[1] if len(segments) > 1 else "Unknown"
    extra_segments = segments[2:]

    isbn_tokens = _extract_isbns(trimmed)

    slug_components = [
        _slug(author_segment) or "unknown",
        _slug(title_segment) or "untitled",
    ]
    slug_components.extend(_slug(segment) for segment in extra_segments)
    slug_components.extend(isbn_tokens)

    filtered = [component for component in slug_components if component]
    slug_body = "__".join(filtered) if filtered else "document"

    max_len = 120
    allowed = max(8, max_len - (len(extension) + 1 if extension else 0))
    truncated = slug_body[:allowed]
    return f"{truncated}.{extension}" if extension else truncated


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


def _split_extension(name: str) -> tuple[str, str]:
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return stem, ext.lower()
    return name, ""


def _strip_branding(value: str, md5: str) -> str:
    without_md5 = MD5_PATTERN.sub("", value)
    without_target_md5 = without_md5.replace(md5, "")
    return ANNA_BRAND_PATTERN.sub("", without_target_md5)


def _extensions_match(expected: str, detected: str) -> bool:
    expected_lower = expected.lower()
    detected_lower = detected.lower()
    if detected_lower == expected_lower:
        return True
    if detected_lower == "zip" and expected_lower in {"epub", "docx"}:
        return True
    return False


def _assert_readable_container(path: Path, extension: str) -> None:
    lowered = extension.lower()
    if lowered in {"epub", "docx"}:
        import zipfile

        if not zipfile.is_zipfile(path):
            raise ArtifactValidationError(f"Corrupt {lowered} container: {path.name}")


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


def _slug(value: str) -> str:
    normal = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in normal if not unicodedata.combining(ch))
    lowered = stripped.lower()
    lowered = lowered.replace("'", "_")
    slug = re.sub(r"[^0-9a-z]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


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


def _detect_extension(path: Path) -> Optional[str]:
    try:
        file_type = detect_filetype(str(path))
    except Exception:  # pylint: disable=broad-except
        file_type = None

    if file_type is not None:
        primary_ext = next(iter(getattr(file_type, "_extensions", [])), None)
        if primary_ext:
            return primary_ext.lstrip(".").lower()

    suffix = path.suffix.lower().lstrip(".")
    return suffix or None


def _deslug(value: str) -> str:
    words = value.replace("_", " ").strip()
    return words.title() if words else ""


def _document_metadata_from_path(path: Path) -> DocumentMetadata:
    stem, extension = _split_extension(path.name)
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
    normalized_author = _deslug(author_slug) or None
    normalized_title = _deslug(title_slug) or path.stem
    normalized_extras = [_deslug(part) for part in extras if part]

    return DocumentMetadata(
        author=normalized_author,
        title=normalized_title,
        filename=path.name,
        extras=normalized_extras,
        format=format_label,
        size_bytes=size_bytes,
    )


def _first_markdown_path(work_dir: Path, md5: str) -> Path:
    target_dir = work_dir / md5
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(target_dir.glob("*.md"))
    if not candidates:
        return target_dir / "converted.md"
    return candidates[0]


if __name__ == "__main__":
    app()
