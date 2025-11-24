"""Utility helpers and Typer CLI for interacting with Anna's Archive content.

The module exposes a Typer-based CLI that can search Anna's Archive, download and
normalize artifacts, derive markdown snapshots, and push document chunks into a
Chroma vector store. All helpers follow the in-house convention of typed,
fail-fast surface areas.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence, Tuple, cast
from urllib.parse import unquote, urlsplit

import backoff
import certifi
from loguru import logger
from pydantic import TypeAdapter
from annas.util import compute_pdf_ocr_ratio, elements_to_markdown
import mobi
import requests
import typer
from rich import box
from rich.console import Console
from rich.pretty import pretty_repr
from rich.table import Table
from rich.text import Text
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
    write_chunk_snapshot,
    store_app,
)

FAST_DOWNLOAD_ENDPOINT = f"{ANNAS_BASE_URL}/dyn/api/fast_download.json"
MD5_PATTERN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
ANNA_BRAND_PATTERN = re.compile(r"anna[’']?s archive", re.IGNORECASE)

console = Console()


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
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit raw JSON instead of a table (useful for scripting)",
        ),
    ] = False,
) -> List[SearchResult]:
    """Search the live Anna's Archive catalog and return structured results."""

    normalized = query.strip()
    assert normalized, "query must be non-empty"
    assert limit >= 1, "limit must be >= 1"
    logger.info("Searching '{}'... limit={}", normalized, limit)

    capped_limit = min(limit, 200)
    results = scrape_search_results(normalized, limit=capped_limit)
    if output_json:
        for entry in results:
            print(entry.model_dump_json())
        logger.debug("Emitted JSONL for {} results", len(results))
        return results
    table = Table(
        title="Search Results",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        row_styles=["none", "dim"],
    )
    table.add_column("Title", overflow="fold", max_width=40, ratio=3)
    table.add_column("Format / Size", ratio=1, max_width=16)
    table.add_column("Lang", width=6, justify="center")
    table.add_column("Year", width=6, justify="center")
    table.add_column("Category", overflow="fold", max_width=24, ratio=2)
    table.add_column("MD5", width=34, overflow="ellipsis")

    for entry in results:
        size_label = entry.file_size_label or (
            f"{entry.file_size_bytes/1_048_576:.1f}MB"
            if entry.file_size_bytes is not None
            else "?"
        )
        fmt_label = entry.file_format or "?"
        format_size = f"{fmt_label.upper()} / {size_label}"
        lang = entry.language_code or entry.language or "?"
        year = str(entry.year) if entry.year is not None else "—"
        category = entry.category or ""
        title_text = Text(entry.title or "Untitled", overflow="fold")
        if entry.url:
            title_text.stylize(f"link {entry.url}")
        table.add_row(title_text, format_size, lang, year, category, entry.md5)

    console.print(table)
    logger.debug("Rendered {} results", len(results))
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
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit raw JSON instead of a table (useful for scripting)",
        ),
    ] = False,
) -> Path:
    """Download a file by md5, normalize artifacts, and optionally ingest chunks."""

    if ocr_limit is not None:
        assert 0.0 <= ocr_limit <= 1.0, "ocr_limit must be between 0.0 and 1.0"

    md5_value = _validate_md5(md5)

    session = _build_session()

    logger.info("Fetching md5={md5}", md5=md5_value)
    logger.debug("Preparing download target dir {}", work_dir / md5_value)
    last_error: Optional[Exception] = None
    download_path: Optional[Path] = None
    raw_path: Optional[Path] = None
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
                logger.debug(
                    "Requesting fast-download URL path={} domain={} params={} ",
                    path_index,
                    domain_index,
                    params,
                )
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
                logger.debug("Streaming payload from {}", download_url)
                download_response = _retrying_get(
                    session,
                    download_url,
                    stream=True,
                    timeout=300,
                )
                artifact_dir = work_dir / md5_value
                artifact_dir.mkdir(parents=True, exist_ok=True)
                download_path, raw_path = _write_stream(
                    md5_value, download_response, artifact_dir
                )
                logger.debug("Raw artifact stored at {}", raw_path)
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
    assert raw_path is not None, f"Failed to retain raw artifact: {last_error}"
    detected_extension = _detect_extension(download_path)
    assert detected_extension is not None, "Unable to detect downloaded file format"
    document_metadata = _document_metadata_from_path(download_path)
    logger.debug("Detected document metadata {}", document_metadata)
    strategy = "hi_res" if download_path.suffix.lower() == ".pdf" else "fast"
    elements = partition(
        filename=str(download_path),
        strategy=strategy,
        metadata_filename=document_metadata.filename,
    )

    if strategy == "hi_res" and ocr_limit is not None:
        ocr_ratio = compute_pdf_ocr_ratio(
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
    markdown = elements_to_markdown(elements)
    markdown_path = _first_markdown_path(work_dir, md5_value, download_path.stem)
    markdown_path.write_text(markdown, encoding="utf-8")
    chunk_path = write_chunk_snapshot(work_dir, md5_value, elements, document_metadata)
    if collection:
        load_elements(
            work_dir,
            md5_value,
            elements,
            collection,
            markdown,
            document_metadata,
        )
    logger.info("Download complete", md5=md5_value)
    if output_json:
        payload = TypeAdapter(Dict[str, str]).dump_json(
            {
                "markdown": str(markdown_path),
                "normalized": str(download_path),
                "raw": str(raw_path),
                "chunks": str(chunk_path),
            }
        )
        print(payload.decode("utf-8"))
        return work_dir / md5_value
    artifact_table = Table(
        title="Saved Artifacts",
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold green",
    )
    artifact_table.add_column("Artifact", style="bold")
    artifact_table.add_column("Path", overflow="fold")
    artifact_table.add_row("Markdown", str(markdown_path))
    artifact_table.add_row("Normalized", str(download_path))
    artifact_table.add_row("Raw download", str(raw_path))
    artifact_table.add_row("Chunks (jsonl)", str(chunk_path))
    console.print(artifact_table)
    return work_dir / md5_value


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

    logger.debug("Searching markdown for md5={} in {}", md5, work_dir)
    md5_value = _validate_md5(md5)
    normalized = needle.strip()
    assert normalized, "needle must be non-empty"
    assert before >= 0, "before must be non-negative"
    assert after >= 0, "after must be non-negative"
    assert limit >= 1, "limit must be >= 1"

    markdown_path = _first_markdown_path(work_dir, md5_value)

    if not markdown_path.exists():
        console.print("", markup=False)
        return ""
    content = markdown_path.read_text(encoding="utf-8")

    lines = content.splitlines()
    if not lines:
        console.print("", markup=False)
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
    console.print(rendered, markup=False)
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


# ---- Anna's Archive-specific internal helpers ---- #


def _validate_md5(md5: str) -> str:
    candidate = md5.strip().lower()
    assert re.fullmatch(r"[0-9a-f]{32}", candidate), f"Invalid md5 hash: {md5}"
    return candidate


def _write_stream(
    md5: str, response: requests.Response, target_dir: Path
) -> Tuple[Path, Path]:
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = _filename_from_response(md5, response)
    raw_file = raw_dir / filename
    logger.debug("Writing stream to {} (chunked)", raw_file)
    logger.info("Writing download to {path}", path=str(raw_file))
    with raw_file.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1_048_576):
            if chunk:
                fh.write(chunk)

    normalized_dir = target_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    final_path = raw_file

    if raw_file.suffix.lower() in [".mobi", ".azw", ".azw3"]:
        try:
            tempdir, name = mobi.extract(str(raw_file))
            tmp_file = Path(tempdir) / name
            logger.info(f"Extracted {raw_file.name} => {tmp_file.name}")
            assert tmp_file.exists(), f"Extracted file missing: {tmp_file}"
            extracted_suffix = tmp_file.suffix or ".html"
            converted_name = raw_file.with_suffix(extracted_suffix).name
            converted_path = normalized_dir / converted_name
            if converted_path.exists():
                converted_path.unlink()
            shutil.move(str(tmp_file), str(converted_path))
            for item in Path(tempdir).iterdir():
                if item == tmp_file or not item.exists():
                    continue
                destination = normalized_dir / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            final_path = converted_path
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("MOBI conversion failed: {}", exc)
            final_path = raw_file
    else:
        candidate = normalized_dir / raw_file.name
        if not candidate.exists():
            shutil.copy2(raw_file, candidate)
        final_path = candidate

    detected_extension = _detect_extension(final_path)
    expected_extension = final_path.suffix.lstrip(".").lower()
    assert (
        expected_extension
    ), "Downloaded file is missing an extension in the Content-Disposition header"
    if detected_extension is not None:
        assert _extensions_match(expected_extension, detected_extension), (
            "Downloaded file extension mismatch",
            expected_extension,
            detected_extension,
        )
    _assert_readable_container(final_path, expected_extension)
    return final_path, raw_file


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

    if slug_components:
        slug_components[0] = _trim_tokens(slug_components[0], 4)
    if len(slug_components) > 1:
        slug_components[1] = _trim_tokens(slug_components[1], 8)

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
    max_segment_length = 48
    return slug[:max_segment_length]


def _trim_tokens(slug: str, max_tokens: int) -> str:
    parts = [segment for segment in slug.split("_") if segment]
    return "_".join(parts[:max_tokens]) if parts else slug


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


def _first_markdown_path(work_dir: Path, md5: str, stem: Optional[str] = None) -> Path:
    target_dir = work_dir / md5
    markdown_dir = target_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(markdown_dir.glob("*.md"))
    if candidates:
        return candidates[0]
    if stem:
        return markdown_dir / f"{stem}.md"
    fallback_candidates = sorted(target_dir.glob("*.md"))
    if fallback_candidates:
        return fallback_candidates[0]
    return markdown_dir / "converted.md"


if __name__ == "__main__":
    app()
