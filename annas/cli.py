"""Utility helpers and Typer CLI for interacting with Anna's Archive content.

The module exposes a Typer-based CLI that can search Anna's Archive, download and
normalize artifacts, derive markdown snapshots, and push document chunks into a
Chroma vector store. All helpers follow the in-house convention of typed,
fail-fast surface areas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from loguru import logger
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from annas.common import work_dir_callback
from annas.core import (
    download as core_download,
    search_catalog as core_search_catalog,
    search_downloaded_text as core_search_downloaded_text,
)
from annas.scrape import SearchResult
from annas.store import (
    StoredChunk,
    fetch_metadata,
    get_or_create_chroma,
    load_elements,
    query_chunks,
    write_chunk_snapshot,
)
console = Console()

app = typer.Typer(help="CLI for Anna's Archive helpers.")
store_app = typer.Typer(name="store", help="Chroma-backed metadata and query helpers.")
app.add_typer(store_app, name="store")


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

    logger.info("Searching '{}'... limit={}", query, limit)
    results = core_search_catalog(query, limit=limit)
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

    result = core_download(
        md5=md5,
        work_dir=work_dir,
        secret_key=secret_key,
        ocr_limit=ocr_limit,
    )
    chunk_path = write_chunk_snapshot(
        work_dir, result.md5, result.elements, result.document_metadata
    )
    if collection:
        client = get_or_create_chroma(work_dir)
        load_elements(
            work_dir,
            result.md5,
            result.elements,
            collection,
            result.document_metadata,
            client=client,
        )
    logger.info("Download complete", md5=result.md5)
    if output_json:
        payload: Dict[str, str] = {
            "markdown": str(result.markdown_path),
            "normalized": str(result.normalized_path),
            "raw": str(result.raw_path),
            "chunks": str(chunk_path),
            "detected_extension": result.detected_extension,
        }
        print(json.dumps(payload))
        return result.normalized_path
    artifact_table = Table(
        title="Saved Artifacts",
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold green",
    )
    artifact_table.add_column("Artifact", style="bold")
    artifact_table.add_column("Path", overflow="fold")
    artifact_table.add_row("Markdown", str(result.markdown_path))
    artifact_table.add_row("Normalized", str(result.normalized_path))
    artifact_table.add_row("Raw download", str(result.raw_path))
    artifact_table.add_row("Chunks (jsonl)", str(chunk_path))
    console.print(artifact_table)
    return result.normalized_path


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

    rendered = core_search_downloaded_text(
        md5=md5,
        needle=needle,
        work_dir=work_dir,
        before=before,
        after=after,
        limit=limit,
    )
    console.print(rendered, markup=False)
    return rendered


def _render_metadata_table(chunks: List[StoredChunk], *, title: str) -> None:
    table = Table(
        title=title,
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold cyan",
        row_styles=["none", "dim"],
    )
    table.add_column("Chunk ID", style="bold", width=18, overflow="ellipsis")
    table.add_column("Title", ratio=2, overflow="fold")
    table.add_column("Chapter", ratio=1, overflow="fold")
    table.add_column("Pages", width=12)
    table.add_column("Filename", ratio=1, overflow="fold")
    for chunk in chunks:
        meta = chunk.metadata
        table.add_row(
            chunk.id,
            meta.get("title", "Untitled"),
            meta.get("chapter", ""),
            _format_pages(meta),
            meta.get("filename", ""),
        )
    console.print(table)


def _render_query_table(chunks: List[StoredChunk]) -> None:
    table = Table(
        title="Query Results",
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold green",
        row_styles=["none", "dim"],
    )
    table.add_column("Chunk ID", style="bold", width=18, overflow="ellipsis")
    table.add_column("Title", ratio=1, overflow="fold")
    table.add_column("Chapter", ratio=1, overflow="fold")
    table.add_column("Snippet", ratio=3, overflow="fold")
    for chunk in chunks:
        meta = chunk.metadata
        table.add_row(
            chunk.id,
            meta.get("title", "Untitled"),
            meta.get("chapter", ""),
            chunk.text.strip(),
        )
    console.print(table)


def _format_pages(meta: Dict[str, str]) -> str:
    start = meta.get("page_start")
    end = meta.get("page_end")
    if start and end:
        return f"{start}-{end}"
    return meta.get("pages_total", "") or ""


def _parse_md5s(md5s: Optional[str]) -> Optional[List[str]]:
    if not md5s:
        return None
    values = []
    for token in md5s.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        assert re.fullmatch(r"[0-9a-f]{32}", cleaned), f"Invalid md5 hash: {token}"
        values.append(cleaned)
    return values or None


@store_app.command()
def metadata(
    collection_name: Annotated[
        str, typer.Argument(help="Chroma collection name to inspect")
    ],
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
    n: Annotated[
        int, typer.Option("--n", min=1, help="Number of documents to return")
    ] = 10,
    md5s: Annotated[
        Optional[str],
        typer.Option(
            "--md5s", help="Comma-separated md5s to filter results", show_default=False
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit raw JSON instead of a table (useful for scripting)",
        ),
    ] = False,
) -> List[StoredChunk]:
    """Return metadata rows from a Chroma collection."""

    client = get_or_create_chroma(work_dir)
    rows = fetch_metadata(client, collection_name, n=n, md5s=_parse_md5s(md5s))
    if output_json:
        for row in rows:
            print(row.model_dump_json())
        return rows
    _render_metadata_table(rows, title=f"Collection: {collection_name}")
    return rows


@store_app.command(name="query-collection")
def query_collection(
    collection: Annotated[str, typer.Argument(help="Chroma collection name to query")],
    query: Annotated[str, typer.Argument(help="Similarity search text")],
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
    n: Annotated[
        int, typer.Option("--n", min=1, help="Number of matching chunks to return")
    ] = 10,
    md5s: Annotated[
        Optional[str],
        typer.Option(
            "--md5s", help="Comma-separated md5s to filter results", show_default=False
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit raw JSON instead of a table (useful for scripting)",
        ),
    ] = False,
) -> List[StoredChunk]:
    """Query a Chroma collection and format the best-matching document chunks."""

    client = get_or_create_chroma(work_dir)
    chunks = query_chunks(client, collection, query, n=n, md5s=_parse_md5s(md5s))
    if output_json:
        for chunk in chunks:
            print(chunk.model_dump_json())
        return chunks
    _render_query_table(chunks)
    return chunks


if __name__ == "__main__":
    app()
