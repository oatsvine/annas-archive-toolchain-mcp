"""FastMCP server exposing core Anna's Archive tools."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

from annas.core import (
    download as core_download,
    search_catalog as core_search_catalog,
    search_downloaded_text as core_search_downloaded_text,
)
from annas.scrape import SearchResult

mcp = FastMCP("Anna's Archive MCP")


def _work_dir(path: Path | None) -> Path:
    """MCP lacks a CLI callback; create a shared work dir on first use."""
    target = path or (Path.cwd() / "annas")
    target.mkdir(parents=True, exist_ok=True)
    return target


@mcp.tool
def search_catalog(query: str, limit: int = 20) -> List[SearchResult]:
    """Search Anna's Archive catalog."""

    return core_search_catalog(query, limit=limit)


@mcp.tool
def download(
    md5: str,
    secret_key: str,
    work_dir: Path | None = None,
    ocr_limit: Optional[float] = None,
) -> dict:
    """Download and normalize an artifact."""

    result = core_download(
        md5=md5,
        work_dir=_work_dir(work_dir),
        secret_key=secret_key,
        ocr_limit=ocr_limit,
    )
    return result.model_dump(exclude={"elements"}, mode="json")


@mcp.tool
def search_downloaded_text(
    md5: str,
    needle: str,
    work_dir: Path | None = None,
    before: int = 2,
    after: int = 2,
    limit: int = 3,
) -> str:
    """Return markdown snippets around a needle for an existing artifact."""

    return core_search_downloaded_text(
        md5=md5,
        needle=needle,
        work_dir=_work_dir(work_dir),
        before=before,
        after=after,
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run()
