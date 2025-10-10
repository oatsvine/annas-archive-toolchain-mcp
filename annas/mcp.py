"""
Anna's Archive operations follow a fixed business loop: researchers first survey
the catalog to locate promising texts, then fetch the definitive files, ingest
them into a shared vector store to power downstream retrieval augmented
generation, sanity-check key passages directly from the cached markdown, and
finally issue semantic similarity queries when drafting deliverables. This MCP
server exposes that workflow end-to-end so assistants can move from discovery to
grounded citation with a single toolkit.

FastMCP represents each workflow step as a tool (https://gofastmcp.com/servers/tools):
decorating a callable registers it, the function name and docstring define the
surface presented to clients, and type annotations generate the JSON schema used
for validation and structured output. The server below reuses the CLI-proven
``Annas`` methods, keeps one instance alive for shared state, and routes every
call through ``asyncio.to_thread`` while capturing request-scoped telemetry with
the provided ``Context``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional
import fire
from fastmcp import Context, FastMCP

from annas.cli import Annas
from annas.scrape import SearchResult

DEFAULT_SERVER_NAME = "AnnasArchiveMCP"
DEFAULT_INSTRUCTIONS = (
    "Start by calling search_catalog to shortlist candidate titles, then use "
    "download_artifact to retrieve the chosen work and (optionally) ingest it "
    "into a vector store collection. With files on disk, confirm key excerpts "
    "via search_downloaded_text, and finish by querying the assembled collection "
    "through query_collection for similarity-ranked context."
)


def build_server(
    annas: Annas,
    name: str = DEFAULT_SERVER_NAME,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> FastMCP:
    """Create a FastMCP server exposing the Annas CLI surface."""
    server = FastMCP(name=name, instructions=instructions)

    @server.tool
    async def search_catalog(
        query: str,
        ctx: Context,
        limit: int = 200,
    ) -> List[SearchResult]:
        """Discover candidate books on Anna's Archive before committing to a download."""
        await ctx.info(f"Searching catalog for {query!r} (limit={limit})")
        return await asyncio.to_thread(annas.search_catalog, query, limit)

    @server.tool
    async def download_artifact(
        md5: str,
        ctx: Context,
        collection: Optional[str] = None,
    ) -> Path:
        """Retrieve the selected book and optionally ingest its markdown into a vector collection."""
        await ctx.info(
            f"Downloading md5={md5}" + (f" into collection={collection}" if collection else "")
        )
        return await asyncio.to_thread(annas.download_artifact, md5, collection)

    @server.tool
    async def search_downloaded_text(
        md5: str,
        needle: str,
        ctx: Context,
        before: int = 2,
        after: int = 2,
        limit: int = 3,
    ) -> str:
        """Extract a contextual snippet from the cached markdown to validate key passages."""
        await ctx.info(
            f"Searching downloaded text for {needle!r} within {md5} "
            f"(window before={before}, after={after}, limit={limit})"
        )
        return await asyncio.to_thread(
            annas.search_downloaded_text,
            md5,
            needle,
            before,
            after,
            limit,
        )

    @server.tool
    async def query_collection(
        collection: str,
        query: str,
        n: int,
        ctx: Context,
        md5s: Optional[str] = None,
    ) -> List[str]:
        """Run a similarity search against the populated collection to supply grounded context."""
        await ctx.info(
            f"Querying collection={collection} for {query!r} "
            f"(n={n}{', filter=' + md5s if md5s else ''})"
        )
        return await asyncio.to_thread(
            annas.query_collection,
            collection,
            query,
            n,
            md5s,
        )

    return server


def run_server(
    work_path: Path | str = os.environ.get("ANNAS_DOWNLOAD_PATH", ""),
    secret_key: Optional[str] = None,
    name: str = DEFAULT_SERVER_NAME,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> None:
    annas = Annas(work_path=Path(work_path).resolve(), secret_key=secret_key)
    server = build_server(annas=annas, name=name, instructions=instructions)
    server.run()


if __name__ == "__main__":
    fire.Fire(run_server)
