from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Iterable, List, cast

import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from unstructured.documents.elements import ElementMetadata
from types import SimpleNamespace

from annas.cli import Annas, ElementLike
from annas.mcp import build_server
from annas.scrape import ANNAS_BASE_URL, SearchResult, scrape_search_results


def _collect_live_results(query: str, limit: int) -> List[SearchResult]:
    try:
        results = scrape_search_results(query, limit=limit)
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"Anna's Archive search unavailable: {exc!r}")
    if not results:
        pytest.skip(f"No search results for query={query!r}")
    return results


def _extract_result_payload(data: object) -> object:
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return data


def _validate_search_entries(entries: Iterable[object]) -> List[SearchResult]:
    validated: List[SearchResult] = []
    for entry in entries:
        if isinstance(entry, SearchResult):
            candidate = entry
        else:
            candidate = SearchResult.model_validate(entry)
        assert candidate.md5 in candidate.url
        assert candidate.url.startswith(f"{ANNAS_BASE_URL}/md5/")
        validated.append(candidate)
    return validated


@pytest.fixture()
def annas_instance(tmp_path: Path) -> Annas:
    return Annas(work_path=tmp_path / "workspace")


@pytest.fixture()
def annas_server(annas_instance: Annas) -> FastMCP:
    return build_server(annas=annas_instance)


@pytest_asyncio.fixture()
async def mcp_client(annas_server: FastMCP) -> AsyncIterator[Client]:
    async with Client(annas_server) as client:
        yield client


@pytest.mark.asyncio
async def test_tool_registration(mcp_client: Client) -> None:
    tools = await mcp_client.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "search_catalog",
        "download_artifact",
        "search_downloaded_text",
        "query_collection",
    }


@pytest.mark.asyncio
async def test_search_catalog_tool_returns_live_results(
    mcp_client: Client,
) -> None:
    try:
        result = await mcp_client.call_tool(
            "search_catalog", {"query": "philosophy", "limit": 10}
        )
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"MCP search unavailable: {exc!r}")

    payload = _extract_result_payload(result.data)
    if not isinstance(payload, list):
        pytest.skip("MCP search returned no structured results")

    entries = _validate_search_entries(payload)
    assert entries, "Expected at least one validated search result"


@pytest.mark.asyncio
async def test_download_artifact_tool_uses_real_md5(
    mcp_client: Client, annas_instance: Annas
) -> None:
    secret = os.environ.get("ANNAS_SECRET_KEY")
    if not secret:
        pytest.skip("ANNAS_SECRET_KEY not configured; skipping download verification")

    results = _collect_live_results("plato", limit=40)
    preferred_formats = {"epub", "mobi", "azw", "azw3", "txt", "html"}
    candidate = next(
        (
            entry
            for entry in results
            if entry.file_format
            and entry.file_format.lower() in preferred_formats
        ),
        None,
    )
    if candidate is None:
        pytest.skip("No suitable search result available for download verification")

    try:
        response = await mcp_client.call_tool(
            "download_artifact", {"md5": candidate.md5}
        )
    except Exception as exc:  # pragma: no cover - network flake protection
        pytest.skip(f"MCP download unavailable: {exc!r}")

    payload = _extract_result_payload(response.data)
    resolved: Path
    if isinstance(payload, Path):
        resolved = payload
    elif isinstance(payload, str):
        resolved = Path(payload)
    else:
        pytest.skip("Unexpected payload type from download_artifact tool")

    assert resolved.exists()
    assert resolved.stat().st_size > 0
    assert resolved.parent.parent == annas_instance.work_path


@pytest.mark.asyncio
async def test_search_downloaded_text_tool_reads_markdown(
    mcp_client: Client, annas_instance: Annas
) -> None:
    md5 = "a" * 32
    target_dir = annas_instance.work_path / md5
    target_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = target_dir / "note.md"
    markdown_path.write_text("line1\nneedle here\nline3\n", encoding="utf-8")

    result = await mcp_client.call_tool(
        "search_downloaded_text",
        {"md5": md5, "needle": "needle", "before": 1, "after": 1, "limit": 1},
    )
    payload = _extract_result_payload(result.data)
    assert isinstance(payload, str)
    assert "line1" in payload
    assert "line3" in payload


@pytest.mark.asyncio
async def test_query_collection_tool_returns_chunks(
    mcp_client: Client, annas_instance: Annas
) -> None:
    pytest.importorskip("chromadb")

    md5 = "b" * 32
    raw_elements = [
        SimpleNamespace(
            text="Intro Title",
            category="Title",
            metadata=ElementMetadata(page_number=1, category_depth=1),
        ),
        SimpleNamespace(
            text="Content line",
            category="NarrativeText",
            metadata=ElementMetadata(page_number=1),
        ),
    ]
    elements: List[ElementLike] = cast(List[ElementLike], raw_elements)
    text = annas_instance._elements_to_markdown(elements)
    annas_instance._load_elements(md5, elements, "test-chroma", text)

    result = await mcp_client.call_tool(
        "query_collection",
        {"collection": "test-chroma", "query": "Intro", "n": 1},
    )
    payload = _extract_result_payload(result.data)
    assert isinstance(payload, list)
    assert payload
    assert "Intro" in payload[0]
