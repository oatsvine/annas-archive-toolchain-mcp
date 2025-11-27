from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp.client import Client

from annas.mcp import mcp


@pytest.fixture
async def mcp_client():
    async with Client(transport=mcp) as client:
        yield client


async def test_tools_registered(mcp_client: Client) -> None:
    tools = await mcp_client.list_tools()
    names = {tool.name for tool in tools}
    assert {"search_catalog", "download", "search_downloaded_text"} <= names


@pytest.fixture
def snippet_workdir(tmp_path: Path) -> Path:
    md5 = "a" * 32
    work_dir = tmp_path / "annas"
    markdown_dir = work_dir / md5
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "sample.md").write_text(
        "line1\nmatch here\nline3\n", encoding="utf-8"
    )
    return work_dir


async def test_snippets_reads_existing_markdown(
    mcp_client: Client, snippet_workdir: Path
) -> None:
    md5 = "a" * 32
    result = await mcp_client.call_tool(
        "search_downloaded_text",
        {
            "md5": md5,
            "needle": "match",
            "work_dir": snippet_workdir,
            "before": 1,
            "after": 1,
            "limit": 1,
        },
    )
    assert "line1" in result.data
    assert "line3" in result.data
