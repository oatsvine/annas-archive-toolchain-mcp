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

# ------ Reusable non-Anna's Archive specific utilities for processing ------ #


def compute_pdf_ocr_ratio(
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


def elements_to_markdown(elements: Sequence[ElementLike]) -> str:
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


# ------ Internal helpers ------ #


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
