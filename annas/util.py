from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, cast

from loguru import logger
from unstructured.documents.elements import Element, ElementMetadata, ListItem, Title
from unstructured.file_utils.filetype import detect_filetype
from unstructured.partition.auto import partition
from unstructured.staging.base import element_to_md

from annas.store import ElementLike


def count_text_characters(elements: Sequence[ElementLike]) -> int:
    """Count non-whitespace characters across text-bearing elements.

    The helper strips whitespace from each element's ``text`` attribute and sums the
    remaining character counts. It is useful for quickly estimating textual density in
    partitioned documents before choosing downstream processing strategies.

    Args:
        elements: Iterable of objects with ``text`` attributes.

    Returns:
        Total number of non-whitespace characters across all elements.
    """

    total = 0
    for element in elements:
        text = getattr(element, "text", "")
        if not text:
            continue
        stripped = str(text).strip()
        if stripped:
            total += len(stripped)
    return total


def compute_pdf_ocr_ratio(
    path: Path,
    metadata_filename: str,
    hi_res_elements: Sequence[ElementLike],
) -> Optional[float]:
    """Estimate how much extracted PDF text depends on OCR.

    Compares character counts between an existing high-resolution partition (typically
    with OCR enabled) and a fresh ``strategy="fast"`` partition of the same file. The
    returned value represents the fraction of characters present only in the OCR-heavy
    output. Non-PDF inputs return ``None``.

    Args:
        path: File path to the document under test.
        metadata_filename: Filename passed through to ``partition`` for metadata.
        hi_res_elements: Elements produced by a high-resolution partition pass.

    Returns:
        ``None`` for non-PDFs, otherwise a float in ``[0.0, 1.0]`` where ``1.0`` means
        all text required OCR and ``0.0`` means the fast pass already contained all
        extracted text.
    """

    if path.suffix.lower() != ".pdf":
        return None

    hi_chars = count_text_characters(hi_res_elements)
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

    fast_chars = count_text_characters(cast(Sequence[ElementLike], fast_elements))
    if fast_chars <= 0:
        return 1.0
    if fast_chars >= hi_chars:
        return 0.0
    return (hi_chars - fast_chars) / hi_chars


def elements_to_markdown(elements: Sequence[ElementLike]) -> str:
    """Render unstructured elements into lightweight Markdown.

    The converter preserves heading levels, list items, and page breaks while leaning on
    ``element_to_md`` for standard element rendering. Empty or whitespace-only elements
    are skipped to avoid noise in downstream chunking.

    Args:
        elements: Sequence of elements produced by ``unstructured`` partitioning.

    Returns:
        Markdown string containing the rendered elements and page markers.
    """

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


def detect_extension(path: Path) -> Optional[str]:
    """Guess a file's extension using ``unstructured`` detection, with suffix fallback.

    Attempts to infer an extension via ``detect_filetype`` (libmagic-backed). When that
    fails, the function falls back to the path's suffix, returning ``None`` only when no
    evidence is available. The returned value omits any leading dot and is lowercased.

    Args:
        path: File path to inspect.

    Returns:
        Lowercased extension string without the leading dot, or ``None`` if undetectable.
    """

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
