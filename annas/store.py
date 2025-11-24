from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import GetResult, Include, Metadata, QueryResult, Where
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Element, ElementMetadata, Title


class ElementLike(Protocol):
    """Minimal element contract used across ingestion helpers."""

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


class ChunkRecord(BaseModel):
    """Persisted chunk snapshot for debuggable ingestion."""

    text: str
    metadata: Dict[str, str]


def _stringify_meta(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    metadata = raw or {}
    cleaned: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[str(key)] = str(value)
    return cleaned


def get_or_create_chroma(work_dir: Path) -> ClientAPI:
    """Return a Chroma client rooted in work_dir."""

    logger.debug("Opening Chroma client at {}", work_dir / "chroma")
    return chromadb.PersistentClient(path=str(work_dir / "chroma"))


def _collection(client: ClientAPI, name: str) -> Collection:
    """Fetch or create the target collection (isolated for reuse in tests)."""

    return client.get_or_create_collection(name)


def _metadata(
    client: ClientAPI,
    collection_name: str,
    n: int,
    md5_filter: Optional[Where],
) -> GetResult:
    """Fetch stored chunk metadata for stable inspection."""

    collection = _collection(client, collection_name)
    logger.debug(
        "Fetching metadata where={} limit={} from collection {}",
        md5_filter,
        n,
        collection_name,
    )
    return collection.get(
        where=md5_filter,
        limit=n,
        include=cast(Include, ["documents", "metadatas"]),
    )


class StoredChunk(BaseModel):
    """Snapshot of a stored chunk and its metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: Dict[str, str]


def fetch_metadata(
    client: ClientAPI,
    collection_name: str,
    *,
    n: int = 10,
    md5s: Optional[Sequence[str]] = None,
) -> List[StoredChunk]:
    """Return stored metadata rows without CLI formatting."""

    assert n >= 1, "limit must be >= 1"
    md5_filter: Optional[Where] = None
    if md5s:
        md5_list = [_validate_md5(m) for m in md5s if m.strip()]
        if md5_list:
            md5_filter = cast(Where, {"md5": {"$in": md5_list}})
    query_result = _metadata(client, collection_name, n, md5_filter)
    docs = query_result["documents"] or []
    metadatas = query_result["metadatas"] or []
    ids = query_result["ids"] or []
    assert len(docs) == len(metadatas) == len(ids), "Metadata count mismatch"
    return [
        StoredChunk(
            id=str(ids[index]),
            text=str(doc or ""),
            metadata=_stringify_meta(metadatas[index]),
        )
        for index, doc in enumerate(docs)
        if doc is not None
    ]


def query_chunks(
    client: ClientAPI,
    collection: str,
    query: str,
    *,
    n: int = 10,
    md5s: Optional[Sequence[str]] = None,
) -> List[StoredChunk]:
    """Query a Chroma collection and return chunk payloads."""

    assert query.strip(), "query must be non-empty"
    assert n >= 1, "limit must be >= 1"
    logger.debug(
        "Querying collection={} text='{}' n={} filter={}",
        collection,
        query,
        n,
        md5s,
    )
    md5_filter: Optional[Where] = None
    if md5s:
        md5_list = [_validate_md5(m) for m in md5s if m.strip()]
        if md5_list:
            md5_filter = cast(Where, {"md5": {"$in": md5_list}})
    collection_client = _collection(client, collection)

    query_result: QueryResult = collection_client.query(
        query_texts=[query],
        n_results=n,
        where=md5_filter,
        include=cast(Include, ["documents", "metadatas"]),
    )
    docs = query_result["documents"]
    assert docs is not None, "Chroma query must return documents"
    assert docs, "Chroma query returned no document rows"
    metadatas = query_result["metadatas"]
    ids = query_result["ids"]
    primary_docs: List[str] = docs[0]
    assert metadatas and len(metadatas[0]) == len(
        primary_docs
    ), "Metadata count mismatch"
    assert ids and len(ids[0]) == len(primary_docs), "ID count mismatch"
    return [
        StoredChunk(
            id=str(ids[0][i]),
            text=str(primary_docs[i] or ""),
            metadata=_stringify_meta(metadatas[0][i]),
        )
        for i in range(len(primary_docs))
    ]


def load_elements(
    work_dir: Path,
    md5: str,
    elements: Sequence[ElementLike],
    collection: str,
    document_metadata: Optional[DocumentMetadata],
    *,
    client: ClientAPI,
) -> None:
    """Upsert chunked elements into Chroma."""

    coll = _collection(client, collection)
    existing = coll.get(where={"md5": md5})
    ids = existing["ids"]
    if ids:
        coll.delete(ids=ids)

    typed_elements = [cast(Element, element) for element in elements]
    if not typed_elements:
        return

    chunk_ids, documents, metadatas, records = _build_chunk_payloads(
        md5, typed_elements, document_metadata
    )
    coll.upsert(
        ids=chunk_ids, documents=documents, metadatas=cast(List[Metadata], metadatas)
    )
    _write_chunk_file(work_dir, md5, records)


def write_chunk_snapshot(
    work_dir: Path,
    md5: str,
    elements: Sequence[ElementLike],
    document_metadata: Optional[DocumentMetadata],
) -> Path:
    """Persist chunk JSONL without touching Chroma (for offline inspection)."""

    typed_elements = [cast(Element, element) for element in elements]
    if not typed_elements:
        return work_dir / md5 / "chunks" / "chunks.jsonl"
    _, _, _, records = _build_chunk_payloads(md5, typed_elements, document_metadata)
    return _write_chunk_file(work_dir, md5, records)


def _build_chunk_payloads(
    md5: str,
    typed_elements: Sequence[Element],
    document_metadata: Optional[DocumentMetadata],
) -> tuple[List[str], List[str], List[Dict[str, str]], List[ChunkRecord]]:
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
        assert isinstance(
            metadata, ElementMetadata
        ), "element.metadata must be ElementMetadata"
        if metadata.page_number is not None:
            document_page_numbers.add(metadata.page_number)
        if not title_found and isinstance(element, Title) and element.text.strip():
            candidate = element.text.strip()
            if not _is_generic_heading(candidate) and not _looks_like_chapter(
                candidate
            ):
                title = candidate
                title_found = True

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
                and not _is_generic_heading(el.text)
            ),
            None,
        )
        if not chapter:
            chapter = _extract_chapter(chunk_text)
        if chapter and chapter.lower() != chunk_metadata["title"].lower():
            chunk_metadata["chapter"] = chapter

        metadatas.append(chunk_metadata)

    assert documents, "All ingestable chunks were empty after processing"
    chunk_ids = [f"{md5}:{index:04d}" for index in range(1, len(documents) + 1)]
    records = [
        ChunkRecord(text=doc, metadata=meta) for doc, meta in zip(documents, metadatas)
    ]
    return chunk_ids, documents, metadatas, records


def _write_chunk_file(work_dir: Path, md5: str, records: Sequence[ChunkRecord]) -> Path:
    chunk_dir = work_dir / md5 / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
    logger.debug("Wrote {} chunks to {}", len(records), chunk_path)
    return chunk_path


def _validate_md5(md5: str) -> str:
    candidate = md5.strip().lower()
    assert re.fullmatch(r"[0-9a-f]{32}", candidate), f"Invalid md5 hash: {md5}"
    return candidate


def _extract_chapter(chunk: str) -> Optional[str]:
    match = re.search(r"^#{1,6}\s+(.+)", chunk, re.MULTILINE)
    if not match:
        return None
    heading = match.group(1).strip()
    if _is_generic_heading(heading):
        return None
    return heading


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


def _looks_like_chapter(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(("chapter", "chap.")):
        return True
    if re.match(r"^\d+\s*[).]", stripped):
        return True
    if re.match(r"(?i)(?:book|part|section)\s+[ivxlcdm\d]+", stripped):
        return True
    return False
