### Source Review (2025-10-12)
- **Document Elements.** Unstructured models expose typed elements (e.g., `Title`, `ListItem`, `NarrativeText`) with structured metadata such as `page_number`, `filename`, and layout coordinates, letting downstream pipelines rely on upstream normalization rather than bespoke wrappers.citeturn1view0
- **Partition Models.** The `partition` entry points share a common interface with pluggable strategies; `fast` uses lightweight parsers while `hi_res` taps layout engines for richer metadata, and both honor `metadata_filename` for provenance.citeturn2view0turn3view0
- **Chunking Guidance.** Unstructured recommends chunk sizes in the ~500–1500 character range with ~10–20% overlap to balance retrieval recall and precision, reinforcing our adoption of `chunk_elements(..., max_characters=1200, overlap=200)`.citeturn4view0
- **Embedding Considerations.** Clean chunk boundaries and metadata fidelity improve embedding quality and downstream ranking, so preserving canonical metadata (author, title, filename) remains central.citeturn5view0
- **EPUB Processing.** `python-epub3` demonstrates direct EPUB XML traversal and spine management, clarifying how unstructured’s EPUB partitioners can map raw XHTML into normalized elements without lossy conversions—helpful when validating fallback extractors like `mobi.extract`.citeturn6view0

### Upstream Unstructured Test Inventory
- **chunking**
  - `test_base.py` – layout-aware chunking smoke tests.
  - `test_basic.py` – baseline `chunk_elements` behavior and `include_orig_elements` metadata.
  - `test_dispatch.py` – routing between chunking strategies.
  - `test_html_output.py` – HTML staging of chunk output.
  - `test_title.py` – title-preserving chunking heuristics.
- **cleaners**
  - `test_core.py` – generic text cleaning utilities (whitespace, bullets).
  - `test_extract.py` – extraction helpers for emails/URLs.
  - `test_translate.py` – translation stubs and mocks.
- **common**
  - `test_html_table.py` – table parsing shared logic.
- **documents**
  - `test_coordinates.py` – coordinate metadata integrity.
  - `test_elements.py` – element type conversions and equality.
  - `test_mappings.py` – ontology ↔ element mappings.
  - `test_ontology_to_unstructured_parsing.py` – ontology import pipeline.
- **embed**
  - `test_mixedbreadai.py`, `test_octoai.py`, `test_openai.py`, `test_vertexai.py`, `test_voyageai.py` – connector-specific embedding adapters.
- **file_utils**
  - `test_encoding.py` – file encoding detection.
  - `test_file_conversion.py` – dependency-backed conversions (e.g., LibreOffice).
  - `test_filetype.py` – libmagic-backed type inference.
  - `test_model.py` – serialization fidelity for file descriptors.
- **metrics**
  - `test_element_type.py`, `test_evaluate.py`, `test_table_alignment.py`, `test_table_detection_metrics.py`, `test_table_formats.py`, `test_table_structure.py`, `test_text_extraction.py`, `test_utils.py` – evaluation tooling for extraction quality.
- **nlp**
  - `test_partition.py` – NLP-aware partition helpers.
  - `test_tokenize.py` – tokenization routines.
- **partition**
  - `test_api.py` – public partition API invariants.
  - Format-specific suites (`test_auto.py`, `test_doc.py`, `test_epub.py`, `test_pdf.py`, etc.) covering parser fidelity.
  - `test_strategies.py` – `fast` vs `hi_res` dispatch logic.
- **staging**
  - `test_base.py` – serialization utilities including `element_to_md`.
  - `test_baseplate.py`, `test_datasaur.py`, `test_huggingface.py`, `test_label_*` – export connectors.
- **testfiles** – fixture corpus used across suites.

### annas/cli.py Concept Map (Canonical vs Bespoke)
#### Partitioning & Source Extraction
- **Canonical**: `partition(..., strategy=strategy, metadata_filename=document_metadata.filename)` embraces the upstream interface and ensures filename provenance (
  `annas/cli.py:254-262`). Tests cover live downloads and sanitized filenames (`tests/test_annas_cli.py:12-34`, `tests/test_annas_filename_sampling.py:67-80`).
- **Bespoke**: `_validate_md5`, `_retrying_get`, `_write_stream`, `_assert_readable_container` (not shown) enforce Anna’s Archive downloads, including MOBI extraction in `_write_stream` (`annas/cli.py:600-646`). These remain repo-specific glue.

#### Chunking & Context Windows
- **Canonical**: `chunk_elements(..., include_orig_elements=True, max_characters=1200, overlap=200)` aligns with upstream recommendations and metadata preservation (`annas/cli.py:508-516`). Tests assert ingestion correctness and dedupe logic (`tests/test_annas_cli.py:93-134`).
- **Bespoke**: Chapter inference still depends on `_extract_chapter`, `_looks_like_chapter`, and generic-heading filters (`annas/cli.py:844-891`), supplementing canonical metadata when partition output lacks rich section titles.

#### Markdown & Staging
- **Canonical**: `element_to_md` drives per-element rendering while respecting upstream heading depth (`annas/util.py`). Markdown formatting test ensures parity with titles and lists (`tests/test_annas_cli.py:58-90`).
- **Bespoke**: Page markers and bullet normalization ensure consistent snippets across downstream tooling (`annas/util.py`). Opportunity exists to validate whether `element_to_md` already prefixes lists, potentially dropping the manual `- ` patch once confirmed by corpus fixtures.

#### Filetype Detection & Container Prep
- **Canonical**: `detect_extension` defers to `detect_filetype` and only falls back to suffix inspection when libmagic is unavailable (`annas/util.py`), mirroring upstream coverage in `test_unstructured/file_utils/test_filetype.py`.
- **Bespoke**: Filename slugging, underscore hex decoding, ISBN pulls, and brand stripping (`annas/cli.py:616-789`) normalize Anna’s Archive payloads. Tests guard both fallback detection and slug parsing (`tests/test_annas_cli.py:152-170`, `tests/test_annas_filename_sampling.py:67-93`).

#### Metadata Normalization & Titles
- **Canonical**: Reliance on `ElementMetadata.page_number` and `Title` elements honors unstructured element contracts (`annas/cli.py:478-507`).
- **Bespoke**: Custom fallback title selection, tag assembly, and size formatting keep Chroma metadata exhaustive while excluding boilerplate headings (`annas/cli.py:519-543`). Unit tests cover `ListItem` handling, `Title` selection, and chapter heuristics (`tests/test_annas_cli.py:58-150`).

### Opportunities (Code Minimization Focus)
1. **List Markdown Prefixing** – Confirm via corpus fixtures whether `element_to_md` already emits list bullets; if so, remove the explicit `- ` prefix to shrink `elements_to_markdown`, documenting the corpus evidence per AGENTS minimalism rule.
2. **Chapter Detection** – Explore `unstructured.chunking.title` or upcoming section-title metadata to replace `_extract_chapter` + regex fallback, reducing bespoke parsing while preserving title accuracy. Reference `test_unstructured/chunking/test_title.py` once validated.
3. **Filename Normalization** – Investigate whether unstructured cleaners (e.g., `cleaners.core` slug/whitespace utilities) can supplant portions of `_sanitize_filename`, provided they maintain the md5/author/title guarantees enforced by `tests/test_annas_filename_sampling.py:35-93`.
4. **MOBI Conversion Dependency** – Compare unstructured’s EPUB pipeline with our MOBI HTML conversion to see if partition `strategy="fast"` on the extracted HTML already fills metadata, allowing us to soften bespoke asset handling.
5. **Metadata Propagation** – Audit chunk metadata to see if `chunk.metadata` now carries section/filename context upstream, potentially eliminating manual `chapter` inference while preserving the “never COPYRIGHT” requirement captured in `tests/test_annas_cli.py:99-130`.
