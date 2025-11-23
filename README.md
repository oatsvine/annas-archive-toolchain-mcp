# Anna's Archive Toolchain

Research teams use the toolkit to execute a repeatable acquisition loop: shortlist new
titles, secure final copies, stage them for retrieval-augmented generation, and surface
grounded citations on demand. The codebase centres on a typed CLI façade
(`annas.cli`) built with Typer that coordinates live scraping, artifact normalization,
optional vector-store ingestion, and a dedicated `store` subcommand group for Chroma
operations.

- Playwright-driven scraping captures search result metadata directly from the public
  site using DOM interactions that match the published UI.
- Fast downloads are normalized into a working directory, converted to Markdown via
  `unstructured`, and optionally chunked into a Chroma collection for retrieval tasks.
- Strict typing (Pyright) and integration tests exercise live search, pagination,
  metadata accuracy, and authenticated downloads.

## Workflow: CLI Commands

The Typer CLI maintains shared state (download path, requests session, Chroma client)
via a callback. Core commands live at the root, with Chroma helpers under `store`:

| Command | Purpose |
| --- | --- |
| `annas search-catalog QUERY --limit 200` | Triage the live catalog to shortlist promising books with rich `SearchResult` metadata. |
| `annas download MD5 [--collection NAME]` | Retrieve the selected book, normalize it to Markdown, and optionally ingest chunks into a shared Chroma collection for RAG. |
| `annas search-downloaded-text MD5 NEEDLE` | Validate quotations or pull quick reference snippets directly from cached Markdown. |
| `annas store metadata COLLECTION [--md5s ...]` | Inspect stored chunk metadata for a collection. |
| `annas store query-collection COLLECTION QUERY [--md5s ...]` | Run similarity search across the ingested corpus to surface context paragraphs and grounded citations. |

All commands assert their preconditions (e.g., non-empty inputs, valid MD5 hashes) and
surface clear errors when remote services fail.

### Environment Expectations

- `ANNAS_DOWNLOAD_PATH` (or `work_path` argument) must point to a writable location for
  downloads and derived artifacts.
- `ANNAS_SECRET_KEY` is required for fast downloads; live search can run without it.
- Chromium Playwright binaries should be available.

## Data Flow Overview

1. **Catalog search** — `annas.scrape.scrape_search_results` launches a headless Chromium
   session, submits the query, paginates through results, and normalizes each card into a
   `SearchResult` that feeds the research reading list.
2. **Artifact ingest** — `download_artifact` validates the MD5, hits the fast-download
   endpoint, streams the payload to disk, and sends it through `unstructured.partition`.
3. **Markdown normalization** — `_elements_to_markdown` arranges titles, lists, page
   markers, and tables into readable Markdown.
4. **Vector-store loading** — When requested, `store.load_elements` chunks the Markdown
   with unstructured's splitter and upserts structured records into Chroma.
5. **Retrieval** — `search_downloaded_text` surfaces quick Markdown snippets for fact checks;
   `store.query_collection` runs semantic queries across the ingested Chroma collection to
   power grounded citations.

## Quality & Testing

- `pyright` enforces strict typing across the codebase.
- `pytest -q` exercises both unit utilities and live Anna's Archive workflows (search,
  pagination, download, metadata validation). Integration tests require network access,
  Playwright Chromium, and a valid `ANNAS_SECRET_KEY`.
