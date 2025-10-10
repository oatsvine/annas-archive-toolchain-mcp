# Anna's Archive Toolchain

Research teams use the toolkit to execute a repeatable acquisition loop: shortlist new
titles, secure final copies, stage them for retrieval-augmented generation, and surface
grounded citations on demand. The codebase centres on a typed CLI façade
(`annas.cli.Annas`) that coordinates live scraping, artifact normalization, optional
vector-store ingestion, and a matching Model Context Protocol (MCP) surface so remote
assistants can drive the same workflow.

- Playwright-driven scraping captures search result metadata directly from the public
  site using DOM interactions that match the published UI.
- Fast downloads are normalized into a working directory, converted to Markdown via
  `unstructured`, and optionally chunked into a Chroma collection for retrieval tasks.
- Strict typing (Pyright) and integration tests exercise live search, pagination,
  metadata accuracy, authenticated downloads, and smoke-test the MCP tool endpoints.

## Workflow: CLI & MCP Tools

The Fire CLI exposes the `Annas` class with a single instance that carries shared state
(`requests.Session`, Playwright scraping helpers, Chroma client). The MCP server wraps
the same instance and publishes one tool per public method so assistants experience
the exact same behaviour. Use the tools sequentially to move from discovery to
retrieval-ready notes.

| Method / Tool | Purpose |
| --- | --- |
| `search_catalog(query, limit=200)` / `tool.search_catalog` | Triage the live catalog to shortlist promising books with rich `SearchResult` metadata (format, language, size, provenance). |
| `download_artifact(md5, collection=None)` / `tool.download_artifact` | Retrieve the selected book, normalize it to Markdown, and optionally ingest chunks into a shared Chroma collection for RAG. |
| `search_downloaded_text(md5, needle, before=2, after=2, limit=3)` / `tool.search_downloaded_text` | Validate quotations or pull quick reference snippets directly from cached Markdown. |
| `query_collection(collection, query, n, md5s=None)` / `tool.query_collection` | Run similarity search across the ingested corpus to surface context paragraphs and grounded citations. |

All public methods assert their preconditions (e.g., non-empty inputs, valid MD5 hashes)
and surface detailed errors if the remote services fail.

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
4. **Vector-store loading** — When requested, `_load_elements` chunks the Markdown with
   LangChain's recursive splitter and upserts the structured records into Chroma.
5. **Retrieval** — `search_downloaded_text` surfaces quick Markdown snippets for fact checks;
   `query_collection` runs semantic queries across the ingested Chroma collection to power
   grounded citations. Both flows are reachable through the CLI and via the MCP tools,
   ensuring the same post-processing semantics in agent-driven conversations.

## MCP Server Workflow

- The MCP server keeps an `Annas` instance in context, so stateful elements (sessions,
  download path, Chroma client) are shared across tool invocations.
- Each MCP tool enforces the same assertions as the CLI, returning structured payloads or
  clear errors so assistants can confidently progress through the four-step pipeline.
- Tool descriptions are written from the perspective of downstream agents, making it easy
  to select the right capability (`search_catalog` for discovery, `download_artifact` for
  ingestion, etc.).

## Quality & Testing

- `pyright` enforces strict typing across the codebase.
- `pytest -q` exercises both unit utilities and live Anna's Archive workflows (search,
  pagination, download, metadata validation) alongside MCP smoke tests that call each tool.
  Integration tests require network access, Playwright Chromium, and a valid `ANNAS_SECRET_KEY`.
