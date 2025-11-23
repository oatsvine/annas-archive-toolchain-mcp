## 0) Precedence & Interpretation

* `NOTE:` comments are **authoritative**. Obey them verbatim; do not remove.
* `TODO:` comments are **actionable**. Implement the specific gap and then remove the `TODO:` line.
* Follow existing file/project conventions first. When this doc and code comments conflict, **code comments win**. Your primary responsibility is to **match the current file’s patterns** (naming, logging, returns).

**Core principle:** Challenge every line of code. If a line is not required for correctness, clarity, or measurable performance, **remove it**. Agents tend to over-abstract: resist helper sprawl, avoid “future-proofing,” and keep the API surface minimal.

## 1) Project Structure

* Use `tree` to get oriented.
* Use appropriate search or package manager CLI for more details.

## 2) Build, Test, and Dev

* Install: `uv sync`
* Run: `uv run` (if `UV_ENV_FILE` not set and cwd includes `.env`, set it)
* Type check: `pyright`
* Format: `black .`
* Tests: `pytest -q`
* Library usage rules: navigate to definition to study usage then use your python REPL to test and inspect before designing and writing code. 

## 3) Style & Naming

* Python ≥ 3.11. **Strict typing everywhere** (no untyped public functions).
* When you need structures, define idiomatic **Pydantic v2** models (no ad-hoc dicts, no dataclasses for runtime models).
* **Typer** for CLIs (typed arguments; treat commands as public functions, not as wrappers).
* **Loguru** for logging (configure once).
* Paths use `pathlib.Path`. Prefer Unicode output.
* Tools (configured in `pyproject.toml`): Black (100 cols), Pyright (strict).

### Typer Style (MANDATORY)
- Always declare parameters with `typing.Annotated` + `typer.Argument`/`typer.Option`; never assign `= Argument(...)` or `= Option(...)`. Example from Typer docs:
  ```python
  @app.command()
  def main(
      user: Annotated[str, Argument(help="Username")],
      token: Annotated[str, Option(envvar="API_TOKEN", help="Auth token")],
      limit: Annotated[int, Option(help="Max items")] = 10,
  ):
      ...
  ```
- Ordering matters: all required params (no defaults) first; defaults after. Avoid “non-default argument follows default argument” by keeping `Option(...` required params before any defaulted ones.
- AWLAYS use env-backed required options instead of manual `_require_env` helpers (`token: Annotated[str, Option(envvar="CIVITAI_API_KEY")]`).
- When migrating from `fire`, move `__init__` level arguments to individual command functions requiring these values. Remove class pattern ENTIRELY, use `envvar` for environment mapped argument and define small `Context` dataclass only if statefullness really needed using composition.
- Do not wrap commands in thin helpers; put command logic directly in the Typer command functions.
- Reference: Typer tutorial on arguments/options with `Annotated` (`https://typer.tiangolo.com/tutorial/arguments/optional/`).

## 4) Testing

* Framework: pytest. Tests live in `tests/`, named `test_*.py`.
* Cover CLI methods (happy paths + edge cases). Keep tests deterministic; mock I/O/LLMs.
* Treat type errors as failures: `pyright` must pass locally and in CI.

## 5) Errors & Contracts

* Fail **fast** with clear exceptions. Validate external input at the **edges** (CLI args, files, network).
* Use typed returns; avoid sentinel `None` where an exception or a result type is clearer.

## 6) Performance & Reliability

* Correctness and clarity first; only then optimize with measurements.
* Keep side effects local; make data flow explicit.
* Batch/stream only when it **materially** improves performance—justify in a short comment.

## 7) Terminal UX

* Use `rich` idiomatically for human-friendly console output.
* If prompting is necessary, prefer **Rich**’s prompt (`rich.prompt.Prompt`).

## 8) Do / Don’t

**Do**

* Use canonical library types and utilities.
* In private functions, always use canonical library types for your argument, never bespoke intermediate types.
* In public functions, use discernment, either use `Typer` idioms or canonical library types or bespoke pydantic v2 models depending the the use case.
* When data validation is necessary, never create bespoke helpers until you ruled out using pydantic validators.
* Never configure **Loguru** use default `from loguru import logger` directly.
* Document why non-obvious choices exist (1–3 lines, not essays).

**Don’t**

* Don’t wrap or duplicate canonical library APIs.
* Don't write bespoke code without first ruling mentally surveying all libraries in project stack to rule out canonical APIs.
* Don't define helpers before your can prove that inlining the code would add significant duplications and more lines of code.
* Don’t ship untyped functions or untyped public interfaces.
* Don’t add dead code, speculative hooks, or “future-proofing.”


