# Repository Guidelines

Keep changes small, typed, and testable. Consider every line of code potentially redundant until proven necessary. 

## 0) Precedence & Interpretation

* `NOTE:` comments are **authoritative**. Obey them verbatim; do not remove.
* `TODO:` comments are **actionable**. Implement the specific gap and then remove the `TODO:` line.
* Follow existing file/project conventions first. When this doc and code comments conflict, **code comments win**. Your primary responsibility is to **match the current file’s patterns** (naming, logging, returns).

**Core principle:** Challenge every line of code. If a line is not required for correctness, clarity, or measurable performance, **remove it**. Agents tend to over-abstract: resist helper sprawl, avoid “future-proofing,” and keep the API surface minimal.

## 1) Project Structure

* Infer from `pyproject.toml`
* Docker/Compose: run `codex` **in the container**; do not rely on host-global installs.

## 2) Build, Test, and Dev

* Install (editable): `uv pip install -e .` (already installed - reinstall only if needed)
* Install extras as needed: `uv pip install -r pyproject.toml --all-extras` (already installed - reinstall only if needed)
* Type check: `pyright`
* Format: `black .`
* Tests: `pytest -q`

> Keep the command surface **copy-pasteable**. If a command changes, update this section in the same PR.

## 3) Style & Naming

* Python ≥ 3.11. **Strict typing everywhere** (no untyped public functions).
* **Pydantic v2** for structured data (no ad-hoc dicts, no dataclasses for runtime models).
* **Fire** for CLIs (single class; instance holds state).
* **Loguru** for logging (configure once).
* Paths use `pathlib.Path`. Prefer Unicode output.
* Tools (configured in `pyproject.toml`): Black (100 cols), Pyright (strict).

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

* Use plain stdout/stderr by default.
* If prompting is necessary, prefer **Rich**’s prompt (`rich.prompt.Prompt`) over `input()`.
  Do **not** introduce Rich beyond legitimate UX needs.

## 8) Do / Don’t

**Do**

* Use **Pydantic v2** models as your contracts; pass plain types across boundaries.
* Use **direct** model constructors; validators enforce invariants.
* Use **pure functions** for stateless logic; put behavior on models **only if it uses `self`**.
* Use **Fire** (class-based) for CLIs; keep global state off module level.
* Configure **Loguru** once; log directly.
* Treat **Pyright (strict)** warnings as defects.
* Document why non-obvious choices exist (1–3 lines, not essays).

**Don’t**

* Don’t wrap or duplicate Pydantic/Fire/Loguru behavior.
* Don’t ship untyped functions or untyped public interfaces.
* Don’t attach methods that don’t use `self`.
* Don’t add dead code, speculative hooks, or “future-proofing.”
* Don’t use `input()`; if you must prompt, use Rich’s prompt.

---

## 9) Lessons Learned — Concise Case Studies

Use these as **nuanced clarifications** of the rules above. Each item: **Context → Lesson**. **Context is not relevant** to this project, retaining the lesson is what matters.

1. **Inline “do not touch” notes**
   **Context:** Renamed `workspace_dir` and related helpers despite explicit “DO NOT TOUCH” notes; broke downstream assumptions.
   **Lesson:** When a file carries a “do not touch” note, **preserve naming and flow**. Extend behavior *around* the stable surface—don’t re-plumb internals. (Applies to §§0, 3, 8)

2. **Respect scaffolded helpers/TODOs**
   **Context:** Rewrote helpers like `choose_score`, `_save` instead of completing TODOs; diverged from incremental plan.
   **Lesson:** Treat `TODO:` as **surgical completion**, not a license to redesign. Fill the gap and keep the rest intact. (Applies to §§0, 2, 8)

3. **Follow file-local conventions**
   **Context:** Changed logging format/returns where a file had an established style.
   **Lesson:** **Mirror the local style** (format strings, return shapes, error patterns). Consistency > preference. (Applies to §§0, 3, 8)

4. **Typed element access, no permissive fallbacks**
   **Context:** Used `getattr` on unstructured objects, hiding type errors.
   **Lesson:** Prefer **typed attributes** (e.g., `Element.text`, `Element.metadata.page_number`). Fail fast if missing—don’t normalize away type issues. (Applies to §§3, 5, 8)

5. **Chunk before indexing**
   **Context:** Ingested entire Markdown files into a vector store; retrieval degraded.
   **Lesson:** Use a **proven splitter** (e.g., recursive char splitter) with overlap before embedding. Keep chunking near ingestion; document parameters. (Applies to §§2, 6, 8)

6. **Asserts for invariants; try/except for the outside world**
   **Context:** Wrapped deterministic imports and config checks in broad `try/except`, creating hidden fallbacks.
   **Lesson:** Guard **deterministic preconditions** with asserts; reserve `try/except` for **non-deterministic** edges (file I/O, network, parsers). (Applies to §§5, 6)

7. **Decode, then sanitize filenames**
   **Context:** Left percent codes (`_20`) in output paths; hurt readability and tooling.
   **Lesson:** **Decode/normalize** to Unicode first, then sanitize. Preserve human readability without weakening safety checks. (Applies to §§2, 3)

**Meta-lesson for agents:** Before adding a helper, ask: *What bug, contract, or measurement requires this line?* If you can’t answer in one sentence, you probably don’t need it.

---

## 10) Minimal Templates

**Model + validation**

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator

class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, pattern=r"^[a-z0-9._-]+$")
    email: str = Field(pattern=r"[^@\s]+@[^@\s]+\.[^@\s]+")
    active: bool = True

    @field_validator("id")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("id must be lowercase")
        return v
```

**CLI (class-based Fire)**

```python
import fire
from loguru import logger

class CLI:
    def hello(self, name: str) -> str:
        msg = f"hello {name}"
        logger.info(msg)
        print(msg)
        return msg

if __name__ == "__main__":
    fire.Fire(CLI)
```

**Test**

```python
from mypkg.models import User

def test_user_roundtrip():
    u = User(id="alice", email="alice@example.com", active=True)
    data = u.model_dump()
    assert data["id"] == "alice"
    assert data["active"] is True
```

---

## 11) Final Notes (decision checklist)

1. Is it **data**? → Pydantic model.
2. Is it **behavior bound to that data**? → Model method (uses `self`).
3. Is it **generic and stateless**? → Pure function.
4. Is it a **command surface**? → Fire class, state on instance.
5. Need logging? → Loguru, configured once.
6. Terminal output? → Plain text; use Rich **only** for real UX needs.
7. Every non-obvious choice gets a short comment. If a line isn’t needed, **delete it**.
