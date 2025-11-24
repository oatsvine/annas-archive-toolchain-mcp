from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pytest

from unstructured.documents.elements import Element


@pytest.fixture(scope="session")
def corpus_root() -> Path:
    return Path(__file__).parent / "corpus"


@pytest.fixture(scope="session")
def corpus_paths(corpus_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for path in sorted(corpus_root.iterdir()):
        if path.is_file():
            mapping.setdefault(path.suffix.lstrip(".").lower(), []).append(path)
    return mapping


@pytest.fixture(scope="session")
def corpus_files(corpus_paths: Dict[str, List[Path]]) -> List[Path]:
    files: List[Path] = []
    for bucket in corpus_paths.values():
        files.extend(bucket)
    return sorted(files)


@pytest.fixture(scope="session")
def partition_elements() -> Callable[[Path], List["Element"]]:
    from unstructured.partition.auto import partition

    def _load(path: Path) -> List["Element"]:
        return partition(filename=str(path), metadata_filename=path.name)

    return _load


@pytest.fixture()
def annas_tmp(tmp_path: Path) -> Iterable[Path]:
    yield tmp_path


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pdf",
        action="store_true",
        default=False,
        help="Enable PDF corpus tests (can be slow)",
    )


@pytest.fixture(scope="session")
def pdf_enabled(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--pdf"))
