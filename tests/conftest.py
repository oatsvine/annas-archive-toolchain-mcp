from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List

import pytest

if TYPE_CHECKING:
    from annas.cli import Annas
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
def partition_elements() -> Callable[[Path], List["Element"]]:
    from unstructured.partition.auto import partition

    def _load(path: Path) -> List["Element"]:
        return partition(filename=str(path), metadata_filename=path.name)

    return _load


@pytest.fixture()
def annas_tmp(tmp_path: Path) -> Iterable["Annas"]:
    from annas.cli import Annas

    annas = Annas(work_path=tmp_path)
    yield annas
