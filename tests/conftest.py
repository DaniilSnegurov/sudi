import os
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _dataset_root() -> Path | None:
    """Корень набора — каталог, где лежит fssp/; разметка может лежать рядом или в courts_anonymized/."""
    candidates = [os.environ.get("COURTDOCS_DATA"), PROJECT / "courts_anonymized", PROJECT]
    return next((Path(c) for c in candidates if c and (Path(c) / "fssp").is_dir()), None)


def _labels() -> Path | None:
    root = _dataset_root()
    candidates = [root / "labels" / "xml.csv" if root else None, PROJECT / "courts_anonymized" / "labels" / "xml.csv"]
    return next((c for c in candidates if c and c.is_file()), None)


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    root = _dataset_root()
    if root is None:
        pytest.skip("учебный набор не найден (нужен каталог fssp/)")
    return root


@pytest.fixture(scope="session")
def xml_labels() -> Path:
    path = _labels()
    if path is None:
        pytest.skip("эталон labels/xml.csv не найден")
    return path


@pytest.fixture(scope="session")
def fssp_batch(dataset_root):
    from courtdocs.pipeline import run

    return run(dataset_root, include=("fssp",))
