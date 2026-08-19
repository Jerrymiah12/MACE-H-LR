from pathlib import Path

import pytest

from maceh.paths import data_root, runs_root


def test_roots_are_explicit_and_canonical(monkeypatch, tmp_path):
    monkeypatch.delenv("MACEH_DATA_ROOT", raising=False)
    monkeypatch.delenv("MACEH_RUNS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="MACEH_DATA_ROOT"):
        data_root()
    with pytest.raises(RuntimeError, match="MACEH_RUNS_ROOT"):
        runs_root()

    monkeypatch.setenv("MACEH_DATA_ROOT", str(tmp_path / "reference"))
    monkeypatch.setenv("MACEH_RUNS_ROOT", str(tmp_path / "generated"))
    assert data_root() == (tmp_path / "reference").resolve()
    assert runs_root() == (tmp_path / "generated").resolve()
    assert data_root(tmp_path / "override") == Path(tmp_path / "override").resolve()
