import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "maceh"
FORBIDDEN = {"workflows", "mgo_lr", "plots", "plots2", "analysis_scripts"}


def _roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
              and node.module):
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=str)
def test_library_never_imports_workflows(path):
    bad = FORBIDDEN & set(_roots(path))
    assert not bad, f"{path} imports workflow-layer module(s): {sorted(bad)}"
