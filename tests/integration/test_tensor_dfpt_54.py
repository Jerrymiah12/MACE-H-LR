import numpy as np

from workflows.training import tensor_dfpt_54 as MODULE


def test_fast_selection_is_split_safe_and_unique():
    rows = MODULE.FAST_SELECTION
    ids = [row[1] for row in rows]
    assert len(ids) == len(set(ids))
    assert rows[0][0:2] == ("train", "snapshot_000386")
    assert {row[0] for row in rows} == {"train", "validation", "test"}


def test_pw_writer_is_54_atom_fixed_occupancy(tmp_path):
    cell = np.eye(3) * 12.0
    species = ["Mg"] * 27 + ["O"] * 27
    cart = np.arange(54 * 3, dtype=float).reshape(54, 3) / 100.0
    qe = {
        "pseudo_dir": "/tmp/pseudo",
        "pseudopotentials": {"Mg": "Mg.upf", "O": "O.upf"},
    }
    path = tmp_path / "pw.in"
    MODULE._write_pw_input(path, qe, MODULE.PROFILES["fast"],
                           cell, cart, species)
    text = path.read_text()
    assert "nat = 54" in text
    assert "occupations = 'fixed'" in text
    assert "2 2 2 0 0 0" in text
    assert text.count("\nMg  ") == 27
    assert text.count("\nO  ") == 27
