import os

import numpy as np

from workflows.mgo_dataset import dfpt
from maceh.config import load_config
from tests.integration.mgo_dataset.test_gen_structures import Args, make_fake_reference

CFG = load_config("workflows/mgo_dataset/configs/mgo.yaml")


def test_config_has_qe_commands():
    assert "pw.x" in CFG["qe"]["pw_command"]
    assert "ph.x" in CFG["qe"]["ph_command"]


def test_init_dfpt_stage(tmp_path):
    ws = str(tmp_path)
    cell, frac, species = make_fake_reference(ws, a=4.19)
    assert dfpt.init_dfpt_stage(CFG, ws, Args()) == 0
    qdir = os.path.join(ws, "reference", "qe")
    pw = open(os.path.join(qdir, "pw.in")).read()
    assert "ibrav = 0" in pw and "nat = 2" in pw and "ntyp = 2" in pw
    assert "occupations = 'fixed'" in pw
    assert "CELL_PARAMETERS angstrom" in pw
    # relaxed lattice vectors present
    row = cell[0]
    assert f"{row[0]:.12f} {row[1]:.12f} {row[2]:.12f}" in pw
    assert "ATOMIC_POSITIONS crystal" in pw
    assert "K_POINTS automatic" in pw
    ph = open(os.path.join(qdir, "ph.in")).read()
    assert "epsil = .true." in ph          # both flags explicit
    assert "zeu = .true." in ph
    assert "trans = .true." in ph
    assert ph.rstrip().endswith("0.0 0.0 0.0")
    job = open(os.path.join(qdir, "job_qe.sh")).read()
    assert CFG["qe"]["pw_command"] in job and CFG["qe"]["ph_command"] in job


def test_write_tensor_only_ph_input(tmp_path):
    path = os.path.join(tmp_path, "ph.in")
    dfpt.write_ph_input(path, CFG, trans=False, zeu=True)
    text = open(path).read()
    assert "epsil = .true." in text
    assert "zeu = .true." in text
    assert "trans = .false." in text
