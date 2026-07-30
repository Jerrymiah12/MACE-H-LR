import copy
import json
import os

import h5py
import numpy as np
import pytest
import yaml

from mgo_lr import abacus_io, convert, lr
from mgo_lr.config import load_config, sha256_file
from mgo_lr.constants import RY_TO_EV
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell, rocksalt_primitive
from mgo_lr.tests.test_abacus_parsers import write_csr
from mgo_lr.tests.test_gen_structures import make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")

SCF_LOG = ("charge density convergence is achieved\n"
           "!FINAL_ETOT_IS -7524.1 eV\nEFERMI = 5.4 eV\n")


class Args:
    set_name = "pilot"
    force = False


def small_cfg():
    """Pilot on the 2-atom primitive cell (n=1) keeps matrices tiny."""
    cfg = copy.deepcopy(CFG)
    cfg["supercells"]["pilot"] = 1
    # Fabricated Hamiltonians carry no real physics, so the Tier-2 response
    # trends (E_sign/E_linear vs amplitude) cannot hold.  Enforcement is
    # covered directly by test_validate.test_tier2_enforce_fails_the_set.
    cfg["validation"]["tier2_enforce"] = False
    return cfg


def fabricate_dft(folder, cfg, sc, seed=0):
    """Write OUT.MgO with a converged log and hermitian synthetic H/S CSRs.
    Shared with the validate/locality/end-to-end tests."""
    types, norb, offsets = convert.species_orbital_info(cfg, sc.species)
    dim = int(offsets[-1])
    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal((dim, dim)) * 0.05
    h0 = 0.5 * (h0 + h0.T)                       # H(0) symmetric
    hp = rng.standard_normal((dim, dim)) * 0.01  # H(R), H(-R) = H(R)^T
    s0 = np.eye(dim) + 0.01 * (lambda m: 0.5 * (m + m.T))(
        rng.standard_normal((dim, dim)))
    sp = 0.01 * rng.standard_normal((dim, dim))
    out = os.path.join(folder, "OUT.MgO")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "running_scf.log"), "w") as f:
        f.write(SCF_LOG)
    write_csr(os.path.join(out, cfg["abacus"]["csr_h_filename"]), dim,
              {(0, 0, 0): h0, (1, 0, 0): hp, (-1, 0, 0): hp.T.copy()})
    write_csr(os.path.join(out, cfg["abacus"]["csr_s_filename"]), dim,
              {(0, 0, 0): s0, (1, 0, 0): sp, (-1, 0, 0): sp.T.copy()},
              name="S")
    return {"h0": h0, "s0": s0, "dim": dim, "offsets": offsets}


def prepared_snapshot(ws, cfg, u=None):
    cell, frac, species = make_fake_reference(ws)
    n = cfg["supercells"]["pilot"]
    sc = make_supercell(cell, frac, species, n)
    store = SnapshotStore(ws, "pilot")
    sid = "snapshot_000001"
    folder = store.folder(sid)
    os.makedirs(folder)
    if u is None:
        u = np.zeros((len(sc.species), 3))
    abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell, sc.cart + u,
                         sc.species, cfg)
    np.save(os.path.join(folder, "displacements.npy"), u)
    meta = {"pattern_class": "equilibrium", "rigid_translation": False,
            "sign_partner_id": None, "amplitude_partner_ids": [],
            "amplitude": 0.0}
    with open(os.path.join(folder, "displacement_metadata.json"), "w") as f:
        json.dump(meta, f)
    store.write_status(sid, "prepared")
    return store, sid, sc


def test_key_roundtrip():
    k = convert.key_str((0, -1, 2), 0, 4)
    assert k == "[0, -1, 2, 1, 5]"                 # 1-based indices
    assert convert.parse_key(k) == (0, -1, 2, 1, 5)


def test_matrices_to_blocks_units_and_transform(tmp_path):
    cfg = small_cfg()
    cell, frac, species = rocksalt_primitive(4.2)
    sc = make_supercell(cell, frac, species, 1)
    types, norb, offsets = convert.species_orbital_info(cfg, sc.species)
    assert norb == [15, 13]                        # Mg 4s2p1d, O 2s2p1d
    dim = int(offsets[-1])
    dense = np.zeros((dim, dim))
    dense[0, 0] = 2.0                              # Mg s <-> Mg s
    import scipy.sparse
    csr = {(0, 0, 0): scipy.sparse.csr_matrix(dense)}
    blocks = convert.matrices_to_blocks(csr, dim, cfg, sc.species, RY_TO_EV)
    assert set(blocks) == {"[0, 0, 0, 1, 1]"}
    b = blocks["[0, 0, 0, 1, 1]"]
    assert b.shape == (15, 15)
    assert abs(b[0, 0] - 2.0 * RY_TO_EV) < 1e-12   # s-channel: U = identity
    # dimension mismatch raises
    with pytest.raises(ValueError, match="dimension"):
        convert.matrices_to_blocks(csr, dim, cfg, ["Mg"], 1.0)


def test_collect_dft_stage(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fab = fabricate_dft(store.folder(sid), cfg, sc)
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    st = store.read_status(sid)
    assert st["state"] == "converted"
    assert st["scf_converged"] is True
    assert st["csr_files"] == [cfg["abacus"]["csr_h_filename"],
                               cfg["abacus"]["csr_s_filename"]]
    assert set(st["raw_sha256"]) == {cfg["abacus"]["csr_h_filename"],
                                     cfg["abacus"]["csr_s_filename"],
                                     "running_scf.log"}
    folder = store.folder(sid)
    with h5py.File(os.path.join(folder, "hamiltonians_full.h5")) as f:
        keys = [json.loads(k) for k in f.keys()]
        assert all(len(k) == 5 and k[3] >= 1 and k[4] >= 1 for k in keys)
        b11 = np.array(f["[0, 0, 0, 1, 1]"])
    # Ry -> eV applied, transform for the s-block diagonal entry is identity
    assert abs(b11[0, 0] - fab["h0"][0, 0] * RY_TO_EV) < 1e-10
    lat = np.loadtxt(os.path.join(folder, "lat.dat"))
    assert np.allclose(lat.T, sc.cell)             # vectors as columns
    rlat = np.loadtxt(os.path.join(folder, "rlat.dat"))
    assert np.allclose(rlat.T @ lat, 2 * np.pi * np.eye(3), atol=1e-10)
    pos = np.loadtxt(os.path.join(folder, "site_positions.dat"))
    assert pos.shape == (3, 2)                     # 3 x N
    ot = open(os.path.join(folder, "orbital_types.dat")).read().splitlines()
    assert len(ot) == 2                            # one line per atom
    info = json.load(open(os.path.join(folder, "info.json")))
    assert info["isspinful"] is False and info["norbits"] == fab["dim"]
    elements = np.loadtxt(os.path.join(folder, "element.dat"))
    assert list(elements) == [12.0, 8.0]


def test_site_positions_are_wrapped_into_the_cell(tmp_path):
    """ABACUS wraps atoms into [0,1) and labels H(R) against the wrapped
    positions, so site_positions.dat must be wrapped too — otherwise every
    distance derived from (R, i, j) is off by a lattice vector for any atom a
    displacement pushed outside the cell.  u must still round-trip."""
    ws = str(tmp_path)
    cfg = small_cfg()
    # push atom 0 (at fractional origin) to a slightly NEGATIVE coordinate
    u = np.zeros((2, 3))
    u[0, 0] = -0.05
    store, sid, sc = prepared_snapshot(ws, cfg, u=u)
    frac_in = (sc.cart + u) @ np.linalg.inv(sc.cell)
    assert frac_in.min() < 0.0                       # precondition
    fabricate_dft(store.folder(sid), cfg, sc)
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    pos = np.loadtxt(os.path.join(store.folder(sid),
                                  "site_positions.dat")).T
    frac_out = pos @ np.linalg.inv(sc.cell)
    assert frac_out.min() >= -1e-12 and frac_out.max() < 1.0
    # the recorded displacement is still recoverable via minimum image
    assert np.abs(lr.minimum_image_displacements(sc.cell, pos, sc.cart)
                  - u).max() < 1e-10


def test_undisplaced_positions_are_not_wrapped(tmp_path):
    """An atom at fractional 0 lands on ~-1e-17, which ABACUS does NOT wrap.
    A bare modulo would send it to ~1.0 — a full lattice vector the wrong way,
    corrupting exactly the equilibrium snapshots the wrap was meant to spare."""
    cfg = small_cfg()
    # A real fcc cell makes an origin atom land a few 1e-17 below zero purely
    # from the cart @ inv(cell) round-trip.  Drive write_structure_files
    # directly so the STRU's 12-decimal format cannot round the effect away.
    cell, _, species = rocksalt_primitive(4.2)
    cart = np.array([[-1e-17, 0.0, 0.0], (0.5 * cell.sum(axis=0)).tolist()])
    frac_in = cart @ np.linalg.inv(cell)
    assert frac_in.min() < 0.0 and frac_in.min() > -1e-9   # noise, not a shift
    folder = str(tmp_path)
    convert.write_structure_files(folder, cell, cart, species, cfg, 0.0)
    pos = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
    # positions must stay put, not jump by a lattice vector
    assert np.abs(pos - cart).max() < 1e-9


def test_collect_dft_skips_and_protects(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fabricate_dft(store.folder(sid), cfg, sc)
    convert.collect_dft_stage(cfg, ws, Args())
    h1 = sha256_file(os.path.join(store.folder(sid), "hamiltonians_full.h5"))
    convert.collect_dft_stage(cfg, ws, Args())     # idempotent skip
    assert sha256_file(os.path.join(store.folder(sid),
                                    "hamiltonians_full.h5")) == h1


def test_collect_dft_rejects_unconverged(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fabricate_dft(store.folder(sid), cfg, sc)
    with open(os.path.join(store.folder(sid), "OUT.MgO",
                           "running_scf.log"), "w") as f:
        f.write("it exploded\n")
    assert convert.collect_dft_stage(cfg, ws, Args()) == 1
    assert store.list() == []                      # moved to rejected/
    rej = os.path.join(ws, "rejected", f"pilot_{sid}")
    st = json.load(open(os.path.join(rej, "status.json")))
    assert st["reason"] == "scf_not_converged"
