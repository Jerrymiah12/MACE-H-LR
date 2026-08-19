"""ABACUS -> DeepH-E3/MACE-H data conversion.

The per-l orbital permutation/sign table is copied verbatim from
DeepH-pack deeph/preprocess/abacus_get_data.py (class OrbAbacus2DeepH).
A silent error here corrupts every matrix — the table is pinned by
tests/integration/mgo_dataset/test_orbital_reorder.py and must not be re-derived.
"""
import json
import os

import h5py
import numpy as np
from scipy.linalg import block_diag

from ...config import atomic_write_text
from ...response.constants import ATOMIC_NUMBERS


def _build_us():
    us = {0: np.eye(1),
          1: np.eye(3)[[1, 2, 0]],
          2: np.eye(5)[[0, 3, 4, 1, 2]],
          3: np.eye(7)}
    minus = {1: [0, 1], 2: [3, 4], 3: [1, 2, 5, 6]}
    for l, rows in minus.items():
        us[l][rows] *= -1.0
    return us


_U_ABACUS2DEEPH = _build_us()


def orbital_u(l):
    if l not in _U_ABACUS2DEEPH:
        raise NotImplementedError(f"only l <= 3 supported, got l={l}")
    return _U_ABACUS2DEEPH[l]


def atom_u(orbital_types_atom):
    """Block-diagonal transform for one atom's full AO set."""
    return block_diag(*[orbital_u(l) for l in orbital_types_atom])


def transform_block(mat, l_left, l_right):
    """U_i @ mat @ U_j.T for an atom-pair block."""
    return atom_u(l_left) @ np.asarray(mat, float) @ atom_u(l_right).T


BLOCK_SKIP_THRESHOLD = 1e-8   # same block-sparsity cutoff as DeepH-pack


def key_str(R, i, j):
    """DeepH-E3 h5 key: JSON list, 1-based atom indices (i, j are 0-based in)."""
    return f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {i + 1}, {j + 1}]"


def parse_key(k):
    v = json.loads(k)
    return (v[0], v[1], v[2], v[3], v[4])


def write_blocks(path, blocks):
    tmp = f"{path}.tmp.{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        for k, v in blocks.items():
            f[k] = np.asarray(v, np.float64)
    os.replace(tmp, path)


def read_blocks(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k], dtype=np.float64)
    return out


def species_orbital_info(cfg, species_list):
    types = [cfg["abacus"]["orbital_types"][s] for s in species_list]
    norb = [sum(2 * l + 1 for l in t) for t in types]
    offsets = np.concatenate([[0], np.cumsum(norb)])
    return types, norb, offsets


def matrices_to_blocks(csr_blocks, dim, cfg, species_list, factor):
    """Slice per-R matrices into atom-pair blocks, apply the orbital
    transform, scale by `factor` (RY_TO_EV for H, 1.0 for S)."""
    types, norb, offsets = species_orbital_info(cfg, species_list)
    if int(offsets[-1]) != dim:
        raise ValueError(f"matrix dimension {dim} != expected {offsets[-1]} "
                         f"from orbital_types for {len(species_list)} atoms")
    n_at = len(species_list)
    out = {}
    for R, m in csr_blocks.items():
        dense = m.toarray()
        if not np.all(np.isfinite(dense)):
            raise ValueError(f"NaN/Inf in matrix block R={R}")
        for i in range(n_at):
            for j in range(n_at):
                blk = dense[offsets[i]:offsets[i + 1],
                            offsets[j]:offsets[j + 1]]
                if np.abs(blk).max() < BLOCK_SKIP_THRESHOLD:
                    continue
                out[key_str(R, i, j)] = factor * transform_block(
                    blk, types[i], types[j])
    return out


def write_structure_files(folder, cell, cart, species, cfg, fermi_ev):
    cell = np.asarray(cell, float)
    types, norb, _ = species_orbital_info(cfg, species)
    np.savetxt(os.path.join(folder, "lat.dat"), cell.T)
    np.savetxt(os.path.join(folder, "rlat.dat"),
               np.linalg.inv(cell) * 2.0 * np.pi)
    # ABACUS wraps atoms into [0,1) internally, so the R labels in its
    # out_mat_hs2 CSR files are relative to WRAPPED positions.  A displacement
    # that pushes an atom at frac 0 slightly negative therefore shifts that
    # atom's R labels by one lattice vector relative to the input STRU.  Write
    # wrapped positions so (R, i, j) + site_positions.dat reproduces the true
    # interatomic vector; without this every distance-based consumer (locality
    # tails, MACE-H edge features) is wrong by a lattice vector for those atoms.
    # Callers that need u still use minimum-image differences, so wrapping here
    # is invisible to them.
    # Snap first: an undisplaced atom at fractional 0 lands on ~-1e-17, which
    # ABACUS treats as 0 but a bare modulo would push a whole lattice vector
    # the wrong way.  Real displacements are ~1e-3, so 1e-9 separates them
    # cleanly.
    frac = np.asarray(cart, float) @ np.linalg.inv(cell)
    near = np.abs(frac - np.round(frac)) < 1e-9
    frac = np.where(near, np.round(frac), frac)
    np.savetxt(os.path.join(folder, "site_positions.dat"),
               ((frac % 1.0) @ cell).T)
    atomic_write_text(os.path.join(folder, "element.dat"),
                      "\n".join(str(ATOMIC_NUMBERS[s]) for s in species) + "\n")
    atomic_write_text(os.path.join(folder, "orbital_types.dat"),
                      "\n".join("  ".join(str(l) for l in t)
                                for t in types) + "\n")
    info = {"nsites": len(species), "isorthogonal": False,
            "isspinful": False, "norbits": int(sum(norb)),
            "fermi_level": fermi_ev if fermi_ev is not None else 0.0}
    atomic_write_text(os.path.join(folder, "info.json"), json.dumps(info))
