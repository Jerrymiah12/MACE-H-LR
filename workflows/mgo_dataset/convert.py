"""Dataset-stage wrapper for ABACUS to MACE-H conversion."""
import os

import numpy as np

from maceh.config import sha256_file
from maceh.data.io import abacus as abacus_io
from maceh.data.io.blocks import (
    key_str,
    matrices_to_blocks,
    write_blocks,
    write_structure_files,
)
from maceh.data.structures import make_supercell
from maceh.response.constants import RY_TO_EV
from workflows.mgo_dataset.snapshot import SnapshotStore, load_reference


def collect_dft_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("collect-dft requires --set pilot|main|large")
    ref = load_reference(workspace)
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    store = SnapshotStore(workspace, args.set_name)
    tau_diag = float(cfg["validation"]["tau_overlap_diag"])
    exit_code, converted, skipped = 0, 0, 0
    for sid in store.list():
        if store.read_status(sid)["state"] == "rejected":
            continue
        if store.state_at_least(sid, "converted") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        out_dir = os.path.join(folder, "OUT.MgO")
        log = os.path.join(out_dir, "running_scf.log")
        if not os.path.exists(log):
            continue                       # DFT not run yet: stay prepared
        scf = abacus_io.parse_running_scf(log)
        if not scf["converged"]:
            store.reject(sid, "scf_not_converged")
            exit_code = 1
            continue
        h_path = os.path.join(out_dir, cfg["abacus"]["csr_h_filename"])
        s_path = os.path.join(out_dir, cfg["abacus"]["csr_s_filename"])
        if not (os.path.exists(h_path) and os.path.exists(s_path)):
            store.reject(sid, "csr_files_missing")
            exit_code = 1
            continue
        cell, cart, species = abacus_io.parse_stru(
            os.path.join(folder, "STRU"))
        u = np.load(os.path.join(folder, "displacements.npy"))
        if species != sc.species:
            store.reject(sid, "atom_order_changed")
            exit_code = 1
            continue
        if not (np.allclose(cell, sc.cell, atol=1e-8)
                and np.allclose(cart, sc.cart + u, atol=1e-6)):
            store.reject(sid, "geometry_mismatch_vs_reference")
            exit_code = 1
            continue
        try:
            dim_h, h_csr = abacus_io.parse_csr(h_path)
            dim_s, s_csr = abacus_io.parse_csr(s_path)
            if dim_h != dim_s:
                raise ValueError(f"H dim {dim_h} != S dim {dim_s}")
            h_blocks = matrices_to_blocks(h_csr, dim_h, cfg, species,
                                          RY_TO_EV)
            s_blocks = matrices_to_blocks(s_csr, dim_s, cfg, species, 1.0)
        except ValueError as e:
            store.reject(sid, f"matrix_parse_failed: {e}")
            exit_code = 1
            continue
        bad_diag = False
        for i in range(len(species)):
            k = key_str((0, 0, 0), i, i)
            if k not in s_blocks or \
                    np.abs(np.diag(s_blocks[k]) - 1.0).max() > tau_diag:
                bad_diag = True
        if bad_diag:
            store.reject(sid, "pathological_overlap_diagonal")
            exit_code = 1
            continue
        full_path = os.path.join(folder, "hamiltonians_full.h5")
        if os.path.exists(full_path) and not args.force:
            raise SystemExit(f"{sid}: hamiltonians_full.h5 exists; "
                             "refusing to overwrite without --force")
        write_blocks(full_path, h_blocks)
        write_blocks(os.path.join(folder, "overlaps.h5"), s_blocks)
        write_structure_files(folder, cell, cart, species, cfg,
                              scf["fermi_ev"])
        store.write_status(
            sid, "converted", scf_converged=True, etot_ev=scf["etot_ev"],
            csr_files=[cfg["abacus"]["csr_h_filename"],
                       cfg["abacus"]["csr_s_filename"]],
            raw_sha256={os.path.basename(p): sha256_file(p)
                        for p in (h_path, s_path, log)})
        converted += 1
    print(f"{args.set_name}: converted {converted}, skipped {skipped}")
    return exit_code
