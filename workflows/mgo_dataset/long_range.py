"""Campaign stages and provenance checks for analytic long-range labels."""
import json
import os

import numpy as np

from maceh.config import atomic_write_text
from maceh.data.io.blocks import read_blocks, write_blocks
from maceh.data.structures import make_supercell, reciprocal, remove_uniform_translation
from maceh.response.long_range import (
    assemble_lr_hamiltonian,
    blocks_diff_norm,
    blocks_norm,
    check_reciprocal_set,
    evaluate_potential,
    gmax_squared,
    imaginary_residual,
    lr_coefficients,
    minimum_image_displacements,
    reciprocal_set,
)
from maceh.response.provenance import (
    expected_lr_definition,
    lr_definition as _lr_definition,
    reference_fingerprints,
    require_current_lr_definition,
)
from workflows.mgo_dataset import __version__
from workflows.mgo_dataset.snapshot import SnapshotStore, load_reference


def _record_lr_definition(workspace, lr_def):
    import yaml
    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    stored = data.get("lr_definition")
    if stored is not None and stored != lr_def:
        raise SystemExit(
            "metadata.yaml already records a different lr_definition — "
            "refusing to mix LR definitions in one workspace (change the "
            "workspace or restore the original Λ/cutoff config)")
    units = {"energy": "eV", "length": "angstrom", "charge": "e"}
    if data.get("units") not in (None, units):
        raise SystemExit(
            "metadata.yaml records incompatible units; expected "
            "energy=eV, length=angstrom, charge=e")
    data["lr_definition"] = lr_def
    data["units"] = units
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def _invalidate_lr_outputs(folder, workspace):
    """Remove derived labels/exports after a failed forced LR calculation."""
    marker = os.path.join(folder, "export_metadata.json")
    exported_lr_target = False
    if os.path.isfile(marker):
        try:
            with open(marker) as f:
                exported_lr_target = json.load(f).get("target") in ("lr", "sr")
        except (json.JSONDecodeError, OSError):
            exported_lr_target = True
    target_path = os.path.join(folder, "hamiltonians.h5")
    if os.path.islink(target_path):
        exported_lr_target = (
            os.path.basename(os.readlink(target_path))
            in ("hamiltonians_lr.h5", "hamiltonians_sr.h5"))

    for name in ("hamiltonians_lr.h5", "hamiltonians_sr.h5",
                 "lr_metadata.json", "quality_checks.json"):
        path = os.path.join(folder, name)
        if os.path.lexists(path):
            os.remove(path)

    if exported_lr_target:
        for name in ("hamiltonians.h5", "export_metadata.json"):
            path = os.path.join(folder, name)
            if os.path.lexists(path):
                os.remove(path)

        import yaml
        meta_path = os.path.join(workspace, "metadata.yaml")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                data = yaml.safe_load(f) or {}
            if data.get("training_target") in ("lr", "sr"):
                data.pop("training_target")
                atomic_write_text(meta_path,
                                  yaml.safe_dump(data, sort_keys=False))


def lr_process_stage(cfg, workspace, args):

    if getattr(args, "set_name", None) is None:
        raise SystemExit("lr-process requires --set pilot|main|large")
    ref = load_reference(workspace)
    ref_dir = os.path.join(workspace, "reference")
    born = np.load(os.path.join(ref_dir, "born_effective_charges.npy"))
    eps = np.load(os.path.join(ref_dir, "dielectric_infinity.npy"))
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    lam = float(cfg["lr"]["ewald_lambda"])
    tol = float(cfg["lr"]["reciprocal_tolerance"])
    tau_imag = float(cfg["lr"]["imaginary_tolerance"])
    factor = float(cfg["lr"]["convergence_factor"])
    delta = float(cfg["validation"]["delta"])
    rec = reciprocal(sc.cell)
    volume = abs(float(np.linalg.det(sc.cell)))
    gmax_sq = gmax_squared(lam, tol)
    n_int, g_cart = reciprocal_set(rec, eps, gmax_sq)
    rep = check_reciprocal_set(n_int)
    if not rep["ok"] or rep["number_of_vectors"] == 0:
        raise SystemExit(f"reciprocal set invalid or empty: {rep}")
    n_int2, g2 = reciprocal_set(rec, eps, gmax_sq * factor ** 2)
    lr_def = _lr_definition(cfg, gmax_sq, reference_fingerprints(workspace))
    _record_lr_definition(workspace, lr_def)

    store = SnapshotStore(workspace, args.set_name)
    exit_code, processed, skipped = 0, 0, 0
    for sid in store.list():
        st = store.read_status(sid)
        if st["state"] == "rejected" \
                or not store.state_at_least(sid, "converted"):
            continue
        if store.state_at_least(sid, "lr_done") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        if args.force:
            # Transaction boundary: both symlink and copy exports must be
            # invalidated before rewriting their source labels.  If any later
            # read/calculation raises unexpectedly, the state remains safely
            # below lr_done and no stale derived label can be published.
            _invalidate_lr_outputs(folder, workspace)
            store.write_status(sid, "converted", lr_failed=None,
                               r_imag=None, lr_convergence=None,
                               lr_convergence_abs=None,
                               lr_rerun_pending=True)
        pos = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        u = minimum_image_displacements(sc.cell, pos, sc.cart)
        u_stored = np.load(os.path.join(folder, "displacements.npy"))
        if np.abs(u - u_stored).max() > 1e-6:
            print(f"WARNING {sid}: recomputed u differs from "
                  f"displacements.npy by {np.abs(u - u_stored).max():.2e} Å")
        u_rel = remove_uniform_translation(u)          # processor-level ASR
        dipoles = np.einsum("nab,nb->na", born[sc.basis_index], u_rel)
        coeffs = lr_coefficients(g_cart, dipoles, sc.cart, eps, lam, volume)
        v_c = evaluate_potential(g_cart, coeffs, pos)  # snapshot AO centers
        r_imag = imaginary_residual(v_c, delta)
        if not np.isfinite(r_imag) or r_imag >= tau_imag:
            _invalidate_lr_outputs(folder, workspace)
            atomic_write_text(
                os.path.join(folder, "lr_failure.json"),
                json.dumps({"r_imag": r_imag, "reciprocal_set": rep,
                            "n_vectors": int(len(n_int)),
                            "lr_definition": lr_def}, indent=1))
            store.write_status(sid, "converted",
                               lr_failed=f"imaginary_residual {r_imag:.3e}",
                               r_imag=None, lr_convergence=None,
                               lr_convergence_abs=None,
                               lr_rerun_pending=False)
            exit_code = 1
            continue
        v_atom = np.real(v_c)
        s_blocks = read_blocks(os.path.join(folder, "overlaps.h5"))
        h_full = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        h_lr = assemble_lr_hamiltonian(s_blocks, v_atom)
        coeffs2 = lr_coefficients(g2, dipoles, sc.cart, eps, lam, volume)
        v2 = np.real(evaluate_potential(g2, coeffs2, pos))
        h_lr2 = assemble_lr_hamiltonian(s_blocks, v2)
        # Report the cutoff difference both ways.  The relative ratio alone is
        # unstable for transverse modes: the analytic dipolar response is
        # essentially zero there (|H_LR| ~ 1e-11 eV), so a converged label with
        # a ~1e-15 eV cutoff difference still divides by a near-zero
        # denominator and reads as a large relative error.
        conv_abs = blocks_diff_norm(h_lr2, h_lr)
        conv_norm = blocks_norm(h_lr2)      # ||H_LR|| at the larger cutoff
        conv = conv_abs / (conv_norm + delta)
        h_sr = {}
        for k in set(h_full) | set(h_lr):
            hf = h_full.get(k)
            hl = h_lr.get(k)
            if hf is None:
                hf = np.zeros_like(hl)
            if hl is None:
                hl = np.zeros_like(hf)
            h_sr[k] = hf - hl
        write_blocks(os.path.join(folder, "hamiltonians_lr.h5"), h_lr)
        write_blocks(os.path.join(folder, "hamiltonians_sr.h5"), h_sr)
        atomic_write_text(os.path.join(folder, "lr_metadata.json"),
                          json.dumps({"lr_definition": lr_def,
                                      "reciprocal_set": rep,
                                      "n_vectors": int(len(n_int)),
                                      "r_imag": r_imag,
                                      "lr_convergence": conv,
                                      "lr_convergence_abs": conv_abs,
                                      "lr_norm_converged": conv_norm,
                                      "code_version": __version__}, indent=1))
        failure_path = os.path.join(folder, "lr_failure.json")
        if os.path.exists(failure_path):
            os.remove(failure_path)
        store.write_status(sid, "lr_done", r_imag=r_imag,
                           lr_convergence=conv, lr_convergence_abs=conv_abs,
                           lr_failed=None, lr_rerun_pending=False)
        processed += 1
    print(f"{args.set_name}: lr-processed {processed}, skipped {skipped}")
    return exit_code
