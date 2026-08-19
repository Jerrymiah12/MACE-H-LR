"""Dataset-level locality-report campaign stage."""
import json
import os

import numpy as np

from maceh.analysis.locality import (
    accumulate_binned_norms,
    blocks_norm,
    controlled_q_comparisons,
    farfield_gate,
    farfield_reference_spread,
    farfield_sensitivity,
    h5_max_block_distance,
    long_range_localizes,
    odd_response,
    summarize_binned_norms,
    summarize_farfield_bins,
    tail_fractions,
)
from maceh.config import atomic_write_text
from maceh.data.io.blocks import read_blocks
from workflows.mgo_dataset.snapshot import SnapshotStore


def locality_report_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("locality-report requires --set pilot|main|large")
    delta = float(cfg["validation"]["delta"])
    bin_width = float(cfg["locality"]["bin_width"])
    store = SnapshotStore(workspace, args.set_name)
    sids = [s for s in store.list()
            if store.read_status(s)["state"] == "validated"]
    if not sids:
        print(f"{args.set_name}: no validated snapshots; nothing to report")
        return 0
    def blocks(sid, kind):
        return read_blocks(os.path.join(store.folder(sid),
                                        f"hamiltonians_{kind}.h5"))

    # Build one radius grid covering every snapshot.  Matrix sparsity can make
    # later snapshots contain longer-R keys than the first one.
    folder0 = store.folder(sids[0])
    cell = np.loadtxt(os.path.join(folder0, "lat.dat")).T   # columns -> rows
    rmax = 0.0
    for sid in sids:
        folder = store.folder(sid)
        cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        for kind in ("full", "lr", "sr"):
            rmax = max(rmax, h5_max_block_distance(
                os.path.join(folder, f"hamiltonians_{kind}.h5"),
                cart, cell))
    radii = [bin_width * i for i in range(1, int(rmax // bin_width) + 2)]

    # Stream one snapshot at a time: accumulate tail fractions and the scalar
    # H_LR norms, holding only the current snapshot's blocks in memory (the
    # planned 400-structure set would need many GB if held all at once).
    metas, lr_norms = {}, {}
    tails = {"full": [], "lr": [], "sr": []}
    binned_acc = {"full": {}, "lr": {}, "sr": {}}
    for sid in sids:
        folder = store.folder(sid)
        with open(os.path.join(folder, "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
        cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        hf = blocks(sid, "full")
        tails["full"].append(tail_fractions(hf, cart, cell, radii))
        accumulate_binned_norms(binned_acc["full"], hf, cart, cell, bin_width)
        del hf
        hl = blocks(sid, "lr")
        tails["lr"].append(tail_fractions(hl, cart, cell, radii))
        accumulate_binned_norms(binned_acc["lr"], hl, cart, cell, bin_width)
        lr_norms[sid] = blocks_norm(hl)
        del hl
        hs = blocks(sid, "sr")
        tails["sr"].append(tail_fractions(hs, cart, cell, radii))
        accumulate_binned_norms(binned_acc["sr"], hs, cart, cell, bin_width)
        del hs
    f_mean = {k: np.mean(np.array(v), axis=0).tolist()
              for k, v in tails.items()}
    f_sr_ok = long_range_localizes(
        f_mean["full"], f_mean["sr"],
        float(cfg["locality"].get("tail_floor", 1e-6)),
        float(cfg["locality"].get("min_tail_improvement", 0.05)))
    binned = {kind: summarize_binned_norms(values, bin_width)
              for kind, values in binned_acc.items()}

    # Odd-response pairs: load only the ± pair members' blocks, momentarily.
    odd = []
    for sid in sids:
        m = metas[sid]
        partner = m.get("sign_partner_id")
        amp = float(m.get("amplitude") or 0.0)
        if partner in metas and amp > 0.0:
            entry = odd_response(blocks(sid, "full"), blocks(partner, "full"),
                                 blocks(sid, "lr"), delta)
            entry.update({"sids": [sid, partner], "amplitude": amp,
                          "family": m.get("comparison_family_id")})
            odd.append(entry)

    families = {}
    for sid in sids:
        m = metas[sid]
        fam = families.setdefault(
            m.get("comparison_family_id"),
            {"q_magnitude": m.get("q_magnitude"), "members": []})
        fam["members"].append({
            "sid": sid, "polarization_class": m.get("polarization_class"),
            "amplitude": m.get("amplitude"),
            "lr_norm": lr_norms[sid]})
    for fam in families.values():
        by_class = {}
        for e in fam["members"]:
            by_class.setdefault(e["polarization_class"] or "none",
                                []).append(e["lr_norm"])
        fam["mean_lr_norm_by_class"] = {c: float(np.mean(v))
                                        for c, v in by_class.items()}

    # Controlled |q| comparisons: wavevector_family_id matches amplitude,
    # phase, normalization, species ratio, polarization, and supercell while
    # deliberately excluding q magnitude.
    q_comparisons = controlled_q_comparisons(metas, lr_norms)

    # Tier-3 gate: far-field sensitivity of H^SR to a localized displacement.
    loc = cfg["locality"]
    references = [s for s in sids
                  if metas[s].get("farfield_role") == "reference"]
    if not references:       # any exact-equilibrium snapshot serves as one
        references = [
            s for s in sids
            if metas[s].get("displaced_atom_index") is None
            and float(np.abs(np.load(os.path.join(
                store.folder(s), "displacements.npy"))).max()) == 0.0][:1]
    probes = [s for s in sids
              if s not in references
              and metas[s].get("displaced_atom_index") is not None]
    farfield = {"references": references, "probes": {}, "unmatched": {},
                "per_combination_pass": {}}
    gate_args = (float(loc.get("farfield_min_radius", 4.0)),
                 float(loc.get("farfield_noise_floor", 1e-6)),
                 int(loc.get("farfield_min_blocks", 20)),
                 float(loc.get("min_farfield_improvement", 0.05)))
    qualifying = []
    # Every probe is compared against EVERY reference.  The references share a
    # geometry but are independent SCF runs, so the spread between them is
    # run-to-run scatter, which the margin over the threshold has to survive.
    for ref_sid in references:
        cart_ref = np.loadtxt(os.path.join(store.folder(ref_sid),
                                           "site_positions.dat")).T
        h_ref, lr_ref = blocks(ref_sid, "full"), blocks(ref_sid, "lr")
        s_ref = read_blocks(os.path.join(store.folder(ref_sid),
                                         "overlaps.h5"))
        for sid in probes:
            cart_p = np.loadtxt(os.path.join(store.folder(sid),
                                             "site_positions.dat")).T
            bins, unmatched = farfield_sensitivity(
                h_ref, cart_ref, blocks(sid, "full"), blocks(sid, "lr"),
                lr_ref, cart_p, cell,
                int(metas[sid]["displaced_atom_index"]), bin_width,
                s_ref=s_ref,
                s_probe=read_blocks(os.path.join(store.folder(sid),
                                                 "overlaps.h5")))
            ok, qual = farfield_gate(bins, *gate_args)
            combo = f"{ref_sid}|{sid}"
            farfield["probes"][combo] = bins
            farfield["unmatched"][combo] = unmatched
            farfield["per_combination_pass"][combo] = ok
            qualifying.extend(qual)
        del h_ref, lr_ref, s_ref
    # every reference x probe combination must pass; no evidence is not a pass
    ff_ok = bool(farfield["per_combination_pass"]) \
        and all(farfield["per_combination_pass"].values())
    farfield["worst_bin_reduction"] = summarize_farfield_bins(
        farfield["probes"], float(loc.get("farfield_min_radius", 4.0)),
        int(loc.get("farfield_min_blocks", 20)))
    farfield["reference_spread"] = farfield_reference_spread(
        farfield["probes"])
    farfield["qualifying_bins"] = qualifying
    farfield["lr_explains_far_field"] = ff_ok

    report = {"set": args.set_name, "n_snapshots": len(sids),
              "farfield": farfield,
              "tail": {"radii": radii, "F_full": f_mean["full"],
                       "F_lr": f_mean["lr"], "F_sr": f_mean["sr"],
                       # retained as a diagnostic only: H^LR = (V_i+V_j)/2 S_ij
                       # inherits H_full's radial shape, so this can never move
                       # much regardless of how good the LR term is.
                       "f_sr_below_f_full": f_sr_ok},
              "binned": binned,
              "odd_response": odd, "families": families,
              "wavevector_comparisons": q_comparisons}
    out_dir = os.path.join(workspace, "generation_logs", "locality")
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_text(os.path.join(out_dir,
                                   f"locality_{args.set_name}.json"),
                      json.dumps(report, indent=1))
    if not references or not probes:
        verdict = ("NOT EVALUATED (no far-field probe pair; regenerate the "
                   "set to add farfield_reference/farfield_probe)")
    else:
        worst = min((b["reduction"]
                     for b in farfield["worst_bin_reduction"]), default=0.0)
        spread = max((e["spread"] for v in farfield["reference_spread"].values()
                      for e in v), default=0.0)
        verdict = (f"{'PASS' if ff_ok else 'NOT YET'} "
                   f"({len(references)} reference(s) x {len(probes)} probe(s), "
                   f"worst bin {100 * worst:.1f}% vs "
                   f"{100 * float(loc.get('min_farfield_improvement', 0.05)):.0f}% "
                   f"threshold, reference spread {100 * spread:.2f} pp)")
    print(f"{args.set_name}: locality report for {len(sids)} snapshots; "
          f"H_SR less sensitive to distant displacement: {verdict}")
    return 0

