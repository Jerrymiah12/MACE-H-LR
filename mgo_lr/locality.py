"""Tier-3 dataset-level physics and locality diagnostics.

Never a per-snapshot rejection: results are reports under
generation_logs/locality/ feeding the dataset-level approval decision
(F_SR(r) < F_full(r) over the long-distance region before scaling up).
Physics comparisons are made only within matched comparison_family_id
groups.
"""
import json
import math
import os

import h5py
import numpy as np

from .config import atomic_write_text
from .convert import parse_key, read_blocks
from .lr import blocks_norm
from .snapshot import SnapshotStore


def block_distance(key, cart, cell):
    r0, r1, r2, i, j = parse_key(key)
    shift = np.array([r0, r1, r2], float) @ np.asarray(cell, float)
    cart = np.asarray(cart, float)
    return float(np.linalg.norm(cart[j - 1] + shift - cart[i - 1]))


def frobenius_inner(a, b):
    return float(sum(np.sum(a[k] * b[k]) for k in set(a) & set(b)))


def odd_response(h_plus, h_minus, h_lr, delta):
    """ΔH_DFT = (H(+A) - H(-A))/2 compared against H_LR(+A).  Diagnostic
    only: ΔH_DFT = ΔH_SR + H_LR, so no exact match is expected."""
    dh = {}
    for k in set(h_plus) | set(h_minus):
        p, m = h_plus.get(k), h_minus.get(k)
        if p is None:
            p = np.zeros_like(m)
        if m is None:
            m = np.zeros_like(p)
        dh[k] = 0.5 * (p - m)
    n_dh, n_lr = blocks_norm(dh), blocks_norm(h_lr)
    return {"cos_theta": frobenius_inner(dh, h_lr) / (n_dh * n_lr + delta),
            "r_lr": n_lr / (n_dh + delta)}


def tail_fractions(blocks, cart, cell, radii):
    dw = [(block_distance(k, cart, cell), float(np.sum(v * v)))
          for k, v in blocks.items()]
    total = sum(w for _, w in dw)
    if total <= 0.0:
        return [0.0 for _ in radii]
    return [sum(w for d, w in dw if d > r) / total for r in radii]


def binned_norms(blocks, cart, cell, bin_width):
    bins = {}
    for k, v in blocks.items():
        b = int(block_distance(k, cart, cell) // bin_width)
        bins.setdefault(b, []).append(float(np.linalg.norm(v)))
    return [{"r_lo": b * bin_width, "r_hi": (b + 1) * bin_width,
             "count": len(ns), "mean": float(np.mean(ns)),
             "median": float(np.median(ns)), "max": float(np.max(ns))}
            for b, ns in sorted(bins.items())]


def accumulate_binned_norms(accumulator, blocks, cart, cell, bin_width):
    for key, value in blocks.items():
        bucket = int(block_distance(key, cart, cell) // bin_width)
        accumulator.setdefault(bucket, []).append(float(np.linalg.norm(value)))


def summarize_binned_norms(accumulator, bin_width):
    return [{"r_lo": b * bin_width, "r_hi": (b + 1) * bin_width,
             "count": len(values), "mean": float(np.mean(values)),
             "median": float(np.median(values)), "max": float(np.max(values))}
            for b, values in sorted(accumulator.items())]


def h5_max_block_distance(path, cart, cell):
    with h5py.File(path, "r") as handle:
        distances = [block_distance(key, cart, cell) for key in handle.keys()]
    return max(distances) if distances else 0.0


def controlled_q_comparisons(metas, lr_norms):
    """Summarize matched-pattern LR norms over distinct |q| shells."""
    families = {}
    for sid, meta in metas.items():
        family_id = meta.get("wavevector_family_id")
        if not family_id:
            continue
        families.setdefault(family_id, []).append({
            "sid": sid, "q_magnitude": float(meta.get("q_magnitude") or 0.0),
            "lr_norm": float(lr_norms[sid]),
            "polarization_class": meta.get("polarization_class")})

    comparisons = []
    for family_id, members in sorted(families.items()):
        by_magnitude = {}
        for entry in members:
            magnitude = round(entry["q_magnitude"], 10)
            by_magnitude.setdefault(magnitude, []).append(entry)
        if len(by_magnitude) < 2:
            continue
        shell_means = []
        for magnitude, shell_members in sorted(by_magnitude.items()):
            shell_means.append({
                "q_magnitude": magnitude,
                "mean_lr_norm": float(np.mean(
                    [entry["lr_norm"] for entry in shell_members])),
                "sids": [entry["sid"] for entry in shell_members],
            })
        small, large = shell_means[0], shell_means[-1]
        comparisons.append({
            "family": family_id, "shells": shell_means,
            "small_q": small, "large_q": large,
            "small_q_has_stronger_lr":
                small["mean_lr_norm"] >= large["mean_lr_norm"]})
    return comparisons


def long_range_localizes(f_full, f_sr, floor, min_improvement):
    """Dataset-level approval criterion for `H^SR` localization.

    Over the long-distance half of the radius grid, and only where `H_full`
    still carries meaningful weight (`F_full > floor` — radii past the largest
    nonzero block are all-zero and carry no evidence), `F_SR` must be *measurably*
    below `F_full` (by at least `min_improvement`, relatively).  Equality is NOT
    a pass: a subtracted-LR Hamiltonian with the same tail as the full one shows
    no localization gain.
    """
    n = len(f_full)
    pairs = [(f_sr[i], f_full[i]) for i in range(n // 2, n) if f_full[i] > floor]
    return bool(pairs) and all(s <= f * (1.0 - min_improvement)
                               for s, f in pairs)


def _blocks_by_pair(blocks, cart, cell):
    """(i, j) -> [(interatomic vector, block)].

    Blocks must be matched between two snapshots by GEOMETRY, not by key:
    ABACUS assigns R against positions it wraps into the cell, so the same
    physical neighbour can carry different R labels in two geometries.
    """
    out = {}
    for key, value in blocks.items():
        rx, ry, rz, i, j = json.loads(key)
        v = cart[j - 1] + np.asarray([rx, ry, rz], float) @ cell - cart[i - 1]
        out.setdefault((i, j), []).append((v, value))
    return out


def farfield_sensitivity(h_ref, cart_ref, h_probe, h_lr_probe, h_lr_ref,
                         cart_probe, cell, atom, bin_width, tol=0.3,
                         s_ref=None, s_probe=None):
    """How much of the DFT response to a localized displacement `H^LR` explains,
    binned by distance from the displaced atom.

    `reduction = 1 - ||dH_SR|| / ||dH_full||` per bin, where `dH_SR` is the
    response left after the analytic long-range term is removed.  This is the
    property the LR split exists to deliver — a finite-cutoff network cannot
    see a distant displacement, so the analytic term must supply it — and it is
    what `H^LR ∝ S` can actually improve, unlike the spatial tail of `H` itself.

    `s_ref` and `s_probe` fix the energy gauge and are required for a
    trustworthy number.  `H` from a periodic SCF carries an arbitrary energy
    zero, so the two runs are free independently:
    `H_ref -> H_ref + c_ref S_ref` and `H_probe -> H_probe + c_probe S_probe`.
    Here their Fermi levels differ by 4 meV while the far-field response is
    ~1.4 meV per block, so that freedom SWAMPS the signal — dropping it
    silently reports whatever gauge ABACUS happened to choose.

    Each Hamiltonian is therefore gauge-fixed against ITS OWN overlap before
    differencing: `H~ = H - (<H,S>/<S,S>) S`, which is exactly invariant since
    adding `c S` shifts the coefficient by exactly `c`.  Projecting both runs
    onto one shared overlap direction would not do it — `S_probe != S_ref`, so
    a `c_probe S_probe` component survives.  The same fixing is applied to the
    `H^LR` labels, which is the gauge `lr.py` already builds `V` in
    (`V(G=0) = 0`).  A Hamiltonian block with no overlap partner means zero
    overlap, not missing data: it contributes 0 to the projection and is still
    counted in the response.
    """
    ref_pairs = _blocks_by_pair(h_ref, cart_ref, cell)
    probe_pairs = _blocks_by_pair(h_probe, cart_probe, cell)
    lr_probe_pairs = _blocks_by_pair(h_lr_probe, cart_probe, cell)
    lr_ref_pairs = _blocks_by_pair(h_lr_ref, cart_ref, cell)
    shifts = (np.array(list(np.ndindex(3, 3, 3)), float) - 1.0) @ cell

    def min_image(v):
        return float(np.min(np.linalg.norm(v + shifts, axis=1)))

    def nearest(pairs, key, v):
        candidates = pairs.get(key)
        if not candidates:
            return None
        w, blk = min(candidates, key=lambda t: np.linalg.norm(t[0] - v))
        return blk if np.linalg.norm(w - v) <= tol else None

    sr_pairs = _blocks_by_pair(s_ref, cart_ref, cell) if s_ref else None
    sp_pairs = _blocks_by_pair(s_probe, cart_probe, cell) if s_probe else None
    rows, unmatched = [], 0
    for (i, j), entries in ref_pairs.items():
        for v, ref_blk in entries:
            probe_blk = nearest(probe_pairs, (i, j), v)
            if probe_blk is None:
                unmatched += 1
                continue
            zero = np.zeros_like(ref_blk)

            def matched(pairs, key=(i, j), vec=v, default=zero):
                if pairs is None:
                    return default
                blk = nearest(pairs, key, vec)
                # absent sparse block == zero overlap, not missing data
                return default if blk is None else blk

            dist = min(min_image(cart_ref[i - 1] - cart_ref[atom]),
                       min_image(cart_ref[i - 1] + v - cart_ref[atom]))
            rows.append({"h_ref": ref_blk, "h_probe": probe_blk,
                         "s_ref": matched(sr_pairs),
                         "s_probe": matched(sp_pairs),
                         "lr_ref": matched(lr_ref_pairs),
                         "lr_probe": matched(lr_probe_pairs),
                         "dist": dist})

    # Gauge fix each snapshot against its OWN overlap, then difference.
    def coefficient(field, overlap):
        ss = sum(float(np.sum(r[overlap] ** 2)) for r in rows)
        if ss <= 0.0:
            return 0.0
        return sum(float(np.sum(r[field] * r[overlap])) for r in rows) / ss

    lam = {f: coefficient(f, s) for f, s in
           (("h_ref", "s_ref"), ("h_probe", "s_probe"),
            ("lr_ref", "s_ref"), ("lr_probe", "s_probe"))} if rows else {}

    acc = {}
    for r in rows:
        d_full = ((r["h_probe"] - lam["h_probe"] * r["s_probe"])
                  - (r["h_ref"] - lam["h_ref"] * r["s_ref"]))
        d_lr = ((r["lr_probe"] - lam["lr_probe"] * r["s_probe"])
                - (r["lr_ref"] - lam["lr_ref"] * r["s_ref"]))
        d_sr = d_full - d_lr
        bucket = int(r["dist"] // bin_width)
        slot = acc.setdefault(bucket, [0.0, 0.0, 0])
        slot[0] += float(np.sum(d_full ** 2))
        slot[1] += float(np.sum(d_sr ** 2))
        slot[2] += 1
    bins = []
    for bucket in sorted(acc):
        full, sr, count = acc[bucket]
        full, sr = math.sqrt(full), math.sqrt(sr)
        bins.append({"r_lo": bucket * bin_width,
                     "r_hi": (bucket + 1) * bin_width,
                     "count": count, "dh_full": full, "dh_sr": sr,
                     "reduction": (1.0 - sr / full) if full > 0.0 else 0.0})
    return bins, unmatched


def farfield_gate(bins, min_radius, noise_floor, min_blocks, min_improvement):
    """Every bin beyond `min_radius` carrying real signal must improve.

    Bins below `noise_floor` are SCF noise, not response — including them
    produces meaningless reductions — and thin bins are excluded outright.
    Requires at least one qualifying bin: no evidence is not a pass.
    """
    qualifying = [b for b in bins
                  if b["r_lo"] >= min_radius
                  and b["count"] >= min_blocks
                  and b["dh_full"] >= noise_floor]
    return (bool(qualifying)
            and all(b["reduction"] >= min_improvement for b in qualifying),
            qualifying)


def summarize_farfield_bins(per_combination, min_radius, min_blocks):
    """Per-bin MINIMUM reduction across every reference x probe combination.

    The gate is only as strong as its worst combination, so report that rather
    than a best case that no single pairing achieves.
    """
    worst = {}
    for bins in per_combination.values():
        for b in bins:
            if b["r_lo"] < min_radius or b["count"] < min_blocks:
                continue
            cur = worst.get(b["r_lo"])
            if cur is None or b["reduction"] < cur["reduction"]:
                worst[b["r_lo"]] = {"r_lo": b["r_lo"], "r_hi": b["r_hi"],
                                    "reduction": b["reduction"]}
    return [worst[k] for k in sorted(worst)]


def farfield_reference_spread(per_combination):
    """Max minus min reduction per bin over the references, per probe.

    The references differ only by being separate SCF runs, so this is the
    run-to-run scatter the margin over the threshold must exceed.
    """
    by_probe = {}
    for combo, bins in per_combination.items():
        probe = combo.split("|", 1)[-1]
        for b in bins:
            by_probe.setdefault(probe, {}).setdefault(
                b["r_lo"], []).append(b["reduction"])
    out = {}
    for probe, per_bin in sorted(by_probe.items()):
        out[probe] = [{"r_lo": r, "spread": max(v) - min(v),
                       "n_references": len(v)}
                      for r, v in sorted(per_bin.items()) if len(v) > 1]
    return out


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
