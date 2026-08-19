"""Generate the five primary SR-vs-full-H result figures.

The scientific comparison is performed on the frozen 37-snapshot main test
split, which was not used for fitting or model selection.  Because the two
networks predict different targets, the SR prediction is always reconstructed
in full-H space before it is scored::

    H_full(SR model) = H_SR(predicted) + H_LR(analytic)

Figure 5 uses the held-out equilibrium 2x2x2 pilot cell.  Its smaller 224-AO
Hamiltonian makes a generalized-eigenvalue band calculation practical while
remaining outside the training and validation sets.

Outputs (PNG + PDF) and ``metrics.json`` are written below
``$MACEH_RUNS_ROOT/result_figures`` by default.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.linalg import eigh


from maceh.analysis.figures import save_figure_formats
from maceh.data.io.blocks import parse_key, read_blocks
from maceh.analysis.locality import block_distance
from workflows.training import paths
from workflows.training.evaluate import build_eval_config, predict, snapshot_dirs


SR_COLOR = "#0072B2"
FULL_COLOR = "#D55E00"
TRUTH_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
DISTANCE_BIN = 1.0
PARITY_STRIDE = 1000
ORBITAL_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}
ELEMENT_NAMES = {8: "O", 12: "Mg"}


class Accumulator:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.n = 0
        self.max_abs = 0.0

    def add(self, err):
        arr = np.asarray(err, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("non-finite prediction error")
        self.abs_sum += float(np.abs(arr).sum())
        self.sq_sum += float(np.square(arr).sum())
        self.n += int(arr.size)
        if arr.size:
            self.max_abs = max(self.max_abs, float(np.abs(arr).max()))

    def summary(self):
        return {
            "mae": self.abs_sum / self.n,
            "rmse": math.sqrt(self.sq_sum / self.n),
            "max_abs": self.max_abs,
            "n_elements": self.n,
        }


def newest_run(training_root, name):
    candidates = sorted(glob.glob(os.path.join(training_root, name, "*")))
    candidates = [p for p in candidates
                  if os.path.isdir(p)
                  and os.path.isfile(os.path.join(p, "best_model.pkl"))]
    if not candidates:
        raise SystemExit(f"no completed run below {training_root}/{name}")
    return candidates[-1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save_figure(fig, out_dir, stem):
    png, pdf = save_figure_formats(fig, out_dir, stem, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}\n  wrote {pdf}")


EPOCH_RE = re.compile(
    r"Epoch #(\d+)\s+\|.*?LR:\s*([0-9.eE+-]+)\s+\|"
    r".*?Train loss:\s*([0-9.eE+-]+)\s+\|"
    r" Val loss:\s*([0-9.eE+-]+)")
BEST_RE = re.compile(
    r"Using best model at epoch (\d+) with val_loss ([0-9.eE+-]+)")


def read_curve(run_dir):
    with open(os.path.join(run_dir, "result.txt"), errors="replace") as fh:
        text = fh.read()
    rows = [(int(e), float(lr), float(t), float(v))
            for e, lr, t, v in EPOCH_RE.findall(text)]
    if not rows:
        raise SystemExit(f"{run_dir}/result.txt contains no epochs")
    best_matches = BEST_RE.findall(text)
    if not best_matches:
        raise SystemExit(f"{run_dir}/result.txt contains no best-model record")
    best_epoch, best_loss = best_matches[-1]
    return np.asarray(rows, dtype=float), (int(best_epoch), float(best_loss))


def smooth_log(values, window=25):
    values = np.asarray(values, float)
    if len(values) < window:
        return np.arange(len(values)), values
    kernel = np.ones(window) / window
    smoothed = 10 ** np.convolve(np.log10(values), kernel, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, smoothed


def figure_training_curves(sr_run, full_run, out_dir):
    curves = {"SR residual target ($H_{SR}$)": (*read_curve(sr_run), SR_COLOR),
              "Full-H target": (*read_curve(full_run), FULL_COLOR)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, column, title in zip(
            axes, (2, 3), ("Training loss", "Validation loss")):
        for label, (curve, best_record, color) in curves.items():
            epochs = curve[:, 0]
            values = curve[:, column]
            ax.plot(epochs, values, color=color, alpha=0.16, lw=0.6)
            sx, sy = smooth_log(values)
            ax.plot(epochs[sx], sy, color=color, lw=2.0, label=label)
            if column == 3:
                best_epoch, best_loss = best_record
                ax.scatter(best_epoch, best_loss, s=38, color=color,
                           edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        style_axes(ax)
    axes[0].set_ylabel("Target-space MSE (eV$^2$)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Optimization history (25-epoch geometric mean)", y=1.01)
    fig.text(0.5, -0.01,
             "Targets differ; cross-model physical accuracy is compared in Figures 2–5.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_1_training_validation_loss")
    return {
        label: {"epochs": int(len(curve)),
                "best_train_loss": float(np.min(curve[:, 2])),
                "best_val_loss": best_record[1],
                "best_val_epoch": best_record[0]}
        for label, (curve, best_record, _) in curves.items()
    }


def orbital_layout(folder):
    numbers = np.loadtxt(os.path.join(folder, "element.dat"), dtype=int)
    with open(os.path.join(folder, "orbital_types.dat")) as fh:
        shells = [list(map(int, line.split())) for line in fh if line.strip()]
    if len(numbers) != len(shells):
        raise ValueError("element/orbital_types atom counts differ")
    labels = []
    ordered = []
    for z, atom_shells in zip(numbers, shells):
        element = ELEMENT_NAMES.get(int(z), f"Z{int(z)}")
        expanded = []
        for ell in atom_shells:
            label = f"{element}-{ORBITAL_NAMES.get(ell, f'l{ell}')}"
            expanded.extend([label] * (2 * ell + 1))
            if label not in ordered:
                ordered.append(label)
        labels.append(expanded)
    # Put chemistry in a stable, readable order rather than atom encounter
    # order (the dataset is species-major: Mg atoms precede O atoms).
    preferred = [f"{element}-{orbital}"
                 for element in ("Mg", "O")
                 for orbital in ("s", "p", "d", "f")]
    ordered = [x for x in preferred if x in ordered]
    index = {name: i for i, name in enumerate(ordered)}
    codes = [np.asarray([index[x] for x in atom], dtype=np.int64)
             for atom in labels]
    return ordered, codes


def score_rich(pred_by_sid, truth_dirs, add_lr, parity_stride=PARITY_STRIDE):
    overall = Accumulator()
    by_distance = defaultdict(Accumulator)
    by_snapshot = defaultdict(Accumulator)
    orbital_names = None
    pair_abs = pair_sq = pair_count = None
    parity_truth, parity_pred = [], []
    stream_offset = 0

    for sid in sorted(pred_by_sid):
        folder = truth_dirs[sid]
        truth = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        lr = read_blocks(os.path.join(folder, "hamiltonians_lr.h5")) \
            if add_lr else {}
        pred = pred_by_sid[sid]
        if set(pred) != set(truth):
            missing = sorted(set(truth) - set(pred))
            extra = sorted(set(pred) - set(truth))
            raise ValueError(
                f"{sid}: prediction/truth block keys differ "
                f"(missing {len(missing)}, extra {len(extra)})")

        names, orbital_codes = orbital_layout(folder)
        if orbital_names is None:
            orbital_names = names
            ncat = len(names)
            pair_abs = np.zeros(ncat * ncat, dtype=np.float64)
            pair_sq = np.zeros(ncat * ncat, dtype=np.float64)
            pair_count = np.zeros(ncat * ncat, dtype=np.int64)
        elif names != orbital_names:
            raise ValueError(f"{sid}: orbital category layout changed")

        cell = np.loadtxt(os.path.join(folder, "lat.dat")).T
        cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        ncat = len(orbital_names)
        pair_code_cache = {}
        for key in sorted(truth):
            r0, r1, r2, i, j = parse_key(key)
            p = pred[key]
            if add_lr and key in lr:
                p = p + lr[key]
            t = truth[key]
            err = np.asarray(p - t, dtype=np.float64)
            if not np.isfinite(err).all():
                raise ValueError(f"{sid}/{key}: non-finite prediction")
            overall.add(err)
            by_snapshot[sid].add(err)
            distance = block_distance(key, cart, cell)
            by_distance[int(distance // DISTANCE_BIN)].add(err)

            cache_key = (i - 1, j - 1)
            codes = pair_code_cache.get(cache_key)
            if codes is None:
                row = orbital_codes[i - 1]
                col = orbital_codes[j - 1]
                codes = (row[:, None] * ncat + col[None, :]).ravel()
                pair_code_cache[cache_key] = codes
            flat_abs = np.abs(err).ravel()
            pair_abs += np.bincount(codes, weights=flat_abs,
                                    minlength=ncat * ncat)
            pair_sq += np.bincount(codes, weights=np.square(err).ravel(),
                                   minlength=ncat * ncat)
            pair_count += np.bincount(codes, minlength=ncat * ncat)

            flat_t = np.asarray(t, dtype=np.float64).ravel()
            flat_p = np.asarray(p, dtype=np.float64).ravel()
            start = (-stream_offset) % parity_stride
            parity_truth.append(flat_t[start::parity_stride])
            parity_pred.append(flat_p[start::parity_stride])
            stream_offset = (stream_offset + flat_t.size) % parity_stride

    pair_abs = pair_abs.reshape(len(orbital_names), -1)
    pair_sq = pair_sq.reshape(len(orbital_names), -1)
    pair_count = pair_count.reshape(len(orbital_names), -1)
    pair_mae = np.divide(pair_abs, pair_count,
                         out=np.full_like(pair_abs, np.nan),
                         where=pair_count > 0)
    pair_rmse = np.sqrt(np.divide(pair_sq, pair_count,
                                  out=np.full_like(pair_sq, np.nan),
                                  where=pair_count > 0))
    distance_rows = []
    for bucket in sorted(by_distance):
        distance_rows.append({
            "r_lo": bucket * DISTANCE_BIN,
            "r_hi": (bucket + 1) * DISTANCE_BIN,
            **by_distance[bucket].summary(),
        })
    snapshot_rows = {sid: acc.summary()
                     for sid, acc in sorted(by_snapshot.items())}
    return {
        "overall": overall.summary(),
        "distance": distance_rows,
        "orbital_names": orbital_names,
        "orbital_pair_mae": pair_mae,
        "orbital_pair_rmse": pair_rmse,
        "orbital_pair_count": pair_count,
        "by_snapshot": snapshot_rows,
        "parity_truth": np.concatenate(parity_truth),
        "parity_pred": np.concatenate(parity_pred),
    }


def figure_parity(results, out_dir):
    all_values = np.concatenate([
        results[k][field]
        for k in ("sr", "full")
        for field in ("parity_truth", "parity_pred")])
    lo, hi = np.quantile(all_values, [0.001, 0.999])
    margin = 0.04 * (hi - lo)
    lo, hi = float(lo - margin), float(hi + margin)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    labels = {"sr": "LR-corrected SR", "full": "Direct full-H"}
    colors = {"sr": "Blues", "full": "Oranges"}
    for ax, key in zip(axes, ("sr", "full")):
        truth = results[key]["parity_truth"]
        pred = results[key]["parity_pred"]
        hb = ax.hexbin(truth, pred, gridsize=85, bins="log", mincnt=1,
                       extent=(lo, hi, lo, hi), cmap=colors[key])
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        residual = pred - truth
        ss_res = float(np.square(residual).sum())
        ss_tot = float(np.square(truth - truth.mean()).sum())
        r2 = 1.0 - ss_res / ss_tot
        m = results[key]["overall"]
        ax.text(0.04, 0.96,
                f"MAE = {1e3*m['mae']:.4f} meV\n"
                f"RMSE = {1e3*m['rmse']:.4f} meV\n$R^2$ = {r2:.6f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.88,
                      "edgecolor": "none", "pad": 4})
        ax.set_title(labels[key])
        ax.set_xlabel("DFT $H_{full}$ matrix element (eV)")
        style_axes(ax)
        fig.colorbar(hb, ax=ax, label="sampled element count")
    axes[0].set_ylabel("Predicted $H_{full}$ matrix element (eV)")
    fig.suptitle("Held-out test parity (37 snapshots)")
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_2_test_parity")


def figure_distance(results, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    rows = {}
    for key, color, label in (("sr", SR_COLOR, "LR-corrected SR"),
                              ("full", FULL_COLOR, "Direct full-H")):
        row_map = {row["r_lo"]: row for row in results[key]["distance"]}
        rows[key] = row_map
        x = np.array(sorted(row_map)) + 0.5 * DISTANCE_BIN
        y = np.array([row_map[r]["mae"] for r in sorted(row_map)]) * 1e3
        axes[0].plot(x, y, "o-", ms=4, lw=1.8, color=color, label=label)
    common = sorted(set(rows["sr"]) & set(rows["full"]))
    x = np.asarray(common) + 0.5 * DISTANCE_BIN
    ratio = np.asarray([rows["full"][r]["mae"] / rows["sr"][r]["mae"]
                        for r in common])
    axes[1].plot(x, ratio, "o-", color="#4B4B4B", ms=3.5, lw=1.5)
    axes[1].axhline(1.0, color="black", linestyle="--", lw=1)
    axes[1].fill_between(x, 1.0, ratio, where=ratio >= 1.0,
                         color=SR_COLOR, alpha=0.14)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Matrix-element MAE (meV)")
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out error versus atomic-pair distance")
    axes[1].set_ylabel("Full / SR\n(<1 favors full-H)")
    axes[1].set_xlabel("Atomic-pair distance (Å)")
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_3_mae_vs_atomic_distance")


def annotate_heatmap(ax, matrix, fmt=".3f", threshold=None):
    finite = matrix[np.isfinite(matrix)]
    if threshold is None and finite.size:
        threshold = float(np.nanmedian(finite))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if threshold is not None and value > threshold \
                else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    fontsize=7.5, color=color)


def figure_orbital_heatmap(results, out_dir):
    names = results["sr"]["orbital_names"]
    if names != results["full"]["orbital_names"]:
        raise ValueError("orbital category mismatch between models")
    sr = results["sr"]["orbital_pair_mae"] * 1e3
    full = results["full"]["orbital_pair_mae"] * 1e3
    improvement = full / sr
    positive = np.concatenate([sr[np.isfinite(sr)], full[np.isfinite(full)]])
    norm = LogNorm(vmin=max(float(np.min(positive)), 1e-8),
                   vmax=float(np.max(positive)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    for ax, matrix, title, cmap in (
            (axes[0], sr, "LR-corrected SR MAE (meV)", "Blues"),
            (axes[1], full, "Direct full-H MAE (meV)", "Oranges")):
        im = ax.imshow(matrix, cmap=cmap, norm=norm)
        annotate_heatmap(ax, matrix, fmt=".3f",
                         threshold=math.sqrt(norm.vmin * norm.vmax))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
    vmax = max(2.0, float(np.nanmax(improvement)))
    im = axes[2].imshow(improvement, cmap="RdBu_r", vmin=0, vmax=vmax)
    for i in range(improvement.shape[0]):
        for j in range(improvement.shape[1]):
            value = improvement[i, j]
            if not np.isfinite(value):
                continue
            # RdBu_r is darkest at the extremes and nearly white at one.
            color = "white" if value < 0.55 or value > 1.55 else "black"
            axes[2].text(j, i, format(value, ".2f"), ha="center",
                         va="center", fontsize=7.5, color=color)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Error ratio: Full / SR\n(>1 favors LR-corrected SR)")
    for ax in axes:
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Column orbital")
    axes[0].set_ylabel("Row orbital")
    fig.suptitle("Held-out orbital-pair error decomposition", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_4_orbital_pair_error_heatmap")


def make_single_snapshot_view(source, root):
    sid = os.path.basename(source)
    dst = os.path.join(root, sid)
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(source)):
        src_path = os.path.abspath(os.path.join(source, name))
        dst_path = os.path.join(dst, name)
        if os.path.lexists(dst_path):
            if os.path.realpath(dst_path) != os.path.realpath(src_path):
                raise ValueError(f"stale analysis view link: {dst_path}")
            continue
        os.symlink(src_path, dst_path)
    return sid


def full_space_blocks(pred, folder, add_lr):
    if not add_lr:
        return pred
    lr = read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    keys = set(pred) | set(lr)
    out = {}
    for key in keys:
        p, l = pred.get(key), lr.get(key)
        if p is None:
            p = np.zeros_like(l)
        if l is not None:
            p = p + l
        out[key] = p
    return out


def orbital_offsets(folder):
    with open(os.path.join(folder, "orbital_types.dat")) as fh:
        orbital_types = [list(map(int, line.split()))
                         for line in fh if line.strip()]
    counts = [sum(2 * ell + 1 for ell in shells)
              for shells in orbital_types]
    return np.concatenate([[0], np.cumsum(counts)]).astype(int)


def assemble_k(blocks, offsets, kpoint):
    n_orb = int(offsets[-1])
    matrix = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for key, block in blocks.items():
        r0, r1, r2, i, j = parse_key(key)
        phase = np.exp(2j * np.pi * np.dot(kpoint, (r0, r1, r2)))
        matrix[offsets[i - 1]:offsets[i],
               offsets[j - 1]:offsets[j]] += block * phase
    return 0.5 * (matrix + matrix.conj().T)


def explicit_band_path(folder, reference_distance=0.055):
    import seekpath

    cell = np.loadtxt(os.path.join(folder, "lat.dat")).T
    cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
    numbers = np.loadtxt(os.path.join(folder, "element.dat"), dtype=int)
    frac = cart @ np.linalg.inv(cell)
    result = seekpath.get_explicit_k_path(
        (cell, frac, numbers), reference_distance=reference_distance,
        symprec=1e-5)
    # Seek-path reduces this 2x2x2 cell to MgO's primitive cell. Convert the
    # primitive reciprocal coordinates back into the input supercell basis;
    # the resulting spectrum is the correctly folded supercell band structure.
    inv_p = np.asarray(result["inverse_primitive_transformation_matrix"], float)
    k_primitive = np.asarray(result["explicit_kpoints_rel"], float)
    k_supercell = k_primitive @ inv_p.T
    x = np.asarray(result["explicit_kpoints_linearcoord"], float)
    labels = list(result["explicit_kpoints_labels"])
    return x, labels, k_supercell, result


def band_eigenvalues(block_sets, overlap, folder):
    offsets = orbital_offsets(folder)
    x, labels, kpoints, path_info = explicit_band_path(folder)
    energies = {name: [] for name in block_sets}
    min_overlap_eigenvalue = float("inf")
    for index, kpoint in enumerate(kpoints):
        sk = assemble_k(overlap, offsets, kpoint)
        s_min = float(np.linalg.eigvalsh(sk)[0])
        min_overlap_eigenvalue = min(min_overlap_eigenvalue, s_min)
        if s_min <= 1e-8:
            raise ValueError(
                f"overlap is not positive definite at k-point {index}: {s_min}")
        for name, blocks in block_sets.items():
            hk = assemble_k(blocks, offsets, kpoint)
            energies[name].append(
                eigh(hk, sk, eigvals_only=True, check_finite=False))
    energies = {name: np.asarray(values) for name, values in energies.items()}
    return x, labels, energies, min_overlap_eigenvalue, path_info


def band_ticks(x, labels):
    ticks, names = [], []
    for value, label in zip(x, labels):
        if not label:
            continue
        label = r"$\Gamma$" if label.upper() == "GAMMA" else label
        if ticks and abs(value - ticks[-1]) < 1e-12:
            names[-1] += "|" + label
        else:
            ticks.append(float(value))
            names.append(label)
    return ticks, names


def figure_bands(sr_run, full_run, workspace, cache_dir, out_dir):
    source = os.path.join(workspace, "pilot", "snapshot_000001")
    with open(os.path.join(source, "displacement_metadata.json")) as fh:
        meta = json.load(fh)
    if meta.get("pattern_class") != "equilibrium":
        raise ValueError("pilot/snapshot_000001 is no longer equilibrium")
    view = os.path.join(cache_dir, "pilot_equilibrium_view")
    sid = make_single_snapshot_view(source, view)
    predictions = {}
    for label, run_dir, add_lr in (("sr", sr_run, True),
                                   ("full", full_run, False)):
        config = build_eval_config(run_dir, view, cache_dir,
                                   "pilot_equilibrium", label, None)
        pred = predict(run_dir, config, [sid])[sid]
        predictions[label] = full_space_blocks(pred, source, add_lr)
    truth = read_blocks(os.path.join(source, "hamiltonians_full.h5"))
    overlap = read_blocks(os.path.join(source, "overlaps.h5"))
    x, labels, energies, s_min, path_info = band_eigenvalues(
        {"truth": truth, **predictions}, overlap, source)

    numbers = np.loadtxt(os.path.join(source, "element.dat"), dtype=int)
    # Pseudopotential valences: Mg 10, O 6. Non-spin-polarized bands hold two
    # electrons each.
    n_electrons = int(sum(10 if z == 12 else 6 if z == 8 else 0
                          for z in numbers))
    n_occ = n_electrons // 2
    truth_e = energies["truth"]
    reference = float(np.max(truth_e[:, n_occ - 1]))
    shifted = {name: value - reference for name, value in energies.items()}
    band_slice = slice(n_occ - 4, n_occ + 4)
    ticks, tick_labels = band_ticks(x, labels)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for band in shifted["truth"][:, band_slice].T:
        ax.plot(x, band, color=TRUTH_COLOR, lw=1.25, alpha=0.9)
    for band in shifted["sr"][:, band_slice].T:
        ax.plot(x, band, color=SR_COLOR, lw=1.0, alpha=0.9,
                linestyle="--")
    for band in shifted["full"][:, band_slice].T:
        ax.plot(x, band, color=FULL_COLOR, lw=1.0, alpha=0.9,
                linestyle=":")
    for tick in ticks:
        ax.axvline(tick, color=GRID_COLOR, lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(x.min(), x.max())
    plotted = shifted["truth"][:, band_slice]
    ylo, yhi = np.quantile(plotted, [0.01, 0.99])
    margin = 0.12 * (yhi - ylo)
    ax.set_ylim(ylo - margin, yhi + margin)
    ax.set_ylabel("Energy relative to DFT VBM (eV)")
    ax.set_xlabel("High-symmetry k path")
    ax.set_title("Held-out equilibrium 2×2×2 MgO bands")
    ax.plot([], [], color=TRUTH_COLOR, lw=1.5, label="DFT")
    ax.plot([], [], color=SR_COLOR, lw=1.5, ls="--",
            label="LR-corrected SR")
    ax.plot([], [], color=FULL_COLOR, lw=1.5, ls=":",
            label="Direct full-H")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    style_axes(ax)

    metrics = {}
    for name in ("sr", "full"):
        err = energies[name] - energies["truth"]
        val_mae = float(np.mean(np.abs(err[:, n_occ - 4:n_occ])))
        cond_mae = float(np.mean(np.abs(err[:, n_occ:n_occ + 4])))
        window_mae = float(np.mean(np.abs(err[:, band_slice])))
        gap = float(np.min(energies[name][:, n_occ])
                    - np.max(energies[name][:, n_occ - 1]))
        metrics[name] = {"valence_mae": val_mae,
                         "conduction_mae": cond_mae,
                         "window_mae": window_mae,
                         "band_gap": gap}
    truth_gap = float(np.min(energies["truth"][:, n_occ])
                      - np.max(energies["truth"][:, n_occ - 1]))
    categories = ["Valence\nMAE", "Conduction\nMAE", "Gap\nabsolute error"]
    sr_values = [metrics["sr"]["valence_mae"],
                 metrics["sr"]["conduction_mae"],
                 abs(metrics["sr"]["band_gap"] - truth_gap)]
    full_values = [metrics["full"]["valence_mae"],
                   metrics["full"]["conduction_mae"],
                   abs(metrics["full"]["band_gap"] - truth_gap)]
    xpos = np.arange(len(categories))
    width = 0.36
    axes[1].bar(xpos - width / 2, np.asarray(sr_values) * 1e3,
                width, color=SR_COLOR, label="LR-corrected SR")
    axes[1].bar(xpos + width / 2, np.asarray(full_values) * 1e3,
                width, color=FULL_COLOR, label="Direct full-H")
    axes[1].set_xticks(xpos, categories)
    axes[1].set_ylabel("Eigenvalue error (meV)")
    axes[1].set_title(f"Near-gap errors\nDFT gap = {truth_gap:.3f} eV")
    axes[1].legend(frameon=False, fontsize=8)
    style_axes(axes[1])
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_5_band_structure_eigenvalue_error")
    return {
        "snapshot": "pilot/snapshot_000001",
        "pattern_class": meta["pattern_class"],
        "n_kpoints": int(len(x)),
        "n_orbitals": int(energies["truth"].shape[1]),
        "n_occupied_bands": int(n_occ),
        "minimum_overlap_eigenvalue": s_min,
        "dft_band_gap": truth_gap,
        "models": metrics,
        "spacegroup_number": int(path_info["spacegroup_number"]),
        "spacegroup_international": path_info["spacegroup_international"],
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def write_readme(out_dir, metrics):
    sr = metrics["heldout_test"]["sr"]["overall"]
    full = metrics["heldout_test"]["full"]["overall"]
    band = metrics["band"]
    full_mae_reduction = 100 * (1 - full["mae"] / sr["mae"])
    orbital_ratio = (
        np.asarray(metrics["heldout_test"]["full"]["orbital_pair_mae"])
        / np.asarray(metrics["heldout_test"]["sr"]["orbital_pair_mae"]))
    full_better_pairs = int(np.sum(orbital_ratio < 1))
    sr_better_pairs = int(np.sum(orbital_ratio > 1))
    sr_band = band["models"]["sr"]
    full_band = band["models"]["full"]
    sr_gap_error = abs(sr_band["band_gap"] - band["dft_band_gap"])
    full_gap_error = abs(full_band["band_gap"] - band["dft_band_gap"])
    text = f"""# Paired SR vs full-H result figures

All matrix-element results use the frozen **37-snapshot main test split**.
The LR-corrected model is evaluated as `H_SR(pred) + H_LR(analytic)` against
the same `H_full(DFT)` target used for the direct full-H baseline.

| Model | Held-out MAE (eV) | Held-out RMSE (eV) |
|---|---:|---:|
| LR-corrected SR | {sr['mae']:.8e} | {sr['rmse']:.8e} |
| Direct full-H | {full['mae']:.8e} | {full['rmse']:.8e} |

Figure 1 shows optimization loss in each model's own target space; because
those targets differ, Figures 2–5 are the valid cross-model comparisons.

## What these runs show

- **Matrix elements:** direct full-H has {full_mae_reduction:.1f}% lower
  held-out MAE overall and lower MAE in every 1-Å distance bin. The present
  data therefore do **not** support the claim that LR-corrected SR improves
  long-range matrix-element error.
- **Orbital channels:** direct full-H is better in {full_better_pairs}/36
  ordered orbital-pair channels; LR-corrected SR is better in
  {sr_better_pairs}/36.
- **Near-gap eigenvalues:** on the equilibrium pilot cell, LR-corrected SR has
  {1e3 * sr_band['window_mae']:.3f} meV eight-band-window MAE versus
  {1e3 * full_band['window_mae']:.3f} meV for full-H. Its gap error is
  {1e3 * sr_gap_error:.3f} meV versus {1e3 * full_gap_error:.3f} meV.
  This physically relevant result favors SR, but it is one held-out
  equilibrium structure and should not be generalized without more cells.

Figure 5 uses the held-out equilibrium 2x2x2 pilot cell
(`pilot/snapshot_000001`, space group {band['spacegroup_international']}) and
solves `H(k)c = E(k)S(k)c` using the DFT overlap. The DFT k-path gap is
{band['dft_band_gap']:.6f} eV.

Machine-readable details, counts, checkpoint hashes, and band metrics are in
`metrics.json`.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(text)


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--sr-run", default=None)
    parser.add_argument("--full-run", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    workspace = paths.resolve("workspace", args.workspace)
    training_root = paths.resolve("training_root", args.training_root)
    sr_run = os.path.abspath(args.sr_run or newest_run(training_root, "run_sr"))
    full_run = os.path.abspath(args.full_run or newest_run(training_root, "run_full"))
    out_dir = os.path.abspath(args.out_dir or
                              os.path.join(training_root, "result_figures"))
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    metadata = os.path.join(workspace, "metadata.yaml")
    with open(metadata) as fh:
        target_lines = [line.split(":", 1)[1].strip() for line in fh
                        if line.startswith("training_target:")]
    if target_lines != ["full"]:
        raise SystemExit(
            f"workspace must be exported as full before held-out graph build; "
            f"metadata says {target_lines}")

    print(f"SR run:     {sr_run}\nFull-H run: {full_run}\nOutput:     {out_dir}")
    metrics = {
        "provenance": {
            "workspace": workspace,
            "sr_run": sr_run,
            "full_run": full_run,
            "sr_best_checkpoint_sha256": sha256(
                os.path.join(sr_run, "best_model.pkl")),
            "full_best_checkpoint_sha256": sha256(
                os.path.join(full_run, "best_model.pkl")),
            "frozen_splits_sha256": sha256(
                os.path.join(paths.PROVENANCE_DIR, "splits.json")),
        }
    }

    print("\n[1/5] training and validation loss")
    metrics["training_curves"] = figure_training_curves(
        sr_run, full_run, out_dir)

    print("\n[2–4/5] held-out test inference and matrix-element analysis")
    sids, graph_root, truth_dirs = snapshot_dirs(workspace, "test")
    heldout = {}
    for label, run_dir, add_lr in (("sr", sr_run, True),
                                   ("full", full_run, False)):
        print(f"\n  predicting {label} on {len(sids)} held-out snapshots")
        config = build_eval_config(run_dir, graph_root, cache_dir,
                                   "test", label, None)
        predictions = predict(run_dir, config, sids)
        heldout[label] = score_rich(predictions, truth_dirs, add_lr)
        del predictions
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        summary = heldout[label]["overall"]
        print(f"  {label}: MAE {summary['mae']:.8e} eV, "
              f"RMSE {summary['rmse']:.8e} eV over "
              f"{summary['n_elements']} elements")

    if not np.allclose(heldout["sr"]["parity_truth"],
                       heldout["full"]["parity_truth"], rtol=0, atol=0):
        raise ValueError("parity sampling was not identical between models")
    figure_parity(heldout, out_dir)
    figure_distance(heldout, out_dir)
    figure_orbital_heatmap(heldout, out_dir)

    # Large parity arrays are plot inputs, not useful JSON payloads.
    metrics["heldout_test"] = {}
    for label, result in heldout.items():
        metrics["heldout_test"][label] = {
            "overall": result["overall"],
            "distance": result["distance"],
            "orbital_names": result["orbital_names"],
            "orbital_pair_mae": result["orbital_pair_mae"],
            "orbital_pair_rmse": result["orbital_pair_rmse"],
            "orbital_pair_count": result["orbital_pair_count"],
            "by_snapshot": result["by_snapshot"],
            "parity_sample_size": int(result["parity_truth"].size),
        }
    metrics["heldout_test"]["snapshots"] = sids

    print("\n[5/5] held-out equilibrium band structure")
    metrics["band"] = figure_bands(
        sr_run, full_run, workspace, cache_dir, out_dir)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(json_ready(metrics), fh, indent=1)
        fh.write("\n")
    write_readme(out_dir, metrics)
    print(f"\nAll five figures complete.\n  metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
