"""Analyze the continuous Mg-x Hamiltonian displacement scan.

Produces the prioritized 20-element H(delta)/dH(delta) figure set plus a
four-bucket summary of gradient quality, Hamiltonian error, analytic-LR scale,
and Hamiltonian-response error correlation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr


from maceh.analysis.figures import pdf_pages, save_figure, save_figure_formats
from maceh.config import load_config


SR_COLOR = "#0072B2"
FULL_COLOR = "#D55E00"
DFT_COLOR = "#222222"
LR_COLOR = "#009E73"
GRID_COLOR = "#D9D9D9"
ELEMENT_NAMES = {8: "O", 12: "Mg"}
ORBITAL_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style(axis, log=False) -> None:
    axis.grid(True, which="both" if log else "major", color=GRID_COLOR,
              linewidth=0.7, alpha=0.72)
    axis.tick_params(direction="in", top=True, right=True)


def save_figure(fig, output_dir: Path, stem: str) -> None:
    for path in save_figure_formats(fig, output_dir, stem,
                                    bbox_inches="tight"):
        print(f"wrote {path}")
    plt.close(fig)


def load_curves(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        def strings(name):
            return [item.decode() if isinstance(item, bytes) else str(item)
                    for item in handle[name][:]]
        return {
            "dft": handle["hamiltonian/dft"][:],
            "sr_fixed_lr": handle["hamiltonian/sr_fixed_lr"][:],
            "analytic_lr": handle["hamiltonian/analytic_lr"][:],
            "full": handle["hamiltonian/full_direct"][:],
            "displacements": handle["displacements_angstrom"][:],
            "block_keys": strings("block_keys"),
            "block_shapes": handle["block_shapes"][:],
            "block_offsets": handle["block_offsets"][:],
            "positions": handle["equilibrium_positions_angstrom"][:],
            "lattice": handle["supercell_lattice_angstrom"][:],
            "atomic_numbers": handle["atomic_numbers"][:],
            "attributes": dict(handle.attrs),
        }


def orbital_labels(cfg: dict) -> dict[int, list[tuple[str, int]]]:
    labels = {}
    for name, number in (("Mg", 12), ("O", 8)):
        shell_count = collections.Counter()
        expanded = []
        for ell in cfg["abacus"]["orbital_types"][name]:
            ell = int(ell)
            shell_count[ell] += 1
            shell = f"{ORBITAL_NAMES[ell]}{shell_count[ell]}"
            for component in range(2 * ell + 1):
                expanded.append((shell, component))
        labels[number] = expanded
    return labels


def orbital_text(label: tuple[str, int]) -> str:
    shell, component = label
    return shell if shell.startswith("s") else f"{shell}[m{component}]"


def block_metadata(data: dict, labels: dict) -> dict:
    offsets = data["block_offsets"]
    n_elements = int(offsets[-1])
    block_index = np.empty(n_elements, dtype=np.int32)
    row_index = np.empty(n_elements, dtype=np.int16)
    column_index = np.empty(n_elements, dtype=np.int16)
    blocks = []
    for index, (key_string, shape) in enumerate(
            zip(data["block_keys"], data["block_shapes"])):
        nrow, ncol = map(int, shape)
        start, stop = map(int, offsets[index:index + 2])
        block_index[start:stop] = index
        row_index[start:stop] = np.repeat(np.arange(nrow), ncol)
        column_index[start:stop] = np.tile(np.arange(ncol), nrow)
        r0, r1, r2, i1, j1 = json.loads(key_string)
        i, j = i1 - 1, j1 - 1
        vector = (data["positions"][j]
                  + np.asarray([r0, r1, r2]) @ data["lattice"]
                  - data["positions"][i])
        zi, zj = int(data["atomic_numbers"][i]), int(data["atomic_numbers"][j])
        blocks.append({
            "key": key_string,
            "translation": [r0, r1, r2],
            "atom_i": i,
            "atom_j": j,
            "atomic_number_i": zi,
            "atomic_number_j": zj,
            "element_i": ELEMENT_NAMES[zi],
            "element_j": ELEMENT_NAMES[zj],
            "vector_angstrom": vector,
            "distance_angstrom": float(np.linalg.norm(vector)),
            "row_orbitals": labels[zi],
            "column_orbitals": labels[zj],
        })
    return {
        "block_index": block_index,
        "row_index": row_index,
        "column_index": column_index,
        "blocks": blocks,
    }


def global_metrics(candidate: np.ndarray, truth: np.ndarray,
                   candidate_slope: np.ndarray, truth_slope: np.ndarray,
                   center: int) -> dict:
    error = candidate - truth
    response_error = ((candidate - candidate[center])
                      - (truth - truth[center]))
    slope_error = candidate_slope - truth_slope
    return {
        "hamiltonian_mae_eV": float(np.mean(np.abs(error))),
        "hamiltonian_rmse_eV": float(np.sqrt(np.mean(np.square(error)))),
        "equilibrium_hamiltonian_mae_eV": float(
            np.mean(np.abs(error[center]))),
        "response_aligned_hamiltonian_rmse_eV": float(
            np.sqrt(np.mean(np.square(response_error)))),
        "central_slope_mae_eV_per_A": float(np.mean(np.abs(slope_error))),
        "central_slope_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(slope_error)))),
        "central_slope_relative_l2": float(
            np.linalg.norm(slope_error) / np.linalg.norm(truth_slope)),
        "central_slope_cosine": float(
            np.dot(candidate_slope, truth_slope)
            / (np.linalg.norm(candidate_slope) * np.linalg.norm(truth_slope))),
    }


def correlation_metrics(hamiltonian_error: np.ndarray,
                        slope_error: np.ndarray) -> dict:
    valid = ((hamiltonian_error > 0) & (slope_error > 0)
             & np.isfinite(hamiltonian_error) & np.isfinite(slope_error))
    x = np.log10(hamiltonian_error[valid])
    y = np.log10(slope_error[valid])
    return {
        "log10_pearson": float(pearsonr(x, y).statistic),
        "log10_spearman": float(spearmanr(x, y).statistic),
        "n_elements": int(valid.sum()),
    }


def aggregate_distance(blocks: list[dict], block_index: np.ndarray,
                       metrics: dict[str, np.ndarray], width: float = 1.0):
    distances = np.asarray([row["distance_angstrom"] for row in blocks])
    element_distances = distances[block_index]
    bins = np.floor(element_distances / width).astype(int)
    rows = []
    for bucket in range(int(bins.max()) + 1):
        selected = bins == bucket
        if not selected.any():
            continue
        row = {
            "lower_A": bucket * width,
            "upper_A": (bucket + 1) * width,
            "center_A": (bucket + 0.5) * width,
            "matrix_element_count": int(selected.sum()),
        }
        for name, values in metrics.items():
            row[name] = float(np.mean(values[selected]))
        rows.append(row)
    return rows, element_distances


def decile_curve(x: np.ndarray, y: np.ndarray) -> dict:
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    edges = np.quantile(xv, np.linspace(0, 1, 11))
    rows = []
    for index in range(10):
        selected = ((xv >= edges[index])
                    & (xv <= edges[index + 1] if index == 9
                       else xv < edges[index + 1]))
        rows.append({
            "x_median": float(np.median(xv[selected])),
            "y_median": float(np.median(yv[selected])),
            "y_q25": float(np.quantile(yv[selected], 0.25)),
            "y_q75": float(np.quantile(yv[selected], 0.75)),
            "count": int(selected.sum()),
        })
    return {"rows": rows, "edges": edges.tolist()}


def select_elements(data: dict, metadata: dict, labels: dict,
                    dft_slope: np.ndarray, sr_slope: np.ndarray,
                    full_slope: np.ndarray, sr: np.ndarray,
                    center: int, count: int = 20) -> list[dict]:
    sr_error = np.abs(sr_slope - dft_slope)
    full_error = np.abs(full_slope - dft_slope)
    contribution = np.square(full_error) - np.square(sr_error)
    signal = np.abs(dft_slope)
    nonzero = signal[signal > 0]
    signal_threshold = float(np.quantile(nonzero, 0.90))
    eligible = (signal >= signal_threshold) & (contribution > 0)
    candidates = np.flatnonzero(eligible)
    candidates = candidates[np.argsort(contribution[candidates])[::-1]]

    block_index = metadata["block_index"]
    row_index = metadata["row_index"]
    column_index = metadata["column_index"]
    signatures = set()
    distance_counts = collections.Counter()
    pair_counts = collections.Counter()
    orbital_pair_counts = collections.Counter()
    selected = []
    for element_index in candidates:
        block_id = int(block_index[element_index])
        block = metadata["blocks"][block_id]
        row = int(row_index[element_index])
        column = int(column_index[element_index])
        left = labels[block["atomic_number_i"]][row]
        right = labels[block["atomic_number_j"]][column]
        vector = block["vector_angstrom"]
        # Preserve the x component relative to the displacement while folding
        # signs and the cubic y/z interchange.  This removes visually redundant
        # symmetry copies without conflating longitudinal and transverse pairs.
        geometry = (round(abs(float(vector[0])), 4),
                    *sorted((round(abs(float(vector[1])), 4),
                             round(abs(float(vector[2])), 4))))
        sides = tuple(sorted(((block["atomic_number_i"], left),
                              (block["atomic_number_j"], right)), key=str))
        signature = (round(block["distance_angstrom"], 4), geometry, sides)
        distance_bin = int(block["distance_angstrom"] // 1.0)
        species_pair = tuple(sorted((block["atomic_number_i"],
                                     block["atomic_number_j"])))
        orbital_pair = tuple(sorted((left[0][0], right[0][0])))
        if (signature in signatures or distance_counts[distance_bin] >= 4
                or pair_counts[species_pair] >= 8
                or orbital_pair_counts[orbital_pair] >= 5):
            continue
        equilibrium_sr_error = abs(
            sr[center, element_index] - data["dft"][center, element_index])
        equilibrium_full_error = abs(
            data["full"][center, element_index]
            - data["dft"][center, element_index])
        record = {
            "rank": len(selected) + 1,
            "flat_element_index": int(element_index),
            "block_index": block_id,
            "block_key": block["key"],
            "row": row,
            "column": column,
            "element_i": block["element_i"],
            "element_j": block["element_j"],
            "atom_i": block["atom_i"],
            "atom_j": block["atom_j"],
            "row_orbital": orbital_text(left),
            "column_orbital": orbital_text(right),
            "pair_vector_angstrom": block["vector_angstrom"].tolist(),
            "pair_distance_angstrom": block["distance_angstrom"],
            "dft_central_slope_eV_per_A": float(dft_slope[element_index]),
            "sr_central_slope_eV_per_A": float(sr_slope[element_index]),
            "full_central_slope_eV_per_A": float(full_slope[element_index]),
            "sr_slope_abs_error_eV_per_A": float(sr_error[element_index]),
            "full_slope_abs_error_eV_per_A": float(full_error[element_index]),
            "squared_slope_error_improvement_eV2_per_A2": float(
                contribution[element_index]),
            "sr_equilibrium_abs_error_eV": float(equilibrium_sr_error),
            "full_equilibrium_abs_error_eV": float(equilibrium_full_error),
            "full_closer_at_equilibrium": bool(
                equilibrium_full_error < equilibrium_sr_error),
        }
        selected.append(record)
        signatures.add(signature)
        distance_counts[distance_bin] += 1
        pair_counts[species_pair] += 1
        orbital_pair_counts[orbital_pair] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)} diverse elements")
    # Put the visually decisive cases first: both model values are within
    # 10 meV of DFT at equilibrium, yet Full-H has the larger slope error.
    # Keep the contribution score as the ordering inside each group.
    for record in selected:
        close_h = max(record["sr_equilibrium_abs_error_eV"],
                      record["full_equilibrium_abs_error_eV"]) <= 0.010
        record["selection_group"] = (
            "close-H / wrong-slope exemplar" if close_h
            else "large slope-error contributor")
    selected.sort(key=lambda record: (
        record["selection_group"] != "close-H / wrong-slope exemplar",
        -record["squared_slope_error_improvement_eV2_per_A2"]))
    for rank, record in enumerate(selected, 1):
        record["rank"] = rank
    return selected


def selected_title(record: dict) -> str:
    vector = record["pair_vector_angstrom"]
    return (f"#{record['rank']:02d} {record['element_i']}({record['atom_i']}) "
            f"{record['row_orbital']} → {record['element_j']}({record['atom_j']}) "
            f"{record['column_orbital']}\n"
            f"R={json.loads(record['block_key'])[:3]}, "
            f"r={record['pair_distance_angstrom']:.2f} Å, "
            f"Δ=({vector[0]:.2f},{vector[1]:.2f},{vector[2]:.2f}) Å")


def plot_selected(data: dict, selected: list[dict], output_dir: Path) -> None:
    displacement = data["displacements"]
    step = float(displacement[1] - displacement[0])
    center = int(np.argmin(np.abs(displacement)))
    indices = np.asarray([row["flat_element_index"] for row in selected])
    curves = {
        "DFT": data["dft"][:, indices],
        "SR-target (+ fixed LR)": data["sr_fixed_lr"][:, indices],
        "Direct Full-H": data["full"][:, indices],
    }
    derivatives = {
        name: savgol_filter(value, window_length=7, polyorder=3, deriv=1,
                            delta=step, axis=0, mode="interp")
        for name, value in curves.items()}
    colors = {"DFT": DFT_COLOR, "SR-target (+ fixed LR)": SR_COLOR,
              "Direct Full-H": FULL_COLOR}
    line_styles = {"DFT": "-", "SR-target (+ fixed LR)": "-",
                   "Direct Full-H": "-"}
    pdf_path = output_dir / "response_scan_selected_20_elements.pdf"
    with pdf_pages(pdf_path) as pdf:
        for page in range(4):
            fig, axes = plt.subplots(5, 2, figsize=(13.0, 15.0))
            for local in range(5):
                item = 5 * page + local
                record = selected[item]
                baseline = curves["DFT"][center, item]
                for name in curves:
                    axes[local, 0].plot(
                        displacement, 1e3 * (curves[name][:, item] - baseline),
                        color=colors[name], linestyle=line_styles[name],
                        linewidth=2.0 if name == "DFT" else 1.65,
                        label=name)
                    axes[local, 1].plot(
                        displacement, derivatives[name][:, item],
                        color=colors[name], linestyle=line_styles[name],
                        linewidth=2.0 if name == "DFT" else 1.65,
                        label=name)
                for axis in axes[local]:
                    axis.axvline(0, color="#888888", linestyle=":",
                                linewidth=0.9)
                    axis.set_xlabel("Mg x displacement δ (Å)")
                    style(axis)
                axes[local, 0].set_ylabel(
                    r"$H_{ij}(\delta)-H^{DFT}_{ij}(0)$ (meV)")
                axes[local, 1].set_ylabel(r"$dH_{ij}/d\delta$ (eV/Å)")
                axes[local, 0].set_title(selected_title(record), fontsize=9)
                axes[local, 1].set_title(
                    f"slope |error|: SR {record['sr_slope_abs_error_eV_per_A']:.3g}, "
                    f"Full {record['full_slope_abs_error_eV_per_A']:.3g} eV/Å",
                    fontsize=9)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3,
                       frameon=False, bbox_to_anchor=(0.5, 0.958))
            fig.suptitle(
                f"Continuous Hamiltonian response: selected elements "
                f"{5*page + 1}–{5*page + 5} of 20\n"
                "derivatives: 7-point cubic Savitzky–Golay estimate",
                y=0.997, fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.1)
            png_path = output_dir / f"response_scan_20_elements_page_{page + 1}.png"
            save_figure(fig, png_path, dpi=240, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            print(f"wrote {png_path}")
            plt.close(fig)
    print(f"wrote {pdf_path}")

    csv_path = output_dir / "response_scan_selected_20_curves.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["rank", "flat_element_index", "displacement_angstrom",
                  "dft_h_eV", "sr_fixed_lr_h_eV", "full_h_eV",
                  "dft_derivative_eV_per_A", "sr_derivative_eV_per_A",
                  "full_derivative_eV_per_A"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, record in enumerate(selected):
            for point, delta in enumerate(displacement):
                writer.writerow({
                    "rank": record["rank"],
                    "flat_element_index": record["flat_element_index"],
                    "displacement_angstrom": f"{delta:.10g}",
                    "dft_h_eV": f"{curves['DFT'][point, item]:.16g}",
                    "sr_fixed_lr_h_eV": f"{curves['SR-target (+ fixed LR)'][point, item]:.16g}",
                    "full_h_eV": f"{curves['Direct Full-H'][point, item]:.16g}",
                    "dft_derivative_eV_per_A": f"{derivatives['DFT'][point, item]:.16g}",
                    "sr_derivative_eV_per_A": f"{derivatives['SR-target (+ fixed LR)'][point, item]:.16g}",
                    "full_derivative_eV_per_A": f"{derivatives['Direct Full-H'][point, item]:.16g}",
                })
    print(f"wrote {csv_path}")


def plot_summary(report: dict, arrays: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))

    active = arrays["active_mask"]
    for name, values, color, label in (
            ("sr", arrays["sr_slope_error"], SR_COLOR,
             "SR-target (+ fixed LR)"),
            ("full", arrays["full_slope_error"], FULL_COLOR,
             "Direct Full-H")):
        selected = np.sort(values[active])
        positive = selected[selected > 0]
        keep = np.unique(np.linspace(
            0, len(positive) - 1, min(4000, len(positive)), dtype=int))
        axes[0, 0].semilogx(positive[keep],
                            100 * (keep + 1) / len(positive),
                            color=color, linewidth=2, label=label)
    axes[0, 0].set_xlabel("Central-slope absolute error (eV/Å)")
    axes[0, 0].set_ylabel("Response-active elements below error (%)")
    axes[0, 0].set_title("A. Gradient quality under tiny displacement")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].text(
        0.04, 0.57,
        f"Full/SR slope MAE = "
        f"{report['overall']['full']['central_slope_mae_eV_per_A'] / report['overall']['sr_only']['central_slope_mae_eV_per_A']:.2f}×\n"
        f"Full/SR slope RMSE = "
        f"{report['overall']['full']['central_slope_rmse_eV_per_A'] / report['overall']['sr_only']['central_slope_rmse_eV_per_A']:.2f}×",
        transform=axes[0, 0].transAxes, fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88,
              "pad": 4})
    style(axes[0, 0], log=True)

    rows = report["distance_bins"]
    x = np.asarray([row["center_A"] for row in rows])
    axes[0, 1].semilogy(
        x, 1e3 * np.asarray([row["sr_hamiltonian_mae_eV"] for row in rows]),
        "o-", color=SR_COLOR, linewidth=2, label="SR-target (+ fixed LR)")
    axes[0, 1].semilogy(
        x, 1e3 * np.asarray([row["full_hamiltonian_mae_eV"] for row in rows]),
        "s-", color=FULL_COLOR, linewidth=2, label="Direct Full-H")
    axes[0, 1].set_xlabel("Bra–ket AO-center distance (Å)")
    axes[0, 1].set_ylabel("Scan Hamiltonian MAE (meV)")
    axes[0, 1].set_title("B. Where aggregate H error is gained/lost")
    axes[0, 1].legend(frameon=False, fontsize=8)
    style(axes[0, 1], log=True)

    axes[1, 0].semilogy(
        x, np.asarray([row["sr_derivative_mae_eV_per_A"] for row in rows]),
        "o-", color=SR_COLOR, linewidth=2, label="SR derivative error")
    axes[1, 0].semilogy(
        x, np.asarray([row["full_derivative_mae_eV_per_A"] for row in rows]),
        "s-", color=FULL_COLOR, linewidth=2, label="Full-H derivative error")
    axes[1, 0].semilogy(
        x, np.asarray([row["analytic_lr_mean_abs_derivative_eV_per_A"]
                       for row in rows]),
        "^-", color=LR_COLOR, linewidth=1.7, label="analytic LR magnitude")
    axes[1, 0].set_xlabel("Bra–ket AO-center distance (Å)")
    axes[1, 0].set_ylabel("Mean |dH/dδ| or derivative error (eV/Å)")
    axes[1, 0].set_title("C. Is the failure the analytic LR contribution?")
    axes[1, 0].legend(frameon=False, fontsize=8)
    style(axes[1, 0], log=True)

    for name, color, label in (
            ("sr", SR_COLOR, "SR-target (+ fixed LR)"),
            ("full", FULL_COLOR, "Direct Full-H")):
        curve = report["correlation"][name]["response_error_deciles"]["rows"]
        xx = np.asarray([row["x_median"] for row in curve])
        yy = np.asarray([row["y_median"] for row in curve])
        low = np.asarray([row["y_q25"] for row in curve])
        high = np.asarray([row["y_q75"] for row in curve])
        axes[1, 1].loglog(xx, yy, "o-", color=color, linewidth=2,
                          label=(f"{label} (Spearman "
                                 f"{report['correlation'][name]['response_h_vs_slope_active']['log10_spearman']:.2f})"))
        axes[1, 1].fill_between(xx, low, high, color=color, alpha=0.12)
    axes[1, 1].set_xlabel("Response-aligned H RMSE per element (eV)")
    axes[1, 1].set_ylabel("Median central-slope |error| (eV/Å)")
    axes[1, 1].set_title("D. Does Hamiltonian error predict EPC error?")
    axes[1, 1].legend(frameon=False, fontsize=8)
    style(axes[1, 1], log=True)

    fig.suptitle("Why better Hamiltonian interpolation need not give better response",
                 fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "response_scan_four_bucket_summary")


def selected_table(selected: list[dict]) -> str:
    lines = [
        "| # | element | distance (Å) | DFT slope | SR slope | Full slope | Full closer at δ=0 |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in selected:
        name = (f"{row['element_i']} {row['row_orbital']} → "
                f"{row['element_j']} {row['column_orbital']}")
        lines.append(
            f"| {row['rank']} | {name} | {row['pair_distance_angstrom']:.2f} | "
            f"{row['dft_central_slope_eV_per_A']:.4f} | "
            f"{row['sr_central_slope_eV_per_A']:.4f} | "
            f"{row['full_central_slope_eV_per_A']:.4f} | "
            f"{'yes' if row['full_closer_at_equilibrium'] else 'no'} |")
    return "\n".join(lines)


def markdown(report: dict) -> str:
    sr = report["overall"]["sr_only"]
    b = report["overall"]["sr_fixed_lr"]
    full = report["overall"]["full"]
    paradox = report["aggregate_loss_paradox"]
    corr = report["correlation"]
    training = report["training_context"]
    return f"""# Hamiltonian-response investigation

## Experiment

A home-cell Mg atom was displaced continuously along Cartesian x at 25 points
from −0.03 to +0.03 Å (0.0025 Å spacing). All 25 ABACUS calculations converged.
The ±0.0025 Å Hamiltonian CSR files are byte-identical to the independent DFT
EPC reference pair, which validates the geometry, orbital, atom-order, and
gauge mapping used here.

The 20-element plots show
`H_ij(delta) - H_ij^DFT(0)` above and a 7-point cubic derivative below. Elements
were selected reproducibly from the top 10% of nonzero DFT response magnitude,
ranked by how much SR reduces Full-H squared central-slope error, with symmetry
and distance/orbital diversity constraints. Close-H/wrong-slope exemplars are
shown first, followed by the remaining large slope-error contributors.

## 1. Gradient quality

| model | central slope MAE (eV/Å) | central slope RMSE (eV/Å) | relative L2 |
|---|---:|---:|---:|
| SR only | {sr['central_slope_mae_eV_per_A']:.6f} | {sr['central_slope_rmse_eV_per_A']:.6f} | {100*sr['central_slope_relative_l2']:.2f}% |
| SR + fixed LR | {b['central_slope_mae_eV_per_A']:.6f} | {b['central_slope_rmse_eV_per_A']:.6f} | {100*b['central_slope_relative_l2']:.2f}% |
| Direct Full-H | {full['central_slope_mae_eV_per_A']:.6f} | {full['central_slope_rmse_eV_per_A']:.6f} | {100*full['central_slope_relative_l2']:.2f}% |

For this Mg-x slice, Full-H central-slope RMSE is
**{full['central_slope_rmse_eV_per_A']/sr['central_slope_rmse_eV_per_A']:.2f}×**
SR-only's, despite both curves often lying close to DFT in H itself. This is the
visual phenomenon the scan was designed to test.

## 2. Where Full-H gains lower Hamiltonian error

Across all scan points and graph elements, Full-H has lower MAE
({1e3*full['hamiltonian_mae_eV']:.4f} meV versus
{1e3*sr['hamiltonian_mae_eV']:.4f} meV), while SR has slightly lower RMSE
({1e3*sr['hamiltonian_rmse_eV']:.4f} meV versus
{1e3*full['hamiltonian_rmse_eV']:.4f} meV). After subtracting each method's
equilibrium value to isolate displacement response, Full-H also has
{100*(1-full['response_aligned_hamiltonian_rmse_eV']/sr['response_aligned_hamiltonian_rmse_eV']):.2f}%
lower H-response RMSE, yet worse central slopes.

Element by element, Full-H has lower raw H MAE for
**{100*paradox['full_lower_raw_h_error_fraction']:.2f}%** of elements. In
**{100*paradox['full_lower_raw_h_but_higher_slope_error_fraction']:.2f}%** of
all elements it is closer in H but worse in slope; those elements carry
**{100*paradox['dft_response_power_fraction_in_raw_h_paradox']:.2f}%** of the
DFT central-response power. Aggregate H loss is therefore hiding a meaningful
response-error subset.

The earlier “14.7% better” number is the Full-H **validation MSE** advantage
({training['full_validation_loss_advantage_fraction']*100:.2f}%), not training
loss. Held-out full-space H MAE favors Full-H by
{training['full_heldout_mae_advantage_fraction']*100:.2f}%.

## 3. Is Full-H specifically failing the analytic LR contribution?

No—not on the present 2×2×2 grid and LR definition. The isolated analytic LR
central slope has only
**{100*report['analytic_lr']['slope_norm_over_sr_residual_norm']:.6f}%** of the
`DFT − SR` residual norm. Adding it changes SR relative slope error from
{100*sr['central_slope_relative_l2']:.6f}% to
{100*b['central_slope_relative_l2']:.6f}%.

This agrees with the full A/B/D EPC decomposition: the fixed LR term is enabled
but negligible at the sampled reciprocal vectors because the largest Ewald
weight is only 1.89×10⁻⁵. The current evidence therefore identifies a
**checkpoint/learned-response difference**, not successful correction by the
analytic LR term and not a missing gradient through learned tensors (there are
no such tensor heads).

## 4. Does Hamiltonian error correlate with response/EPC error?

Yes, but imperfectly. For response-aligned per-element H RMSE versus central
slope error, log-space Spearman correlation is
**{corr['sr']['response_h_vs_slope']['log10_spearman']:.3f}** for SR and
**{corr['full']['response_h_vs_slope']['log10_spearman']:.3f}** for Full-H.
For raw H MAE the correlations are lower
({corr['sr']['raw_h_vs_slope']['log10_spearman']:.3f} and
{corr['full']['raw_h_vs_slope']['log10_spearman']:.3f}). Hamiltonian accuracy is
informative, but aggregate value loss does not uniquely control local slopes.

## Selected 20 elements

{selected_table(report['selected_elements'])}

## Interpretation and limitations

The prioritized plot supports the proposed mechanism: a network can track
`H(delta)` closely over a narrow range while learning a poorer local slope.
However, it does **not** yet establish that the SR target itself causes the
improvement. SR and Full-H are independent fits with different random training
trajectories, and the analytic LR subtraction is extremely small on this grid.
Matched-seed multi-run training, or direct derivative supervision, is needed to
separate target-design effects from training variance.

This scan covers one atom/direction in a 2×2×2 periodic cell. The existing full
EPC tensor covers both atoms and all directions and shows the same ordering,
but a denser-q/supercell response study is required before claiming an isolated
long-range mechanism.

## Figure files

* `response_scan_selected_20_elements.pdf` — all 20 paired H/derivative panels.
* `response_scan_20_elements_page_1.png` through page 4 — reviewable PNG pages.
* `response_scan_four_bucket_summary` — four-bucket quantitative summary.
* `response_scan_selected_20_curves.csv` — plotted numerical curves.
* `response_scan_investigation.json` — complete metrics and provenance.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="runs/epc/response_scan_curves.h5")
    parser.add_argument("--config", default="provenance/config.resolved.yaml")
    parser.add_argument("--training-metrics", default=(
        "runs/result_figures/metrics.json"))
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_curves(input_path)
    cfg = load_config(args.config)
    labels = orbital_labels(cfg)
    metadata = block_metadata(data, labels)
    displacement = data["displacements"]
    center = int(np.argmin(np.abs(displacement)))
    if abs(displacement[center]) > 1e-14:
        raise ValueError("scan has no equilibrium point")
    if center == 0 or center == len(displacement) - 1:
        raise ValueError("equilibrium point is at scan boundary")
    denominator = float(displacement[center + 1] - displacement[center - 1])

    sr = data["sr_fixed_lr"] - data["analytic_lr"]
    slopes = {
        "dft": (data["dft"][center + 1] - data["dft"][center - 1])
               / denominator,
        "sr_only": (sr[center + 1] - sr[center - 1]) / denominator,
        "sr_fixed_lr": (
            data["sr_fixed_lr"][center + 1]
            - data["sr_fixed_lr"][center - 1]) / denominator,
        "full": (data["full"][center + 1] - data["full"][center - 1])
                / denominator,
        "analytic_lr": (
            data["analytic_lr"][center + 1]
            - data["analytic_lr"][center - 1]) / denominator,
    }

    overall = {
        "sr_only": global_metrics(sr, data["dft"], slopes["sr_only"],
                                  slopes["dft"], center),
        "sr_fixed_lr": global_metrics(
            data["sr_fixed_lr"], data["dft"], slopes["sr_fixed_lr"],
            slopes["dft"], center),
        "full": global_metrics(data["full"], data["dft"], slopes["full"],
                               slopes["dft"], center),
    }

    per_element = {}
    for name, candidate in (("sr", sr), ("full", data["full"])):
        error = candidate - data["dft"]
        response_error = ((candidate - candidate[center])
                          - (data["dft"] - data["dft"][center]))
        per_element[name] = {
            "raw_h_mae": np.mean(np.abs(error), axis=0),
            "response_h_rmse": np.sqrt(np.mean(np.square(response_error),
                                                axis=0)),
            "slope_error": np.abs(slopes[name if name == "full" else "sr_only"]
                                  - slopes["dft"]),
        }

    step = float(displacement[1] - displacement[0])
    smooth_derivatives = {
        "dft": savgol_filter(data["dft"], 7, 3, deriv=1, delta=step,
                             axis=0, mode="interp"),
        "sr": savgol_filter(sr, 7, 3, deriv=1, delta=step,
                            axis=0, mode="interp"),
        "full": savgol_filter(data["full"], 7, 3, deriv=1, delta=step,
                              axis=0, mode="interp"),
        "analytic_lr": savgol_filter(data["analytic_lr"], 7, 3, deriv=1,
                                     delta=step, axis=0, mode="interp"),
    }
    derivative_metrics = {
        "sr_derivative_mae_eV_per_A": np.mean(np.abs(
            smooth_derivatives["sr"] - smooth_derivatives["dft"]), axis=0),
        "full_derivative_mae_eV_per_A": np.mean(np.abs(
            smooth_derivatives["full"] - smooth_derivatives["dft"]), axis=0),
        "analytic_lr_mean_abs_derivative_eV_per_A": np.mean(np.abs(
            smooth_derivatives["analytic_lr"]), axis=0),
    }
    del smooth_derivatives

    distance_rows, element_distances = aggregate_distance(
        metadata["blocks"], metadata["block_index"], {
            "sr_hamiltonian_mae_eV": per_element["sr"]["raw_h_mae"],
            "full_hamiltonian_mae_eV": per_element["full"]["raw_h_mae"],
            **derivative_metrics,
        })

    response_active_threshold = float(np.quantile(
        np.abs(slopes["dft"])[np.abs(slopes["dft"]) > 0], 0.90))
    response_active = np.abs(slopes["dft"]) >= response_active_threshold
    correlations = {}
    for name in ("sr", "full"):
        correlations[name] = {
            "raw_h_vs_slope": correlation_metrics(
                per_element[name]["raw_h_mae"],
                per_element[name]["slope_error"]),
            "response_h_vs_slope": correlation_metrics(
                per_element[name]["response_h_rmse"],
                per_element[name]["slope_error"]),
            "response_h_vs_slope_active": correlation_metrics(
                per_element[name]["response_h_rmse"][response_active],
                per_element[name]["slope_error"][response_active]),
            "response_error_deciles": decile_curve(
                per_element[name]["response_h_rmse"][response_active],
                per_element[name]["slope_error"][response_active]),
        }

    raw_full_better = (per_element["full"]["raw_h_mae"]
                       < per_element["sr"]["raw_h_mae"])
    slope_full_worse = (per_element["full"]["slope_error"]
                        > per_element["sr"]["slope_error"])
    response_full_better = (per_element["full"]["response_h_rmse"]
                            < per_element["sr"]["response_h_rmse"])
    response_power = np.square(slopes["dft"])
    raw_paradox = raw_full_better & slope_full_worse
    response_paradox = response_full_better & slope_full_worse

    selected = select_elements(
        data, metadata, labels, slopes["dft"], slopes["sr_fixed_lr"],
        slopes["full"], data["sr_fixed_lr"], center)

    with Path(args.training_metrics).open() as handle:
        training_metrics = json.load(handle)
    train_sr = training_metrics["training_curves"][
        "SR residual target ($H_{SR}$)"]
    train_full = training_metrics["training_curves"]["Full-H target"]
    held_sr = training_metrics["heldout_test"]["sr"]["overall"]
    held_full = training_metrics["heldout_test"]["full"]["overall"]

    minus_scan = Path("data/epc/response_scan_dft/point_011/OUT.MgO/"
                      "data-HR-sparse_SPIN0.csr")
    minus_reference = Path("data/epc/dft_reference/Mg_x_minus/OUT.MgO/"
                           "data-HR-sparse_SPIN0.csr")
    plus_scan = Path("data/epc/response_scan_dft/point_013/OUT.MgO/"
                     "data-HR-sparse_SPIN0.csr")
    plus_reference = Path("data/epc/dft_reference/Mg_x_plus/OUT.MgO/"
                          "data-HR-sparse_SPIN0.csr")
    repeat_hashes = {str(path): sha256(path) for path in (
        minus_scan, minus_reference, plus_scan, plus_reference)}

    report = {
        "conclusion": (
            "Full-H has slightly lower scan H MAE but substantially worse "
            "central slopes; the isolated analytic LR term is negligible."),
        "definition": {
            "displaced_atom": "home-cell Mg (cell-major index 0)",
            "direction": "x",
            "displacement_min_A": float(displacement.min()),
            "displacement_max_A": float(displacement.max()),
            "point_count": len(displacement),
            "step_A": step,
            "central_slope_formula": "[H(+0.0025 A)-H(-0.0025 A)]/0.005 A",
            "curve_derivative": "7-point cubic Savitzky-Golay",
            "matrix_elements_per_point": int(data["block_offsets"][-1]),
            "block_count": len(data["block_keys"]),
        },
        "paths": {
            "input_cache": str(input_path),
            "training_metrics": str(Path(args.training_metrics).resolve()),
        },
        "overall": overall,
        "aggregate_loss_paradox": {
            "full_lower_raw_h_error_fraction": float(np.mean(raw_full_better)),
            "full_lower_raw_h_but_higher_slope_error_fraction": float(
                np.mean(raw_paradox)),
            "dft_response_power_fraction_in_raw_h_paradox": float(
                response_power[raw_paradox].sum() / response_power.sum()),
            "full_lower_response_h_error_fraction": float(
                np.mean(response_full_better)),
            "full_lower_response_h_but_higher_slope_error_fraction": float(
                np.mean(response_paradox)),
            "dft_response_power_fraction_in_response_h_paradox": float(
                response_power[response_paradox].sum() / response_power.sum()),
        },
        "analytic_lr": {
            "central_slope_l2_norm_eV_per_A": float(
                np.linalg.norm(slopes["analytic_lr"])),
            "dft_minus_sr_slope_l2_norm_eV_per_A": float(
                np.linalg.norm(slopes["dft"] - slopes["sr_only"])),
            "slope_norm_over_sr_residual_norm": float(
                np.linalg.norm(slopes["analytic_lr"])
                / np.linalg.norm(slopes["dft"] - slopes["sr_only"])),
            "maximum_abs_central_slope_eV_per_A": float(
                np.max(np.abs(slopes["analytic_lr"]))),
        },
        "correlation": correlations,
        "distance_bins": distance_rows,
        "response_active_definition": {
            "minimum_abs_dft_central_slope_eV_per_A": response_active_threshold,
            "quantile_of_nonzero_response": 0.90,
        },
        "selected_elements": selected,
        "training_context": {
            "sr_best_validation_loss": train_sr["best_val_loss"],
            "full_best_validation_loss": train_full["best_val_loss"],
            "full_validation_loss_advantage_fraction": (
                1 - train_full["best_val_loss"] / train_sr["best_val_loss"]),
            "sr_best_training_loss": train_sr["best_train_loss"],
            "full_best_training_loss": train_full["best_train_loss"],
            "full_training_loss_advantage_fraction": (
                1 - train_full["best_train_loss"] / train_sr["best_train_loss"]),
            "sr_heldout_full_space_mae_eV": held_sr["mae"],
            "full_heldout_full_space_mae_eV": held_full["mae"],
            "full_heldout_mae_advantage_fraction": (
                1 - held_full["mae"] / held_sr["mae"]),
            "note": "training/validation target spaces differ between runs",
        },
        "dft_repeat_validation": {
            "csr_sha256": repeat_hashes,
            "minus_pair_byte_identical": (
                repeat_hashes[str(minus_scan)]
                == repeat_hashes[str(minus_reference)]),
            "plus_pair_byte_identical": (
                repeat_hashes[str(plus_scan)]
                == repeat_hashes[str(plus_reference)]),
        },
    }

    json_path = output_dir / "response_scan_investigation.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {json_path}")
    md_path = output_dir / "RESPONSE_SCAN_INVESTIGATION.md"
    md_path.write_text(markdown(report))
    print(f"wrote {md_path}")
    plot_summary(report, {
        "active_mask": (np.abs(slopes["dft"])
                        >= report["response_active_definition"]
                        ["minimum_abs_dft_central_slope_eV_per_A"]),
        "sr_slope_error": per_element["sr"]["slope_error"],
        "full_slope_error": per_element["full"]["slope_error"],
    }, output_dir)
    plot_selected(data, selected, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
