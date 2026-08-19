"""Generate the post-tensor-model Hamiltonian and EPC comparison figures."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from maceh.analysis.figures import save_figure_formats
from workflows.epc.compare_epc import load_epc, metric, validate_grids
from workflows.training.make_result_figures import read_curve, smooth_log


OUT = ROOT / "results/learned_response"
COLORS = {
    "dft": "#222222",
    "full": "#D55E00",
    "sr": "#0072B2",
    "tensor": "#009E73",
}
LABELS = {
    "dft": "Actual DFT",
    "full": "Direct Full-H",
    "sr": "LR-corrected SR",
    "tensor": r"SR + predicted $Z^*$ + $\epsilon_\infty$",
}
ORDER = ("dft", "full", "sr", "tensor")
LINESTYLES = {"dft": "-", "full": "-", "sr": "--", "tensor": ":"}
MARKERS = {"dft": "o", "full": "s", "sr": "^", "tensor": "D"}

SR_RUN = ROOT / "runs/run_sr/2026-08-06_15-56-09_sr"
FULL_RUN = ROOT / "runs/run_full/2026-08-08_13-03-35_full"
TENSOR_RUN = (ROOT / "runs/run_sr_tensors_partial/"
              "2026-08-13_19-04-59_sr_born_epsilon_partial")
REGULAR_METRICS = ROOT / "runs/result_figures/metrics.json"
LOCKED_REPORT = TENSOR_RUN / "locked_test_report.json"
EPC_PATHS = {
    "dft": ROOT / "runs/epc/actual/structure_primitive/epc_cartesian_actual.h5",
    "full": ROOT / "runs/epc/full/structure_primitive/epc_cartesian_pred.h5",
    "sr": ROOT / "runs/epc/sr/structure_primitive/epc_cartesian_pred.h5",
    "tensor": (ROOT / "runs/epc/sr_tensors_geometry/structure_primitive/"
               "epc_cartesian_pred.h5"),
    "tensor_frozen": (ROOT / "runs/epc/sr_tensors_frozen/structure_primitive/"
                      "epc_cartesian_pred.h5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig, stem):
    save_figure_formats(fig, OUT, stem, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def style(ax, log=False):
    ax.set_axisbelow(True)
    ax.grid(True, which="both" if log else "major", color="#D9D9D9",
            linewidth=0.7, alpha=0.8)
    ax.tick_params(direction="in", top=True, right=True)


def combine(rows):
    count = sum(int(row["n_elements"]) for row in rows)
    absolute = sum(float(row["mae"]) * int(row["n_elements"])
                   for row in rows)
    square = sum(float(row["rmse"]) ** 2 * int(row["n_elements"])
                 for row in rows)
    return {"mae": absolute / count, "rmse": (square / count) ** 0.5,
            "n_elements": count,
            "max_abs": max(float(row["max_abs"]) for row in rows)}


def load_hamiltonian_metrics():
    regular = json.loads(REGULAR_METRICS.read_text())
    locked = json.loads(LOCKED_REPORT.read_text())
    ids = locked["test_ids"]
    output = {"dft": {"mae": 0.0, "rmse": 0.0, "n_elements": 0,
                      "max_abs": 0.0}}
    snapshots = {"dft": {sid: {"mae": 0.0, "rmse": 0.0}
                         for sid in ids}}
    for name in ("full", "sr"):
        rows = regular["heldout_test"][name]["by_snapshot"]
        selected = [rows[sid] for sid in ids]
        output[name] = combine(selected)
        snapshots[name] = {sid: rows[sid] for sid in ids}
    output["tensor"] = locked["metrics"]["reconstructed_total_h"]
    snapshots["tensor"] = {
        sid: locked["per_snapshot"][sid]["reconstructed_total_h"]
        for sid in ids
    }
    return ids, output, snapshots, locked


TASK_RE = re.compile(
    r"Epoch #(\d+).*?Multitask validation: H MSE=([0-9.eE+-]+); "
    r"Born MAE/RMSE=([0-9.eE+-]+)/([0-9.eE+-]+) e; "
    r"epsilon MAE/RMSE=([0-9.eE+-]+)/([0-9.eE+-]+)", re.S)


def tensor_curve():
    text = (TENSOR_RUN / "result.txt").read_text(errors="replace")
    rows = [[int(value[0]), *map(float, value[1:])]
            for value in TASK_RE.findall(text)]
    if not rows:
        raise ValueError("partial tensor run has no multitask validation rows")
    return np.asarray(rows)


def plot_training():
    sr_curve, sr_best = read_curve(str(SR_RUN))
    full_curve, full_best = read_curve(str(FULL_RUN))
    tensor = tensor_curve()
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.5))

    ax = axes[0]
    for curve, best, name in ((full_curve, full_best, "full"),
                              (sr_curve, sr_best, "sr")):
        x, y = smooth_log(curve[:, 3])
        ax.plot(curve[:, 0], curve[:, 3], color=COLORS[name], alpha=0.12,
                linewidth=0.55)
        ax.plot(curve[x, 0], y, color=COLORS[name], linewidth=2,
                label=LABELS[name])
        ax.scatter(*best, color=COLORS[name], s=30, edgecolor="white",
                   linewidth=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation MSE (eV$^2$)")
    ax.set_title("A. Regular H training")
    ax.legend(frameon=False)
    style(ax, log=True)

    ax = axes[1]
    ax.plot(tensor[:, 0], tensor[:, 1], "o-", color=COLORS["tensor"],
            linewidth=2, markersize=4)
    ax.axhline(3e-6, color="#666666", linestyle="--", linewidth=1.2,
               label="H validation gate")
    ax.set_xlabel("Partial fine-tuning epoch")
    ax.set_ylabel(r"SR H validation MSE (eV$^2$)")
    ax.set_title("B. Shared-H stability")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    style(ax)

    ax = axes[2]
    ax.plot(tensor[:, 0], tensor[:, 2] / 0.00250115, "o-",
            color="#CC79A7", linewidth=2, markersize=4,
            label=r"Born MAE / strict baseline")
    ax.plot(tensor[:, 0], tensor[:, 4] / 0.00118457, "s-",
            color="#56B4E9", linewidth=2, markersize=4,
            label=r"$\epsilon_\infty$ MAE / strict baseline")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2,
               label="Pass threshold")
    ax.set_xlabel("Partial fine-tuning epoch")
    ax.set_ylabel("Normalized validation MAE")
    ax.set_title("C. Tensor-head gates")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.suptitle("Optimization history and ten-epoch multitask acceptance",
                 fontsize=14)
    fig.tight_layout()
    save(fig, "01_training_and_multitask_gates")


def plot_hamiltonian_summary(ids, metrics, snapshots):
    models = list(ORDER)
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, field, title in zip(
            axes, ("mae", "rmse"), ("Mean absolute error", "Root mean-square error")):
        values = np.asarray([metrics[name][field] * 1e3 for name in models])
        bars = ax.bar(x, values, color=[COLORS[name] for name in models],
                      edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(value, values.max() * 0.015), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, ["DFT\nreference", "Direct\nFull-H",
                         "LR-corrected\nSR",
                         "SR + $Z^*$ +\n$\\epsilon_\\infty$"])
        ax.set_ylabel(f"{title} (meV)")
        ax.set_title(title)
        ax.set_ylim(0, values.max() * 1.20)
        style(ax)
    fig.suptitle("Full-H matrix-element accuracy on the five locked snapshots",
                 fontsize=14)
    fig.tight_layout()
    save(fig, "02_locked_hamiltonian_error")

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    xpos = np.arange(len(ids))
    width = 0.22
    for offset, name in zip((-1.5, -0.5, 0.5, 1.5), ORDER):
        values = [snapshots[name][sid]["mae"] * 1e3 for sid in ids]
        ax.bar(xpos + offset * width, values, width, color=COLORS[name],
               label=LABELS[name], edgecolor="white", linewidth=0.5)
    ax.set_xticks(xpos, [sid.replace("snapshot_", "") for sid in ids])
    ax.set_xlabel("Locked snapshot ID")
    ax.set_ylabel("Full-H matrix-element MAE (meV)")
    ax.set_title("Configuration-resolved held-out Hamiltonian error")
    ax.legend(frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    save(fig, "03_locked_hamiltonian_by_snapshot")


def load_epc_data():
    data = {name: load_epc(str(EPC_PATHS[name])) for name in ORDER}
    frozen = load_epc(str(EPC_PATHS["tensor_frozen"]))
    validate_grids(data["dft"], {name: data[name]
                                  for name in ("full", "sr", "tensor")})
    validate_grids(data["dft"], {"tensor_frozen": frozen})
    summaries = {"dft": metric(data["dft"]["g"], data["dft"]["g"])}
    for name in ("full", "sr", "tensor"):
        summaries[name] = metric(data[name]["g"], data["dft"]["g"])
    return data, frozen, summaries


def plot_epc_summary(data, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(len(ORDER))
    for ax, field, ylabel, scale in (
            (axes[0], "relative_l2", "Relative L2 error (%)", 100),
            (axes[1], "complex_mae_eV_per_A", "Complex MAE (eV/Å)", 1)):
        values = np.asarray([summaries[name][field] * scale for name in ORDER])
        bars = ax.bar(x, values, color=[COLORS[name] for name in ORDER],
                      edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(value, values.max() * 0.015), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, ["DFT", "Direct\nFull-H", "LR-corrected\nSR",
                         "SR + $Z^*$ +\n$\\epsilon_\\infty$"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, values.max() * 1.18)
        style(ax)
    axes[0].set_title("A. Overall tensor error")
    axes[1].set_title("B. Complex component error")
    fig.suptitle("Cartesian-AO EPC accuracy against actual DFT", fontsize=14)
    fig.tight_layout()
    save(fig, "04_epc_overall_error")

    truth = data["dft"]["g"]
    qpoints = data["dft"]["qpoints"]
    qlabels = ["(" + ",".join("½" if abs(v - .5) < 1e-12 else "0"
                                if abs(v) < 1e-12 else f"{v:g}" for v in q)
               + ")" for q in qpoints]
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 8.0), sharex=True)
    for name in ORDER:
        norms = [np.linalg.norm(data[name]["g"][:, iq])
                 for iq in range(len(qpoints))]
        axes[0].plot(range(len(qpoints)), norms,
                     color=COLORS[name], linestyle=LINESTYLES[name],
                     marker=MARKERS[name], linewidth=2, markersize=4,
                     label=LABELS[name])
        errors = [np.mean(np.abs(data[name]["g"][:, iq] - truth[:, iq]))
                  for iq in range(len(qpoints))]
        axes[1].plot(range(len(qpoints)), errors,
                     color=COLORS[name], linestyle=LINESTYLES[name],
                     marker=MARKERS[name], linewidth=2, markersize=4,
                     label=LABELS[name])
    axes[0].set_ylabel(r"$\|g(q)\|_2$ (eV/Å)")
    axes[0].set_title("A. EPC strength")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylabel("Complex MAE (eV/Å)")
    axes[1].set_xlabel("q point (fractional reciprocal coordinates)")
    axes[1].set_title("B. Error relative to actual DFT")
    axes[1].set_xticks(range(len(qpoints)), qlabels, rotation=25, ha="right")
    for ax in axes:
        style(ax)
    fig.suptitle("q-resolved Cartesian-AO EPC comparison", fontsize=14)
    fig.tight_layout()
    save(fig, "05_epc_q_resolved")


def plot_epc_parity_and_distribution(data, summaries):
    truth_complex = data["dft"]["g"].ravel()
    truth = np.concatenate([truth_complex.real, truth_complex.imag])
    all_values = [truth]
    flat = {}
    for name in ("full", "sr", "tensor"):
        current = data[name]["g"].ravel()
        flat[name] = np.concatenate([current.real, current.imag])
        all_values.append(flat[name])
    limit = float(np.quantile(np.abs(np.concatenate(all_values)), 0.9995))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharex=True,
                             sharey=True)
    cmaps = {"full": "Oranges", "sr": "Blues", "tensor": "Greens"}
    for ax, name in zip(axes, ("full", "sr", "tensor")):
        shown = ((np.abs(truth) <= limit) & (np.abs(flat[name]) <= limit))
        density = ax.hexbin(truth[shown], flat[name][shown], gridsize=75,
                            bins="log", mincnt=1,
                            extent=(-limit, limit, -limit, limit),
                            cmap=cmaps[name])
        ax.plot([-limit, limit], [-limit, limit], "--", color=COLORS["dft"],
                linewidth=1.1, label="Actual DFT identity")
        ax.text(0.04, 0.96,
                f"relative L2 = {100*summaries[name]['relative_l2']:.2f}%\n"
                f"MAE = {summaries[name]['complex_mae_eV_per_A']:.3f} eV/Å",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.9})
        ax.set_title(LABELS[name])
        ax.set_xlabel("Actual DFT EPC component (eV/Å)")
        ax.set_aspect("equal", adjustable="box")
        style(ax)
        fig.colorbar(density, ax=ax, label="component count")
    axes[0].set_ylabel("Predicted EPC component (eV/Å)")
    fig.suptitle("Actual DFT versus predicted Cartesian-AO EPC\n"
                 "(real and imaginary components; common 99.95% range)",
                 fontsize=14)
    fig.tight_layout()
    save(fig, "06_epc_dft_parity")

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for name in ORDER:
        values = np.sort(np.abs(data[name]["g"]).ravel())
        probability = np.linspace(0, 1, len(values), endpoint=False)
        positive = values > 0
        ax.semilogx(values[positive], probability[positive],
                    color=COLORS[name], linestyle=LINESTYLES[name],
                    linewidth=2, label=LABELS[name])
    ax.set_xlabel(r"Cartesian-AO EPC magnitude $|g|$ (eV/Å)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Distribution of EPC magnitudes")
    ax.legend(frameon=False)
    style(ax, log=True)
    fig.tight_layout()
    save(fig, "07_epc_magnitude_distribution")


def write_outputs(ids, h_metrics, snapshots, locked, epc_data, frozen,
                  epc_metrics):
    frozen_difference = metric(frozen["g"], epc_data["tensor"]["g"])
    payload = {
        "scope": {
            "hamiltonian_snapshots": ids,
            "hamiltonian_quantity": "full-H matrix elements",
            "epc_quantity": "Cartesian AO EPC before phonon/band contraction",
            "tensor_epc_mode_plotted": "geometry_dependent",
        },
        "sources": {
            "regular_training_metrics": str(REGULAR_METRICS),
            "locked_tensor_report": str(LOCKED_REPORT),
            "epc": {name: str(path) for name, path in EPC_PATHS.items()},
            "sha256": {
                "locked_tensor_report": sha256(LOCKED_REPORT),
                **{f"epc_{name}": sha256(path)
                   for name, path in EPC_PATHS.items()},
            },
        },
        "hamiltonian": h_metrics,
        "hamiltonian_by_snapshot": snapshots,
        "tensor_targets": {
            "born": locked["metrics"]["born"],
            "epsilon": locked["metrics"]["epsilon"],
            "constant_baseline": locked["constant_training_mean_baseline"],
        },
        "epc": epc_metrics,
        "tensor_epc_frozen_vs_geometry": frozen_difference,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = f"""# Full-H, LR-corrected SR, and predicted-response comparison

This folder compares four consistent references/outputs:

- actual DFT;
- the direct Full-H model;
- the original LR-corrected SR model, using fixed reference response tensors;
- the new SR model reconstructed with its predicted Born charges and
  dielectric tensor. The plotted EPC result is the geometry-dependent mode.

The regular Hamiltonian comparison uses the five locked tensor-test snapshots,
which are also members of the original 37-snapshot held-out split. No training
or validation structures are mixed into these plots. DFT has zero prediction
error by definition and is shown as the reference.

## Headline values

| Model | Full-H MAE (meV) | EPC relative L2 | EPC complex MAE (eV/A) |
|---|---:|---:|---:|
| Actual DFT | 0 | 0 | 0 |
| Direct Full-H | {1e3*h_metrics['full']['mae']:.6f} | {epc_metrics['full']['relative_l2']:.6f} | {epc_metrics['full']['complex_mae_eV_per_A']:.6f} |
| LR-corrected SR | {1e3*h_metrics['sr']['mae']:.6f} | {epc_metrics['sr']['relative_l2']:.6f} | {epc_metrics['sr']['complex_mae_eV_per_A']:.6f} |
| SR + predicted Z* + epsilon_inf | {1e3*h_metrics['tensor']['mae']:.6f} | {epc_metrics['tensor']['relative_l2']:.6f} | {epc_metrics['tensor']['complex_mae_eV_per_A']:.6f} |

The equilibrium-frozen and geometry-dependent predicted-tensor EPC results are
nearly identical at the 5e-6 A finite-difference displacement: relative L2
difference `{frozen_difference['relative_l2']:.3e}` and complex MAE
`{frozen_difference['complex_mae_eV_per_A']:.3e} eV/A`.

Every figure is provided as both PNG and vector PDF. Exact metrics, input paths,
and SHA-256 hashes are in `metrics.json`; rerun with `python
the pre-refactor learned-response plot generator.
"""
    (OUT / "README.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, h_metrics, snapshots, locked = load_hamiltonian_metrics()
    epc_data, frozen, epc_metrics = load_epc_data()
    plot_training()
    plot_hamiltonian_summary(ids, h_metrics, snapshots)
    plot_epc_summary(epc_data, epc_metrics)
    plot_epc_parity_and_distribution(epc_data, epc_metrics)
    write_outputs(ids, h_metrics, snapshots, locked, epc_data, frozen,
                  epc_metrics)
    print(f"completed plots in {OUT}")


if __name__ == "__main__":
    main()
