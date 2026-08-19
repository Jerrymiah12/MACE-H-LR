"""Compare SR-only, SR + fixed analytic LR, and direct Full-H EPC.

This is the controlled A/B/D decomposition supported by the checkpoints in
this repository.  There is no C case with learned Born charges or dielectric
tensors because neither trained model has those heads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


from maceh.analysis.figures import save_figure_formats
from maceh.epc.build_tensor import compute_epc_cartesian
from maceh.epc.derivative import finite_difference
from maceh.epc.lr_correction import (
    load_reindexed_equilibrium_overlap, make_gauge_fixed_predict_fn,
    make_lr_corrected_predict_fn)
from maceh.epc.supercell import (build_supercell, load_structure,
                                 uniform_grid)
from maceh.config import load_config
from maceh.response.long_range import gmax_squared, reciprocal_set
from maceh.data.structures import reciprocal


COLORS = {"A": "#56B4E9", "B": "#0072B2", "D": "#D55E00"}
GRID_COLOR = "#D9D9D9"


def load_epc(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "path": str(path.resolve()),
            "g": handle["g_real"][:] + 1j * handle["g_imag"][:],
            "kpoints": handle["kpoints"][:],
            "qpoints": handle["qpoints"][:],
            "atom_indices": handle["atom_indices"][:],
            "orbital_indices": handle["orbital_indices"][:],
            "lattice": handle["lattice"][:],
            "positions": handle["positions"][:],
            "attributes": {key: (value.item() if isinstance(value, np.generic)
                                  else value.tolist()
                                  if isinstance(value, np.ndarray) else value)
                           for key, value in handle.attrs.items()},
        }


def validate(reference: dict, others: dict[str, dict]) -> None:
    for name, current in others.items():
        if current["g"].shape != reference["g"].shape:
            raise ValueError(f"{name}: tensor shape differs from DFT")
        for field in ("kpoints", "qpoints", "atom_indices", "orbital_indices",
                      "lattice", "positions"):
            if not np.allclose(current[field], reference[field], rtol=0,
                               atol=1e-12):
                raise ValueError(f"{name}: {field} differs from DFT")


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    error = np.asarray(candidate - reference)
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    return {
        "relative_l2": float(np.linalg.norm(error) / reference_norm),
        "complex_mae_eV_per_A": float(np.mean(np.abs(error))),
        "complex_rmse_eV_per_A": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "max_abs_eV_per_A": float(np.max(np.abs(error))),
        "cosine_similarity": float(
            np.vdot(reference.ravel(), candidate.ravel()).real
            / (reference_norm * candidate_norm)),
        "norm_ratio": candidate_norm / reference_norm,
        "candidate_l2_norm_eV_per_A": candidate_norm,
        "reference_l2_norm_eV_per_A": reference_norm,
    }


def component_mae(candidate: np.ndarray, reference: np.ndarray) -> list[float]:
    return [float(np.mean(np.abs(candidate[:, :, :, alpha] -
                                 reference[:, :, :, alpha])))
            for alpha in range(3)]


def q_label(qpoint: np.ndarray) -> str:
    values = []
    for value in qpoint:
        if abs(value) < 1e-12:
            values.append("0")
        elif abs(value - 0.5) < 1e-12:
            values.append("½")
        else:
            values.append(f"{value:g}")
    return "(" + ",".join(values) + ")"


def analytic_lr_diagnostic(structure_dir: Path, overlap_dir: Path,
                           workspace: Path, config_path: Path,
                           delta: float) -> dict:
    """Construct the analytic LR derivative without evaluating either network."""
    primitive = load_structure(str(structure_dir))
    supercell, smap = build_supercell(primitive, (2, 2, 2))
    positions = torch.tensor(supercell.positions, dtype=torch.float64)
    overlaps = load_reindexed_equilibrium_overlap(str(overlap_dir), supercell)

    def zero_predictor(_positions):
        return {key: np.zeros_like(value) for key, value in overlaps.items()}

    raw_predictor = make_lr_corrected_predict_fn(
        zero_predictor, positions, supercell, str(workspace), str(overlap_dir),
        str(config_path))
    projected_predictor = make_gauge_fixed_predict_fn(
        raw_predictor, supercell, str(overlap_dir))
    grids = uniform_grid((2, 2, 2))
    values = {}
    # Mg contributes 15 AO functions and O contributes 13 in this basis.
    orbital_offsets = np.asarray([0, 15, 28], dtype=int)
    for name, predictor in (("raw", raw_predictor),
                            ("gauge_projected", projected_predictor)):
        derivative = finite_difference(
            predictor, positions, smap, orbital_offsets, delta,
            grad_threshold=0.0)
        g = compute_epc_cartesian(derivative, grids, grids)["g"]
        values[name] = {
            "l2_norm_eV_per_A": float(np.linalg.norm(g)),
            "mean_abs_eV_per_A": float(np.mean(np.abs(g))),
            "max_abs_eV_per_A": float(np.max(np.abs(g))),
            "by_q_l2_norm_eV_per_A": [
                float(np.linalg.norm(g[:, iq])) for iq in range(len(grids))],
        }

    cfg = load_config(str(config_path))
    eps = np.load(workspace / "reference" / "dielectric_infinity.npy")
    lam = float(cfg["lr"]["ewald_lambda"])
    tolerance = float(cfg["lr"]["reciprocal_tolerance"])
    n_int, g_cart = reciprocal_set(
        reciprocal(supercell.lattice), eps, gmax_squared(lam, tolerance))
    geg = np.einsum("ga,ab,gb->g", g_cart, eps, g_cart)
    weights = np.exp(-geg / (4.0 * lam ** 2))
    return {
        "definition": {
            "ewald_lambda_inverse_A": lam,
            "reciprocal_tolerance": tolerance,
            "supercell_reciprocal_vector_count": int(len(n_int)),
            "minimum_nonzero_G_inverse_A": float(
                np.min(np.linalg.norm(g_cart, axis=1))),
            "minimum_G_dot_epsilon_dot_G_inverse_A2": float(np.min(geg)),
            "maximum_ewald_weight_on_2x2x2_grid": float(np.max(weights)),
            "G_zero_included": False,
        },
        **values,
        "gauge_projection_norm_ratio": (
            values["gauge_projected"]["l2_norm_eV_per_A"]
            / values["raw"]["l2_norm_eV_per_A"]),
    }


def style(axis) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    axis.tick_params(direction="in")


def make_figure(report: dict, output_dir: Path) -> None:
    labels = report["case_labels"]
    cases = ("A", "B", "D")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    overall = [100 * report["overall"][case]["relative_l2"] for case in cases]
    bars = axes[0, 0].bar(cases, overall,
                          color=[COLORS[case] for case in cases])
    axes[0, 0].bar_label(bars, labels=[f"{value:.3f}%" for value in overall],
                         padding=3, fontsize=9)
    axes[0, 0].set_xticks(range(3), [labels[case] for case in cases])
    axes[0, 0].set_ylabel("Relative L2 error (%)")
    axes[0, 0].set_title("Overall EPC error")
    axes[0, 0].set_ylim(0, 1.14 * max(overall))

    x = np.arange(len(report["qpoints"]))
    for case in cases:
        values = [100 * row["relative_l2"]
                  for row in report["by_q"][case]]
        axes[0, 1].plot(x, values, "o-", linewidth=1.8, markersize=4,
                        color=COLORS[case], label=labels[case])
    axes[0, 1].set_xticks(x, report["q_labels"], rotation=40, ha="right")
    axes[0, 1].set_ylabel("Relative L2 error (%)")
    axes[0, 1].set_title("Error by q point")
    axes[0, 1].legend(frameon=False, fontsize=8)

    directions = np.arange(3)
    width = 0.25
    for index, case in enumerate(cases):
        values = report["component_mae_eV_per_A"][case]
        axes[1, 0].bar(directions + (index - 1) * width, values, width,
                       color=COLORS[case], label=labels[case])
    axes[1, 0].set_xticks(directions, ("x", "y", "z"))
    axes[1, 0].set_ylabel("Complex MAE (eV/Å)")
    axes[1, 0].set_title("Cartesian component error")
    axes[1, 0].legend(frameon=False, fontsize=8)

    residual = report["analytic_lr_test"]["dft_minus_A_l2_norm_eV_per_A"]
    correction = report["analytic_lr_test"]["B_minus_A_l2_norm_eV_per_A"]
    full_delta = report["analytic_lr_test"]["D_minus_A_l2_norm_eV_per_A"]
    names = ("DFT − A\nneeded residual", "B − A\nanalytic LR",
             "D − A\nmodel difference")
    values = (residual, correction, full_delta)
    bars = axes[1, 1].bar(range(3), values,
                          color=("#333333", COLORS["B"], COLORS["D"]))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(range(3), names)
    axes[1, 1].set_ylabel("Tensor L2 norm (eV/Å)")
    axes[1, 1].set_title("What changes the SR response?")
    axes[1, 1].bar_label(
        bars, labels=[f"{value:.3g}" for value in values], padding=3,
        fontsize=9)
    axes[1, 1].text(
        0.03, 0.96,
        "Analytic LR / needed residual = "
        f"{100 * report['analytic_lr_test']['correction_to_residual_norm_ratio']:.5f}%",
        transform=axes[1, 1].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
              "pad": 4})

    for axis in axes.flat:
        style(axis)
    fig.suptitle("EPC pipeline decomposition: SR-only vs fixed LR vs direct Full-H",
                 fontsize=14)
    fig.tight_layout()
    for path in save_figure_formats(fig, output_dir,
                                    "epc_pipeline_decomposition",
                                    bbox_inches="tight"):
        print(f"wrote {path}")
    plt.close(fig)


def markdown(report: dict) -> str:
    a, b, d = (report["overall"][case] for case in ("A", "B", "D"))
    lr = report["analytic_lr_test"]
    definition = report["analytic_lr_standalone"]["definition"]
    q0 = report["by_q"]["D"][0]
    nonzero_full = [100 * row["relative_l2"]
                    for row in report["by_q"]["D"][1:]]
    nonzero_sr = [100 * row["relative_l2"]
                  for row in report["by_q"]["A"][1:]]
    return f"""# EPC A/B/D pipeline decomposition

## Decisive result

The analytic LR wrapper is enabled and uses the frozen reference tensors, but
it is numerically negligible on this 2×2×2 EPC grid.  The EPC advantage comes
from the independently trained **SR-target checkpoint itself**, not from adding
the analytic LR term during EPC evaluation.

| Case | Definition | Relative L2 | Complex MAE (eV/Å) | Cosine |
|---|---|---:|---:|---:|
| A | SR checkpoint only | {100*a['relative_l2']:.6f}% | {a['complex_mae_eV_per_A']:.6f} | {a['cosine_similarity']:.6f} |
| B | SR + analytic LR, fixed DFT/DFPT tensors | {100*b['relative_l2']:.6f}% | {b['complex_mae_eV_per_A']:.6f} | {b['cosine_similarity']:.6f} |
| D | independent direct Full-H checkpoint | {100*d['relative_l2']:.6f}% | {d['complex_mae_eV_per_A']:.6f} | {d['cosine_similarity']:.6f} |

Going from A to B changes relative L2 error by only
**{100*(b['relative_l2']-a['relative_l2']):.8f} percentage points**.  The
analytic correction `B − A` has L2 norm
`{lr['B_minus_A_l2_norm_eV_per_A']:.6g} eV/Å`, only
**{100*lr['correction_to_residual_norm_ratio']:.6f}%** of the
`DFT − A` residual norm (`{lr['dft_minus_A_l2_norm_eV_per_A']:.6g} eV/Å`).
Its cosine with the needed residual is {lr['correction_residual_cosine']:.4f}.

## Why the fixed LR contribution is tiny here

The analytic definition uses `lambda = {definition['ewald_lambda_inverse_A']:.3g} Å⁻¹`.
For the 2×2×2 displacement supercell, the smallest nonzero reciprocal vector
has magnitude `{definition['minimum_nonzero_G_inverse_A']:.6f} Å⁻¹`; after
dielectric screening, even the largest sampled Ewald weight is only
`{definition['maximum_ewald_weight_on_2x2x2_grid']:.6e}`.  `G=0` is excluded
by definition.  A standalone LR-only finite difference gives norm
`{report['analytic_lr_standalone']['raw']['l2_norm_eV_per_A']:.6g} eV/Å` both
before and after gauge projection (norm ratio
`{report['analytic_lr_standalone']['gauge_projection_norm_ratio']:.9f}`), so
the common gauge projection is not suppressing it.

## Where direct Full-H fails

At Gamma, direct Full-H is slightly better than A:
{100*q0['relative_l2']:.3f}% relative L2 versus
{100*report['by_q']['A'][0]['relative_l2']:.3f}%.  Its failure appears at the
nonzero q points: Full-H spans {min(nonzero_full):.2f}%–{max(nonzero_full):.2f}%
error, versus {min(nonzero_sr):.2f}%–{max(nonzero_sr):.2f}% for SR-only.
This localizes the discrepancy to the learned displacement response away from
Gamma, not to a missing `dZ*/dR` or `d epsilon/dR` path.

## Interpretation

* Full-H is not a composed SR + learned-tensor model; it is a separate direct
  Hamiltonian fit.
* There are no predicted Born-charge or dielectric tensors to detach.
* Case C with learned tensors cannot be evaluated from these checkpoints.
* The earlier causal wording that “analytic LR reconstruction removes the
  Full-H error” is not supported by this decomposition.  The accurate wording
  is that the **SR-target checkpoint has the lower EPC error**.
* The next response scan should therefore test why two near-identical
  equilibrium Hamiltonian fits develop very different slopes, especially at
  nonzero q.  Matched-seed repeat training would also test whether this is a
  target effect or ordinary training variance.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actual", default=(
        "runs/epc/actual/structure_primitive/epc_cartesian_actual.h5"))
    parser.add_argument("--sr-only", default=(
        "runs/epc/sr_only/structure_primitive/epc_cartesian_pred.h5"))
    parser.add_argument("--sr-fixed-lr", default=(
        "runs/epc/sr/structure_primitive/epc_cartesian_pred.h5"))
    parser.add_argument("--full", default=(
        "runs/epc/full/structure_primitive/epc_cartesian_pred.h5"))
    parser.add_argument("--workspace", default="data")
    parser.add_argument("--structure", default="workflows/epc/structure_primitive")
    parser.add_argument("--overlap-dir", default="data/pilot/snapshot_000001")
    parser.add_argument("--lr-config", default="provenance/config.resolved.yaml")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    datasets = {
        "actual": load_epc(Path(args.actual)),
        "A": load_epc(Path(args.sr_only)),
        "B": load_epc(Path(args.sr_fixed_lr)),
        "D": load_epc(Path(args.full)),
    }
    validate(datasets["actual"], {key: datasets[key]
                                  for key in ("A", "B", "D")})
    truth = datasets["actual"]["g"]
    overall = {case: metrics(datasets[case]["g"], truth)
               for case in ("A", "B", "D")}
    by_q = {case: [metrics(datasets[case]["g"][:, iq], truth[:, iq])
                   for iq in range(len(datasets["actual"]["qpoints"]))]
            for case in ("A", "B", "D")}
    components = {case: component_mae(datasets[case]["g"], truth)
                  for case in ("A", "B", "D")}

    correction = datasets["B"]["g"] - datasets["A"]["g"]
    needed = truth - datasets["A"]["g"]
    full_change = datasets["D"]["g"] - datasets["A"]["g"]
    correction_norm = float(np.linalg.norm(correction))
    needed_norm = float(np.linalg.norm(needed))
    full_change_norm = float(np.linalg.norm(full_change))
    correction_cosine = float(
        np.vdot(needed.ravel(), correction.ravel()).real
        / (needed_norm * correction_norm))

    standalone = analytic_lr_diagnostic(
        Path(args.structure), Path(args.overlap_dir), Path(args.workspace),
        Path(args.lr_config), float(datasets["A"]["attributes"]["delta"]))
    report = {
        "case_labels": {
            "A": "SR only",
            "B": "SR + fixed LR",
            "D": "Direct Full-H",
        },
        "paths": {key: datasets[key]["path"] for key in datasets},
        "qpoints": datasets["actual"]["qpoints"].tolist(),
        "q_labels": [q_label(q) for q in datasets["actual"]["qpoints"]],
        "overall": overall,
        "by_q": by_q,
        "component_mae_eV_per_A": components,
        "analytic_lr_test": {
            "B_minus_A_l2_norm_eV_per_A": correction_norm,
            "dft_minus_A_l2_norm_eV_per_A": needed_norm,
            "D_minus_A_l2_norm_eV_per_A": full_change_norm,
            "correction_to_residual_norm_ratio": correction_norm / needed_norm,
            "correction_residual_cosine": correction_cosine,
            "relative_l2_error_change_A_to_B": (
                overall["B"]["relative_l2"] - overall["A"]["relative_l2"]),
            "relative_error_reduction_A_to_B": (
                1.0 - overall["B"]["relative_l2"]
                / overall["A"]["relative_l2"]),
            "measured_correction_vs_standalone_relative_difference": abs(
                correction_norm - standalone["gauge_projected"]
                ["l2_norm_eV_per_A"]) / standalone["gauge_projected"]
                ["l2_norm_eV_per_A"],
        },
        "analytic_lr_standalone": standalone,
        "conclusion": (
            "The SR checkpoint's EPC advantage is already present without "
            "analytic LR; the fixed analytic term is negligible on this grid."),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "epc_pipeline_decomposition.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {json_path}")
    md_path = output_dir / "EPC_PIPELINE_DECOMPOSITION.md"
    md_path.write_text(markdown(report))
    print(f"wrote {md_path}")
    make_figure(report, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
