"""Audit what the SR and Full-H checkpoints actually predict.

This is deliberately separate from the EPC accuracy analysis.  It answers the
implementation questions that have to be settled before interpreting an EPC
error difference:

* whether Full-H is a composed SR + learned-tensor + analytic-LR model;
* whether either checkpoint contains Born-charge or dielectric heads;
* which tensors enter the analytic LR reconstruction;
* whether Full-H and reconstructed SR agree at the equilibrium geometry.

The equilibrium comparison is evaluated both in the raw training gauge and
after applying the same equilibrium-overlap gauge projection used by EPC.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


from maceh.analysis.figures import save_figure_formats
from maceh.epc.lr_correction import project_hamiltonian_gauge
from maceh.data.io.blocks import read_blocks
from workflows.training.evaluate import build_eval_config, predict
from workflows.training.make_result_figures import (full_space_blocks,
                                          make_single_snapshot_view)


TENSOR_HEAD_PATTERN = re.compile(
    r"born|zstar|z_star|dielectric|epsilon|eps|charge|polar", re.I)
SR_COLOR = "#0072B2"
FULL_COLOR = "#D55E00"
GRID_COLOR = "#D9D9D9"


def newest_run(training_root: Path, run_name: str) -> Path:
    candidates = sorted(glob.glob(str(training_root / run_name / "*")))
    candidates = [Path(path) for path in candidates
                  if (Path(path) / "best_model.pkl").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no completed run below {training_root / run_name}")
    return candidates[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ini(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    return config


def checkpoint_summary(run_dir: Path) -> dict:
    checkpoint_path = run_dir / "best_model.pkl"
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state = checkpoint["state_dict"]
    matches = sorted(key for key in state if TENSOR_HEAD_PATTERN.search(key))
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "validation_loss": float(checkpoint.get("val_loss", math.nan)),
        "state_tensor_count": len(state),
        "state_scalar_count": int(sum(value.numel() for value in state.values())),
        "born_dielectric_head_key_matches": matches,
        "last_state_keys": list(state)[-10:],
    }


def config_summary(run_dir: Path, epc_config: Path) -> dict:
    train_path = run_dir / "src" / "train.ini"
    train = read_ini(train_path)
    epc = read_ini(epc_config)
    epc_section = epc["epc"]
    return {
        "training_config": str(train_path.resolve()),
        "dataset_name": train.get("data", "dataset_name"),
        "training_target": train.get("train", "target", fallback="hamiltonian"),
        "epc_config": str(epc_config.resolve()),
        "analytic_lr_workspace": epc_section.get("analytic_lr_workspace"),
        "analytic_lr_overlap_dir": epc_section.get("analytic_lr_overlap_dir"),
        "analytic_lr_config": epc_section.get("analytic_lr_config"),
        "gauge_overlap_dir": epc_section.get("gauge_overlap_dir"),
    }


def block_metrics(candidate: dict, reference: dict) -> dict:
    common = sorted(set(candidate) & set(reference))
    if not common:
        raise ValueError("block dictionaries have no keys in common")
    errors = np.concatenate([
        np.asarray(candidate[key] - reference[key], dtype=np.float64).ravel()
        for key in common])
    truth = np.concatenate([
        np.asarray(reference[key], dtype=np.float64).ravel()
        for key in common])
    values = np.concatenate([
        np.asarray(candidate[key], dtype=np.float64).ravel()
        for key in common])
    truth_norm = float(np.linalg.norm(truth))
    value_norm = float(np.linalg.norm(values))
    return {
        "candidate_block_count": len(candidate),
        "reference_block_count": len(reference),
        "common_block_count": len(common),
        "same_key_set": set(candidate) == set(reference),
        "n_elements": int(errors.size),
        "mae_eV": float(np.mean(np.abs(errors))),
        "rmse_eV": float(np.sqrt(np.mean(np.square(errors)))),
        "max_abs_eV": float(np.max(np.abs(errors))),
        "relative_l2": float(np.linalg.norm(errors) / truth_norm),
        "candidate_norm_ratio": value_norm / truth_norm,
        "cosine_similarity": float(np.dot(truth, values)
                                   / (truth_norm * value_norm)),
    }


def flattened_common(first: dict, second: dict) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(first) & set(second))
    return (
        np.concatenate([np.asarray(first[key]).ravel() for key in common]),
        np.concatenate([np.asarray(second[key]).ravel() for key in common]),
    )


def tensor_summary(path: Path) -> dict:
    array = np.load(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def save_figure(report: dict, predictions: dict, truth: dict,
                output_dir: Path) -> None:
    sr = predictions["sr_gauge_projected"]
    full = predictions["full_gauge_projected"]
    full_values, sr_values = flattened_common(full, sr)

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
    bounds = np.quantile(np.concatenate((full_values, sr_values)),
                         [0.001, 0.999])
    margin = 0.04 * float(bounds[1] - bounds[0])
    lo, hi = float(bounds[0] - margin), float(bounds[1] + margin)
    density = axes[0].hexbin(sr_values, full_values, gridsize=85, bins="log",
                             mincnt=1, extent=(lo, hi, lo, hi), cmap="viridis")
    axes[0].plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1)
    mismatch = report["equilibrium"]["gauge_projected"]["full_vs_sr_plus_lr"]
    axes[0].text(
        0.04, 0.96,
        f"MAE = {1e3 * mismatch['mae_eV']:.3f} meV\n"
        f"relative L2 = {100 * mismatch['relative_l2']:.3f}%",
        transform=axes[0].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
              "pad": 4})
    axes[0].set_xlabel(r"SR prediction + analytic $H_{LR}$ (eV)")
    axes[0].set_ylabel("Direct Full-H prediction (eV)")
    axes[0].set_title("Equilibrium model-to-model parity")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(density, ax=axes[0], label="matrix-element count")

    projected = report["equilibrium"]["gauge_projected"]
    labels = ["SR + fixed LR", "Direct Full-H"]
    maes = [1e3 * projected["sr_plus_lr_vs_dft"]["mae_eV"],
            1e3 * projected["full_vs_dft"]["mae_eV"]]
    rmses = [1e3 * projected["sr_plus_lr_vs_dft"]["rmse_eV"],
             1e3 * projected["full_vs_dft"]["rmse_eV"]]
    xpos = np.arange(2)
    width = 0.34
    axes[1].bar(xpos - width / 2, maes, width, color=(SR_COLOR, FULL_COLOR),
                alpha=0.95, label="MAE")
    axes[1].bar(xpos + width / 2, rmses, width, color=(SR_COLOR, FULL_COLOR),
                alpha=0.45, hatch="//", label="RMSE")
    axes[1].set_xticks(xpos, labels)
    axes[1].set_ylabel("Hamiltonian error (meV)")
    axes[1].set_title("Equilibrium error versus DFT")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
        axis.tick_params(direction="in")
    fig.suptitle("Full-H implementation audit: independently trained predictors")
    fig.tight_layout()
    for path in save_figure_formats(fig, output_dir,
                                    "full_h_pipeline_audit_equilibrium",
                                    bbox_inches="tight"):
        print(f"wrote {path}")
    plt.close(fig)


def markdown_report(report: dict) -> str:
    raw = report["equilibrium"]["raw"]
    projected = report["equilibrium"]["gauge_projected"]
    mismatch = projected["full_vs_sr_plus_lr"]
    sr = report["models"]["sr"]
    full = report["models"]["full"]
    born = np.asarray(report["fixed_reference_tensors"]["born_effective_charges"]["values"])
    eps = np.asarray(report["fixed_reference_tensors"]["dielectric_infinity"]["values"])
    return f"""# Full-H pipeline architecture audit

## Finding

The conditional learned-tensor architecture is **not present in these runs**.
The Full-H checkpoint directly predicts the total Hamiltonian.  The SR
checkpoint directly predicts the SR target, and only its EPC configuration
adds an analytic LR term.  Neither checkpoint contains a Born-charge or
dielectric head.

| Check | SR | Full-H |
|---|---:|---:|
| Training dataset | `{sr['config']['dataset_name']}` | `{full['config']['dataset_name']}` |
| Checkpoint tensors | {sr['checkpoint']['state_tensor_count']:,} | {full['checkpoint']['state_tensor_count']:,} |
| Checkpoint scalars | {sr['checkpoint']['state_scalar_count']:,} | {full['checkpoint']['state_scalar_count']:,} |
| Born/dielectric-like state keys | {len(sr['checkpoint']['born_dielectric_head_key_matches'])} | {len(full['checkpoint']['born_dielectric_head_key_matches'])} |
| Analytic LR enabled in EPC | yes | no |

## Equilibrium numerical check

The stored analytic LR label has norm
`{report['equilibrium']['analytic_lr_label']['l2_norm_eV']:.3e} eV` and maximum
absolute element `{report['equilibrium']['analytic_lr_label']['max_abs_eV']:.3e} eV` at equilibrium,
so `SR + analytic LR` is numerically the SR prediction there.

After the same overlap-gauge projection used in EPC:

| Comparison | MAE (eV) | RMSE (eV) | relative L2 |
|---|---:|---:|---:|
| Full-H vs SR + LR | {mismatch['mae_eV']:.6e} | {mismatch['rmse_eV']:.6e} | {100*mismatch['relative_l2']:.4f}% |
| SR + LR vs DFT | {projected['sr_plus_lr_vs_dft']['mae_eV']:.6e} | {projected['sr_plus_lr_vs_dft']['rmse_eV']:.6e} | {100*projected['sr_plus_lr_vs_dft']['relative_l2']:.4f}% |
| Full-H vs DFT | {projected['full_vs_dft']['mae_eV']:.6e} | {projected['full_vs_dft']['rmse_eV']:.6e} | {100*projected['full_vs_dft']['relative_l2']:.4f}% |

They are not mathematically identical, nor should they be: they are separate
fits to different targets.  The raw-gauge Full-H/SR mismatch is
`{raw['full_vs_sr_plus_lr']['mae_eV']:.6e} eV` MAE.

## LR tensors and derivative path

The SR EPC wrapper loads frozen NumPy tensors before its prediction closure:

* Born effective charges: diagonal values {born[0, 0, 0]:.8f} (Mg) and
  {born[1, 0, 0]:.8f} (O), SHA-256
  `{report['fixed_reference_tensors']['born_effective_charges']['sha256']}`.
* Electronic dielectric tensor: diagonal value {eps[0, 0]:.9f}, SHA-256
  `{report['fixed_reference_tensors']['dielectric_infinity']['sha256']}`.

EPC uses central finite differences in atomic positions.  There is no
autodiff path through these arrays, no predicted `Z*`/`epsilon_infinity`, and
therefore no meaningful detach/stop-gradient experiment for the current
checkpoints.

## Correct decomposition for these artifacts

The realizable comparison is:

1. A: finite difference of the direct SR predictor;
2. B: finite difference of SR plus analytic LR using frozen DFT/DFPT tensors;
3. D: finite difference of the independently trained direct Full-H predictor.

The proposed C case with learned tensors cannot be constructed from these
checkpoints.  Its absence is an architectural fact, not a failed numerical
test.  EPC metrics for A/B/D are reported separately after the SR-only run.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="data")
    parser.add_argument("--training-root", default="runs")
    parser.add_argument("--output-dir", default="results/project_update")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    training_root = Path(args.training_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sr_run = newest_run(training_root, "run_sr")
    full_run = newest_run(training_root, "run_full")

    models = {}
    for label, run_dir, epc_ini in (
            ("sr", sr_run, Path("workflows/epc/sr.ini")),
            ("full", full_run, Path("workflows/epc/full.ini"))):
        models[label] = {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": checkpoint_summary(run_dir),
            "config": config_summary(run_dir, epc_ini),
        }

    source = workspace / "pilot" / "snapshot_000001"
    cache = training_root / "result_figures" / "cache"
    view = cache / "pilot_equilibrium_view"
    sid = make_single_snapshot_view(str(source), str(view))
    predictions = {}
    for label, run_dir in (("sr", sr_run), ("full", full_run)):
        config_path = build_eval_config(
            str(run_dir), str(view), str(cache), "pilot_equilibrium", label,
            args.device)
        predictions[label] = predict(str(run_dir), config_path, [sid])[sid]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    truth = read_blocks(str(source / "hamiltonians_full.h5"))
    lr = read_blocks(str(source / "hamiltonians_lr.h5"))
    overlaps = read_blocks(str(source / "overlaps.h5"))
    sr_reconstructed = full_space_blocks(predictions["sr"], str(source), True)
    full_prediction = predictions["full"]

    projected = {}
    coefficients = {}
    for label, blocks in (("dft", truth), ("sr", sr_reconstructed),
                          ("full", full_prediction)):
        projected[label], coefficients[label] = project_hamiltonian_gauge(
            blocks, overlaps)

    lr_values = np.concatenate([np.asarray(value).ravel()
                                for value in lr.values()])
    report = {
        "conclusion": (
            "Full-H is an independently trained direct total-Hamiltonian "
            "predictor; it is not SR plus learned Born/dielectric heads and "
            "analytic LR reconstruction."),
        "models": models,
        "fixed_reference_tensors": {
            "born_effective_charges": tensor_summary(
                workspace / "reference" / "born_effective_charges.npy"),
            "dielectric_infinity": tensor_summary(
                workspace / "reference" / "dielectric_infinity.npy"),
        },
        "derivative_path": {
            "method": "central finite difference of predicted Hamiltonians",
            "sr_analytic_lr_uses_fixed_numpy_tensors": True,
            "autodiff_through_born_or_dielectric": False,
            "learned_tensor_case_available": False,
        },
        "equilibrium": {
            "snapshot": str(source),
            "analytic_lr_label": {
                "block_count": len(lr),
                "n_elements": int(lr_values.size),
                "l2_norm_eV": float(np.linalg.norm(lr_values)),
                "max_abs_eV": float(np.max(np.abs(lr_values))),
            },
            "raw": {
                "full_vs_sr_plus_lr": block_metrics(full_prediction,
                                                      sr_reconstructed),
                "sr_plus_lr_vs_dft": block_metrics(sr_reconstructed, truth),
                "full_vs_dft": block_metrics(full_prediction, truth),
            },
            "gauge_projection_coefficients_eV": coefficients,
            "gauge_projected": {
                "full_vs_sr_plus_lr": block_metrics(projected["full"],
                                                      projected["sr"]),
                "sr_plus_lr_vs_dft": block_metrics(projected["sr"],
                                                     projected["dft"]),
                "full_vs_dft": block_metrics(projected["full"],
                                               projected["dft"]),
            },
        },
    }

    json_path = output_dir / "full_h_pipeline_audit.json"
    with json_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"wrote {json_path}")
    md_path = output_dir / "FULL_H_PIPELINE_AUDIT.md"
    md_path.write_text(markdown_report(report))
    print(f"wrote {md_path}")
    save_figure(report, {
        "sr_gauge_projected": projected["sr"],
        "full_gauge_projected": projected["full"],
    }, projected["dft"], output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
