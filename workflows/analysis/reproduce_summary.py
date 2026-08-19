"""Regenerate a compact comparison figure from committed metrics only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maceh.analysis.figures import save_figure

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "learned_response" / "metrics.json"
DEFAULT_OUTPUT = ROOT / "results" / "learned_response" / "reproduced_summary"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output stem; .png and .pdf are written")
    args = parser.parse_args(argv)

    metrics = json.loads(args.input.read_text(encoding="utf-8"))
    names = ("full", "sr", "tensor")
    labels = ("Direct Full-H", "LR-corrected SR", "SR + learned response")
    h_mae = np.array([metrics["hamiltonian"][name]["mae"] for name in names]) * 1000
    epc_mae = np.array([
        metrics["epc"][name]["complex_mae_eV_per_A"] for name in names])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    colors = ("#D55E00", "#0072B2", "#009E73")
    axes[0].bar(labels, h_mae, color=colors)
    axes[0].set_ylabel("Hamiltonian MAE (meV)")
    axes[0].set_title("Locked full-H test")
    axes[1].bar(labels, epc_mae, color=colors)
    axes[1].set_ylabel(r"EPC complex MAE (eV $\AA^{-1}$)")
    axes[1].set_title("Cartesian AO EPC")
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.spines[["top", "right"]].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        save_figure(fig, args.output.with_suffix(suffix), dpi=200)
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
