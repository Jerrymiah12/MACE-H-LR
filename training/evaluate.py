"""Held-out evaluation of the paired SR / full-H checkpoints, in full-H space.

The two runs predict different things, so they are only comparable after the
SR run's prediction is put back together:

    SR run     H_full_pred = H_SR_pred + H_LR      (H_LR is an analytic label,
                                                    read from the snapshot, not
                                                    predicted)
    full run   H_full_pred = H_pred

Both are then scored against the same ground truth, `hamiltonians_full.h5`.
`H_LR` is stored only on the blocks where it is nonzero -- about 13.5k of the
24.9k blocks on a 3x3x3 snapshot -- and `H_full - H_SR` is exactly zero on the
rest, so a missing key contributes nothing.

Two held-out sets, neither ever seen in training:

    --set test    the 37 main-set 3x3x3 snapshots
    --set large   all 44 4x4x4 snapshots (the cell-size extrapolation)

Metrics are MAE and RMSE over Hamiltonian matrix elements in eV, reported
overall and broken down three ways:

    distance bin        block interatomic distance, 1 A bins, using the same
                        `mgo_lr.locality.block_distance` convention as the
                        locality reports, so the numbers line up with them
    displacement family `pattern_class` from displacement_metadata.json
    |q| shell           `q_magnitude`, rounded into shells

Usage
-----
    python training/evaluate.py --sr-run DIR --full-run DIR --set test
    python training/evaluate.py --sr-run DIR --full-run DIR --set large
    python training/evaluate.py --sr-run DIR --set test        # one model only
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import paths

BIN_WIDTH = 1.0        # Angstrom, matches locality.bin_width in configs/mgo.yaml
Q_DECIMALS = 4         # |q| shell rounding


class ErrorAccumulator:
    """Streams sum|e|, sum e^2 and counts so nothing is held per element."""

    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.n = 0
        self.max_abs = 0.0

    def add(self, err):
        err = np.asarray(err, dtype=np.float64)
        self.abs_sum += float(np.abs(err).sum())
        self.sq_sum += float((err ** 2).sum())
        self.n += err.size
        if err.size:
            self.max_abs = max(self.max_abs, float(np.abs(err).max()))

    def summary(self):
        if not self.n:
            return None
        return {"mae": self.abs_sum / self.n,
                "rmse": (self.sq_sum / self.n) ** 0.5,
                "max_abs": self.max_abs,
                "n_elements": self.n}


class Report:
    def __init__(self):
        self.overall = ErrorAccumulator()
        self.by_distance = defaultdict(ErrorAccumulator)
        self.by_family = defaultdict(ErrorAccumulator)
        self.by_q = defaultdict(ErrorAccumulator)
        self.by_snapshot = defaultdict(ErrorAccumulator)

    def as_dict(self):
        def dump(d, key_name, transform=lambda k: k):
            out = []
            for key in sorted(d):
                s = d[key].summary()
                if s:
                    out.append({key_name: transform(key), **s})
            return out

        return {
            "overall": self.overall.summary(),
            "by_distance_bin": dump(
                self.by_distance, "r_lo",
                lambda b: round(b * BIN_WIDTH, 6)),
            "by_displacement_family": dump(self.by_family, "pattern_class"),
            "by_q_shell": dump(self.by_q, "q_magnitude"),
            "by_snapshot": dump(self.by_snapshot, "snapshot"),
        }


def snapshot_dirs(workspace, which):
    """(graph_root, {sid: truth_folder}) for a held-out set."""
    splits_path = os.path.join(paths.PROVENANCE_DIR, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)
    if which == "test":
        sids = sorted(splits["main"]["test"])
        graph_root = os.path.join(workspace, "loader_splits", "test")
        # the loader view symlinks only what the loader needs; hamiltonians_lr
        # and hamiltonians_full live with the snapshot itself
        truth = {s: os.path.join(workspace, "main", s) for s in sids}
    elif which == "large":
        sids = sorted(splits["large_test"])
        graph_root = os.path.join(workspace, "test_large_cell")
        truth = {s: os.path.join(workspace, "test_large_cell", s)
                 for s in sids}
    else:
        raise SystemExit(f"unknown set {which}")
    missing = [s for s, d in truth.items() if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"{which}: {len(missing)} snapshots missing, "
                         f"e.g. {missing[:3]}")
    return sids, graph_root, truth


def build_eval_config(run_dir, graph_root, training_root, dataset_name, device):
    """The run's own train.ini, repointed at the evaluation set."""
    src_ini = os.path.join(run_dir, "src", "train.ini")
    if not os.path.isfile(src_ini):
        raise SystemExit(f"{src_ini}: missing; is {run_dir} a finished run?")
    cp = paths.read_config(src_ini)
    cp["data"]["processed_data_dir"] = graph_root
    cp["data"]["save_graph_dir"] = os.path.join(training_root, "graphs_eval")
    cp["data"]["dataset_name"] = dataset_name
    cp["basic"]["save_dir"] = os.path.join(training_root, "eval_scratch",
                                           dataset_name)
    cp["train"]["extra_validation"] = "[]"
    if device:
        cp["basic"]["device"] = device
    out = os.path.join(training_root, f"eval_{dataset_name}.ini")
    os.makedirs(training_root, exist_ok=True)
    with open(out, "w") as f:
        cp.write(f)
    return out


def predict(run_dir, config_path, sids):
    """{sid: {block key: predicted block}} from the run's best checkpoint."""
    import torch
    from maceh import DeepHE3Kernel
    from maceh.graph import Collater

    kernel = DeepHE3Kernel()
    kernel.load_config(train_config_path=config_path)
    config = kernel.train_config
    torch.set_default_dtype(config.torch_dtype)

    dataset = kernel.get_graph(config)
    kernel.config_set_target()
    dataset.set_mask(config.target_blocks,
                     convert_to_net=config.convert_net_out)
    construct_kernel = kernel.register_constructor(device=config.device)
    net = kernel.load_model(os.path.join(run_dir, "src"), device=config.device)

    ckpt_path = os.path.join(run_dir, "best_model.pkl")
    ckpt = torch.load(ckpt_path, map_location=config.device,
                      weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"  checkpoint: epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss')}")

    wanted = set(sids)
    collate = Collater()
    out = {}
    with torch.no_grad():
        for data in dataset:
            if data.stru_id not in wanted:
                continue
            batch = collate([data]).to(device=config.device)
            _, output_edge = net(batch)
            H_pred = construct_kernel.get_H(output_edge).cpu()
            blocks = {}
            kernel.update_hopping(blocks, H_pred, batch.x.cpu(),
                                  batch.edge_index.cpu(),
                                  batch.edge_key.cpu())
            out[data.stru_id] = {k: np.asarray(v, dtype=np.float64)
                                 for k, v in blocks.items()}
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"the graph dataset is missing {len(missing)} "
                         f"snapshots, e.g. {sorted(missing)[:3]}")
    return out


def score(pred_by_sid, truth_dirs, add_lr, label):
    """Accumulate errors of H_full_pred against hamiltonians_full.h5."""
    from mgo_lr.convert import read_blocks
    from mgo_lr.locality import block_distance

    report = Report()
    n_nan = 0
    for sid in sorted(pred_by_sid):
        folder = truth_dirs[sid]
        truth = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        lr = read_blocks(os.path.join(folder, "hamiltonians_lr.h5")) \
            if add_lr else {}
        with open(os.path.join(folder, "displacement_metadata.json")) as f:
            meta = json.load(f)
        family = meta.get("pattern_class", "unknown")
        q_shell = round(float(meta.get("q_magnitude") or 0.0), Q_DECIMALS)
        cell = np.loadtxt(os.path.join(folder, "lat.dat")).T
        cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T

        pred = pred_by_sid[sid]
        common = set(pred) & set(truth)
        if len(common) != len(truth):
            print(f"  {sid}: WARNING predicted {len(pred)} blocks, truth has "
                  f"{len(truth)}, scoring the {len(common)} in common")
        for key in common:
            p = pred[key]
            if add_lr and key in lr:
                p = p + lr[key]
            err = p - truth[key]
            if not np.isfinite(err).all():
                n_nan += int((~np.isfinite(err)).sum())
                err = err[np.isfinite(err)]
            bucket = int(block_distance(key, cart, cell) // BIN_WIDTH)
            report.overall.add(err)
            report.by_distance[bucket].add(err)
            report.by_family[family].add(err)
            report.by_q[q_shell].add(err)
            report.by_snapshot[sid].add(err)
        del truth, lr
    if n_nan:
        print(f"  WARNING {label}: {n_nan} non-finite predicted elements "
              "were excluded")
    return report


def print_table(name, rows, key_name):
    if not rows:
        return
    print(f"\n  -- {name} --")
    print(f"    {key_name:>22}  {'MAE (eV)':>12}  {'RMSE (eV)':>12}  "
          f"{'max|e|':>12}  {'n':>12}")
    for row in rows:
        print(f"    {str(row[key_name]):>22}  {row['mae']:12.4e}  "
              f"{row['rmse']:12.4e}  {row['max_abs']:12.4e}  "
              f"{row['n_elements']:12d}")


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--sr-run", default=None,
                        help="save_dir of the finished SR-target run")
    parser.add_argument("--full-run", default=None,
                        help="save_dir of the finished full-H baseline run")
    parser.add_argument("--set", dest="which", choices=("test", "large"),
                        required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None,
                        help="write the full report as JSON here")
    args = parser.parse_args()
    if not args.sr_run and not args.full_run:
        raise SystemExit("give --sr-run and/or --full-run")
    workspace = paths.resolve("workspace", args.workspace)
    training_root = paths.resolve("training_root", args.training_root)

    sids, graph_root, truth_dirs = snapshot_dirs(workspace, args.which)
    print(f"evaluating on the {args.which} set: {len(sids)} snapshots "
          f"from {graph_root}")

    results = {}
    for label, run_dir, add_lr in (("sr", args.sr_run, True),
                                   ("full", args.full_run, False)):
        if not run_dir:
            continue
        print(f"\n[{label}] {run_dir}")
        cfg = build_eval_config(run_dir, graph_root, training_root,
                                f"eval{args.which}{label}", args.device)
        preds = predict(run_dir, cfg, sids)
        print(f"  predicted {len(preds)} snapshots; scoring in full-H space"
              f"{' (H_SR_pred + H_LR)' if add_lr else ' (direct)'}")
        report = score(preds, truth_dirs, add_lr, label)
        results[label] = report.as_dict()
        del preds

    print(f"\n{'=' * 70}\nHELD-OUT RESULTS -- {args.which} set "
          f"({len(sids)} snapshots), full-H space\n{'=' * 70}")
    for label, res in results.items():
        o = res["overall"]
        head = "H_SR_pred + H_LR" if label == "sr" else "direct H_full_pred"
        print(f"\n[{label}] {head}")
        print(f"  overall MAE {o['mae']:.6e} eV   RMSE {o['rmse']:.6e} eV   "
              f"max|e| {o['max_abs']:.4e} eV   over {o['n_elements']} elements")
        print_table("by distance bin (r_lo, A)", res["by_distance_bin"],
                    "r_lo")
        print_table("by displacement family",
                    res["by_displacement_family"], "pattern_class")
        print_table("by |q| shell", res["by_q_shell"], "q_magnitude")

    if "sr" in results and "full" in results:
        s, f = results["sr"]["overall"], results["full"]["overall"]
        print(f"\n{'=' * 70}\nSR vs full-H baseline, {args.which} set")
        for metric in ("mae", "rmse"):
            ratio = s[metric] / f[metric] if f[metric] else float("nan")
            better = "SR better" if ratio < 1 else "baseline better"
            print(f"  {metric.upper():5} SR {s[metric]:.6e}  "
                  f"baseline {f[metric]:.6e}  ratio {ratio:.4f}  ({better})")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"set": args.which, "n_snapshots": len(sids),
                       "snapshots": sids, "results": results}, fh, indent=1)
        print(f"\nfull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
