"""One-shot locked evaluation for the SR/Born/dielectric prototype.

The model and all hyperparameters must be frozen before this program is run.
It intentionally evaluates only the tensor manifest's declared test split and
refuses to overwrite its report unless ``--force`` is explicitly supplied.
"""
import argparse
import configparser
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch


from maceh import DeepHE3Kernel
from maceh.epc.lr_correction import reconstruct_total_hamiltonian
from maceh.graph import Collater
from maceh.kernel import assert_sr_tensor_checkpoint
from maceh.config import load_config
from maceh.data.io.blocks import read_blocks
from maceh.data.structures import remove_uniform_translation
from maceh.response.long_range import (assemble_lr_hamiltonian, evaluate_potential,
                       gmax_squared, lr_coefficients, reciprocal_set)
from workflows.mgo_dataset.snapshot import load_reference
from maceh.data.structures import make_supercell, reciprocal


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Moments:
    def __init__(self):
        self.absolute = 0.0
        self.square = 0.0
        self.count = 0
        self.maximum = 0.0

    def add(self, error):
        value = np.asarray(error, dtype=np.float64)
        self.absolute += float(np.abs(value).sum())
        self.square += float(np.square(value).sum())
        self.count += int(value.size)
        if value.size:
            self.maximum = max(self.maximum, float(np.abs(value).max()))

    def report(self):
        return {
            "mae": self.absolute / self.count,
            "rmse": (self.square / self.count) ** 0.5,
            "max_abs": self.maximum,
            "n_elements": self.count,
        }


def manifest_records(path):
    with open(path) as handle:
        manifest = json.load(handle)
    root = os.path.dirname(os.path.abspath(path))
    records = {}
    train_born, train_epsilon = [], []
    test_born, test_epsilon = [], []
    for item in manifest["entries"]:
        folder = os.path.join(root, item.get("profile", "fast"),
                              item["snapshot_id"])
        born = np.load(os.path.join(folder, "born_effective_charges.npy"))
        epsilon = np.load(os.path.join(folder, "dielectric_infinity.npy"))
        if item["split"] == "train":
            train_born.append(born)
            train_epsilon.append(epsilon)
        if item["split"] != "test":
            continue
        source_path = os.path.join(folder, "source.json")
        with open(source_path) as handle:
            source = json.load(handle)
        test_born.append(born)
        test_epsilon.append(epsilon)
        records[item["snapshot_id"]] = {
            "born": born,
            "epsilon": epsilon,
            "source": source,
        }
    if not records:
        raise ValueError("manifest has no locked tensor-test entries")
    if not train_born:
        raise ValueError("manifest has no tensor-training entries")
    born_delta = np.stack(test_born) - np.stack(train_born).mean(axis=0)
    epsilon_delta = (np.stack(test_epsilon)
                     - np.stack(train_epsilon).mean(axis=0))
    baseline = {
        "born_mae": float(np.abs(born_delta).mean()),
        "born_rmse": float(np.sqrt(np.square(born_delta).mean())),
        "epsilon_mae": float(np.abs(epsilon_delta).mean()),
        "epsilon_rmse": float(np.sqrt(np.square(epsilon_delta).mean())),
    }
    return records, baseline


def evaluation_config(run_dir, graph_root, graph_cache):
    source = os.path.join(run_dir, "src", "train.ini")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(source)
    parser["data"]["processed_data_dir"] = os.path.abspath(graph_root)
    parser["data"]["save_graph_dir"] = os.path.abspath(graph_cache)
    parser["data"]["dataset_name"] = "evaltest"
    parser["basic"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", prefix="sr_tensor_locked_", delete=False)
    try:
        parser.write(handle)
        return handle.name
    finally:
        handle.close()


def predicted_lr(folder, positions, born, epsilon, lr_config, workspace):
    cfg = load_config(lr_config)
    reference = load_reference(workspace)
    supercell = make_supercell(reference["prim_cell"], reference["frac"],
                               reference["species"], 3)
    if not np.allclose(supercell.cell, np.loadtxt(
            os.path.join(folder, "lat.dat")).T, atol=1e-7, rtol=0):
        raise ValueError("locked-test cell differs from the 3x3x3 LR cell")
    displacement = positions - supercell.cart
    fractional = displacement @ np.linalg.inv(supercell.cell)
    displacement = (fractional - np.rint(fractional)) @ supercell.cell
    displacement = remove_uniform_translation(displacement)
    dipoles = np.einsum("nab,nb->na", born, displacement)
    lam = float(cfg["lr"]["ewald_lambda"])
    cutoff = gmax_squared(lam, float(cfg["lr"]["reciprocal_tolerance"]))
    _, vectors = reciprocal_set(reciprocal(supercell.cell), epsilon, cutoff)
    coefficients = lr_coefficients(
        vectors, dipoles, supercell.cart, epsilon, lam,
        abs(float(np.linalg.det(supercell.cell))))
    potential = np.real(evaluate_potential(vectors, coefficients, positions))
    overlaps = read_blocks(os.path.join(folder, "overlaps.h5"))
    return assemble_lr_hamiltonian(overlaps, potential)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--lr-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.force:
        raise SystemExit(f"locked report already exists: {output}")

    run_dir = os.path.abspath(args.run)
    manifest = os.path.abspath(args.manifest)
    workspace = os.path.abspath(args.workspace)
    records, baseline = manifest_records(manifest)
    config_path = evaluation_config(run_dir, args.graph_root,
                                    args.graph_cache)
    try:
        kernel = DeepHE3Kernel()
        kernel.load_config(train_config_path=config_path)
        config = kernel.train_config
        torch.set_default_dtype(config.torch_dtype)
        dataset = kernel.get_graph(config)
        kernel.config_set_target()
        dataset.set_mask(config.target_blocks,
                         convert_to_net=config.convert_net_out)
        constructor = kernel.register_constructor(device=config.device)
        net = kernel.load_model(os.path.join(run_dir, "src"),
                                device=config.device)
        checkpoint_path = os.path.join(run_dir, "best_model.pkl")
        checkpoint = torch.load(checkpoint_path, map_location=config.device,
                                weights_only=False)
        assert_sr_tensor_checkpoint(checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        wanted = set(records)
        found = set()
        collate = Collater()
        h_error, born_error, epsilon_error = Moments(), Moments(), Moments()
        total_error = Moments()
        asr_max = 0.0
        epsilon_min, epsilon_max = float("inf"), -float("inf")
        identity_max = 0.0
        block_scored = block_truth = 0
        per_snapshot = {}
        with torch.no_grad():
            for data in dataset:
                sid = data.stru_id
                if sid not in wanted:
                    continue
                found.add(sid)
                batch = collate([data]).to(device=config.device)
                prediction = net(batch)
                edge_prediction = constructor.get_H(
                    prediction["hamiltonian"])
                mask = batch.mask.bool()
                h_delta = (edge_prediction - batch.label)[mask].cpu().numpy()
                h_error.add(h_delta)

                born = prediction["born"].cpu().numpy()
                epsilon = prediction["epsilon"][0].cpu().numpy()
                born_delta = born - records[sid]["born"]
                epsilon_delta = epsilon - records[sid]["epsilon"]
                born_error.add(born_delta)
                epsilon_error.add(epsilon_delta)
                asr = float(np.abs(born.sum(axis=0)).max())
                eigenvalues = np.linalg.eigvalsh(epsilon)
                asr_max = max(asr_max, asr)
                epsilon_min = min(epsilon_min, float(eigenvalues.min()))
                epsilon_max = max(epsilon_max, float(eigenvalues.max()))

                h_pred = {}
                kernel.update_hopping(
                    h_pred, edge_prediction.cpu(), batch.x.cpu(),
                    batch.edge_index.cpu(), batch.edge_key.cpu())
                h_pred = {key: value.numpy() for key, value in h_pred.items()}
                folder = os.path.join(workspace, "main", sid)
                lr_all = predicted_lr(
                    folder, data.pos.cpu().numpy(), born, epsilon,
                    args.lr_config, workspace)
                lr = {key: value for key, value in lr_all.items()
                      if key in h_pred}
                total = reconstruct_total_hamiltonian(h_pred, lr)
                truth = read_blocks(os.path.join(folder,
                                                 "hamiltonians_full.h5"))
                common = set(total) & set(truth)
                block_scored += len(common)
                block_truth += len(truth)
                snapshot_total = Moments()
                for key in common:
                    delta = total[key] - truth[key]
                    total_error.add(delta)
                    snapshot_total.add(delta)
                    expected = h_pred[key] + lr.get(key, 0.0)
                    identity_max = max(
                        identity_max,
                        float(np.abs(total[key] - expected).max()))
                per_snapshot[sid] = {
                    "h_sr": _moments(h_delta).report(),
                    "born_mae": float(np.abs(born_delta).mean()),
                    "born_rmse": float(np.sqrt(np.square(born_delta).mean())),
                    "epsilon_mae": float(np.abs(epsilon_delta).mean()),
                    "epsilon_rmse": float(np.sqrt(
                        np.square(epsilon_delta).mean())),
                    "born_asr_max_abs": asr,
                    "epsilon_eigenvalues": eigenvalues.tolist(),
                    "reconstructed_total_h": snapshot_total.report(),
                    "scored_blocks": len(common),
                    "truth_blocks": len(truth),
                }
        missing = wanted - found
        if missing:
            raise ValueError(f"locked graph cache is missing {sorted(missing)}")

        contract = checkpoint.get("checkpoint_metadata", {})
        report = {
            "status": "frozen-model locked test; do not use for retuning",
            "checkpoint": {
                "path": checkpoint_path,
                "sha256": sha256(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "model_contract": contract,
            },
            "manifest": {"path": manifest, "sha256": sha256(manifest)},
            "test_ids": sorted(wanted),
            "metrics": {
                "h_sr": h_error.report(),
                "born": born_error.report(),
                "epsilon": epsilon_error.report(),
                "born_asr_max_abs": asr_max,
                "epsilon_eigenvalue_range": [epsilon_min, epsilon_max],
                "reconstructed_total_h": total_error.report(),
                "reconstruction_identity_max_abs": identity_max,
                "reconstructed_block_coverage": block_scored / block_truth,
                "scored_blocks": block_scored,
                "truth_blocks": block_truth,
            },
            "constant_training_mean_baseline": baseline,
            "beats_constant_baseline": {
                "born_mae": (baseline is not None and
                             born_error.report()["mae"] < baseline["born_mae"]),
                "epsilon_mae": (baseline is not None and
                                epsilon_error.report()["mae"] <
                                baseline["epsilon_mae"]),
            },
            "per_snapshot": per_snapshot,
        }
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = output + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
        print(json.dumps(report["metrics"], indent=2, sort_keys=True))
        print(f"Locked report written once to {output}")
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def _moments(error):
    result = Moments()
    result.add(error)
    return result


if __name__ == "__main__":
    main()
