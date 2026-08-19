"""Production-sized training smoke: a few real epochs on the real splits.

`smoke_onebatch.py` proves the pipeline runs. This proves the *production*
configuration runs, at full model size, on the frozen data, and that what it
trains on is exactly what it is supposed to train on.

    [1] resolve the production config with num_epoch overridden and a
        throwaway save_dir, so the real run directory stays untouched
    [2] pre-flight: build the dataset and loaders in-process and assert the
        loaders' *membership* is exactly the frozen 330 train / 37 validation,
        and that no held-out test or large-cell snapshot is present at all
    [3] run the real `maceh train` as a subprocess, sampling GPU
        utilisation while it works
    [4] post-flight: the epoch losses parsed from the log are all finite and
        the best checkpoint reloads bit-identically

The graph cache is shared with the production run by design -- it is the same
data, and building it is the expensive part. Only `save_dir` and `num_epoch`
differ, so a passing smoke is evidence about the run that follows it.

Usage
-----
    python -m workflows.training.smoke_production --target sr   --epochs 2
    python -m workflows.training.smoke_production --target full --epochs 2
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time


from workflows.training import paths

LOSS_RE = re.compile(r"(train|val)(?:idation)?[ _]loss[:= ]+([0-9eE.+-]+)",
                     re.IGNORECASE)


class GpuSampler(threading.Thread):
    """Poll nvidia-smi while the training subprocess runs."""

    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                continue
            for line in out.splitlines():
                try:
                    util, mem = (int(x) for x in line.split(","))
                except ValueError:
                    continue
                self.samples.append((util, mem))

    def stop(self):
        self._stop.set()

    def report(self):
        if not self.samples:
            return "no GPU samples collected"
        utils = [u for u, _ in self.samples]
        mems = [m for _, m in self.samples]
        return (f"{len(self.samples)} samples: GPU util "
                f"max {max(utils)}% / mean {sum(utils)/len(utils):.0f}%, "
                f"memory max {max(mems)} MiB")


def resolve_config(target, epochs, workspace, training_root, device):
    src = paths.SR_CONFIG if target == "sr" else paths.FULL_CONFIG
    cp = paths.read_config(src)
    cp["DEFAULT"]["workspace"] = workspace
    cp["DEFAULT"]["training_root"] = training_root
    cp["train"]["num_epoch"] = str(epochs)
    if device:
        cp["basic"]["device"] = device
    save_dir = os.path.join(training_root, f"smoke_prod_{target}")
    cp["basic"]["save_dir"] = save_dir
    out = os.path.join(training_root, f"smoke_prod_{target}.resolved.ini")
    os.makedirs(training_root, exist_ok=True)
    with open(out, "w") as f:
        cp.write(f)
    return out, save_dir, src


def preflight(config_path, workspace):
    """Assert the loaders hold exactly the frozen splits, by membership."""
    from maceh import DeepHE3Kernel

    splits = paths.load_frozen_splits(workspace, "main")
    want_train = sorted(splits["train"])
    want_val = sorted(splits["validation"])
    # Snapshot ids are unique only *within* a set -- `main`, `pilot` and
    # `large_test` each number from snapshot_000001, so all 44 large ids also
    # name unrelated main-set structures. Comparing ids across sets is
    # therefore meaningless; the large set is excluded structurally instead,
    # by the view being built only from main's train and validation loader
    # views, which the exact-membership assertions below already pin.
    forbidden = set(splits["test"])

    import torch

    kernel = DeepHE3Kernel()
    kernel.load_config(train_config_path=config_path)
    config = kernel.train_config
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "config asks for cuda but torch.cuda.is_available() is False. "
            f"This is torch {torch.__version__}; a +cpu build cannot use the "
            "GPU. Install a CUDA build, or pass --device cpu.")
    dataset = kernel.get_graph(config)
    kernel.config_set_target()
    dataset.set_mask(config.target_blocks,
                     convert_to_net=config.convert_net_out)
    train_loader, val_loader, extra_val_loader, test_loader = \
        kernel.get_loader()

    def ids_of(loader):
        return sorted(dataset.data.stru_id[i] for i in loader.sampler.indices)

    got_train, got_val = ids_of(train_loader), ids_of(val_loader)
    present = set(dataset.data.stru_id)

    assert len(got_train) == 330, \
        f"train loader holds {len(got_train)} snapshots, expected 330"
    assert len(got_val) == 37, \
        f"validation loader holds {len(got_val)} snapshots, expected 37"
    assert got_train == want_train, "train loader is not the frozen 330"
    assert got_val == want_val, "validation loader is not the frozen 37"
    leaked = sorted(present & forbidden)
    assert not leaked, ("held-out main-set test snapshots are in the training "
                        f"dataset: {leaked[:5]}"
                        f"{'...' if len(leaked) > 5 else ''}")
    assert not set(got_train) & set(got_val), "train and validation overlap"
    assert len(present) == 367, \
        (f"dataset holds {len(present)} snapshots, expected exactly 367 "
         "(330 train + 37 validation) -- anything else means the view picked "
         "up structures it should not have")
    print(f"  train loader      330 snapshots == frozen main.train")
    print(f"  validation loader  37 snapshots == frozen main.validation")
    print(f"  dataset holds {len(present)} snapshots (330+37), "
          f"none from main.test")
    return config


def parse_losses(log_path):
    losses = []
    with open(log_path, errors="replace") as f:
        for line in f:
            for _, value in LOSS_RE.findall(line):
                try:
                    losses.append(float(value))
                except ValueError:
                    pass
    return losses


def find_run_dir(save_dir):
    """Resolve MACE-H's timestamped run directory below ``save_dir``."""
    if os.path.isfile(os.path.join(save_dir, "best_model.pkl")):
        return save_dir
    candidates = []
    if os.path.isdir(save_dir):
        for name in os.listdir(save_dir):
            path = os.path.join(save_dir, name)
            if (os.path.isdir(path) and
                    os.path.isfile(os.path.join(path, "best_model.pkl")) and
                    os.path.isdir(os.path.join(path, "src"))):
                candidates.append(path)
    assert len(candidates) == 1, \
        (f"expected one completed timestamped run under {save_dir}, found "
         f"{len(candidates)}: {candidates}")
    return candidates[0]


def postflight(run_dir, device):
    import torch

    ckpt_path = os.path.join(run_dir, "best_model.pkl")
    assert os.path.isfile(ckpt_path), \
        f"{ckpt_path}: training produced no checkpoint"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt, "checkpoint has no state_dict"
    val_loss = ckpt.get("val_loss")
    print(f"  best_model.pkl: epoch {ckpt.get('epoch')}, "
          f"val_loss {val_loss}, {len(ckpt['state_dict'])} tensors")
    assert val_loss is None or (val_loss == val_loss and
                                abs(float(val_loss)) != float("inf")), \
        f"checkpoint val_loss is not finite: {val_loss}"

    # Reload into a fresh model built from the run's own saved source. A bare
    # kernel is enough: load_model() reads NetOutInfo from src/ and only
    # touches train_config when a train_loader is passed.
    from maceh import DeepHE3Kernel

    net = DeepHE3Kernel().load_model(os.path.join(run_dir, "src"),
                                     device=torch.device("cpu"))
    net.load_state_dict(ckpt["state_dict"])
    again = net.state_dict()
    cmp = [k for k in ckpt["state_dict"]
           if again[k].is_floating_point() and again[k].numel() > 0]
    worst = max(float((ckpt["state_dict"][k].cpu() - again[k]).abs().max())
                for k in cmp)
    print(f"  reloaded into a fresh model from {run_dir}/src, "
          f"{len(cmp)} tensors compared, worst delta {worst:.3e}")
    assert worst == 0.0, "checkpoint reload changed the weights"


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--target", choices=("sr", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=2,
                        help="epochs to run (1-5 is the point of a smoke)")
    parser.add_argument("--device", default=None,
                        help="override the config's [basic] device")
    parser.add_argument("--skip-train", action="store_true",
                        help="run the pre-flight membership check only")
    args = parser.parse_args()
    workspace = paths.resolve("workspace", args.workspace)
    training_root = paths.resolve("training_root", args.training_root)

    print(f"[1] resolving train_{args.target}.ini "
          f"with num_epoch = {args.epochs}")
    config_path, save_dir, src = resolve_config(
        args.target, args.epochs, workspace, training_root, args.device)
    print(f"  from {src}\n  -> {config_path}\n  save_dir {save_dir}")
    # A pre-flight-only run never creates or reuses save_dir, so an earlier
    # smoke output must not prevent us from validating the production cache.
    if not args.skip_train and os.path.exists(save_dir):
        raise SystemExit(f"{save_dir} exists; remove it before re-running "
                         "(kernel.train() refuses to reuse a save_dir)")

    print("[2] pre-flight: loader membership against the frozen splits")
    config = preflight(config_path, workspace)
    if args.skip_train:
        print("\nPRE-FLIGHT PASSED (--skip-train; no training run)")
        return 0

    print(f"[3] running {args.epochs} real epochs")
    log_path = os.path.join(training_root,
                            f"smoke_prod_{args.target}.log")
    sampler = GpuSampler()
    if config.device.type == "cuda":
        sampler.start()
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "maceh", "train", config_path],
            cwd=paths.REPO_ROOT, stdout=log,
            stderr=subprocess.STDOUT)
    sampler.stop()
    dt = time.time() - t0
    print(f"  finished in {dt:.0f}s ({dt / max(args.epochs, 1):.0f}s/epoch), "
          f"exit {proc.returncode}, log {log_path}")
    if config.device.type == "cuda":
        print(f"  GPU: {sampler.report()}")
        assert sampler.samples and max(u for u, _ in sampler.samples) > 0, \
            "GPU utilisation never rose above 0% -- training did not use the GPU"
    if proc.returncode != 0:
        raise SystemExit(f"training exited {proc.returncode}; see {log_path}")

    run_dir = find_run_dir(save_dir)
    result_path = os.path.join(run_dir, "result.txt")
    losses = parse_losses(result_path) if os.path.isfile(result_path) \
        else parse_losses(log_path)
    finite = [x for x in losses if x == x and abs(x) != float("inf")]
    print(f"  parsed {len(losses)} loss values, {len(finite)} finite")
    assert losses, "no loss values found in the training log"
    assert len(finite) == len(losses), "some losses were NaN/Inf"

    print("[4] post-flight: checkpoint reload")
    postflight(run_dir, config.device)

    print(f"\nPRODUCTION SMOKE PASSED ({args.target}, {args.epochs} epochs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
