"""One-batch training smoke test: does the whole pipeline run on this box?

No workarounds -- `AijData` is loaded through the normal kernel path, the model
is built by the same `save_script`/`load_model` round trip `kernel.train()`
uses, and one real forward/backward/optimizer step runs on the configured
device.  Steps mirror `DeepHE3Kernel.train()` up to the training loop.

What it proves, in order:

    [1] graphs build/load through AijData
    [2] the loaders populate
    [3] the model constructs and reports its parameter count
    [4] one forward/backward/step produces a finite loss and finite gradients,
        with the tensors actually resident on the requested device
    [5] a checkpoint round-trips bit-identically

It reads the 2x2x2 pilot set by default -- outside the main experiment, so it
cannot touch the frozen splits -- and throws its model away.  Run it before
the production-sized smoke.

Usage
-----
    python training/smoke_onebatch.py [--device cuda] [--config FILE]
                                      [--workspace DIR] [--training-root DIR]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import paths

DEFAULT_CONFIG = os.path.join(paths.TRAINING_DIR, "smoke_train.ini")


def report_device(device, note=""):
    print(f"[0] device = {device} {note}")
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        print(f"[0] {props.name}, {props.total_memory / 2**30:.1f} GiB, "
              f"capability sm_{props.major}{props.minor}, "
              f"torch {torch.__version__} (cuda {torch.version.cuda})")


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"training .ini (default: {DEFAULT_CONFIG})")
    parser.add_argument("--device", default=None,
                        help="override the config's [basic] device")
    args = parser.parse_args()

    # materialise the config with resolved paths, so the .ini can stay generic
    cp = paths.read_config(args.config)
    cp["DEFAULT"]["workspace"] = paths.resolve("workspace", args.workspace)
    cp["DEFAULT"]["training_root"] = paths.resolve("training_root",
                                                   args.training_root)
    if args.device:
        cp["basic"]["device"] = args.device
    resolved = os.path.join(
        paths.resolve("training_root", args.training_root),
        "smoke_onebatch.resolved.ini")
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w") as f:
        cp.write(f)

    from maceh import DeepHE3Kernel
    from maceh.utils import MaskMSELoss

    kernel = DeepHE3Kernel()
    kernel.load_config(train_config_path=resolved)
    config = kernel.train_config
    torch.set_default_dtype(config.torch_dtype)

    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "config asks for cuda but torch.cuda.is_available() is False. "
            f"This is torch {torch.__version__}; a +cpu build cannot use the "
            "GPU. Install a CUDA build, or pass --device cpu.")
    report_device(config.device)

    print("[1] building/loading graphs through AijData")
    dataset = kernel.get_graph(config)
    print(f"[1] dataset size = {len(dataset)}")
    assert len(dataset) > 0, "no structures found; check processed_data_dir"

    kernel.config_set_target()
    dataset.set_mask(config.target_blocks,
                     convert_to_net=config.convert_net_out)

    print("[2] data loaders")
    train_loader, val_loader, extra_val_loader, test_loader = \
        kernel.get_loader()
    print(f"[2] train batches = {len(train_loader)}, "
          f"val batches = {len(val_loader)}")
    assert len(train_loader) > 0, "empty training loader"

    kernel.register_constructor(device=config.device)
    # save_script() copies the maceh source into save_dir/src as maceh_1, which
    # the generated build_model.py imports -- load_model() fails without it.
    kernel.save_script()
    src = os.path.join(config.save_dir, "src")
    kernel.save_model(src)
    net = kernel.load_model(src, device=config.device)
    n_params = sum(int(np.prod(p.size()))
                   for p in net.parameters() if p.requires_grad)
    print(f"[3] model built: {n_params} trainable parameters")
    on_device = {p.device.type for p in net.parameters()}
    assert on_device == {config.device.type}, \
        f"model parameters live on {on_device}, expected {config.device.type}"

    print("[4] one forward / backward / optimizer step")
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = MaskMSELoss()          # same criterion kernel.train() uses
    net.train()
    batch = next(iter(train_loader))
    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)
    t0 = time.time()
    # mirrors kernel.py:691-702
    output, output_edge = net(batch.to(device=config.device))
    H_pred = output_edge if config.convert_net_out \
        else kernel.construct_kernel.get_H(output_edge)
    assert H_pred.device.type == config.device.type, \
        f"predictions came back on {H_pred.device}, not {config.device}"
    loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients at all"
    assert all(torch.isfinite(g).all() for g in grads), \
        "some gradients are NaN/Inf"
    gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9)
    assert torch.isfinite(gnorm), f"grad norm is not finite: {gnorm}"
    assert float(gnorm) > 0.0, "gradients are all zero; nothing would train"
    opt.step()
    dt = time.time() - t0
    print(f"[4] H_pred {tuple(H_pred.shape)}  label {tuple(batch.label.shape)}"
          f"  masked entries {int(batch.mask.sum())}")
    print(f"[4] loss {loss.item():.6e}  grad-norm {float(gnorm):.6e}  "
          f"({dt:.1f}s), {len(grads)} gradient tensors all finite")
    if config.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(config.device) / 2**30
        print(f"[4] peak CUDA memory {peak:.2f} GiB -- nonzero confirms the "
              "step really ran on the GPU")
        assert peak > 0, "no CUDA memory was allocated; the step did not use " \
                         "the GPU"

    print("[5] checkpoint save + reload")
    ckpt = os.path.join(config.save_dir, "smoke_ckpt.pkl")
    torch.save({"state_dict": net.state_dict(), "epoch": 0}, ckpt)
    reloaded = torch.load(ckpt, weights_only=False)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(reloaded["state_dict"])
    after = net.state_dict()
    assert set(before) == set(after), "state_dict keys changed across reload"
    # some state_dict entries are zero-element buffers; .max() has no reduction
    # dim on those, so compare only the non-empty floating tensors.
    cmp = [k for k in before
           if before[k].is_floating_point() and before[k].numel() > 0]
    worst = max(float((before[k] - after[k]).abs().max()) for k in cmp)
    print(f"[5] {len(after)} tensors round-tripped "
          f"({len(cmp)} non-empty float compared), worst delta {worst:.3e}")
    assert worst == 0.0, "checkpoint round-trip changed weights"

    print("\nONE-BATCH TRAINING SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
