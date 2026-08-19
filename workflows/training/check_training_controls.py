"""Fast standalone checks for production training-control settings.

This deliberately uses plain assertions rather than pytest so it runs in the
minimal CUDA environment. It does not load graph data or build MACE-H.
"""
import math
import os
import sys
import tempfile

import torch


from maceh import DeepHE3Kernel
from maceh.parse_configs import TrainConfig
from maceh.utils import LossThresholdStopper, RevertDecayLR
from workflows.training import paths


EXPECTED_PRODUCTION = {
    "float32_matmul_precision": "high",
    "batch_size": 2,
    "checkpoint_interval": 10,
    "early_stop_val_loss": 1e-6,
    "early_stop_min_epochs": 200,
    "early_stop_patience": 10,
}


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_config(cp, path):
    with open(path, "w") as handle:
        cp.write(handle)


def check_production_configs(tmp):
    print("1. production config controls")
    old_precision = torch.get_float32_matmul_precision()
    try:
        for source in (paths.SR_CONFIG, paths.FULL_CONFIG):
            cp = paths.read_config(source)
            cp["DEFAULT"]["training_root"] = tmp
            cp["basic"]["save_dir"] = os.path.join(
                tmp, os.path.basename(source) + "_out")
            config_path = os.path.join(tmp, os.path.basename(source))
            write_config(cp, config_path)

            kernel = DeepHE3Kernel()
            kernel.load_config(train_config_path=config_path)
            config = kernel.train_config
            for name, expected in EXPECTED_PRODUCTION.items():
                got = getattr(config, name)
                assert got == expected, \
                    f"{source}: {name}={got!r}, expected {expected!r}"
            assert torch.get_float32_matmul_precision() == "high"
            print(f"  {os.path.basename(source)}: controls parsed")
    finally:
        torch.set_float32_matmul_precision(old_precision)


def check_default_compatibility(tmp):
    print("2. defaults and invalid-value rejection")
    cp = paths.read_config(os.path.join(paths.TRAINING_DIR,
                                        "smoke_train.ini"))
    cp["DEFAULT"]["training_root"] = tmp
    cp["basic"]["save_dir"] = os.path.join(tmp, "legacy_out")
    legacy_path = os.path.join(tmp, "legacy.ini")
    write_config(cp, legacy_path)
    legacy = TrainConfig(legacy_path)
    assert legacy.float32_matmul_precision == "highest"
    assert legacy.checkpoint_interval == 1
    assert legacy.early_stop_val_loss == -1
    assert legacy.early_stop_min_epochs == 0
    assert legacy.early_stop_patience == 1

    invalid = (
        ("basic", "float32_matmul_precision", "fastest"),
        ("train", "checkpoint_interval", "0"),
        ("train", "early_stop_val_loss", "nan"),
        ("train", "early_stop_min_epochs", "-1"),
        ("train", "early_stop_patience", "0"),
    )
    for index, (section, key, value) in enumerate(invalid):
        bad = paths.read_config(paths.SR_CONFIG)
        bad["DEFAULT"]["training_root"] = tmp
        bad["basic"]["save_dir"] = os.path.join(tmp, f"bad_{index}")
        bad[section][key] = value
        bad_path = os.path.join(tmp, f"bad_{index}.ini")
        write_config(bad, bad_path)
        try:
            TrainConfig(bad_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{section}.{key}={value!r} was accepted")
    print("  legacy defaults and validation: OK")


def check_loss_threshold():
    print("3. validation-loss threshold semantics")
    stopper = LossThresholdStopper(1e-6, min_epochs=5, patience=3)
    assert not stopper.step(0, 1e-6, 1)  # inclusive threshold
    assert stopper.consecutive == 1
    assert not stopper.step(1, math.nan, 1)
    assert stopper.consecutive == 0
    assert not stopper.step(2, 0.0, 0)  # empty validation is ineligible
    assert not stopper.step(3, 9e-7, 1)
    assert not stopper.step(4, 1e-6, 1)
    assert stopper.step(5, 8e-7, 1)

    for invalid in (math.inf, -math.inf, 2e-6):
        assert not stopper.step(6, invalid, 1)
        assert stopper.consecutive == 0

    resumed = LossThresholdStopper(1e-6, min_epochs=0, patience=3)
    resumed.load_state_dict({"consecutive": 2})
    assert resumed.step(10, 1e-6, 1)
    changed = LossThresholdStopper(5e-7, min_epochs=0, patience=3)
    changed.load_state_dict(resumed.state_dict())
    assert changed.consecutive == 0
    disabled = LossThresholdStopper(-1)
    assert not disabled.step(10, 0.0, 1)
    print("  finite, non-empty, consecutive and resume controls: OK")


def check_checkpoint_interval(tmp):
    print("4. best/latest checkpoint cadence")
    checkpoint_dir = os.path.join(tmp, "checkpoints")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RevertDecayLR(
        model, optimizer, checkpoint_dir, decay_patience=99,
        scheduler_type=0, checkpoint_interval=3)

    def set_weight(value):
        with torch.no_grad():
            model.weight.fill_(value)

    set_weight(1.0)
    scheduler.step(1.0, early_stop_state={"consecutive": 0}, global_step=0)
    latest = os.path.join(checkpoint_dir, "model.pkl")
    best = os.path.join(checkpoint_dir, "best_model.pkl")
    assert load_checkpoint(latest)["epoch"] == 0
    assert load_checkpoint(latest)["global_step"] == 0
    assert load_checkpoint(best)["epoch"] == 0

    set_weight(2.0)
    scheduler.step(2.0)
    assert load_checkpoint(latest)["epoch"] == 0
    assert load_checkpoint(best)["epoch"] == 0

    set_weight(3.0)
    scheduler.step(3.0)
    assert load_checkpoint(latest)["epoch"] == 2
    assert load_checkpoint(best)["epoch"] == 0
    stale_latest = load_checkpoint(latest)

    set_weight(0.5)
    scheduler.step(0.5, early_stop_state={"consecutive": 1})
    assert load_checkpoint(latest)["epoch"] == 2
    assert load_checkpoint(best)["epoch"] == 3
    assert load_checkpoint(best)["early_stop_state"]["consecutive"] == 1

    scheduler.save_latest(3, 0.5,
                          early_stop_state={"consecutive": 1},
                          global_step=3)
    assert load_checkpoint(latest)["epoch"] == 3
    assert load_checkpoint(latest)["global_step"] == 3
    assert not os.path.exists(latest + ".tmp")
    assert not os.path.exists(best + ".tmp")

    state = scheduler.state_dict()
    assert "checkpoint_interval" not in state
    replacement = RevertDecayLR(
        model, optimizer, os.path.join(tmp, "replacement"),
        scheduler_type=0, checkpoint_interval=7)
    replacement.load_state_dict(state)
    assert replacement.checkpoint_interval == 7

    stale_resume = RevertDecayLR(
        model, optimizer, os.path.join(tmp, "stale_resume"),
        scheduler_type=0, checkpoint_interval=10)
    stale_resume.load_state_dict(stale_latest["scheduler_state_dict"])
    assert stale_resume.best_loss == 1.0
    current_best = load_checkpoint(best)
    stale_resume.reconcile_best(current_best["epoch"],
                                current_best["val_loss"])
    assert stale_resume.best_loss == 0.5
    assert stale_resume.best_epoch == 3

    rewind_dir = os.path.join(tmp, "no_rewind")
    rewind_model = torch.nn.Linear(1, 1, bias=False)
    rewind_optimizer = torch.optim.SGD(rewind_model.parameters(), lr=0.1)
    rewind = RevertDecayLR(
        rewind_model, rewind_optimizer, rewind_dir, decay_patience=1,
        decay_rate=0.8, scheduler_type=0, checkpoint_interval=10)
    rewind.step(1.0)
    rewind.step(3.0)  # triggers revert to epoch 0
    assert rewind.next_epoch == 2, "revert rewound the physical epoch counter"
    print("  immediate best, periodic latest, atomic files and resume: OK")


def main():
    with tempfile.TemporaryDirectory(prefix="maceh_controls_") as tmp:
        check_production_configs(tmp)
        check_default_compatibility(tmp)
        check_loss_threshold()
        check_checkpoint_interval(tmp)
    print("\nTRAINING CONTROLS OK")


if __name__ == "__main__":
    main()
