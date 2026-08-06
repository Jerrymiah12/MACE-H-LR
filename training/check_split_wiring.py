"""Prove that train_sr.ini / train_full.ini reproduce the frozen splits.

Three checks, cheapest first, so the ones that matter most for a handoff run
without a GPU, without MACE-H's dependencies and without the 150 GB workspace:

1. **Frozen validation set.**  Both production configs' `extra_validation`
   must equal the 37 validation ids of `provenance/splits.json`, exactly --
   same length, no duplicates, no strays, and disjoint from train and test.
   If a workspace is given, its `splits.json` must agree with the frozen copy
   too.  This is the assertion that stops a stale or hand-edited id list from
   quietly redefining the model-selection set.

2. **Paired baselines.**  The two configs must be identical outside the four
   keys that name where a run writes.  Same architecture, seed, optimizer,
   schedule and stopping rule is the whole premise of the comparison, so it
   is checked mechanically rather than by eye.

3. **Loader membership** (needs MACE-H + a workspace).  `get_loader()` builds
   its train/val/test index sets from ratios *plus* `extra_validation`, and
   the interaction is easy to get wrong.  This runs the real config parsing
   and the real `get_loader()` on a small stand-in view and asserts the
   *membership* of each loader, not just its size:

       train loader == the view's train snapshots, exactly
       val   loader == the view's validation snapshots, exactly
       test  loader touches no training snapshot

   The stand-in view is built the way `make_trainval_view.py` builds the real
   one, so the wiring proved here is the wiring the production runs use.

Usage
-----
    python training/check_split_wiring.py [--workspace DIR] [--training-root DIR]
    python training/check_split_wiring.py --configs-only   # checks 1-2 only
"""
import argparse
import ast
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import paths

N_TRAIN, N_VAL = 6, 3

#: the only keys allowed to differ between the paired configs -- they name
#: where a run writes, nothing that could change what it learns.
PAIRED_EXEMPT = {
    ("basic", "save_dir"),
    ("basic", "additional_folder_name"),
    ("data", "save_graph_dir"),
    ("data", "dataset_name"),
}


def check_frozen_validation(workspace):
    """Check 1: both configs name exactly the frozen 37 validation ids."""
    splits = paths.load_frozen_splits(workspace, "main")
    train = sorted(splits["train"])
    val = sorted(splits["validation"])
    test = sorted(splits["test"])
    assert len(val) == 37, f"frozen validation set is {len(val)} ids, want 37"
    assert len(train) == 330, f"frozen train set is {len(train)} ids, want 330"
    assert len(test) == 37, f"frozen test set is {len(test)} ids, want 37"

    for path in (paths.SR_CONFIG, paths.FULL_CONFIG):
        name = os.path.basename(path)
        raw = paths.read_config(path).get("train", "extra_validation")
        got = ast.literal_eval(raw)
        assert isinstance(got, list) and all(isinstance(s, str) for s in got), \
            f"{name}: extra_validation is not a list of strings"
        assert len(got) == len(set(got)), \
            f"{name}: extra_validation has duplicate ids"
        missing = sorted(set(val) - set(got))
        extra = sorted(set(got) - set(val))
        assert not missing and not extra, (
            f"{name}: extra_validation is not the frozen validation set "
            f"(missing {missing}, unexpected {extra})")
        assert not set(got) & set(train), \
            f"{name}: extra_validation overlaps the training split"
        assert not set(got) & set(test), \
            f"{name}: extra_validation overlaps the held-out test split"
        print(f"  {name}: extra_validation == the frozen 37 validation ids")
    return val


def check_paired_configs():
    """Check 2: the baselines differ only in where they write."""
    sr = paths.read_config(paths.SR_CONFIG)
    full = paths.read_config(paths.FULL_CONFIG)
    assert sr.defaults() == full.defaults(), \
        "the configs' [DEFAULT] path blocks differ; they would read different data"
    assert set(sr.sections()) == set(full.sections()), \
        f"section mismatch: {set(sr.sections()) ^ set(full.sections())}"
    diffs = []
    for section in sr.sections():
        keys = set(sr[section]) | set(full[section])
        for key in sorted(keys):
            a = sr[section].get(key)
            b = full[section].get(key)
            if a != b and (section, key) not in PAIRED_EXEMPT:
                diffs.append(f"[{section}] {key}: {a!r} vs {b!r}")
    assert not diffs, ("the paired runs differ in more than their output "
                       "paths:\n    " + "\n    ".join(diffs))
    print(f"  train_sr.ini / train_full.ini differ only in "
          f"{len(PAIRED_EXEMPT)} output-path keys")


def build_view(root, workspace, train, val):
    for subset, sids in (("train", train), ("validation", val)):
        for sid in sids:
            src = os.path.join(workspace, "loader_splits", subset, sid)
            dst = os.path.join(root, sid)
            os.makedirs(dst)
            for name in sorted(os.listdir(src)):
                target = os.path.realpath(os.path.join(src, name))
                os.symlink(target, os.path.join(dst, name))


def ids_of(loader, dataset):
    return sorted(dataset.data.stru_id[i] for i in loader.sampler.indices)


def check_loader_membership(workspace):
    """Check 3: the real get_loader() puts the right snapshots in each loader."""
    from maceh import DeepHE3Kernel

    splits = paths.load_frozen_splits(workspace, "main")
    train = sorted(splits["train"])[:N_TRAIN]
    val = sorted(splits["validation"])[:N_VAL]

    tmp = tempfile.mkdtemp(prefix="split_wiring_")
    try:
        view = os.path.join(tmp, "view")
        os.makedirs(view)
        build_view(view, workspace, train, val)

        # the production config, with only the paths and sizes made small
        cp = paths.read_config(paths.SR_CONFIG)
        cp["DEFAULT"]["training_root"] = tmp
        cp["DEFAULT"]["workspace"] = workspace
        cp["basic"]["save_dir"] = os.path.join(tmp, "out")
        cp["basic"]["device"] = "cpu"
        cp["data"]["processed_data_dir"] = view
        cp["data"]["save_graph_dir"] = os.path.join(tmp, "graphs")
        cp["data"]["dataset_name"] = "splitwiring"
        cp["train"]["extra_validation"] = repr(val)
        cp["train"]["num_epoch"] = "1"
        cp["network"]["irreps_embed"] = "8x0e"
        cp["network"]["irreps_mid"] = "8x0e+4x1o+2x2e"
        cp["network"]["num_blocks"] = "1"
        ini = os.path.join(tmp, "check.ini")
        with open(ini, "w") as f:
            cp.write(f)

        kernel = DeepHE3Kernel()
        kernel.load_config(train_config_path=ini)
        config = kernel.train_config
        assert config.train_ratio == 1.0 and config.val_ratio == 0.0, \
            "ratios were not read as expected"
        assert config.extra_val_test_only is False, \
            "extra_val_test_only must be False or the val loader stays empty"

        dataset = kernel.get_graph(config)
        kernel.config_set_target()
        dataset.set_mask(config.target_blocks,
                         convert_to_net=config.convert_net_out)
        train_loader, val_loader, extra_val_loader, test_loader = \
            kernel.get_loader()

        got_train = ids_of(train_loader, dataset)
        got_val = ids_of(val_loader, dataset)
        got_test = ids_of(test_loader, dataset)
        print(f"  train  {len(got_train)}: {got_train}")
        print(f"  val    {len(got_val)}: {got_val}")
        print(f"  test   {len(got_test)}: {got_test}")
        assert got_train == train, f"train set is {got_train}, want {train}"
        assert got_val == val, f"val set is {got_val}, want {val}"
        assert ids_of(extra_val_loader, dataset) == val
        # `get_loader()` appends extra_val to the *test* indices too, so with
        # ratios of zero MACE-H's "test" loader is the validation set.  That is
        # harmless during training, but it means the "test" numbers in a
        # training log are NOT held-out performance -- the 37 held-out test
        # snapshots are not in this dataset at all.  Pin both halves of that.
        assert got_test == val, \
            (f"test loader is {got_test}; expected it to be the validation set "
             "(get_loader appends extra_val to test_indices)")
        assert not set(got_test) & set(train), \
            "test loader overlaps the training set"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--configs-only", action="store_true",
                        help="skip the get_loader() check, which needs MACE-H "
                             "and a populated workspace")
    args = parser.parse_args()
    training_root = paths.resolve("training_root", args.training_root)
    workspace = paths.resolve("workspace", args.workspace)
    del training_root  # resolved for the error message it raises, not used here

    have_workspace = os.path.isdir(
        os.path.join(workspace, "loader_splits", "train"))

    print("1. frozen validation set")
    check_frozen_validation(workspace if have_workspace else None)
    print("2. paired baseline configs")
    check_paired_configs()

    print("3. loader membership")
    if args.configs_only:
        print("  skipped (--configs-only)")
    elif not have_workspace:
        print(f"  SKIPPED: {workspace} has no loader_splits/train. "
              "Pass --workspace to run this check.")
    else:
        check_loader_membership(workspace)

    print("\nSPLIT WIRING OK -- extra_validation is the frozen 37, "
          "the baselines are paired, ratios do not re-split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
