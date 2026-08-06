"""Path resolution shared by the training helper scripts.

Two locations are configurable, and they are the same two keys the `[DEFAULT]`
section of `train_sr.ini` / `train_full.ini` interpolates into the rest of the
config:

    workspace       the `mgo_lr` run workspace -- the directory holding
                    `splits.json`, `metadata.yaml`, `main/`, `loader_splits/`.
    training_root   where the training artefacts go: the `data_trainval`
                    loader view, the graph caches, the run output dirs.

Resolution order, highest first:

    1. an explicit --workspace / --training-root command-line argument
    2. $MGO_LR_WORKSPACE / $MGO_LR_TRAINING_ROOT
    3. the `[DEFAULT]` value in `training/train_sr.ini`

Step 3 keeps the scripts and the .ini files from drifting apart: edit the
config's `[DEFAULT]` block and the helpers follow.  Relative values resolve
against the working directory (the repo root, by convention), matching how
`ConfigParser` hands them to MACE-H.
"""
import argparse
import configparser
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
PROVENANCE_DIR = os.path.join(REPO_ROOT, "provenance")
SR_CONFIG = os.path.join(TRAINING_DIR, "train_sr.ini")
FULL_CONFIG = os.path.join(TRAINING_DIR, "train_full.ini")

#: name of the marker file `make_trainval_view.py` drops in a view it owns
VIEW_MARKER = ".mgo_lr_trainval_view"


def read_config(path):
    """Parse a training .ini the same way MACE-H's `BaseConfig` does."""
    cp = configparser.ConfigParser(inline_comment_prefixes=(';',))
    if not cp.read(path):
        raise SystemExit(f"{path}: not readable")
    return cp


def _config_default(key, config_path=SR_CONFIG):
    return read_config(config_path).defaults().get(key, "")


def resolve(name, cli_value=None, config_path=SR_CONFIG):
    """Resolve `workspace` or `training_root` by the documented precedence."""
    assert name in ("workspace", "training_root"), name
    value = cli_value or os.environ.get(f"MGO_LR_{name.upper()}") \
        or _config_default(name, config_path)
    if not value:
        raise SystemExit(
            f"cannot resolve {name}: pass --{name.replace('_', '-')}, set "
            f"$MGO_LR_{name.upper()}, or give it a [DEFAULT] value in "
            f"{config_path}")
    return os.path.abspath(os.path.expanduser(value))


def add_path_args(parser):
    """Add the standard --workspace / --training-root pair to a parser."""
    parser.add_argument("--workspace", default=None,
                        help="mgo_lr run workspace (default: $MGO_LR_WORKSPACE,"
                             " else [DEFAULT] workspace in train_sr.ini)")
    parser.add_argument("--training-root", default=None,
                        help="training artefact root (default: "
                             "$MGO_LR_TRAINING_ROOT, else [DEFAULT] "
                             "training_root in train_sr.ini)")
    return parser


def resolved_args(parser=None, **kwargs):
    """Parse argv and return `args` with `workspace`/`training_root` absolute."""
    parser = parser or add_path_args(argparse.ArgumentParser(**kwargs))
    args = parser.parse_args()
    args.workspace = resolve("workspace", args.workspace)
    args.training_root = resolve("training_root", args.training_root)
    return args


def trainval_view(training_root):
    """The dataset directory both production configs read."""
    return os.path.join(training_root, "data_trainval")


def load_splits(workspace, subset="main"):
    """Return the frozen split lists from the workspace `splits.json`."""
    import json
    path = os.path.join(workspace, "splits.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: no splits.json -- is {workspace} an "
                         "mgo_lr workspace? (run `mgo_lr organize` first)")
    with open(path) as f:
        return json.load(f)[subset]


def load_frozen_splits(workspace=None, subset="main"):
    """Splits from `provenance/splits.json`, cross-checked against a workspace.

    The repo carries a frozen copy so the split assertions run on a machine
    that has the code but not the 150 GB workspace.  When a workspace *is*
    given and has its own `splits.json`, the two must agree exactly -- a
    mismatch means the workspace was re-`organize`d away from the frozen
    dataset this repo describes, which would silently change what "the 37
    validation snapshots" means.
    """
    import json
    frozen_path = os.path.join(PROVENANCE_DIR, "splits.json")
    if not os.path.isfile(frozen_path):
        raise SystemExit(f"{frozen_path}: missing frozen splits")
    with open(frozen_path) as f:
        frozen = json.load(f)[subset]
    if workspace and os.path.isfile(os.path.join(workspace, "splits.json")):
        live = load_splits(workspace, subset)
        for key in sorted(set(frozen) | set(live)):
            if sorted(frozen.get(key, [])) != sorted(live.get(key, [])):
                raise SystemExit(
                    f"{subset}/{key}: workspace splits.json disagrees with "
                    f"{frozen_path} ({len(live.get(key, []))} vs "
                    f"{len(frozen.get(key, []))} ids). The workspace is not "
                    "the frozen dataset; refusing to guess which is right.")
    return frozen
