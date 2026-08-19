"""Path resolution shared by the training helper scripts.

Two locations are configurable, and they are the same two keys the `[DEFAULT]`
section of `train_sr.ini` / `train_full.ini` interpolates into the rest of the
config:

    workspace       the `mgo_lr` run workspace -- the directory holding
                    `splits.json`, `metadata.yaml`, `main/`, `loader_splits/`.
    training_root   where the training artefacts go: the `data_trainval`
                    loader view, the graph caches, the run output dirs.

Resolution is explicit CLI value first, then ``MACEH_DATA_ROOT`` for the
dataset workspace or ``MACEH_RUNS_ROOT`` for generated training artifacts.
The environment contract itself is implemented only in :mod:`maceh.paths`.
"""
import argparse
import configparser
import json
import os

from maceh.paths import data_root, runs_root

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TRAINING_DIR = os.path.join(REPO_ROOT, "workflows", "training")
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


def resolve(name, cli_value=None):
    """Resolve a campaign path through the canonical library contract."""
    assert name in ("workspace", "training_root"), name
    resolver = data_root if name == "workspace" else runs_root
    try:
        return str(resolver(cli_value))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def add_path_args(parser):
    """Add the standard --workspace / --training-root pair to a parser."""
    parser.add_argument("--workspace", default=None,
                        help="dataset workspace (default: $MACEH_DATA_ROOT)")
    parser.add_argument("--training-root", default=None,
                        help="training artefact root (default: "
                             "$MACEH_RUNS_ROOT)")
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
    path = os.path.join(workspace, "splits.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path}: no splits.json -- is {workspace} an MgO dataset "
            "workspace? (run `python -m workflows.mgo_dataset organize` first)")
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
