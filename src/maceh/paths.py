"""Canonical resolution of the project's two external storage roots."""

import os
from pathlib import Path

DATA_ENV = "MACEH_DATA_ROOT"
RUNS_ENV = "MACEH_RUNS_ROOT"


def _resolve(env_name, explicit=None):
    value = explicit if explicit is not None else os.environ.get(env_name)
    if not value:
        raise RuntimeError(
            f"{env_name} is not set. Point it at the external "
            f"{'DFT/DFPT dataset' if env_name == DATA_ENV else 'cache/checkpoint/run'} "
            "tree, or pass an explicit path to the calling command. See docs/DATA.md.")
    return Path(value).expanduser().resolve()


def data_root(explicit=None):
    """Reference DFT/DFPT data, snapshots, and loader splits."""
    return _resolve(DATA_ENV, explicit)


def runs_root(explicit=None):
    """Graph caches, checkpoints, logs, and evaluation outputs."""
    return _resolve(RUNS_ENV, explicit)


__all__ = ["DATA_ENV", "RUNS_ENV", "data_root", "runs_root"]
