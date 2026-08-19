# Data and run storage

The project uses two external roots with different lifecycles.

| Variable | Contents | Approximate size | Regeneration cost |
|---|---|---:|---|
| `MACEH_DATA_ROOT` | reference DFT/DFPT, snapshots, loader splits | 134 GB | expensive; some parts represent weeks of DFT |
| `MACEH_RUNS_ROOT` | graph caches, checkpoints, logs, evaluations | 65 GB | generated, but training is costly |

`maceh.paths` is the sole environment-variable resolver and raises an
actionable error when a root is unset. CLI arguments may supply an explicit
path. The ignored repository paths `data/` and `runs/` are convenience
pointers only and may be symlinked to the two roots.

## Existing dataset vocabulary

The data root retains the established layout: `reference/`, `pilot/`, `main/`,
`test_large_cell/`, `loader_splits/`, and `rejected/`. Do not rename these into
a generic raw/interim/processed hierarchy: configs, provenance checks, and
long-running jobs use their scientific meanings directly.

- `reference/`: relaxed cell, Born charges, dielectric tensor, DFT settings.
- `pilot/`: approval and calibration structures; relatively cheap to rebuild.
- `main/`: the principal 400-snapshot campaign; expensive DFT output.
- `test_large_cell/`: extrapolation structures; expensive DFT output.
- `loader_splits/`: organized views; cheap to recreate after validation.
- `rejected/`: failed structures retained for audit, not training.

Within the runs root, graph caches can be rebuilt from validated/exported data.
Checkpoints and frozen evaluations should be retained with their manifests;
retraining is not numerically identical unless the full environment and random
state are also preserved.

Tiny deterministic fixtures live beside the tests that read them under
`tests/` — for example `tests/smoke/golden.json`. Never commit a checkpoint, a
production HDF5 tensor, or a symlink whose target contains a machine- or
user-specific path.
