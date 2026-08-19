# Frozen dataset provenance

The dataset run workspace is ~150 GB and lives outside this repository. What
is kept here is the small, archival part: enough to tie any checkpoint back to
exactly the data it saw, long after the workspace is gone.

Written and checked by `workflows/training/freeze_provenance.py`:

    python -m workflows.training.freeze_provenance --workspace DIR   # rewrite this directory
    python -m workflows.training.freeze_provenance --verify          # re-hash against SHA256SUMS

`--verify` needs no workspace and no GPU. Run it on the training box before
building graph caches, and again before reporting results.

## Files

| file | what it fixes |
|---|---|
| `splits.json` | the frozen train / validation / test membership, by snapshot id |
| `config.resolved.yaml` | the fully resolved `mgo_lr` configuration of the `organize` run that produced those splits |
| `metadata.yaml` | LR definition (Ewald λ, cutoffs, gauge and sign conventions), DFT settings, supercell matrices, and sha256 of the DFPT reference artefacts |
| `validation_main.json`, `validation_pilot.json`, `validation_large.json` | Tier-1/Tier-2 verdicts and per-snapshot metrics |
| `locality_main.json`, `locality_pilot.json`, `locality_large.json` | Tier-3 far-field reports |
| `SHA256SUMS` | checksum manifest over all of the above |

The same four `organize` / `validate` / `locality-report` / `export-target`
config snapshots were byte-identical in the workspace, so one
`config.resolved.yaml` covers all of them.

## Final counts

| set | cell | count | rejected |
|---|---|---|---|
| main — train | 3×3×3 | **330** | — |
| main — validation | 3×3×3 | **37** | — |
| main — test | 3×3×3 | **37** | — |
| main — total validated | 3×3×3 | **404** | **0** |
| large_test | 4×4×4 | **44** | **0** |
| pilot | 2×2×2 | 20 | **0** |

`freeze_provenance.py` asserts every one of these numbers, plus that the three
main splits are pairwise disjoint and sum to 404, before it will write this
directory. A re-`organize` that moves any of them fails the freeze loudly
rather than publishing a different dataset under the same name.

Splits are from seed `20260720` at a 0.1 / 0.1 validation / test fraction,
grouped by generation-time split-specific q shells and pattern groups so that
related structures cannot straddle the boundary.

## Never train on these

- the **37 main-set test snapshots** (`splits.json` → `main.test`)
- the **44 large-cell 4×4×4 snapshots** (`splits.json` → `large_test`)

The 37 are the held-out test set for the 3×3×3 comparison. The 44 are the
cell-size extrapolation benchmark, and they are the *only* set the Tier-3
far-field gate actually approves — see "Which set the far-field gate actually
approves" in `workflows/mgo_dataset/README.md`. Folding either into training destroys the
claim the experiment exists to make.

Neither set appears in the `data_trainval` view the training configs read, and
`workflows/training/check_split_wiring.py` asserts the validation ids are disjoint from
both. Note also that MACE-H's *"test"* loader during training holds the
validation snapshots, not these — see `workflows/training/README.md`.

## Change notes

Events that alter what a recorded number or checksum means, newest first:

- [figure-determinism-2026-08-18.md](figure-determinism-2026-08-18.md)
  — figure output is now byte-reproducible; every checksum under `results/` is
  superseded and must be regenerated in one run.
