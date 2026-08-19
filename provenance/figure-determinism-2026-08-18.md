# Figure determinism change and manifest supersession — 2026-08-18

## What was wrong

Matplotlib stamped wall-clock time into every PDF it wrote (`/CreationDate`).
Two runs of the same command therefore produced different bytes, which meant the
SHA-256 recorded in every `*.pdf.manifest.yaml` under `results/` could never be
verified — including immediately after it was written. The PNG artifacts were
byte-stable on a fixed matplotlib version but carried a `Software` key holding
that version, so they were not stable across an upgrade.

This was found while auditing the repository restructure, by regenerating a
committed figure and comparing it against its own manifest.

## What changed

`src/maceh/analysis/figures.py` is now the single chokepoint for figure output.
It omits the metadata keys that carry a timestamp or a toolchain version rather
than freezing them to a constant:

| Format | Keys removed | Notes |
|---|---|---|
| PDF | `CreationDate`, `Producer` | `Producer` held the matplotlib version |
| PS / EPS | `CreationDate` | |
| SVG | `Date` | also needs a fixed `svg.hashsalt`, which the module sets |
| PNG | `Software` | held the matplotlib version |

Every `savefig` call site in `src/maceh/` and `workflows/` was routed through it,
and the near-identical `save`/`save_figure` helpers that each figure script
carried were replaced by `save_figure_formats`. `SOURCE_DATE_EPOCH` is set as a
fallback for any figure that escapes the chokepoint in future.

Verified: two consecutive runs of `python -m workflows.analysis.reproduce_summary`
now produce byte-identical PNG **and** PDF output. Before the change the PDF
differed on every run.

## Consequence: all manifests are superseded

Removing the metadata keys changes the bytes of **both** PNG and PDF artifacts.
Every SHA-256 currently recorded under `results/` therefore describes a
pre-change artifact and will not match a regeneration.

**All artifacts and manifests must be regenerated together, in one run, on a
machine with `MACEH_DATA_ROOT` and `MACEH_RUNS_ROOT` available.** Do not
reconcile them individually.

## Disclosure: one hand-edited checksum

During the audit, `results/learned_response/reproduced_summary.pdf` was
regenerated and its original bytes were not recoverable (the tree was
uncommitted, so Git held no copy). Its manifest checksum was updated by hand at
that point. Its PNG regenerated bit-identically to its manifest, which confirmed
the plot content was unchanged and that only the embedded timestamp differed.

That hand-edit is superseded by the pending full regeneration and should not
survive into the tagged release. `results/README.md` states the scope these
manifests actually claim.
