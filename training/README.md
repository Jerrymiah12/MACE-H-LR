# SR-target vs full-H baseline runs

Both configs are production-sized (`64x0e` embed,
`64x0e+32x1o+16x2e+8x3o+8x4e`, 3 blocks, 1350936 trainable parameters) and
point at the frozen 404-snapshot 3x3x3 main set.

    train_sr.ini             SR-target run   (label = H_SR)
    train_full.ini           baseline run    (label = H_full)
    paths.py                 resolves the two configurable locations
    make_trainval_view.py    builds the dataset directory both configs read
    check_split_wiring.py    proves the configs reproduce the frozen splits
    freeze_provenance.py     writes / verifies `provenance/`
    setup_gpu_env.sh         builds the CUDA environment
    smoke_onebatch.py        does the pipeline run at all on this box?
    smoke_train.ini            (its config -- pilot data, tiny model)
    smoke_production.py      do a few real epochs run, on the real splits?

Run with `python deephe3-train.py training/<config>.ini` from the repo root.

## Runbook

    ./training/setup_gpu_env.sh /path/to/venv_cuda
    export MGO_LR_WORKSPACE=/path/to/run
    export MGO_LR_TRAINING_ROOT=/path/to/training_runs

    python training/freeze_provenance.py --verify        # dataset unchanged?
    python training/make_trainval_view.py                # 330 + 37
    python training/check_split_wiring.py                # splits wired right?
    python training/smoke_onebatch.py                    # pipeline runs?
    python training/smoke_production.py --target sr --epochs 2
    python deephe3-train.py training/train_sr.ini        # the real run

then switch the workspace export to `full` and repeat from
`check_split_wiring.py` for the baseline (see "The two runs cannot share a
workspace export" below).

`smoke_production.py` deliberately shares `save_graph_dir` with the production
run: the graph cache is the same data either way and building it is the
expensive step, so the smoke warms exactly the cache the real run will use.
Only `save_dir` and `num_epoch` differ. Pass `--skip-train` to do the cache
build and the membership assertions without training.

## Paths

Two locations are configurable, and everything else is derived from them:

    workspace       the `mgo_lr` run workspace: `splits.json`, `metadata.yaml`,
                    `main/`, `loader_splits/`.  ~150 GB, lives outside the repo.
    training_root   where training artefacts go: the `data_trainval` view, the
                    graph caches, the run output directories.

Set them once in the `[DEFAULT]` block at the top of **both** `.ini` files --
the rest of each config interpolates `%(training_root)s`.  The helper scripts
resolve the same two values, in this order:

    1. --workspace / --training-root on the command line
    2. $MGO_LR_WORKSPACE / $MGO_LR_TRAINING_ROOT
    3. the `[DEFAULT]` values in `train_sr.ini`

so exporting the two environment variables is enough to drive the whole
handoff without editing a file. Relative values resolve against the working
directory; use absolute paths for production runs.

## The splits must come from `organize`, not from ratios

MACE-H's `get_loader()` splits **one** dataset by ratio, while `mgo_lr
organize` publishes three *separate* loader views. Pointing a config at
`loader_splits/train` with `train_ratio = 0.67` therefore re-splits the 330
training snapshots at random and never reads the frozen 37-snapshot validation
set at all -- discarding the group-aware split (q-shell + pattern groups) that
`organize` computed precisely to keep related structures out of validation.

So both configs instead read one directory holding train+validation and name
the validation snapshots explicitly:

    processed_data_dir = %(training_root)s/data_trainval    # 330 + 37
    extra_validation   = [the 37 organize validation ids]
    extra_val_test_only = False
    train_ratio = 1.0 ; val_ratio = 0 ; test_ratio = 0

`get_loader()` removes `extra_validation` from the pool *before* applying the
ratios, so `train_ratio = 1.0` means "exactly the 330 organize train
snapshots", and the validation loader is exactly the 37. The 37 test snapshots
are never loaded during training.

Build the directory (a tree of real dirs holding file symlinks -- `os.walk`
does not follow directory symlinks) and verify the wiring:

    python training/make_trainval_view.py     # -> 330 train + 37 validation
    python training/check_split_wiring.py     # 3 checks, see below

The view's symlinks resolve to `run/main/<sid>/hamiltonians.h5`, which
`export-target` rewrites in place, so the view survives an export switch and a
re-`organize`; only the *labels* behind it change. It needs rebuilding only if
the main splits themselves change.

### What `check_split_wiring.py` asserts

1. **The frozen validation set.** Both configs' `extra_validation` must equal
   the 37 validation ids in `provenance/splits.json` exactly -- no strays, no
   duplicates, disjoint from train and test. If a workspace is given, its
   `splits.json` must agree with the frozen copy too. This is what stops a
   stale or hand-edited id list from quietly redefining model selection.
2. **The baselines are paired.** The two configs must be byte-identical
   outside four keys (`save_dir`, `additional_folder_name`, `save_graph_dir`,
   `dataset_name`). Same architecture, seed, optimizer, schedule and stopping
   rule is the premise of the comparison, so it is checked mechanically.
3. **Loader membership.** Runs the real config parser and the real
   `get_loader()` on a small stand-in view and asserts *which* snapshots land
   in each loader, not just how many.

Checks 1-2 need neither a GPU nor the workspace; run them anywhere with
`--configs-only`. Check 3 needs MACE-H's dependencies and a populated
workspace, and skips itself with a message if the workspace is absent.

### MACE-H's "test" loader is the validation set

`get_loader()` appends `extra_val_indices` to the *test* indices as well. With
`test_ratio = 0` that makes the test loader hold the same 37 validation
snapshots. So **any "test" number in a training log is validation performance,
not held-out performance** -- the 37 held-out test snapshots are not in
`data_trainval` at all. Held-out evaluation is a separate step against
`loader_splits/test`. `check_split_wiring.py` pins this so it cannot drift
into being read as a held-out score.

One other cost of this wiring: with `extra_validation` set, `train()`
evaluates `val_loader` and `extra_val_loader` each epoch, and here they hold
the same 37 structures -- about 10% of an epoch spent computing the validation
loss twice. Setting `extra_val_test_only = True` would leave `val_loader`
empty and starve the LR scheduler, so the duplicate stays.

## These need a GPU

Measured on a 32-core CPU box (torch 2.13.0+cpu), production size,
batch_size=1:

    model construction        480 s
    forward+backward+step     157.65 s / batch  (median of 4)
    330-batch epoch           867 min  = 14.5 h
    3000 epochs               43354 h  ~ 5 years

CPU is not an option -- a single epoch costs over half a day. Both configs
therefore ship with `device = cuda`; set it back to `cpu` only for the small
wiring checks.

`setup_gpu_env.sh` builds the environment. There is **no torch_scatter** in
it: MACE-H aggregates with `torch_geometric.utils.scatter`, which is pure
PyTorch, so nothing has to be compiled against the exact torch build and
`nvcc` is not needed. Verified on `torch 2.13.0+cu129` / Python 3.14 against
an RTX 5090 (sm_120, 31.8 GiB).

The graph caches (`graphs_sr/`, `graphs_full/`) are deliberately **not**
prebuilt: a 330-snapshot cache is ~8 GB, cheap to regenerate, and awkward to
move. Build them on the target box, in the export order below.

## The two runs cannot share a workspace export

`export-target` is workspace-wide and all-or-nothing: it points every
snapshot's `hamiltonians.h5` at one source and records the choice in
`run/metadata.yaml` as `training_target`. Build one graph cache at a time:

    # SR run
    python -m mgo_lr export-target --target sr   --config run.yaml --workspace $MGO_LR_WORKSPACE
    python -m mgo_lr organize                    --config run.yaml --workspace $MGO_LR_WORKSPACE
    python training/make_trainval_view.py
    python training/check_split_wiring.py
    # ... build graphs / train with train_sr.ini ...

    # full-H baseline
    python -m mgo_lr export-target --target full --config run.yaml --workspace $MGO_LR_WORKSPACE
    python -m mgo_lr organize                    --config run.yaml --workspace $MGO_LR_WORKSPACE
    python training/check_split_wiring.py
    # ... build graphs / train with train_full.ini ...

Once a run's `save_graph_dir` cache exists it is self-contained, so the two
trainings can then proceed in parallel -- but the *graph building* steps must
not overlap, or the second will cache whichever labels happen to be exported
at that moment. `dataset_name` differs between the configs (`mgo404sr` /
`mgo404full`) so the two caches cannot collide on filename, but that does not
protect against building both while the workspace is in one export state.

Sanity check before training: `run/metadata.yaml` -> `training_target` must
match the config you are about to run.

## Reconstruction and evaluation

The SR run predicts `H_SR`. To compare against the baseline in `H_full` terms,
reconstruct `H_full_pred = H_SR_pred + H_LR`, taking `H_LR` from each
snapshot's `hamiltonians_lr.h5` -- it is an analytic label, not a prediction.
The identity `H_full - H_SR - H_LR` holds to 7.0e-15 eV on the stored labels.

`H_LR` is stored only where it is nonzero (13554 of 24894 blocks on a 3x3x3
snapshot), and `H_full - H_SR` is exactly 0 on the rest, so a missing key
contributes nothing to the sum.

`evaluate.py` does all of this:

    python training/evaluate.py --sr-run DIR --full-run DIR --set test  --out test.json
    python training/evaluate.py --sr-run DIR --full-run DIR --set large --out large.json

It loads each run's `best_model.pkl` through that run's own `src/`, predicts,
reconstructs, and scores against `hamiltonians_full.h5` -- MAE and RMSE in eV,
overall and by distance bin, displacement family and |q| shell. Distance bins
use `mgo_lr.locality.block_distance` with the same 1 A width as the locality
reports, so the two line up.

Evaluation is independent of the workspace export state: it takes the label
files by name and only needs the graph, whose edges come from the Hamiltonian
keys, which are the same 24894 blocks under either export.

### How big is the effect you are looking for?

`||H_LR|| / ||H_full||` varies over orders of magnitude across the set, and
that is the point -- it tracks the displacement family:

    3x3x3 transverse modes      ~1e-15   (max|H_LR| ~ 5e-14 eV)
    3x3x3 polar modes           ~1e-4    (max|H_LR| ~ 5e-3 eV)
    4x4x4 polar modes           ~3e-4    (max|H_LR| ~ 1.5e-2 eV)

On snapshots where `H_LR` is ~1e-14 eV the two targets are identical to far
below any achievable model error, and no difference between the runs is
possible even in principle. The signal lives in the polar/longitudinal
families and the low-|q| shells, which is exactly why the report is
stratified. Read the family and |q| breakdowns before the overall number.

## Never train on these

    37 main-set test snapshots       provenance/splits.json -> main.test
    44 large-cell (4x4x4) snapshots  provenance/splits.json -> large_test

`test_large_cell` is the cell-size extrapolation benchmark and holds all 44
processed 4x4x4 snapshots. Do not fold it into training -- see "Which set the
far-field gate actually approves" in `mgo_lr/README.md`. Neither set appears
in `data_trainval`, and `check_split_wiring.py` asserts the validation ids are
disjoint from both.

See `provenance/README.md` for the frozen dataset identity and counts.
