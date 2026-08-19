# SR-target vs full-H baseline runs

Both configs are production-sized (`64x0e` embed,
`64x0e+32x1o+16x2e+8x3o+8x4e`, 3 blocks, 1350936 trainable parameters) and
point at the frozen 404-snapshot 3x3x3 main set.

    train_sr.ini             SR-target run   (label = H_SR)
    train_full.ini           baseline run    (label = H_full)
    paths.py                 resolves the two configurable locations
    make_trainval_view.py    builds the dataset directory both configs read
    check_split_wiring.py    proves the configs reproduce the frozen splits
    check_training_controls.py  checks speed/stop/checkpoint controls
    freeze_provenance.py     writes / verifies `provenance/`
    freeze_cache.py          records / verifies which target a cache contains
    run_sr_then_full.sh      launches the guarded sequential production pair
    keep_awake.ps1           prevents Windows auto-sleep during that pair
    setup_gpu_env.sh         builds the CUDA environment
    smoke_onebatch.py        does the pipeline run at all on this box?
    smoke_train.ini            (its config -- pilot data, tiny model)
    smoke_production.py      do a few real epochs run, on the real splits?
    evaluate.py              score checkpoints on frozen held-out sets
    make_result_figures.py   generate the five primary comparison figures

Run with `python -m maceh train workflows/training/<config>.ini` from the repo root.

## Runbook

    ./workflows/training/setup_gpu_env.sh /path/to/venv_cuda
    export MACEH_DATA_ROOT=/path/to/run
    export MACEH_RUNS_ROOT=/path/to/generated-runs

    python -m workflows.training.freeze_provenance --verify        # dataset unchanged?
    python -m workflows.training.make_trainval_view                # 330 + 37
    python -m workflows.training.check_split_wiring                # splits wired right?
    python -m workflows.training.smoke_onebatch                    # pipeline runs?
    python -m workflows.training.smoke_production --target sr --epochs 2
    python -m workflows.training.freeze_cache --target sr           # pin cache labels
    python -m maceh train workflows/training/train_sr.ini        # the real run

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

    workspace       the MgO dataset workspace: `splits.json`, `metadata.yaml`,
                    `main/`, `loader_splits/`.  ~150 GB, lives outside the repo.
    training_root   where training artefacts go: the `data_trainval` view, the
                    graph caches, the run output directories.

Set them once in the `[DEFAULT]` block at the top of **both** `.ini` files --
the rest of each config interpolates `%(training_root)s`. The helper scripts
resolve the same two values, in this order:

    1. `--workspace` / `--training-root` on the command line
    2. `$MACEH_DATA_ROOT` / `$MACEH_RUNS_ROOT`

The helper scripts require either an explicit CLI value or the corresponding
environment variable. `maceh train`
reads the `.ini` directly and does not import `workflows.training.paths`. Before the real
run, either put absolute locations in both `[DEFAULT]` blocks or make the
config defaults (`./data` and `./runs`) local symlinks to the ext4
locations. Relative values resolve against the working directory; never let a
production cache or output silently land on `/mnt/c`.

## The splits must come from `organize`, not from ratios

MACE-H's `get_loader()` splits **one** dataset by ratio, while
`workflows.mgo_dataset organize` publishes three *separate* loader views.
Pointing a config at
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

    python -m workflows.training.make_trainval_view     # -> 330 train + 37 validation
    python -m workflows.training.check_split_wiring     # 3 checks, see below

The view's symlinks retain the logical hop through
`data/loader_splits/<subset>/<sid>/hamiltonians.h5`, which ultimately points at
`data/main/<sid>/hamiltonians.h5`. `export-target` rewrites the latter and
`organize` safely recreates the loader views, so the combined view follows an
export switch. The guarded paired runner still rebuilds and verifies all 367
links after each export as a fail-safe against stale labels.

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

With `extra_validation` set, upstream MACE-H evaluates `val_loader` and
`extra_val_loader` separately. Here their memberships are exactly the same 37
structures. `get_loader()` now proves that equality, aliases the two loaders,
and `train()` reuses the first result. Generic configurations with different
memberships still get two independent validation passes. This preserves the
loss used by the LR scheduler and saves one 37-structure forward pass per
epoch.

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

### RTX 5090 production benchmark, acceleration and ETA

Measured on the real 367-structure train+validation view, with the production
model and an already-built SR cache:

    graph-cache load             < 1 s (mmap metadata only)
    mask preparation             15.16 s
    model construction            4.98 s
    original FP32/batch-1 epoch 153.15 s
    TF32/batch-2 full epoch     102.84 s (measured before val-pass reuse)
    optimized production smoke 100.72 s = 1.68 min (reuse enabled)
    final 37-structure test        6.40 s
    TF32/batch-2 CUDA allocated    17.99 GiB peak
    TF32/batch-2 CUDA reserved     20.81 GiB peak

The production configs use `float32_matmul_precision = high` (TF32 tensor-core
matmuls) and `batch_size = 2`. SR and full-H have identical graph shapes,
model and batch counts, so their runtime should be effectively the same. At
100.72 seconds per epoch, all 3000 epochs would take about **83.9 hours = 3.50
days per target**, or **7.0 days for SR and full-H sequentially**, plus cache
building (historically about 20 minutes for a missing production cache). A
flat validation curve reaches the configured LR floor after about 1088 epochs,
or roughly 30.4 hours per target at this rate; validation improvements extend
that schedule toward the 3000-epoch cap.

TF32 is not bit-identical to full FP32: sampled predictions differed by
`8.3e-4` in relative L2 and sampled losses by at most 0.94%. Batch size 2 also
halves the number of Adam updates per epoch. Both controls are identical in SR
and full-H, but their convergence must still be checked through the frozen
validation and held-out reports.

### Absolute validation-loss stop

Both production configs stop after the frozen-validation MSE is at or below
`1e-6 eV^2` for 10 consecutive epochs, once at least 200 epochs have completed.
This is approximately a 1 meV global masked-element RMSE. The current two-epoch
SR smoke checkpoint is at `1.2795e-4 eV^2` (11.31 meV RMSE), so it is nowhere
near the gate yet and cannot predict when the gate will be reached. The
200-epoch floor alone is about 5.6 hours per target at the optimized rate.

This is a provisional global accuracy gate, not proof that every sparse
orbital channel or the long-range polar signal is converged. The final claim
still comes from `evaluate.py` on the frozen held-out test and large sets,
especially the displacement-family and low-|q| breakdowns. Set
`early_stop_val_loss = -1` to disable the gate.

`checkpoint_interval = 10` writes the restart `model.pkl` every ten completed
epochs and once on a clean exit, while every new `best_model.pkl` is still
saved immediately. This cuts routine latest-checkpoint traffic from at least
637 GB to about 64 GB over 3000 epochs, plus writes for validation
improvements. Checkpoint files are replaced atomically.

After an abrupt stop, compare the `epoch` fields in `model.pkl` and
`best_model.pkl` and resume from the newer one: the periodic restart can be
older than an immediately saved best. Resume logic preserves the better model
metadata and the checkpoint's own TensorBoard step even when the two files
come from different epochs.

Run the fast control checks after editing either production config:

    python -m workflows.training.check_training_controls
    python -m workflows.training.check_split_wiring --configs-only

### Why WSL was being killed

The production cache has 9,107,206 edges. Its raw `Aij` tensor is 7.634 GiB,
one dense label tensor is another 7.634 GiB, and one mask is 1.908 GiB. The old
`set_mask()` retained per-graph labels/masks and then `torch.cat` allocated a
second complete pair: a 26.72 GiB floor before Python/model overhead, already
above this machine's 26 GB (25.4 GiB) WSL limit. The kernel therefore invoked
the host OOM killer; this was not a CUDA OOM.

`AijData` now memory-maps the trusted local cache and fills one final collated
label/mask allocation in place. Shift/scale statistics are also streamed one
structure at a time instead of materializing two more 7.6 GiB transforms. The
full mask-plus-statistics check now peaks at 13.18 GiB RSS. Keep the cache,
training output and virtual environment on WSL's native ext4 filesystem, not
under `/mnt/c`, and run only one of these trainings at a time on the 26 GB WSL
VM.

The graph caches (`graphs_sr/`, `graphs_full/`) are deliberately **not**
prebuilt: a 330-snapshot cache is ~8 GB, cheap to regenerate, and awkward to
move. Build them on the target box, in the export order below.

## The two runs cannot share a workspace export

`export-target` is workspace-wide and all-or-nothing: it points every
snapshot's `hamiltonians.h5` at one source and records the choice in
`data/metadata.yaml` as `training_target`. Build one graph cache at a time:

    # SR run
    python -m workflows.mgo_dataset export-target --target sr   --config provenance/config.resolved.yaml --workspace $MACEH_DATA_ROOT
    python -m workflows.mgo_dataset organize                    --config provenance/config.resolved.yaml --workspace $MACEH_DATA_ROOT
    python -m workflows.training.make_trainval_view
    python -m workflows.training.check_split_wiring
    # ... build graphs / train with train_sr.ini ...

    # full-H baseline
    python -m workflows.mgo_dataset export-target --target full --config provenance/config.resolved.yaml --workspace $MACEH_DATA_ROOT
    python -m workflows.mgo_dataset organize                    --config provenance/config.resolved.yaml --workspace $MACEH_DATA_ROOT
    python -m workflows.training.make_trainval_view
    python -m workflows.training.check_split_wiring
    # ... build graphs / train with train_full.ini ...

Once a run's `save_graph_dir` cache exists it is self-contained, so the two
trainings can proceed in parallel on a host with enough RAM and GPU capacity.
On this 26 GB WSL VM they must run sequentially. The *graph building* steps
must never overlap, or the second will cache whichever labels happen to be
exported at that moment. `dataset_name` differs between the configs
(`mgo404sr` / `mgo404full`) so the two caches cannot collide on filename, but
that does not protect against building both while the workspace is in one
export state.

Sanity check before training: `data/metadata.yaml` -> `training_target` must
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

    python -m workflows.training.evaluate --sr-run DIR --full-run DIR --set test  --out test.json
    python -m workflows.training.evaluate --sr-run DIR --full-run DIR --set large --out large.json

It loads each run's `best_model.pkl` through that run's own `src/`, predicts,
reconstructs, and scores against `hamiltonians_full.h5` -- MAE and RMSE in eV,
overall and by distance bin, displacement family and |q| shell. Distance bins
use `maceh.analysis.locality.block_distance` with the same 1 A width as the locality
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
far-field gate actually approves" in `workflows/mgo_dataset/README.md`. Neither set appears
in `data_trainval`, and `check_split_wiring.py` asserts the validation ids are
disjoint from both.

See `provenance/README.md` for the frozen dataset identity and counts.

## Geometry-dependent Born/dielectric prototype

The original SR checkpoint has no Born-charge or dielectric head.  The
existing dataset also has only one equilibrium pair of tensors under
`data/reference/`; it does not contain per-snapshot tensor labels.

`tensor_dfpt_54.py` prepares a separate, partially labelled 54-atom campaign
without changing any Hamiltonian files.  The fast prototype contains 10
tensor-training structures, 2 tensor-validation structures, and 5 held-out
tensor-test structures.  The Hamiltonian head will still use all 330 existing
training snapshots; its tensor losses will be masked on snapshots without
DFPT labels.

Prepare the fast inputs and their one-snapshot convergence anchor:

    python -m workflows.training.tensor_dfpt_54 prepare --profile fast
    python -m workflows.training.tensor_dfpt_54 prepare --profile anchor

Run one snapshot in tmux (the script refuses to overlap another QE job and
requires at least 18 GiB available WSL memory):

    tmux new-session -d -s mgo_tensor54 \
      'bash workflows/training/run_tensor_dfpt_54.sh fast snapshot_000386'

After inspecting that first benchmark, the gated campaign runner executes the
3x3x3 anchor comparison and proceeds through the other 16 fast jobs only when
the comparison passes:

    bash workflows/training/run_tensor_dfpt_54_campaign.sh

For an explicitly exploratory run, a failed convergence gate can be recorded
and overridden with `MGO_TENSOR_ACCEPT_FAST=1`.  Do not use labels produced
under that override as converged production targets without reporting the
fast-versus-anchor discrepancy.

The fast profile uses 80 Ry and a 2x2x2 supercell k-grid.  Before launching
the rest, run the same snapshot with the 3x3x3 `anchor` profile and compare:

    bash workflows/training/run_tensor_dfpt_54.sh anchor snapshot_000386
    python -m workflows.training.tensor_dfpt_54 compare snapshot_000386

The collector checks completion, atom ordering, finite values, positive
dielectric eigenvalues and species-mean Born-charge signs.  It writes both raw
and corrected tensors; `born_effective_charges.npy` has the acoustic sum rule
applied and `dielectric_infinity.npy` is explicitly symmetrized.

The optimized QE path uses `epsil=.true.`, `zeu=.true.`, and
`trans=.false.`.  It computes all 54 Born tensors from the three electric-field
responses and deliberately skips the unnecessary dynamical matrix.  The
runner uses 16 MPI ranks across the 16 physical CPU cores; do not reintroduce
`OMP_PLACES=cores`, which pins every independent MPI rank to the first WSL
core unless rank-specific OpenMP places are supplied.

The implemented shared-head warmup is `train_sr_tensors.ini`. It audits all 17
sidecars against the manifest and source STRU hashes, attaches only the ten
training and two validation labels to the existing graph cache, derives
residual baselines/scales from the ten training labels, and refuses a cache
containing any of the five locked tensor-test IDs. Launch it only after the
focused tests pass:

    PY=python
    $PY -m pytest -q -s tests/unit/test_sr_tensors.py tests/integration/epc/test_mgo_reference.py
    $PY -m maceh train workflows/training/train_sr_tensors.ini

The head-only config freezes the SR encoder and Hamiltonian head, repeats
tensor-labelled structures four times per epoch, and retains every ordinary
Hamiltonian structure. `train_sr_tensors_partial.ini` is the accepted partial
stage: it unfreezes the final interaction block and H head at `1e-5` while the
tensor heads use `1e-4`. Checkpoints record the model contract and
tensor-manifest SHA-256.

Predicted tensors can drive EPC by setting these keys in an EPC config:

    analytic_lr_tensor_source = model
    analytic_lr_tensor_mode = equilibrium_frozen

Use `geometry_dependent` for the second explicit mode. Every EPC HDF5 records
the source, mode, and tensor provenance. The locked test remains a
post-selection action and is never evaluated by either training config. The
completed prototype decisions, validation trajectory, locked metrics, and EPC
comparison are recorded in `SR_BORN_EPSILON_TRAINING_REPORT.md`.
