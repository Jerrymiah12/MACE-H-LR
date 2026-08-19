# SR + Born + dielectric prototype report

Completed 2026-08-13 from
`SR_BORN_EPSILON_TRAINING_PLAN.md`. The selected model is the Stage-3 partial
fine-tuning checkpoint:

`<MACEH_RUNS_ROOT>/run_sr_tensors_partial/2026-08-13_19-04-59_sr_born_epsilon_partial/best_model.pkl`

- Checkpoint SHA-256:
  `0281160e3a1cd8a5a318444f5820e2b845899819795bbf2368a089c8a4fa1974`
- Tensor-manifest SHA-256:
  `ecd3cf8e5de65a123d77b9782888104417bf1b472929b7c25b4a7b7f05ebb78d`
- Contract: `sr_born_epsilon`, analytic LR reconstruction enabled, no direct
  full-H head.
- State dictionary: 19 Born-head keys and 15 dielectric-head keys.

## Training decision

The head-only warmup passed all three validation gates for ten consecutive
evaluations. Partial fine-tuning initially oscillated with a `3e-4` tensor-head
learning rate and was rejected. The accepted restart used `1e-4` for tensor
heads and `1e-5` for the final interaction block and H head. It passed every
gate for epochs 0--9 and stopped automatically.

Selected-checkpoint validation metrics at epoch 9:

| Metric | Value |
|---|---:|
| SR Hamiltonian MSE | `2.85743e-6` |
| Born MAE | `1.79936e-3 e` |
| Born RMSE | `2.72296e-3 e` |
| Dielectric MAE | `3.38648e-4` |
| Dielectric RMSE | `5.30948e-4` |

Stage 4 was not run. Full-encoder unfreezing was optional, the partial model
was already stable, and selecting another joint stage from only two validation
structures would add overfitting risk. The model choice and hyperparameters
were frozen before opening the five tensor-test labels.

## Locked test

The one-shot machine-readable report is stored beside the checkpoint as
`locked_test_report.json`. It covers exactly `snapshot_000080`,
`snapshot_000229`, `snapshot_000295`, `snapshot_000333`, and
`snapshot_000334`.

| Metric | Model | Training-mean baseline |
|---|---:|---:|
| SR Hamiltonian MAE | `4.85004e-4 eV` | -- |
| SR Hamiltonian MSE | `3.26059e-6 eV^2` | -- |
| Born MAE | `2.36865e-3 e` | `3.20100e-3 e` |
| Born RMSE | `4.03303e-3 e` | `5.01108e-3 e` |
| Dielectric MAE | `3.89317e-4` | `1.46639e-3` |
| Dielectric RMSE | `6.01951e-4` | `2.02497e-3` |
| Reconstructed full-H MAE | `4.84999e-4 eV` | -- |
| Reconstructed full-H MSE | `3.26041e-6 eV^2` | -- |

The model beats both locked tensor baselines. Born ASR maximum absolute
residual is `1.88351e-5 e` in float32 inference. Predicted dielectric
eigenvalues span `[3.326313, 3.332940]`, so every dielectric is symmetric
positive definite. Full-H reconstruction covers all 123,716 truth blocks, and
the checked `H_reported = H_SR + H_LR(predicted tensors)` identity has exactly
zero maximum residual.

## EPC comparison

Both EPC results use the selected checkpoint, predicted tensors, the existing
2x2x2 k/q grids, and the DFT Cartesian-AO EPC reference. Each HDF5 records its
tensor source, mode, model directory, analytic-reconstruction flag, and
no-direct-full-H contract.

| Mode | Relative L2 | Complex MAE (eV/A) | Complex RMSE (eV/A) |
|---|---:|---:|---:|
| Equilibrium frozen | `0.244327754` | `0.268813168` | `0.860036576` |
| Geometry dependent | `0.244327754` | `0.268813168` | `0.860036576` |

At the `5e-6 A` finite-difference displacement, geometry-dependent tensors
change EPC by `4.25717e-10 eV/A` complex MAE and `4.27330e-7 eV/A` maximum.
The EPC finite-difference ASR violation is `3.521e-6 eV/A` in both modes.

## Verification

- All 17 tensor labels were audited; only 10 training and 2 validation labels
  were attached during optimization.
- A production-cache mixed labelled/unlabelled forward, backward, and optimizer
  step passed; unlabelled structures contributed only Hamiltonian loss.
- Physical constraints, rotational equivariance, old-checkpoint compatibility,
  reconstructed-H identity, and EPC configuration tests passed.
- The consolidated `tests/unit`, `tests/integration`, and `tests/smoke` suite passed with four
  expected skips before the final training stages.
