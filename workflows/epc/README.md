# MgO Cartesian-AO EPC comparison

This directory compares electron-phonon Hamiltonian derivatives from the trained
SR-target and direct Full-H checkpoints with a matched finite-displacement
ABACUS reference. The SR-target evaluation has a fixed analytic LR wrapper
enabled. The EPC implementation was copied from the sibling
`MACE-H-epc` project at commit `43eaf1d`; that source project was not modified.

## Result

The SR-target checkpoint is substantially closer to DFT over the full 2x2x2
k/q grid, but the controlled decomposition shows that adding the fixed analytic
LR term is numerically negligible on this grid:

| method | relative L2 error | complex MAE (eV/A) | cosine similarity | norm / DFT |
|---|---:|---:|---:|---:|
| SR only | **24.3559%** | **0.268** | **0.9700** | 0.954 |
| SR + fixed analytic LR | **24.3558%** | **0.268** | **0.9700** | 0.954 |
| Direct Full-H | 91.93% | 0.552 | 0.7306 | 1.346 |

The SR-target pipeline has 73.50% lower relative L2 error than the direct
Full-H model. This is a checkpoint comparison, not an isolated LR effect: the
analytic correction has only 0.001606% of the norm of the residual needed to
bring SR to DFT. At Gamma alone the methods are almost tied (SR 6.07%, Full
5.82%); the difference comes from nonzero q, where SR is 24.48--26.13% and Full
is 55.85--116.99%. Full therefore slightly wins the Gamma-only comparison but
substantially overestimates finite-q coupling.

The audit established that Full-H is a separate direct total-Hamiltonian
network. It is not SR plus learned Born-charge/dielectric heads, and neither
checkpoint contains such heads. The SR wrapper uses frozen DFT/DFPT
`Z* = +/-2.01451` and `epsilon_infinity = 3.305289435`; there is no autodiff
path through those NumPy tensors. See `plots/FULL_H_PIPELINE_AUDIT.md` and
`plots/EPC_PIPELINE_DECOMPOSITION.md` for the numerical evidence.

The fine DFT reference uses delta = 0.0025 A. Repeating all 12 DFT calculations
at delta = 0.005 A changes the resulting tensor by only 0.00656% in relative L2,
far below either model error. All tensors satisfy
`g(k,q)^dagger = g(k+q,-q)`; the largest residual is 2.9e-11 of the DFT tensor
peak. These checks support the finite-difference and indexing pipeline.

The complete machine-readable results, including per-q and per-component
metrics, SHA-256 hashes, acoustic-sum diagnostics, and Hermiticity residuals,
are in `comparison_metrics.json`.

## What was calculated

The stored quantity is

```
g_ij,kappa,alpha(k,q) = dH_ij(k,q) / d tau_kappa,alpha
```

in the Cartesian atomic-orbital basis, in eV/A. Each tensor has shape
`[8 k, 8 q, 2 atoms, 3 directions, 28 AO, 28 AO]`, or 301,056 complex
components. The setup uses the primitive two-atom rocksalt MgO cell and a
2x2x2 displacement supercell.

For SR, every displaced prediction is reconstructed as
`H_SR(predicted) + H_LR(analytic)` before differentiation. DFT and both models
are placed in the same scalar-energy gauge,
`H - <H,S0>/<S0,S0> S0`, using the equilibrium DFT overlap. This removes the
arbitrary SCF/model energy zero without changing orbital-resolved structure.

The DFT reference consists of 12 converged ABACUS 3.7.4 SCFs: plus/minus one
displacement for each of two primitive atoms and three Cartesian directions.
The 16-atom supercell uses PBE, 100 Ry, a 4x4x4 k mesh, the same Mg/O numerical
atomic orbitals and pseudopotentials as the dataset, and `scf_thr = 1e-8`.

## Measured runtime

On this machine, with four simultaneous four-rank ABACUS jobs and the CUDA GPU:

| stage | wall time |
|---|---:|
| Fine DFT reference, 12 SCFs | 2 min 03 s |
| SR-target (+ fixed LR) EPC | 1 min 22 s |
| SR-only diagnostic EPC | about 1 min |
| Direct Full-H EPC | 1 min 30 s |
| Optional coarse DFT convergence reference | 2 min 03 s |
| Collection and plots | seconds |

A fresh primary comparison is therefore about **5 minutes** after the
environment and trained checkpoints exist. Including the independent DFT
step-size check is about **7 minutes**. Model runs peaked near 3.5 GiB resident
memory in the measured process. A denser q grid will cost much more because its
displacement supercell grows as the product of the three q-grid dimensions.

## Outputs

Generated tensors (ignored by git because each is about 116 MiB):

```
runs/epc/actual/structure_primitive/epc_cartesian_actual.h5
runs/epc/actual_d005/structure_primitive/epc_cartesian_actual.h5
runs/epc/sr/structure_primitive/epc_cartesian_pred.h5
runs/epc/sr_only/structure_primitive/epc_cartesian_pred.h5
runs/epc/full/structure_primitive/epc_cartesian_pred.h5
```

The six main results figures are in `plots/` as both PNG and PDF. Their exact
definitions, bin values, and interpretation are recorded in
`plots/EPC_RESULTS_ANALYSIS.md` and `plots/epc_results_analysis.json`.

| figure | contents |
|---|---|
| `epc_results_01_actual_predicted_parity` | actual-vs-predicted complex EPC for SR and Full-H |
| `epc_results_02_cartesian_component_mae` | x/y/z complex MAE |
| `epc_results_03_magnitude_parity` | actual-vs-predicted EPC magnitude for SR and Full-H |
| `epc_results_04_mae_vs_ao_distance` | real-space EPC MAE versus perturbation-propagation and AO-pair distances |
| `epc_results_05_full_error_removed_vs_distance` | Full-H error reduction of the SR-target pipeline versus both distances |
| `epc_results_06_angular_direction_accuracy` | power-weighted Cartesian-vector angular error and per-q direction accuracy |

Four diagnostic figures are retained alongside them:

| figure | contents |
|---|---|
| `epc_01_q_resolved_comparison` | per-q relative error and coupling-norm ratio |
| `epc_02_dft_parity` | SR/Full component parity against actual DFT EPC |
| `epc_03_component_error_heatmap` | q-, atom-, and direction-resolved error |
| `epc_04_finite_difference_convergence` | model and DFT displacement-step checks |

## Reproduce

Run from the repository root. `PY` below is the prepared CUDA environment.

```bash
PY=python

# Fine DFT reference.
$PY epc/prepare_dft_reference.py --delta 0.0025 --output data/epc/dft_reference
$PY epc/run_dft_reference.py --root data/epc/dft_reference --jobs 4 --mpi-ranks 4
$PY epc/collect_dft_epc.py --root data/epc/dft_reference --output runs/epc/actual

# Model predictions.
$PY -m maceh epc workflows/epc/sr.ini -n 8
$PY -m maceh epc workflows/epc/sr_only.ini -n 8  # controlled A/B diagnostic
$PY -m maceh epc workflows/epc/full.ini -n 8

# Independent DFT finite-difference check (optional but used in the report).
$PY epc/prepare_dft_reference.py --delta 0.005 --output data/epc/dft_reference_d005
$PY epc/run_dft_reference.py --root data/epc/dft_reference_d005 --jobs 4 --mpi-ranks 4
$PY epc/collect_dft_epc.py --root data/epc/dft_reference_d005 \
    --output runs/epc/actual_d005

# Metrics and figures.
$PY epc/compare_epc.py
$PY epc/make_results_figures.py
$PY epc/audit_full_h_pipeline.py
$PY epc/analyze_epc_decomposition.py

# Continuous Mg-x Hamiltonian-response investigation.
$PY epc/prepare_response_scan.py
$PY epc/run_dft_reference.py --root data/epc/response_scan_dft \
    --jobs 4 --mpi-ranks 4
$PY epc/collect_response_scan.py
$PY epc/analyze_response_scan.py
```

`run_dft_reference.py` defaults to the local ABACUS bundle under
``; use `--dft-bin` if that
bundle moves. Completed, converged DFT folders are skipped safely.

The production model finite-difference step is 5e-6 A. It was selected from
`model_delta_sweep.py`: the SR derivative is already stable there, and the Full
derivative changes by only 0.00319% between 5e-6 and 2e-6 A (and 0.000455%
between 2e-6 and 1e-6 A). The much larger DFT step is appropriate because an
SCF calculation has a different numerical-noise scale.

## Scope and limitations

- This is an AO Cartesian Hamiltonian derivative, not the final band- and
  phonon-mode-resolved EPC. Phonon eigenvectors/frequencies, mass factors,
  electronic eigenvectors, and the non-orthogonal-basis `dS/dtau` term are not
  included.
- The 2x2x2 q grid is a deliberately small first comparison. The Gamma sum is
  valid, but the supercell image-free radius is only 2.47 A while both models
  respond farther away, so individual nonzero-q values are coarsely resolved.
  A denser q grid is required before treating the q dependence as converged.
- This evaluates one equilibrium MgO geometry. It establishes a large
  checkpoint-level response difference, not successful analytic-LR correction
  on this grid and not a statistical generalization across structures or
  materials.
- The continuous 25-point Mg-x scan shows the same qualitative mismatch:
  Full-H has slightly lower Hamiltonian MAE but 1.67x the SR central-slope RMSE.
  Because the checkpoints are independent fits, matched-seed repeated training
  is still needed to distinguish a target-design effect from training variance.
- The reference and model matrices share the same AO convention and scalar
  energy alignment. Comparing unaligned finite differences would mix physical
  derivatives with changes in an arbitrary energy zero.

## Checkpoint provenance

```
SR   03168b3db8b5c61abf1a14375d6cd5488f66856fdc6ecd44ca552f3410b35583
Full ef8d56cadc2b899c66d9162b0ca80c8f2ab2717309a83f0b588440e8464ed56f
```
