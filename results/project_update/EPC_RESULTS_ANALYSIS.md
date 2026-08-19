# EPC results-figure analysis

## Main result

Across the full Cartesian-AO tensor, the SR-target checkpoint with the fixed
analytic LR wrapper enabled has
**24.36%** relative L2 error,
versus **91.93%** for direct
Full-H. Its relative L2 error is
**73.51% lower**
than Full-H's.

This is a checkpoint comparison, not an isolated analytic-LR effect. The
controlled A/B decomposition in `EPC_PIPELINE_DECOMPOSITION.md` shows that
adding the fixed LR term to this SR checkpoint changes relative L2 error by
only 0.000078 percentage points on the 2×2×2 grid. The EPC advantage is already
present in the SR-only checkpoint.

The result is not driven by one Cartesian axis:

| direction | SR MAE (eV/Å) | Full-H MAE (eV/Å) |
|---|---:|---:|
| x | 0.269 | 0.554 |
| y | 0.269 | 0.554 |
| z | 0.266 | 0.549 |

## Long-range behavior

The distance figures report two complementary quantities from the stored
real-space Cartesian `dH/dtau` blocks before the k/q Fourier transform:

1. Minimum periodic distance from the displaced atom to the bra–ket AO-pair
   midpoint. This measures perturbation propagation. The 2×2×2 displacement
   supercell has an image-free radius of
   **2.47 Å**, so distances beyond
   it mix the perturbation with its periodic images and are shaded accordingly.
2. Unwrapped bra–ket separation `|r_j + R - r_i|`. This measures errors on
   long-hopping blocks, not propagation distance.

Every occupied atom-pair block contributes all of its AO matrix elements,
including exact zeros. Within the whole bins inside the image-free radius, SR
has **69.8% lower**
error than Full-H. In the periodic-image bins its error is
**80.2% lower**,
but that latter percentage is a supercell-periodic diagnostic rather than an
isolated-distance result.

For AO pairs at or beyond 7 Å, the SR MAE is
**0.002306 eV/Å**, compared with
**0.02494 eV/Å** for Full-H, a
**90.8% lower** error for the
SR-target pipeline. The A/B decomposition shows this difference cannot be
causally assigned to the analytic LR addition on this grid. It is also not
proof that the tail is quantitatively accurate: beyond about 8 Å the mean DFT
block magnitude is smaller than either model's MAE. The SR-target checkpoint
mostly suppresses the much larger spurious Full-H tail.

### Perturbation-to-pair-midpoint distance

| distance (Å) | SR MAE | Full-H MAE | Full-H error removed by SR |
|---:|---:|---:|---:|
| 0.0–0.5 | 0.003026 | 0.003703 | 18.3% |
| 1.0–1.5 | 0.004322 | 0.00552 | 21.7% |
| 1.5–2.0 | 0.006717 | 0.03835 | 82.5% |
| 2.0–2.5 | 0.003908 | 0.005488 | 28.8% |
| 2.5–3.0 | 0.007466 | 0.0489 | 84.7% |
| 3.0–3.5 | 0.003761 | 0.01621 | 76.8% |
| 3.5–4.0 | 0.003625 | 0.004949 | 26.7% |
| 4.0–4.5 | 0.001786 | 0.002083 | 14.2% |

### Bra–ket AO separation

| distance (Å) | SR MAE | Full-H MAE | Full-H error removed by SR |
|---:|---:|---:|---:|
| 0–1 | 0.04111 | 0.04382 | 6.2% |
| 2–3 | 0.02964 | 0.03397 | 12.8% |
| 3–4 | 0.02741 | 0.0287 | 4.5% |
| 4–5 | 0.009952 | 0.01066 | 6.7% |
| 5–6 | 0.01079 | 0.01053 | -2.5% |
| 6–7 | 0.003805 | 0.003934 | 3.3% |
| 7–8 | 0.00215 | 0.003621 | 40.6% |
| 8–9 | 0.00282 | 0.06596 | 95.7% |
| 9–10 | 0.001342 | 0.009516 | 85.9% |
| 10–11 | 0.004033 | 0.005029 | 19.8% |

## Cartesian direction accuracy

For each fixed `(k,q,kappa,i,j)`, the three complex x/y/z values form one
Cartesian vector. The angle uses the real Hermitian inner product and is
weighted by DFT coupling power `||g_DFT||^2`, so symmetry-zero elements do not
dominate the statistic.

| method | mean angle | median angle | 90% power angle | mean cosine |
|---|---:|---:|---:|---:|
| SR-target (+ fixed LR) | 4.57° | 1.45° | 11.63° | 0.9860 |
| Direct Full-H | 8.04° | 1.67° | 18.42° | 0.9507 |

## Figure files

1. `epc_results_01_actual_predicted_parity` — actual-vs-predicted complex EPC.
2. `epc_results_02_cartesian_component_mae` — x/y/z complex MAE.
3. `epc_results_03_magnitude_parity` — actual-vs-predicted `|g|`.
4. `epc_results_04_mae_vs_ao_distance` — real-space EPC MAE vs perturbation and AO-pair distances.
5. `epc_results_05_full_error_removed_vs_distance` — percent of Full-H error removed by SR under both distances.
6. `epc_results_06_angular_direction_accuracy` — Cartesian-vector angular error.

Every figure is available as PNG and PDF in this directory. Numerical values
and definitions are in `epc_results_analysis.json`.

## Scope

These are Cartesian AO Hamiltonian derivatives on the 2×2×2 k/q grid. They are
not yet band- and phonon-mode-resolved EPCs and do not include the downstream
phonon eigenvectors, mass/frequency factors, electronic eigenvectors, or
`dS/dtau` contribution.
