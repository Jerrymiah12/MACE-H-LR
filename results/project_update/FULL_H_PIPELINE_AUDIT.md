# Full-H pipeline architecture audit

## Finding

The conditional learned-tensor architecture is **not present in these runs**.
The Full-H checkpoint directly predicts the total Hamiltonian.  The SR
checkpoint directly predicts the SR target, and only its EPC configuration
adds an analytic LR term.  Neither checkpoint contains a Born-charge or
dielectric head.

| Check | SR | Full-H |
|---|---:|---:|
| Training dataset | `mgo404sr` | `mgo404full` |
| Checkpoint tensors | 1,352 | 1,352 |
| Checkpoint scalars | 50,045,998 | 50,045,998 |
| Born/dielectric-like state keys | 0 | 0 |
| Analytic LR enabled in EPC | yes | no |

## Equilibrium numerical check

The stored analytic LR label has norm
`7.378e-16 eV` and maximum
absolute element `6.343e-17 eV` at equilibrium,
so `SR + analytic LR` is numerically the SR prediction there.

After the same overlap-gauge projection used in EPC:

| Comparison | MAE (eV) | RMSE (eV) | relative L2 |
|---|---:|---:|---:|
| Full-H vs SR + LR | 1.302255e-04 | 3.702117e-04 | 0.0749% |
| SR + LR vs DFT | 3.455777e-04 | 2.043755e-03 | 0.4134% |
| Full-H vs DFT | 3.302419e-04 | 1.994670e-03 | 0.4035% |

They are not mathematically identical, nor should they be: they are separate
fits to different targets.  The raw-gauge Full-H/SR mismatch is
`1.287802e-04 eV` MAE.

## LR tensors and derivative path

The SR EPC wrapper loads frozen NumPy tensors before its prediction closure:

* Born effective charges: diagonal values 2.01451000 (Mg) and
  -2.01451000 (O), SHA-256
  `1adb05f216df8d6feb287e6e86dbd41b17a3cc468747d920e1e308e7b09b7b6c`.
* Electronic dielectric tensor: diagonal value 3.305289435, SHA-256
  `3dd0884d085fe9435557d347246f10d3cca035538125432b003d3f2c6fb3f6a0`.

EPC uses central finite differences in atomic positions.  There is no
autodiff path through these arrays, no predicted `Z*`/`epsilon_infinity`, and
therefore no meaningful detach/stop-gradient experiment for the current
checkpoints.

## Correct decomposition for these artifacts

The realizable comparison is:

1. A: finite difference of the direct SR predictor;
2. B: finite difference of SR plus analytic LR using frozen DFT/DFPT tensors;
3. D: finite difference of the independently trained direct Full-H predictor.

The proposed C case with learned tensors cannot be constructed from these
checkpoints.  Its absence is an architectural fact, not a failed numerical
test.  EPC metrics for A/B/D are reported separately after the SR-only run.
