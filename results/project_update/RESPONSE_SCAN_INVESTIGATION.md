# Hamiltonian-response investigation

## Experiment

A home-cell Mg atom was displaced continuously along Cartesian x at 25 points
from −0.03 to +0.03 Å (0.0025 Å spacing). All 25 ABACUS calculations converged.
The ±0.0025 Å Hamiltonian CSR files are byte-identical to the independent DFT
EPC reference pair, which validates the geometry, orbital, atom-order, and
gauge mapping used here.

The 20-element plots show
`H_ij(delta) - H_ij^DFT(0)` above and a 7-point cubic derivative below. Elements
were selected reproducibly from the top 10% of nonzero DFT response magnitude,
ranked by how much SR reduces Full-H squared central-slope error, with symmetry
and distance/orbital diversity constraints. Close-H/wrong-slope exemplars are
shown first, followed by the remaining large slope-error contributors.

## 1. Gradient quality

| model | central slope MAE (eV/Å) | central slope RMSE (eV/Å) | relative L2 |
|---|---:|---:|---:|
| SR only | 0.006673 | 0.027998 | 27.82% |
| SR + fixed LR | 0.006673 | 0.027997 | 27.82% |
| Direct Full-H | 0.008721 | 0.046878 | 46.58% |

For this Mg-x slice, Full-H central-slope RMSE is
**1.67×**
SR-only's, despite both curves often lying close to DFT in H itself. This is the
visual phenomenon the scan was designed to test.

## 2. Where Full-H gains lower Hamiltonian error

Across all scan points and graph elements, Full-H has lower MAE
(0.7851 meV versus
0.8416 meV), while SR has slightly lower RMSE
(7.0892 meV versus
7.3385 meV). After subtracting each method's
equilibrium value to isolate displacement response, Full-H also has
7.54%
lower H-response RMSE, yet worse central slopes.

Element by element, Full-H has lower raw H MAE for
**52.19%** of elements. In
**24.07%** of
all elements it is closer in H but worse in slope; those elements carry
**16.41%** of the
DFT central-response power. Aggregate H loss is therefore hiding a meaningful
response-error subset.

The earlier “14.7% better” number is the Full-H **validation MSE** advantage
(14.73%), not training
loss. Held-out full-space H MAE favors Full-H by
24.47%.

## 3. Is Full-H specifically failing the analytic LR contribution?

No—not on the present 2×2×2 grid and LR definition. The isolated analytic LR
central slope has only
**0.000928%** of the
`DFT − SR` residual norm. Adding it changes SR relative slope error from
27.822146% to
27.822046%.

This agrees with the full A/B/D EPC decomposition: the fixed LR term is enabled
but negligible at the sampled reciprocal vectors because the largest Ewald
weight is only 1.89×10⁻⁵. The current evidence therefore identifies a
**checkpoint/learned-response difference**, not successful correction by the
analytic LR term and not a missing gradient through learned tensors (there are
no such tensor heads).

## 4. Does Hamiltonian error correlate with response/EPC error?

Yes, but imperfectly. For response-aligned per-element H RMSE versus central
slope error, log-space Spearman correlation is
**0.827** for SR and
**0.772** for Full-H.
For raw H MAE the correlations are lower
(0.640 and
0.657). Hamiltonian accuracy is
informative, but aggregate value loss does not uniquely control local slopes.

## Selected 20 elements

| # | element | distance (Å) | DFT slope | SR slope | Full slope | Full closer at δ=0 |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | O p2[m0] → O s2 | 0.00 | 3.7264 | 2.5516 | 2.1861 | yes |
| 2 | Mg d1[m1] → O p2[m0] | 2.14 | -1.1515 | -0.7250 | -0.1132 | no |
| 3 | O p2[m0] → Mg s3 | 2.14 | 1.4402 | 0.7593 | 0.4089 | yes |
| 4 | O p2[m0] → Mg s3 | 3.71 | 1.4012 | 0.8548 | 0.4638 | yes |
| 5 | Mg d1[m2] → Mg s4 | 3.03 | 3.6117 | 3.3708 | 3.0963 | no |
| 6 | Mg s4 → Mg d1[m3] | 3.03 | 3.6117 | 3.3708 | 3.0963 | no |
| 7 | Mg d1[m1] → Mg s4 | 4.29 | 3.1929 | 3.1845 | 3.6343 | no |
| 8 | Mg d1[m1] → Mg s4 | 4.29 | 0.0583 | 0.0696 | -0.3058 | no |
| 9 | O p2[m0] → Mg d1[m1] | 6.43 | 0.4315 | 0.2768 | 0.2121 | yes |
| 10 | Mg d1[m0] → O p2[m0] | 6.43 | -0.2491 | -0.1598 | -0.1225 | yes |
| 11 | O s1 → O s1 | 0.00 | 2.5929 | 0.6411 | 0.0313 | no |
| 12 | O p2[m0] → O p2[m0] | 0.00 | 6.7881 | 4.7899 | 4.5399 | yes |
| 13 | O d1[m1] → O d1[m1] | 0.00 | 3.5421 | 1.9090 | 1.6415 | yes |
| 14 | Mg s3 → Mg s4 | 8.02 | 0.0699 | -0.1376 | -0.8115 | yes |
| 15 | O p2[m0] → Mg s1 | 2.14 | -3.4523 | -2.6804 | -2.3300 | no |
| 16 | O p2[m0] → Mg s3 | 2.14 | 1.3896 | 1.0296 | 0.5429 | yes |
| 17 | Mg s1 → Mg s3 | 4.29 | -1.8081 | -1.4557 | -1.0212 | yes |
| 18 | Mg s3 → Mg s1 | 4.29 | 0.4644 | 0.2968 | -0.0592 | yes |
| 19 | O p2[m2] → O p2[m0] | 3.03 | 2.4518 | 1.7366 | 1.6054 | yes |
| 20 | Mg s4 → Mg s4 | 8.02 | -0.0208 | -0.1657 | 0.4360 | yes |

## Interpretation and limitations

The prioritized plot supports the proposed mechanism: a network can track
`H(delta)` closely over a narrow range while learning a poorer local slope.
However, it does **not** yet establish that the SR target itself causes the
improvement. SR and Full-H are independent fits with different random training
trajectories, and the analytic LR subtraction is extremely small on this grid.
Matched-seed multi-run training, or direct derivative supervision, is needed to
separate target-design effects from training variance.

This scan covers one atom/direction in a 2×2×2 periodic cell. The existing full
EPC tensor covers both atoms and all directions and shows the same ordering,
but a denser-q/supercell response study is required before claiming an isolated
long-range mechanism.

## Figure files

* `response_scan_selected_20_elements.pdf` — all 20 paired H/derivative panels.
* `response_scan_20_elements_page_1.png` through page 4 — reviewable PNG pages.
* `response_scan_four_bucket_summary` — four-bucket quantitative summary.
* `response_scan_selected_20_curves.csv` — plotted numerical curves.
* `response_scan_investigation.json` — complete metrics and provenance.
