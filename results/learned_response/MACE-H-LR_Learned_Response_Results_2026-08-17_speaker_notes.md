# Learned response tensors — speaker notes

Presentation date: August 17, 2026  
Target length: 9–10 minutes

## 1. Learned response tensors

0:00–0:35. This update closes the loop on the learned Born-charge and
dielectric architecture. I will show three levels of evidence: whether the
heads generalize on locked tensor labels, whether reconstructed full-H values
improve, and whether geometry-dependent tensors change Cartesian-AO EPC.

## 2. The heads learned the tensors; EPC is unchanged so far

0:35–1:25. The headline is nuanced. The tensor heads are not trivial constant
predictors: both beat the locked baselines. Training preserved the short-range
Hamiltonian. But at the current five-microångström EPC displacement, replacing
fixed tensors with predicted geometry-dependent tensors produces essentially
the same EPC. The direct full-H model remains best for H values and worst for
the derivative observable.

## 3. A shared encoder now predicts HSR, Z*, and ε∞

1:25–2:10. The architectural change is explicit: one equivariant encoder
branches to the short-range Hamiltonian, full atom-resolved Born tensors, and
a symmetric positive-definite dielectric tensor. Acoustic sum rule and SPD
constraints are built in. Training progressed from head-only to a limited
final-block unfreeze, and the locked test was opened only after the model and
learning rates were frozen.

## 4. Partial fine-tuning passed every gate for ten epochs

2:10–2:55. The left panel gives context from the original Hamiltonian runs.
The center panel shows that partial unfreezing did not push SR validation MSE
through the three-times-ten-to-the-minus-six ceiling. On the right, both tensor
tasks remain below their stricter zero-residual baselines for all ten epochs.
The stop is a consecutive-gate criterion, not selection from one lucky epoch.

## 5. The heads generalize beyond a constant predictor

2:55–3:45. On the locked structures, Born MAE is 0.002369 electron versus a
0.003201 training-mean baseline, a 26 percent reduction. Dielectric MAE falls
by 73 percent. The physical outputs remain constrained: Born satisfies ASR to
float32 accumulation precision, and all dielectric eigenvalues are positive.
This is evidence that the heads learned signal, but only five locked examples
are not enough for a production claim.

## 6. Predicted tensors barely change full-H accuracy

3:45–4:30. On the same five locked structures, direct full-H remains best on
matrix elements at 0.417 meV MAE. The old fixed-LR SR reconstruction is 0.487,
and the predicted-tensor reconstruction is 0.485. So the new tensors make a
small improvement to the SR reconstruction, but they do not close the value
gap. The reconstruction is nevertheless complete and exact on every truth
block.

## 7. SR still wins the response test; learned tensors are neutral

4:30–5:15. The central EPC result remains stable. Direct full-H has 91.93
percent relative L2 error. Both SR pipelines are around 24.4 percent. But the
new geometry-dependent tensor pipeline is slightly worse, not better, than
fixed-reference tensors by 0.077 percentage points. At this sensitivity, the
new heads have neither helped nor damaged the EPC conclusion.

## 8. The two SR curves overlap across the full 2×2×2 grid

5:15–6:00. The q-resolved view shows where the separation occurs. Direct
full-H greatly overestimates EPC strength at finite q. The old fixed-tensor SR
and new predicted-tensor SR curves overlap at every sampled q. Different line
styles and markers are used because the numerical differences are otherwise
too small to see.

## 9. Predicted tensors preserve the SR response distribution

6:00–6:45. The parity view confirms the same hierarchy: direct full-H has a
much broader cloud around the DFT identity line, while both SR variants are
tighter and nearly indistinguishable. The magnitude distributions show that
the tensor change does not alter the current response manifold; both SR runs
retain the same systematic differences from DFT.

## 10. Three statements are supported—and three are not yet

6:45–7:35. The clean conclusion is not that tensor learning failed. The heads
learned held-out structure dependence and the composed physics path works. The
result we do not have is an EPC improvement from geometry-dependent tensors.
We also cannot generalize from ten tensor-training structures, and a 2×2×2 q
grid is not a small-q convergence study. These boundaries matter for how we
describe the result.

## 11. Increase sensitivity before increasing model complexity

7:35–8:45. The immediate priority is a sensitivity experiment, not a larger
network. The geometry-dependent correction may be invisible because the EPC
displacement is only five microångströms. I would run a controlled displacement
ladder, then add tensor labels chosen for response variation, and use matched
initializations for the causal ablation. Dense q should come only after the
effect is resolvable above finite-difference uncertainty.

## 12. What we see so far

8:45–9:30. The tensor-head implementation is successful as a prototype. It
learns held-out tensor variation, preserves Hamiltonian stability, and closes
the intended composed architecture. The current scientific outcome is that it
does not yet change EPC. The next useful result is to establish whether that
neutrality is physical, data-limited, or hidden below the finite-difference
sensitivity. Then invite discussion on the next compute campaign.
