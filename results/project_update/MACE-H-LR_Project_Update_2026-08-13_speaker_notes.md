# MACE-H-LR project update — speaker notes

Presentation date: August 13, 2026  
Target length: 10 minutes  
Slides 13–15 are backup.

## 1. MACE-H-LR

0:00–0:30. The project has moved through three layers: I first tested the
model machinery on bulk gold, then built and validated Cartesian
electron–phonon coupling, and the MgO results exposed an architecture issue.
The rest of the week is about correcting that issue with learned Born-charge
and dielectric heads and leaving a reproducible handoff by August 20.

## 2. Three stages changed the question

0:30–1:10. Stage one was the gold machinery test. Stage two was the first
end-to-end Cartesian-AO EPC calculation for MgO. That calculation did not
just produce a metric—it changed the question. The remaining problem is not
whether the plumbing runs; it is whether the training objective represents
the intended composed long-range model.

## 3. The intended model is composed—not a direct full-H fit

1:10–2:00. This is the target architecture. The short-range Hamiltonian and
the two response tensors depend on structure. The analytic long-range term is
then reconstructed from the predicted Born charges and electronic dielectric
tensor. Cartesian EPC is the derivative of this entire composition. A direct
full-H network is a useful baseline, but it is not this model.

## 4. Hamiltonian metrics looked excellent—and hid the response problem

2:00–2:45. Both runs converged cleanly. On 37 held-out snapshots, the direct
full-H model even had the lower matrix-element MAE, and both parity plots look
nearly perfect. If I had stopped at Hamiltonian values, I would have concluded
that the direct full-H baseline was better. EPC tests the local slope with
respect to atomic displacement, which is a stricter observable.

## 5. Cartesian EPC exposed a finite-q failure in direct full-H

2:45–3:35. Across the full 2×2×2 Cartesian-AO tensor, the SR checkpoint is at
24.36 percent relative L2 while direct full-H is at 91.93 percent. At Gamma
they are essentially tied: 6.07 versus 5.82 percent. The failure appears at
nonzero q, where direct full-H both overestimates the coupling strength and
has much larger relative error. This is the central experimental result.

## 6. The machinery is already tested: gold first, then Cartesian MgO EPC

3:35–4:20. I am not starting from an untested code path. Bulk gold exercised
the model machinery first. For MgO, the Cartesian EPC reference uses twelve
converged ABACUS calculations and contains 301,056 complex components. Doubling
the DFT displacement changes the tensor by only 0.00656 percent, and the
Hermiticity residual is 2.9 times 10 to the minus 11 of the tensor peak. These
checks support the finite-difference and indexing pipeline.

## 7. The previous “full-H” run was not the intended long-range model

4:20–5:05. The audit is decisive. Both checkpoints have the same parameter
count and neither contains a Born-charge or dielectric head. Direct full-H is
an independently trained total-H predictor. The SR model gets fixed DFPT
tensors through an external wrapper. Therefore the previous training did not
test the proposed learned-tensor architecture. That is the correction I am
making now.

## 8. The SR advantage is not yet an analytic long-range result

5:05–5:45. The controlled A/B/D decomposition prevents an incorrect causal
claim. A is the SR checkpoint alone, B adds fixed analytic LR, and D is direct
full-H. A and B are visually and numerically identical on this 2×2×2 grid.
The analytic correction is only 0.001606 percent of the residual needed to
match DFT. The current advantage belongs to the SR checkpoint, not yet to the
analytic correction.

## 9. A model can match H while learning the wrong local slope

5:45–6:45. A continuous 25-point Mg-x scan explains the mismatch. Direct
full-H has slightly better raw Hamiltonian MAE, but its central-slope RMSE is
1.67 times worse. In 24.1 percent of matrix elements it is closer in H but
worse in slope. The mechanism is plausible, but because these are independent
training runs, matched-seed repeats or derivative-aware supervision are still
needed to distinguish target design from ordinary training variance.

## 10. Train HSR, Z*, and ε∞ together—then reconstruct full H

6:45–8:00. This is the corrected implementation. A shared MACE-H backbone
feeds three outputs: the short-range Hamiltonian, an atom-resolved Born-charge
head with the acoustic sum rule enforced, and a global symmetric
positive-definite electronic dielectric head. Those tensors feed a
differentiable analytic long-range reconstruction. The training objective
includes direct tensor supervision and full-H reconstruction; if schedule
allows, I also want a finite-difference response-consistency term. The key
ablation is A/B/C/D: SR only, fixed tensors, learned tensors, and direct full-H.

## 11. What I will finish before August 20—and what the group should decide

8:00–9:15. The critical path is: validate the fast tensor-label profile against
the anchor, implement and test the two equivariant heads, run a joint-training
smoke and matched-seed ablations, then evaluate in the order tensor accuracy,
Hamiltonian reconstruction, Cartesian EPC. By August 20 I will leave clean
interfaces, configs, tests, provenance, and a restartable runbook. I need the
group to decide whether the continuation is primarily a paper result or a
reusable platform, whether scope stays MgO-only, and whether the next compute
budget goes to denser q or more tensor-labelled structures.

## 12. Takeaways

9:15–10:00. The machinery and Cartesian EPC pipeline are complete. The key
learning is that the previous direct full-H training was not the proposed
long-range architecture. I am correcting that now with explicit Born-charge
and dielectric heads plus analytic reconstruction. Before I leave, I want to
deliver a tested minimum viable implementation and a handoff aligned with the
group's preferred scientific direction. Then stop and invite questions.

## 13. What the current EPC tensor does—and does not—establish

Backup. The current result is a Cartesian atomic-orbital Hamiltonian
derivative. It does not yet include phonon eigenvectors and frequencies,
electronic eigenvectors, mass factors, or the overlap derivative, and the
2×2×2 q grid is not a converged small-q study. The direction metric still
shows a consistent checkpoint-level difference across Cartesian components.

## 14. The comparison that will isolate the learned-tensor contribution

Backup. A and B exist now and show that fixed LR is negligible on this grid.
D is the current direct full-H baseline. C is the missing learned-tensor case.
B minus A isolates the fixed analytic contribution; C minus A measures the
effect of learned structure-dependent tensors; C versus D answers whether the
composed objective improves the physical response.

## 15. If time compresses, preserve the causal test and the handoff

Backup. If the week compresses, the minimum viable outcome is not the largest
campaign. It is valid tensor labels, constrained heads, one matched A/B/C/D
comparison, and a reproducible handoff. Band- and mode-resolved EPC, a large
tensor dataset, and dense-q production can follow after ownership is clear.
