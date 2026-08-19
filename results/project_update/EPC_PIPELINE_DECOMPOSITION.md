# EPC A/B/D pipeline decomposition

## Decisive result

The analytic LR wrapper is enabled and uses the frozen reference tensors, but
it is numerically negligible on this 2×2×2 EPC grid.  The EPC advantage comes
from the independently trained **SR-target checkpoint itself**, not from adding
the analytic LR term during EPC evaluation.

| Case | Definition | Relative L2 | Complex MAE (eV/Å) | Cosine |
|---|---|---:|---:|---:|
| A | SR checkpoint only | 24.355916% | 0.267903 | 0.970016 |
| B | SR + analytic LR, fixed DFT/DFPT tensors | 24.355837% | 0.267903 | 0.970016 |
| D | independent direct Full-H checkpoint | 91.928323% | 0.552189 | 0.730601 |

Going from A to B changes relative L2 error by only
**-0.00007840 percentage points**.  The
analytic correction `B − A` has L2 norm
`0.00755629 eV/Å`, only
**0.001606%** of the
`DFT − A` residual norm (`470.405 eV/Å`).
Its cosine with the needed residual is 0.2004.

## Why the fixed LR contribution is tiny here

The analytic definition uses `lambda = 0.35 Å⁻¹`.
For the 2×2×2 displacement supercell, the smallest nonzero reciprocal vector
has magnitude `1.269777 Å⁻¹`; after
dielectric screening, even the largest sampled Ewald weight is only
`1.890706e-05`.  `G=0` is excluded
by definition.  A standalone LR-only finite difference gives norm
`0.00755629 eV/Å` both
before and after gauge projection (norm ratio
`1.000000000`), so
the common gauge projection is not suppressing it.

## Where direct Full-H fails

At Gamma, direct Full-H is slightly better than A:
5.825% relative L2 versus
6.069%.  Its failure appears at the
nonzero q points: Full-H spans 55.84%–116.98%
error, versus 24.48%–26.13% for SR-only.
This localizes the discrepancy to the learned displacement response away from
Gamma, not to a missing `dZ*/dR` or `d epsilon/dR` path.

## Interpretation

* Full-H is not a composed SR + learned-tensor model; it is a separate direct
  Hamiltonian fit.
* There are no predicted Born-charge or dielectric tensors to detach.
* Case C with learned tensors cannot be evaluated from these checkpoints.
* The earlier causal wording that “analytic LR reconstruction removes the
  Full-H error” is not supported by this decomposition.  The accurate wording
  is that the **SR-target checkpoint has the lower EPC error**.
* The next response scan should therefore test why two near-identical
  equilibrium Hamiltonian fits develop very different slopes, especially at
  nonzero q.  Matched-seed repeat training would also test whether this is a
  target effect or ordinary training variance.
