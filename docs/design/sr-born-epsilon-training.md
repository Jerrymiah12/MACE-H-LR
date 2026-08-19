# SR + Born Charge + Dielectric Training Plan

## Intended model

The required model is one shared, multi-head network:

$$
R \longrightarrow \{H_{\mathrm{SR}}(R), Z^*(R), \epsilon_\infty(R)\},
$$

with the total Hamiltonian reconstructed analytically:

$$
H_{\mathrm{total}} =
H_{\mathrm{SR}} + H_{\mathrm{LR}}(R,Z^*,\epsilon_\infty).
$$

There must be **no learned direct-$H_{\mathrm{full}}$ head** in this model.

## Current repository state

- `workflows/training/train_sr.ini` trains only $H_{\mathrm{SR}}$.
- `workflows/training/train_full.ini` is a separate direct-$H_{\mathrm{full}}$ baseline.
- Neither existing checkpoint has Born-charge or dielectric heads.
- The current network returns node features and one Hamiltonian edge output.
- The current training loop computes only one masked Hamiltonian MSE.
- The current EPC LR wrapper loads fixed tensors from `$MACEH_DATA_ROOT/reference`.

This cannot be corrected only through an INI setting. The data loader, model,
loss calculation, checkpoint handling, inference, and EPC reconstruction must
be changed together.

## Available data

The completed fast tensor-DFPT campaign contains 17 labelled 54-atom
snapshots:

- 10 tensor-training snapshots
- 2 tensor-validation snapshots
- 5 locked tensor-test snapshots

Each contains:

- born_effective_charges.npy with shape (54, 3, 3)
- dielectric_infinity.npy with shape (3, 3)

All 17 passed finite-value, acoustic-sum-rule, dielectric-symmetry,
positive-definiteness, atom-ordering, and species-sign checks.

The fast-versus-anchor comparison found:

- Born MAE: 0.0049636 e
- Born maximum absolute error: 0.0289794 e
- Born relative L2 error: 0.6055%
- Dielectric relative L2 error: 0.7267%

The dielectric result exceeded the original 0.5% convergence gate. These
should therefore be described as fast-prototype labels, not fully converged
production labels.

No Hamiltonian regeneration is needed for the prototype. The Hamiltonian head
can use all 330 existing training snapshots, with tensor losses masked on
unlabelled structures.

### Tensor split

Training:

- snapshot_000386
- snapshot_000011
- snapshot_000033
- snapshot_000063
- snapshot_000175
- snapshot_000244
- snapshot_000275
- snapshot_000312
- snapshot_000353
- snapshot_000354

Validation:

- snapshot_000028
- snapshot_000327

Locked test:

- snapshot_000080
- snapshot_000229
- snapshot_000295
- snapshot_000333
- snapshot_000334

## Required changes

### 1. Dataset loading

Update maceh/data.py so structures may optionally contain:

- born_target: (N_atoms, 3, 3)
- epsilon_target: (3, 3)
- has_born_label: Boolean
- has_epsilon_label: Boolean

The Hamiltonian target remains hamiltonians_sr.h5.

Load tensors through a sidecar manifest keyed by snapshot ID and attach the
small targets after loading the existing graph cache. This avoids rebuilding
the multi-gigabyte graph cache.

The loader must verify snapshot and split membership, atom count/order,
coordinate provenance, shapes, finite values, Born ASR, dielectric symmetry,
and positive dielectric eigenvalues. Test snapshots must never be sampled for
training or checkpoint selection.

### 2. Multi-head architecture

Refactor maceh/maceh.py so the shared MACE encoder feeds:

1. The existing edge-equivariant $H_{\mathrm{SR}}$ head.
2. A per-atom equivariant $Z^*$ head.
3. A pooled structure-level equivariant $\epsilon_\infty$ head.

Preserve backward compatibility for Hamiltonian-only checkpoints. Prefer an
explicit multi-head wrapper or encoder interface rather than silently changing
the output type of every legacy model.

#### Born head

The labels are not exactly symmetric; the largest observed antisymmetric
contribution is about 0.02881. Preserve all nine components. Use the
rank-two decomposition

$$
Z^*: 0e \oplus 1e \oplus 2e.
$$

Enforce the acoustic sum rule by projection:

$$
Z^*_\kappa \leftarrow Z^*_\kappa
-\frac{1}{N}\sum_{\kappa'}Z^*_{\kappa'}.
$$

#### Dielectric head

The global symmetric dielectric tensor uses

$$
\epsilon_\infty: 0e \oplus 2e.
$$

Pool atom features permutation-invariantly. Enforce positive definiteness with
a differentiable, rotationally consistent eigenvalue floor or equivalent
positive-definite construction.

#### Residual targets

Because both tensors vary only slightly, predict residuals:

$$
Z^*(R)=Z^*_0+\Delta Z^*(R),
$$

$$
\epsilon_\infty(R)=\epsilon_{\infty,0}
+\Delta\epsilon_\infty(R).
$$

Use equilibrium or tensor-training means as the fixed baselines.

### 3. Multitask loss

Replace the single loss in maceh/kernel.py with

$$
L=L_{H_{\mathrm{SR}}}+\lambda_ZL_{Z^*}
+\lambda_\epsilon L_{\epsilon_\infty}.
$$

Requirements:

- Hamiltonian loss uses every SR training snapshot.
- Born and dielectric losses use only their labelled snapshots.
- Normalize each target from training-split statistics before combining losses.
- Log all three losses and physical-unit metrics separately.
- Sample labelled structures regularly without discarding ordinary H data.
- Monitor or limit auxiliary gradient contributions to the shared encoder.
- Start with unit weights after unit-variance normalization, then inspect
  per-task encoder gradient norms before finalizing the weights.

### 4. Configuration and checkpoints

Add `workflows/training/train_sr_tensors.ini` with:

- Hamiltonian target set to SR
- Born and dielectric prediction enabled
- tensor manifest and split IDs
- independent loss weights
- separate backbone and head learning rates
- freeze/unfreeze controls
- multitask early-stopping settings

Checkpoint metadata must record:

    model_type = sr_born_epsilon
    direct_full_h_head = false
    analytic_lr_reconstruction = true
    tensor_label_manifest = path and SHA-256

Initialize the encoder and H head from:

    <MACEH_RUNS_ROOT>/run_sr/2026-08-06_15-56-09_sr/best_model.pkl

That checkpoint reached epoch 1681 with validation MSE
2.866399023414435e-6. The new heads start separately. Loading should report
missing/unexpected keys and permit only the two new heads to be absent.

### 5. LR reconstruction and EPC

Update maceh/epc/mgo_long_range.py so LR reconstruction can consume tensors
predicted by the model instead of always loading fixed reference arrays.

Provide two explicit EPC modes:

- **equilibrium_frozen:** Predict tensors once at equilibrium and hold them
  fixed during the finite-difference sweep. This should be the default
  comparison with the existing analytic LR formulation.
- **geometry_dependent:** Recompute tensors at every displaced geometry,
  including effective $dZ^*/dR$ and $d\epsilon_\infty/dR$ contributions.

The mode and tensor provenance must be saved with every EPC result.

Add an identity test:

$$
H_{\mathrm{reported}} =
H_{\mathrm{SR,pred}}+
H_{\mathrm{LR}}(R,Z^*_{\mathrm{pred}},
\epsilon_{\infty,\mathrm{pred}})
$$

to machine precision. Also assert that a new checkpoint has both tensor heads
and no direct-full-H output head.

## Training sequence

### Stage 1: Smoke tests

- Attach and audit all 17 tensor labels.
- Confirm splits and atom ordering.
- Run one forward/backward/optimizer step.
- Confirm unlabelled graphs produce only H loss.
- Test rotational equivariance.
- Test Born ASR and dielectric symmetry/positivity.
- Test checkpoint save/load and reconstructed-H identity.
- Do not launch a long run until every test passes.

### Stage 2: Head-only warmup

- Load the best SR checkpoint.
- Freeze the encoder and H head.
- Train only tensor heads for 50-100 epochs.
- Confirm H outputs remain unchanged.

Expected time: less than one hour.

### Stage 3: Partial fine-tuning

- Unfreeze the final interaction block.
- Use a much lower backbone LR than head LR.
- Train for about 100-200 epochs.
- Reject checkpoints that materially degrade SR validation loss.

### Stage 4: Short joint fine-tuning

- If stable, optionally unfreeze the full encoder.
- Use a small backbone LR.
- Train another 100-300 epochs.
- Stop on a plateau or after all gates pass for ten evaluations.

### Stage 5: Locked test

Evaluate once on the five tensor-test snapshots and report:

- SR Hamiltonian MSE and MAE
- Born component MAE and RMSE
- dielectric component MAE and RMSE
- Born ASR residual
- dielectric eigenvalue range
- reconstructed total-H error
- EPC error in equilibrium-frozen mode
- EPC error in geometry-dependent mode

## Baselines and stopping gates

A model trained from only ten tensor structures must beat a constant
training-mean predictor before it can be said to learn geometry dependence.

| Split | Born MAE | Born RMSE | Dielectric MAE | Dielectric RMSE |
|---|---:|---:|---:|---:|
| Training | 0.003008 e | 0.004347 e | 0.002332 | 0.003478 |
| Validation | 0.002778 e | 0.003950 e | 0.001532 | 0.002082 |
| Test | 0.003201 e | 0.005011 e | 0.001466 | 0.002025 |

Suggested prototype gates:

- H validation MSE no worse than approximately 3.0e-6.
- Born validation MAE below 0.002778 e.
- Dielectric validation MAE below 0.001532.
- ASR residual near machine precision.
- All dielectric eigenvalues positive.
- All gates pass for ten consecutive evaluations.

Use plateau/patience as the main stopping rule. A raw combined-loss threshold
is unsafe unless all tasks are normalized because their units and element
counts differ.

Two validation structures are enough for wiring decisions, not a production
claim. Keep the five test structures untouched until all choices are frozen.

## Runtime estimate

Measured full-Hamiltonian epochs take about 95-101 seconds.

| Work | Estimated time |
|---|---:|
| Implementation and verification | 1-2 working days |
| Head warmup | Less than 1 hour |
| 200 joint epochs | About 5.5 hours |
| 500 joint epochs | About 13-14 hours |
| Full retraining from scratch | About 45-84 hours |

The recommended prototype is checkpoint initialization plus 200-500
fine-tuning epochs: approximately 6-14 GPU hours after implementation passes
its tests.

## Dataset decision

The existing 17 labels are sufficient to:

- implement and validate the architecture;
- test masked multitask training;
- verify predicted-tensor LR reconstruction;
- compare against constant tensor baselines;
- run an initial EPC comparison.

They are not enough to establish robust production generalization. If the
prototype succeeds, add configurations selected for displacement type,
amplitude, and configuration-space coverage. All 400 tensor calculations are
not required before the prototype, but more labels will likely be required for
publication-quality geometry-dependent response results.

## Definition of done

The pipeline is complete only when:

1. The checkpoint contains SR Hamiltonian, Born, and dielectric heads.
2. It contains no direct-full-H head.
3. H loss uses the complete frozen SR training split.
4. Tensor losses use only the declared tensor-training split.
5. Validation and test tensor structures do not leak into training.
6. Predicted Born tensors satisfy ASR.
7. Predicted dielectric tensors are symmetric and positive definite.
8. Total H is constructed only as predicted SR plus analytic LR.
9. EPC results record frozen versus geometry-dependent tensor treatment.
10. The locked test report gives separate H, tensor, reconstructed-H, and EPC
    errors and compares tensor heads with the constant baseline.
