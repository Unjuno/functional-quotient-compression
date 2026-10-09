# MA-292 — Task-vector Mirror basis

## H — Hypothesis

A shared basis and compact task coordinates can reconstruct held-out task functions and additive compositions with lower bytes and less interference than raw task vectors; PCA/SVD and generic coefficients may explain the result.

## T — Conditions

Synthetic 256-dimensional linear task deltas from a shared rank-4 basis. Twelve task identities expose 64 support examples; four identities are held out; eight composition pairs are evaluated on separate queries. Fresh worlds 29210–29212 × seeds 0–2; 72 rows. Compared PCA basis, Mirror coordinates, generic coefficients, raw task vectors, ordinary vector addition and independent upper. Basis fitting uses train task deltas; held-out coefficients use only support input/output examples. Fresh protocol/source commit `5cfa9d2b` preceded fresh. CPU only.

## D — FAIL for Mirror-specific claim; shared-basis compression replicated

| Split | Method | Mean NRMSE | Payload B | Fit seconds |
|---|---|---:|---:|---:|
| heldout_tasks | raw_vectors | 0 | 4,159 | 0.000014 |
| heldout_tasks | pca_basis | 3.437437e-07 | 4,356 | 0.000408 |
| heldout_tasks | mirror_basis | 1.991657e-07 | 4,359 | 0.000298 |
| heldout_tasks | generic_coeff | 1.991657e-07 | 4,360 | 0.000329 |
| heldout_tasks | task_addition | 0 | 12,354 | 0.000037 |
| heldout_tasks | independent_upper | 0 | 4,165 | 0.000012 |
| heldout_compositions | raw_vectors | 0 | 8,255 | 0.000012 |
| heldout_compositions | pca_basis | 3.381385e-07 | 4,356 | 0.000580 |
| heldout_compositions | mirror_basis | 2.005201e-07 | 4,359 | 0.000495 |
| heldout_compositions | generic_coeff | 2.005201e-07 | 4,360 | 0.000518 |
| heldout_compositions | task_addition | 0 | 12,354 | 0.000097 |
| heldout_compositions | independent_upper | 0 | 8,261 | 0.000013 |

Fact: held-out task recovery: Mirror/generic/PCA NRMSE ≈2.0–3.4e-7 at 4,356–4,360 B. Held-out compositions: ≈2.0–3.4e-7 at the same payload. Raw/independent upper are exact but store ~4,159–8,261 B; ordinary task addition is exact and uses 12,354 B in the charged representation. Mirror is 3 B larger than PCA and 1 B smaller than generic coefficients.

Interpretation: a shared rank-4 task-vector basis recovers the held-out synthetic task family and additive compositions with substantially fewer bytes than raw/additive storage. PCA and generic coefficients are equally effective and slightly smaller; no Mirror-specific gain is established.

## C — Strongest counter-hypothesis

Task deltas were generated from an exactly rank-4 shared basis and compositions are exactly additive. This is a favorable synthetic subspace problem, not evidence for natural task-vector arithmetic or model behavior.

## U — Unknown

No pretrained model task arithmetic, nonlinear task interactions, human-evaluated composition quality, or larger task basis frontier was tested.
