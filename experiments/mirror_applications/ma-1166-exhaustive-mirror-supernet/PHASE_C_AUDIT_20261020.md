# MA-1166 Phase C: corrective parameterization audit (2026-10-20)

**Status: forensic check of earlier fixed-update synthetic observations, not a new positive experiment.** Original Phase C protocol and historical results are not altered.

## Critical implementation limitation

The container model owns **one trainable 253-float tensor**, not 63 independent trained 110-float tensors. The 253 coordinates consist of **64 global shared coefficients, 63 per-edge sigmoid gates, and 126 per-edge Mirror orientation coefficients (2 each)**. Therefore, **189/253 (~74.7%) trainable coefficients are edge-specific**, and the physical globally shared core is only 64 coefficients. The generated virtual weights (63 edges × 110 = 6,930 logical float weights) are recomputed and executed as **63 ordinary virtual 6→8→6 tanh MLP calls**. The mirror decoder and fixed edge basis are seeded/fixed, not an optimizable many-to-many physical-address-to-virtual-edge mapping.

The implementation includes all 21 forward i<j node-event pairs times 3 Mirror views, but it does **not** learn arbitrary new pointer integer indices, full topology rewiring, input-dependent routing, or free physical-to-virtual mapping. Training is 400 Adam updates, and there is no pruning or gate sparsity penalty.

**Conclusion:** This screen does NOT faithfully test the proposed fully optimizable Mirror/Loop/pointer correspondence graph. An equal-253-float comparison to a directly trained 6→19→6 MLP is not proof against that broader hypothesis.

## New oracle representability audit

The *aligned* teacher uses only **11 of the 63 edge/view instances**. Copying its shared core and corresponding Mirror codes into the student, while setting its 11 oracle connection logits to +20 and the other 52 logits to -20, recreates the teacher on held-out fresh inputs to FP32 precision. Oracle information is used *only for this representability diagnostic*, not in prior student training.

| aligned seed | oracle fresh NMSE | trained full Mirror fresh NMSE | directly trained native253 NMSE | learned gate mean at 11 teacher edges | learned gate mean at other 52 edges |
|---|---:|---:|---:|---:|---:|
| 421 | 2.67e-17 | 0.222526 | 0.051556 | 0.4953 | 0.4030 |
| 433 | 5.64e-17 | 0.169385 | 0.028909 | 0.4527 | 0.5588 |
| 447 | 2.37e-17 | 0.241742 | 0.036426 | 0.5046 | 0.5667 |

Maximum oracle output absolute error: 4.77e-7 in each seed. This **refutes an expressivity-impossibility explanation on the aligned teacher**; the existing function class contains exact-reproduction solutions, but the trained no-prune model did not discover them under the frozen 400-update optimization. The independent-teacher world is not covered by this oracle claim.

## Gradient diagnosis (single first-train-batch gradient at trained checkpoint)

| seed | shared-core grad L2 | edge-gate grad L2 | Mirror-code grad L2 |
|---|---:|---:|---:|
| 421 | 0.10390 | 0.002588 | 0.002067 |
| 433 | 0.25531 | 0.006330 | 0.001806 |
| 447 | 0.31676 | 0.005112 | 0.002568 |

The one-batch group norm contrast is diagnostic only; it is not a trajectory-wide proof of gradient starvation.

## Mandatory follow-on isolation

1. Oracle-known 11 edge **indices only** (not teacher weights or codes) vs hidden-edge graph discovery, at matched training budget.
2. Freeze/unfreeze shared core, gates, and Mirror codes separately; assess optimization failure and gate identifiability.
3. Replace fixed source-to-virtual addressing with a trainable many-to-many mapping, count all addressing bytes and compute.
4. Compare native and non-Mirror controls at the same learned coefficient, actual byte, and active compute budgets.
5. Never promote an aligned-oracle reproduction to a general capacity or efficiency proof.

**Evidence scope:** Verified container artifacts `/mnt/data/ma1166_no_prune_20261020/source/model.py`, `audit_parameterization.py`, `AUDIT_PARAMETERIZATION.json`, 36 Phase C checkpoints, and passing 8/8 model tests. The underlying code/checkpoints are **not yet uploaded to this repository branch**, so this note is a branch-tracked corrective audit, not a claim that GitHub alone enables full replay.
