# MA-690 — Orbit/private frontier across View families

Status: SCREENING  
Evidence lane: MECHANISM / OPERATOR-SPACE / ACTUAL BYTES / COMPUTE  
Base commit: `53040b26`

## Hypothesis

H: Across multiple target generator families, one shared matrix plus a compact family View code and a low-rank private residual can reach lower normalized operator error at lower complete serialized bytes than the best residual-only SVD bank at the same byte budget.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-task coordinate `m` over a shared family-specific operator basis around a common matrix `W`, so that multiple logical task matrices can be represented without storing every full matrix independently.

The logical functions are eight task-specific linear operators. The native control stores the same direct family parameters that define `m`; this tests Mirror-specific attribution as well as the shared/private frontier. A shared code plus residual that wins over residual-only is still not a Mirror-specific result if the native parameterization exactly matches it.

## Prior-art delta and controls

PA19 (IA3), PA17 (BatchEnsemble), PA18 (VeRA), and PA21 (BOFT) are direct parameterization controls. Each candidate family is evaluated against every generator bank, including a family-neutral bank. Controls are independent full matrices, native family coordinates, native family plus private residual, and residual-only truncated SVD chosen at the largest rank whose actual serialized payload does not exceed the candidate payload.

## Frozen synthetic scope

Use one frozen 16x16 backbone, 8 logical tasks, two independent development seeds, seven target generators (IA3, BatchEnsemble, VeRA, low-rank, Givens, BOFT, and family-neutral), and alpha amplitude mixtures `{0,.25,.5,.75,1}`. Every structured family is represented by its local operator-space basis around the shared backbone; Givens/BOFT are explicitly tangent screens, not full nonlinear finetuning. Structured and private components are normalized to equal Frobenius norm; the private residual is orthogonalized against the union of all structured spans. Cross-evaluate all six family coordinate systems on all seven target banks.

Sweep private residual ranks `{0,1,2,4,8,full}`. Fit coordinates by deterministic least squares and private factors by truncated SVD; this is an oracle representation screen with zero optimizer updates, not a learned adaptation or downstream task benchmark. Report operator reconstruction error and held-out isotropic-probe error, exact serialized bytes, per-task amortization, residual-only matched-budget error, matrix-decode operation proxy, per-query apply proxy, fit/serialization wall time, and in-memory replay.

## Gates and decision

Useful shared/private frontier point: normalized operator error <=.02, actual bank bytes <=75% of independent full matrices, and decode/apply proxy <=2x the independent bank. A family-level generic storage result additionally needs lower error than the same-byte residual-only SVD control on at least three alpha values in at least two target generators.

Mirror-specific PASS requires a strict Pareto improvement over the corresponding native family coordinate control. Exact function/compute/byte alias means FAIL for Mirror-specific value; fresh seeds remain sealed. If a family fails its own-generator exact-reconstruction calibration at full residual rank or serialization replay fails, classify the affected comparison NOT ESTABLISHED and investigate before cross-family interpretation.

## Limits

This experiment measures local linear operator-bank representation. It does not establish language quality, learned adaptation, task generalization, full nonlinear Givens/BOFT behavior, hardware kernel speed, or practical amortization under a pretrained base already resident in memory.
