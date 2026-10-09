# MA-508 — Activation-addition Mirror basis

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / FUNCTIONALITY  
Base commit: `f85b2963` (integrated through MA-504)  
Prior art: PA97 (Representation Engineering), PA98 (Activation Addition)

## Hypothesis

H: A shared rank-4 activation-steering basis plus a per-behavior low-description Givens View can store and apply a bank of fixed-norm steering behaviors with lower actual bytes than explicit steering vectors and generic per-behavior rank-4 coefficients, while retaining on-target efficacy and off-target specificity.

## Mirror insertion

> **Mirror insertion:** this experiment adds two per-behavior Givens angles `m_i` to one shared activation-addition vector so that multiple logical behavior directions are represented by Views of one physical steering code rather than storing one full vector per behavior.

- Physical object: one shared synthetic 64×4 orthonormal representation basis and one rank-4 seed vector.
- Mirror coordinate: two per-behavior Givens angles; optional FP16 serialization is charged.
- Logical multiplicity: 64 behavior vectors applied as additive activation interventions.
- Simple controls: direct full vectors; generic shared basis plus FP32 or FP16 per-behavior coefficients.
- Task reads out behavior efficacy and off-target effects as well as vector reconstruction.

## Frozen protocol

- D=64, rank=4, 64 behavior vectors; 20% of behavior IDs are held out from any basis/seed fitting.
- Fresh worlds: 50820/50821/50822 × seeds 0/1/2. Development worlds: 50800/50801 × seeds 0/1/2.
- Two regimes: fixed-norm two-plane Givens orbit (`rho=0`) and 10% coefficient-space private residual (`rho=.1`).
- `B` is a known shared synthetic basis charged in each low-rank payload. The seed code is fitted using development behaviors only. Fresh test behavior angles are not used to tune anything; their compact codes are their stored intervention state.
- A fixed random set of 32 unit readout probes is frozen per world. Efficacy is target-readout activation change relative to the explicit-vector teacher. Off-target drift is RMS change under 31 non-target probes. Also report vector NRMSE and exact payload bytes.

## Methods

1. Explicit full 64D steering-vector bank (exact control).
2. Shared basis + per-behavior FP32 rank-4 coefficient table.
3. Shared basis + per-behavior FP16 coefficient table.
4. Mirror shared basis + one FP32 seed code + per-behavior two FP16 Givens angles.
5. Mirror plus per-behavior FP16 rank-4 private residual fallback; this maps the private-state boundary and charges every residual.

## Gates

- PASS for aligned compression if every fresh `rho=0` world has vector NRMSE ≤0.01, efficacy ≥0.99, off-target drift ≤0.01, and payload ≤80% of both the explicit bank and the cheapest generic coefficient control.
- PROMISING if it beats explicit storage/quality but a generic coefficient control matches the rate-quality point, or if CPU runtime is worse.
- FAIL if aligned quality or byte gates fail or generic coefficients dominate.
- `rho=.1` is a boundary test; record the residual fraction and cost needed to recover efficacy.

## Boundaries

This is a controlled synthetic Activation Addition screen, not a pretrained-language-model behavior-steering result. The aligned orbit is deliberately favorable to Mirror. Codes for the 64 test behavior directions are paid inference state. Logical behavior count is not capacity.
