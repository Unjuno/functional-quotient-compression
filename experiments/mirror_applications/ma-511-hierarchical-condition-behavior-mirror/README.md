# MA-511 — Hierarchical condition × behavior Mirror codes

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / HELD-OUT COMPOSITION  
Base commit: `25c4f935`  
Prior art: PA101 (Conditional Activation Steering)

## Hypothesis

H: A condition-specific Mirror View composed with a shared behavior code predicts held-out condition-behavior interventions with lower bytes than storing a separate intervention for every pair, while matching generic full-matrix layer factorization.

## Mirror insertion

> **Mirror insertion:** this experiment adds a condition coordinate `m_c` as a Givens View over a per-behavior rank-4 code, so an unseen condition×behavior combination can be expressed without storing its pair-specific activation vector.

- Shared object: fixed 64×4 activation basis plus one code per behavior.
- View: two Givens angles per condition.
- Logical objects: 12 conditions × 10 behaviors; hold out 20% of pairs while each condition and behavior remains represented in training.
- Controls: explicit pair table, hard behavior tying, diagonal factorization, generic full 4×4 condition matrix × behavior code.

## Frozen protocol

- D=64, rank=4, C=12, B=10.
- Target coefficient function: `c[c,b]=R(theta_c) z_b`; `rho=.1` adds coefficient-space private residual.
- Development worlds 51100/51101 × seeds 0/1/2; choose common Adam steps from {400,800} based on Mirror held-out NRMSE.
- Fresh worlds 51120/51121/51122 × seeds 0/1/2. Fresh settings are locked before evaluation.
- Train all factorized models only on observed pairs; evaluate all held-out combinations. The explicit pair table is an oracle storage upper control, clearly labeled.

## Gates

- PASS at `rho=0` if every fresh world has held-out NRMSE ≤0.05, Mirror bytes ≤80% of the explicit pair table, and Mirror lies on a better quality/byte point than generic full condition matrices.
- PROMISING if pair-table compression and held-out quality pass but generic factorization matches Mirror or runtime is worse.
- FAIL if held-out quality/bytes fail or simpler factorization dominates.
- Report actual torch-serialized bytes, held-out/observed NRMSE, optimizer updates, MAC proxy and decode latency.

## Boundaries

Synthetic Givens-aligned condition/behavior functions only. No pretrained LM, natural behavior composition or independent-capacity claim. `rho=.1` maps the off-orbit/private boundary.
