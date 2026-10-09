# MA-463 status

- Status: FAIL
- Branch: `research/ma-463-factorized-task-layer-position-20261009`
- Base commit: `add72cbe`
- Initial protocol freeze: `905eed28`
- A1 deterministic-init freeze: `f5f9485a`
- Development selected 500 updates
- Valid fresh worlds: 46320–46322 × seeds 0–2
- Initial worlds 46310–46312: exploratory, excluded for unkeyed stochastic initialization

## H / T / D / C / U

- H: Factorized Mirror codes generalize task × layer × position composition while reducing adapter payload.
- T: Rank-one 8×4×3 synthetic adapter tensor, 72 train and 24 held-out combinations; HyperFormer MLP, Mirror product codes, generic CP, additive and independent controls.
- D (Fact): N96 held-out NRMSE Mirror/CP 0.3064, HyperFormer 0.1559, additive 0.3775, independent 1.75e-7. Bytes/combo: 33.55 Mirror/CP, 55.59 HyperFormer, 25.72 additive, 39.05 independent. Mirror is 60.4% of HyperFormer bytes and exactly equals CP hashes/results. One fresh world has Mirror mean NRMSE 0.919 vs HyperFormer 0.149.
- D (Interpretation): Both quality and strict byte gates miss; the coordinate product is ordinary CP and showed seed-sensitive optimization.
- C: The teacher is exactly a rank-one CP tensor, and CP explains the factorized code.
- U: Natural adapters and robust higher-rank fitting remain unknown.

## Next action

Commit checked evidence and continue to MA-464.

## Blockers

None.
