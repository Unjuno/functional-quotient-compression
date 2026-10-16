# GIC-001: gradient-interference intervention results — 2026-10-16

New same-container mechanism experiment, not a rerun of historical AMA/MN checkpoints. No main merge or MA registry/status changes.

[Full experiment report, source and frozen protocol](../../experiments/mirror_applications/research_intake/gic001_gradient_interference_20261016/README.md)

- 704 trained/serialized spline-bank models: 44 development and 660 fresh (5 worlds × 3 initializations × 4 regimes × 11 methods).
- 12 tests passed; all 704 payloads replayed; 11 full-training runs reproduced identical payload hashes. Same-container verification, not external replication.
- Partial-sharing rank2 ordinary training reaches its known empirical global optimum, maximum gap 1.79e-12, yet population MSE remains positive. This isolates a representation limit from optimizer stagnation.
- PCGrad-only rescue failed the preregistered quality gate in all tested regime/architecture comparisons. This is a fixed-setting shared-block Adam experiment, not a general PCGrad refutation.
- Adding private splines selected by conflict improved partial-world mean and worst MSE in 5/5 worlds, but increased actual bytes from 464 to 664.
- At equal 664-byte storage, conflict selection beat random in 5/5 but failed to beat training-loss selection in every world. Conflict-specific selection advantage: FAIL.
- Training-only exhaustive selection of 2 among 8 tasks recovers the true exception pair in 15/15 nested world/init settings. Population mean-MSE world median: 0.0000387 vs conflict 0.057132 and loss 0.051131 at the same bytes. The reference pays for 28 exact fits; it is not a free large-model selector.
- Shared low-rank spline coordinates are an ordinary low-rank dictionary alias. No Mirror-specific novelty, full-KAN/Transformer, natural-data or production-speed claim.

Interpretation: measuring conflict magnitude does not determine which sharing constraint should be removed. The next hypothesis is a cheap estimate of validation-loss improvement per added byte, not conflict-count minimization alone. This next selector is NOT implemented here.

Proofs, variable definitions and dimensional checks: [mathematical contract](../../experiments/mirror_applications/research_intake/gic001_gradient_interference_20261016/MATHEMATICAL_CONTRACT.md).

All raw payloads, metrics and logs remain in the conversation archive `GIC001_FULL_EVIDENCE_2026-10-16.zip`; SHA256 `80927f11426d7927980b3b2d041b2b651379a0bfa247c03ade434fdf76a8896c`.
