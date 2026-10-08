# MA-012 — residual Mirror-MoE

Status: **PROMISING**; registered aligned fresh gate passed 3/3.
Evidence lane: mechanism / storage / compute.
Branch: `research/ma-012-residual-mirror-moe-20261007`.
Base commit: `099fae1d2e66d940ad9a6b451889242ea7e81d9a`.

## H — Hypothesis

A small per-role output residual added to a shared nonlinear expert with Mirror input/output views can recover aligned expert behavior more accurately than either Mirror-only sharing or an equally small residual on hard tying, while using fewer actual bytes than independent experts.

## T — Test

Four balanced oracle-routed 16→32→16 GELU experts. The aligned teacher contains one shared MLP under Givens input/output views plus rank-1 private output residuals. Independent MLPs are the upper/private-function control. Controls include hard tying, tied rank-1/rank-2 residuals, Mirror with no residual, and Mirror with rank-1/rank-2 residuals. Development world 120000 selects the learning rate and residual rank; fresh worlds 120001–120003 remain sealed until the preregistered gate is met.

Actual serialized inference payload includes full MLP weights, all Givens coordinates, private residual factors, and metadata. Report active compute, training wall, and eager CPU inference throughput separately.

## Development result

- Selected learning rate: 0.003. Selected Mirror private rank: 2, chosen by aligned held-out MSE on development world 120000.
- Mirror-rank2 MSE was 3.54e-4 vs 6.96e-4 full independent (0.508×); actual payload was 8,347B vs 18,218B (0.458×). The preregistered pre-fresh gate passed.
- Mirror-rank2 beat tied-rank2 MSE (3.54e-4 vs 5.52e-4) at 6.5% higher payload, inside the 10% comparison margin. Rank-1 Mirror used 7,579B and MSE 3.75e-4.
- On independent experts, Mirror-rank2 MSE was 1.45e-2 vs 8.10e-4 for full independent weights; private/richer functions remain a clear boundary.
- Fresh worlds 120001–120003 use the frozen LR and rank selection. Do not retune from fresh results.

## D — Decision: PASS on registered synthetic gate; PROMISING overall

The selected Mirror-rank2 variant passed useful-sharing and matched-rank Mirror-specific gates in all three fresh aligned worlds. It is not a strict runtime Pareto win: median eager CPU throughput was 0.365M examples/s vs 0.701M for tied-rank2 and 0.720M for independent experts; median training wall was 2.48s vs 1.37s and 1.00s. Its compute proxy was 1,120 MACs plus about 12 Givens coordinate FLOPs per example, compared with 1,120 MACs for tied-rank2 and 1,024 for independent. The teacher deliberately shares a Givens orbit and rank-1 residual; on independent teachers Mirror-rank2 MSE was 0.0135–0.0139 vs 0.00068–0.00074 for full independent.

All 42 deterministic fresh rows replayed exactly (timing excluded); eight pre-access freeze hashes matched commit `cfee04ee5047b9dcfe49ad47405e0840b2bd828b`. A first replay attempt computed all rows but failed while opening a missing temporary output directory; the same frozen run then completed and replayed exactly after the directory was created. This affected result persistence only.

## C — Strongest counter-hypothesis

The aligned teacher explicitly contains the same view-plus-residual structure. A result may reflect teacher construction, while arbitrary expert functions may still require full private matrices.

## U — Unconfirmed

Learned routing, larger/deeper experts, near-convergence capacity, optimized kernels, and language quality remain untested.

## Fact / interpretation / hypothesis

- **Fact:** fresh Mirror-rank2 MSE ratios to full independent were 0.438, 0.393, and 0.597, at 8,347B vs 18,218B (0.458×). Against tied-rank2, bytes were 6.5% higher and MSE 18.7–33.1% lower in each world.
- **Interpretation:** the small private residual and view coordinates recovered aligned roles at a useful storage frontier; arbitrary roles still needed private/richer capacity and CPU execution was slower.
- **Hypothesis:** this view-plus-residual pattern may help trained expert pools only when roles share a low-description orbit.
