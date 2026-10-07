# MA-012 — residual Mirror-MoE

Status: SCREENING; development gate passed and fresh settings frozen.  
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

## D — Decision

Fresh results pending. Development alone is not capacity evidence.

## C — Strongest counter-hypothesis

The aligned teacher explicitly contains the same view-plus-residual structure. A result may reflect teacher construction, while arbitrary expert functions may still require full private matrices.

## U — Unconfirmed

Fresh replication, learned routing, larger/deeper experts, near-convergence capacity, optimized kernels, and language quality remain untested.

## Fact / interpretation / hypothesis

- **Fact:** the development gate passed for Mirror-rank2 and its payload was within 10% of tied-rank2.
- **Interpretation:** adding a small private residual improved the aligned view fit and beat the same-rank hard-tied residual control on development.
- **Hypothesis:** residual plus view codes may recover related roles while arbitrary roles still need private capacity.
