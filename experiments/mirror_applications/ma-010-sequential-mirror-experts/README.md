# MA-010 — sequential Mirror experts

Status: **FAIL** at the pre-fresh development gate.  
Evidence lane: mechanism / storage / compute.  
Branch: `research/ma-010-sequential-mirror-experts-20261007`.  
Protocol base: `d1d7eec6e4b059e077a7ffa7116a26adba0c670e`.

## H — Hypothesis

For ordered compositions of two expert transformations, one shared linear expert with per-role input/output Givens coordinates can preserve four independent expert functions at lower actual payload bytes, including order-sensitive compositions.

## T — Test

The synthetic task applies one of four 8×8 linear transformations twice, in a supplied order. All 16 ordered role pairs are uniform. The aligned teacher is generated from one shared matrix under role-specific 2D Givens views. A second independent-matrix teacher mode measures the private-function boundary. Controls are independent experts, hard tying, per-role rank-2 residuals, and scalar gates. Development uses world 100000 to choose one learning rate; fresh worlds are 100001–100003 and stay sealed until the setting and implementation are frozen.

The primary metric is held-out MSE. Five methods ran 1,800 updates × 128 examples in each teacher mode. Development world 100000 selected LR 0.01 from pooled held-out MSE. The audit reports all ordered-pair errors and the output change under swapping sequence order. Actual inference payload bytes include all matrices, residuals, gates, view angles, and config metadata. Runtime and active MAC proxy are reported separately.

## Development result

- The selected learning rate was 0.01 from development world 100000, using pooled held-out MSE across all five methods and both teacher modes.
- On the aligned teacher, Mirror MSE was 7.98e-4 vs independent 3.75e-4 (2.13×); Mirror payload was 2,330B vs 2,663B (0.875×). The preregistered gate required MSE ≤1.25× and payload ≤0.65×, so both conditions failed.
- On aligned data, hard tying was smaller (1,888B) but had 2.16× Mirror's MSE. Rank-2 residual had lower MSE than Mirror (3.43e-4 vs 7.98e-4) at 2,777B; it is a simpler quality counter-control, though 447B larger than Mirror.
- On independent teachers, Mirror did not recover unrelated functions: MSE 5.88e-3 vs near-zero independent fit. This is a private-function boundary under the fixed update budget.
- Order swapping changed aligned teacher outputs (mean squared gap 2.40e-3), confirming the task did exercise sequence order.
- Fresh worlds 100001–100003 remain sealed because the development gate failed. This is not fresh evidence.

## D — Decision: FAIL

The candidate missed both preregistered development access conditions. The small 12.5% byte reduction is insufficient for this protocol, and Mirror was over twice the independent MSE. Do not interpret the fixed-budget screen as a capacity limit.

## Fact / interpretation / hypothesis

- **Fact:** the development gate failed on both quality and actual payload. No fresh worlds were accessed.
- **Interpretation:** this small sequential linear task did not show useful Mirror compression at the registered threshold; rank-2 residual fit the aligned teacher more accurately.
- **Hypothesis:** larger nonlinear expert blocks may reduce serializer overhead and change the frontier, but that requires a separately registered experiment.

## C — Strongest counter-hypothesis

The task is deliberately small and linear, while serialized tensor container overhead is a large fraction of total bytes. This may make a 35% reduction unattainable despite sharing weights. Rank-2 residual also has a quality advantage on this development world.

## U — Unconfirmed

Fresh aligned replication, larger nonlinear experts, learned routing, more sequential steps, near-convergence capacity, optimized kernels, and language quality remain untested.
