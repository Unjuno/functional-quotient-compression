# MA-005 — signed Mirror expert mixture

Status: **PROMISING** (aligned synthetic mechanism and storage; CPU runtime regression).
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-005-signed-mirror-mixture-20261007`.
Base commit: `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`.

## H — Hypothesis

A signed mixture of four Givens-coordinate views of one shared expert matrix can reproduce an aligned four-expert mixture teacher with fewer actual serialized bytes than independent signed full experts. Independently parameterized expert mixtures should expose a private-parameter boundary.

## T — Test

Synthetic 16D-to-12D linear experts. Every input activates all four experts using signed Walsh coefficients derived from signs of `x[0]` and `x[1]`; this deterministic address is identical and free of learned-router variation across methods. Compared full independent experts, hard tying, shared scalar amplitudes, rank-1 per-expert residuals, and Mirror views in both aligned-view and independent-matrix teacher modes. Development world 50000 selected LR 0.01. Fresh worlds 50001–50003 used 1,200 AdamW updates and matched minibatches. Source/protocol/tests/selection were hash-frozen before fresh access.

## D — Decision: PROMISING

### Fact

- Aligned Mirror MSE was 1.53e-9 / 2.45e-9 / 2.74e-9 vs full MoE 2.04e-9 / 1.38e-8 / 7.53e-8. The <=1.10x MSE and <=0.70x payload gates passed in all 3 worlds.
- Actual serialized inference payload was 2,853B vs 4,841B (-41.1%). Mirror beat tied, scalar amplitude, and rank-1 residual quality in all aligned worlds; the closest shared baseline (rank-1) had MSE 0.175–0.276.
- In independent mode, full MoE was near numerical zero (2.0e-9–3.4e-9), while Mirror was 3.40–3.80 and rank-1 residual 2.80–3.40. This task requires private expert degrees of freedom for arbitrary matrices.
- Mirror active-compute proxy was 67.58M vs 62.67M for full MoE (+7.8%). Median training time was 9.91s vs 2.85s aligned; median CPU inference throughput was 48k vs 564k examples/s (0.085x). This eager Givens implementation has a large runtime penalty.
- 30 fresh rows replayed with exact payload bytes; maximum MSE delta 3.3e-10, R² delta 4.6e-9, worst-pattern MSE delta 4.4e-10. Tests: 4 passed.

### Interpretation

The experiment supports a narrow structural claim: signed composition of shared-base Givens views can encode aligned role functions while reducing payload by 41.1%. This does not increase independent capacity; arbitrary expert matrices require private parameters. Current implementation trades storage for substantially worse compute and latency, so it does not improve the overall storage/quality/runtime frontier.

### Strongest counter-hypothesis

The teacher is deliberately generated from the Mirror coordinate family. Full MoE also reaches essentially exact aligned quality, and fixed-budget alignment favors the matching inductive bias. The result may not transfer to learned experts or other structured transforms.

### Unconfirmed

Near-convergence capacity across broader tasks, nonlinear experts, a learned router, top-k sparsity, GPU kernels, and whether a cheaper structured coordinate implementation keeps the byte gain without the runtime cost remain untested. The signed coefficients are supplied by an input-sign rule, so this is not evidence about routing quality.

## Prior art delta

PA01 motivates hard expert tying. PA02/PA03 motivate shared/path-constrained and low-rank routing controls. MA-003 tested top-1 expert choice; MA-005 isolates joint signed functional composition, with all four components evaluated per example.

## Fact / interpretation / hypothesis

- **Fact:** frozen fresh comparison passed aligned quality/storage 3/3, exact replay passed, independent matrices defeated shared views, and current CPU runtime regressed.
- **Interpretation:** learned Givens coordinates recovered a deliberately aligned signed expert basis with fewer stored bytes, but do not replace arbitrary independent experts and currently reduce runtime efficiency.
- **Hypothesis:** a larger nonlinear MoE with optimized fused views may preserve this storage advantage; this remains untested.
