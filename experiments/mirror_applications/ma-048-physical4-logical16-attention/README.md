# MA-048 — physical-4 to logical-16 attention heads

Status: **FAIL** under the preregistered aggregate quality gate. Payload and per-head contribution results were informative, but output quality missed full MHA in all fresh worlds.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-048-physical4-logical16-attention-20261007`.
Base commit: full SHA in `PROTOCOL.json`.

## H — Hypothesis

Four physical QKV sets expanded into sixteen logical heads with per-head Givens views can recover an aligned sixteen-head teacher at lower serialized payload than ordinary MHA; hard GQA and rank-1 generated residuals are controls.

## T — Test

Synthetic 32D full-attention block, sequence length 6, sixteen 2D heads, four physical QKV groups and shared output projection. Compared full 16-head MHA, hard four-group GQA, per-logical-head rank-1 QKV residual, and Mirror views. Teacher modes were four shared QKV sets plus sixteen Givens views and sixteen independent QKV sets. Development world 48000 selected LR 0.003. Fresh worlds 48001–48003 used 600 updates.

## D — Decision: FAIL (formal output-quality gate)

### Fact

- Aligned output MSE: Mirror 0.03033 / 0.02936 / 0.02539; full MHA 0.01223 / 0.01354 / 0.01487. Ratios 2.48x / 2.17x / 1.71x missed the <=1.10x gate in all worlds.
- Actual payload was 10,401B vs 18,405B full MHA (0.565x; 43.5% fewer bytes), passing the <=0.65x storage gate.
- Hard GQA had worse aggregate MSE in worlds 2 and 3 but was slightly better in world 1. Rank-1 residual was more accurate than Mirror in worlds 1 and 3, less accurate in world 2, at 16,093B.
- Per-head contribution audit was better for Mirror than full MHA in all three worlds: worst-head MSE 0.00645–0.00708 vs 0.00727–0.01923. This did not translate into aggregate output quality.
- Independent16 mode remained difficult: Mirror MSE 0.04199–0.04673 vs full MHA 0.03546–0.04053; rank-1 residual was more accurate (0.03750–0.04302).
- Compute proxy was 20.0M vs 32.9M full MHA (-39.2%), but median aligned training time was 36.34s vs 7.00s and CPU inference throughput 6.9k vs 36.4k examples/s (0.189x).
- 24 fresh rows replayed; payload bytes exact; max output-MSE delta 4.9e-12, worst-head delta 1.4e-12, R² delta 5.0e-9. Tests: 3 passed.

### Interpretation

Four physical projections plus Mirror coordinates saved bytes and preserved per-head contribution patterns better than MHA, but the combined attention output was substantially worse under the fixed budget. Lower dense-compute proxy did not improve runtime in this eager rotation implementation. This is a negative result for the registered physical-4 to logical-16 quality target.

### Strongest counter-hypothesis

The fixed 600-update schedule may undertrain the transformed QKV path; the observed discrepancy between per-head and aggregate quality may also reflect head/output-projection compensation. Longer training and fused views could change runtime, but neither can be inferred here.

### U — Unconfirmed

Near-convergence quality, causal attention, NLL, optimized kernels, ablations, larger head dimensions, and whether private residuals can repair the aggregate gap remain untested.

## Fact / interpretation / hypothesis

- **Fact:** storage passed; aggregate output quality failed 3/3; per-head contribution audit favored Mirror; CPU runtime regressed substantially.
- **Interpretation:** this view family retained head-local structure better than its output composition, but not enough to meet the task quality gate.
- **Hypothesis:** fused rotation kernels or a different training schedule could close the aggregate gap; untested.
