# MA-061 — one KV head to many logical KV heads

Status: **FAIL at development screen**. Fresh worlds stayed sealed because ordinary MQA dominated Mirror on quality, model bytes, cache bytes, and compute at the selected condition.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-061-single-kv-logical-views-20261007`.
Base commit: full SHA in `PROTOCOL.json`.

## H — Hypothesis

One physical K/V projection and one persistent K/V cache stream, expanded through per-query-head Mirror coordinates, can preserve a shared-view teacher while reducing actual cache bytes versus MQA/GQA/MHA. Independent K/V roles should need additional state.

## T — Test

Synthetic 16D full attention, four query heads of dimension 4, sequence length 8. Compared full MHA, GQA-2, MQA, MQA with per-head rank-1 K/V residuals, and MQA cache with K/V Mirror views. Teacher modes used one physical K/V pair plus per-head Givens views or four independent K/V pairs. Development world 61000 tested LR 0.001 and 0.003 for 700 updates; 0.003 was selected by the preregistered pooled MSE criterion. Fresh worlds 61001–61003 were not opened after a simpler control dominated.

## D — Decision: FAIL (development)

### Fact

- At selected LR in the aligned mode, Mirror MSE was 1.43e-3, while MQA was 6.68e-6 and rank-1 residual was 6.60e-6. At LR 0.001, Mirror MSE 6.14e-3 was still slightly worse than MQA 5.78e-3.
- Mirror and MQA stored the same 256B persistent K/V cache per sequence. Mirror's serialized model was larger: 5,333B vs 4,893B. Combined model+cache storage was 5,589B vs 5,149B.
- MQA active-compute proxy was identical to Mirror (41.57M), while inference throughput was 73.2k vs 50.6k examples/s.
- The cache formula charges exact float32 state for one sequence of eight tokens: 1,024B MHA, 512B GQA-2, and 256B MQA/Mirror. Cache bandwidth is an estimate, not measured GPU traffic.
- Twenty development rows replayed with exact model and cache bytes; max MSE delta 4.9e-12, worst-head delta 5.0e-12, R² delta 4.8e-9. Tests: 3 passed. Fresh worlds were not opened.

### Interpretation

The Mirror coordinate added stored metadata and eager transform cost without reducing persistent cache bytes beyond MQA. MQA also fitted the aligned attention output much better under this fixed-budget screen. The tested design therefore offers no quality/storage/compute improvement over the simple baseline.

### Strongest counter-hypothesis

The per-head rotation parameters may have been difficult to optimize from zero initialization at the selected schedule. However, development at both tested learning rates failed to beat MQA, which used the same physical KV count and cache size.

### U — Unconfirmed

Longer optimization, quantized cache, larger head dimensions, causal decoding, GPU memory bandwidth, NLL, and alternative coordinates remain untested.

## Fact / interpretation / hypothesis

- **Fact:** MQA used less model storage, equal cache bytes/compute proxy, and far lower aligned MSE than Mirror in development.
- **Interpretation:** the current transform is a functional overhead without a benefit over MQA for this task.
- **Hypothesis:** a different view that compresses persistent cache values or reduces memory traffic could still help; untested.
