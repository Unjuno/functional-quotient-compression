# MA-063 — causal Mirror-MQA

Status: **FAIL at development screen**. This confirms the same structural issue as MA-061: Mirror views did not improve the MQA frontier when one physical cache was already sufficient. Fresh worlds remained sealed.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-063-causal-mirror-mqa-20261007`.
Base commit: full SHA in `PROTOCOL.json`.

## H — Hypothesis

A vectorized head view over one causal MQA cache may recover multiple logical K/V roles without expanding persistent cache state. MQA/GQA quality and cache traffic are the primary controls.

## T — Test

Synthetic causal attention, 16D, four query heads of dimension 4, sequence length 16. Compared full MHA, GQA-2, MQA, rank-1 K/V residual, and vectorized Mirror-MQA. Teacher modes used one shared causal KV pair plus per-head rotations or independent KV pairs. Development world 63000 tested LR 0.001 and 0.003 for 600 updates; the preregistered pooled metric selected 0.003. Fresh worlds 63001–63003 were not opened after MQA dominated both development rates.

## D — Decision: FAIL (development)

### Fact

- At LR 0.003 in the aligned mode, MQA MSE was 3.27e-6 and Mirror MSE 1.34e-5. At LR 0.001, MQA was 0.00317 and Mirror 0.00357. Rank-1 residual MSE was lower than Mirror at both rates.
- MQA and Mirror stored the same 512B persistent cache per sequence. MQA model payload was 4,893B vs 5,333B Mirror; combined model+cache was 5,405B vs 5,845B.
- MQA and Mirror had identical active-compute proxy. MQA measured higher CPU throughput at selected LR.
- Exact float32 causal cache sizes: MHA 2,048B, GQA-2 1,024B, MQA/Mirror 512B. The optimized vectorized Givens path did not change the MQA storage frontier.
- Twenty development rows replayed with exact model/cache bytes; max MSE delta 4.3e-12, worst-head delta 5.0e-12, R² delta 5.0e-9. Tests: 4 passed. Fresh worlds stayed sealed.

### Interpretation

This repeats MA-061's result with causal masking, longer cached context, and vectorized rotations. MQA already stores one physical K/V state; Mirror adds model coordinates without reducing cache bytes, while output quality and runtime are worse under the tested budget. This is the second consecutive KV-family failure with the same cause.

### Strongest counter-hypothesis

The learned rotations may need different initialization or longer training, but the simpler MQA model uses the same cache state and already had better quality at both tested development rates. That does not support opening fresh worlds for this exact screen.

### U — Unconfirmed

Hardware cache bandwidth, quantized cache, longer optimization, language NLL, and transforms that reduce actual stored K/V tensor bytes remain untested.

## Fact / interpretation / hypothesis

- **Fact:** MQA dominates Mirror on model bytes and output MSE with equal cache bytes/compute; vectorized rotations did not change that.
- **Interpretation:** logical KV views do not create a cache advantage when MQA already shares the persistent state.
- **Hypothesis:** cache-view mechanisms may need to compress value precision or data layout to improve the MQA frontier; not tested.
