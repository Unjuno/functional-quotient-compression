# MA-693 — Pre-RoPE canonical cache with deferred position and View transforms

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `dfd5c63c`

## Hypothesis

H: A content-space canonical cache can serve noncommuting logical key Views by applying the View before RoPE inside attention, exactly matching explicit re-encoding while avoiding per-view persistent KV copies.

## Mirror insertion

> **Mirror insertion:** adds per-view structured rotations at the content-cache read point before position encoding, retaining one physical cache and applying position/View algebra lazily.

- Native baseline: ordinary RoPE attention with shared content K/V.
- Exact insertion: per-view cross-plane Givens on keys before RoPE; value map after attention.
- `m` is persistent per-view angle metadata.
- Cheapest native equivalent: explicit structured Givens coordinates, expected to alias Mirror.

## Physical-to-logical claim

One position-free K/V cache supports eight logical attention views. Unlike MA-692, the key transforms intentionally do not commute with RoPE; the test applies both in order at read time. This is synthetic algebra evidence only.

## Prior art delta

PA156 motivates deferring positional encoding; PA157 strips RoPE before mapping caches. MA-693 tests a known exact structured View under deferred position encoding and measures serialized cache payload, compute proxy, and a position-independent query approximation as a negative control.

## Comparisons and gates

See frozen [PROTOCOL.json](PROTOCOL.json) for controls and gates. Audit seeds are fixed before any measurements. Native Givens parameterization is an attribution control; exact alias means no Mirror-specific result.

## Storage and compute

Actual NPZ bytes include K/V, view codes, positions/frequencies, causal/shape metadata. Report incremental transform MAC proxy separately from attention, estimate materialization break-even by dwell length, and treat CPU wall time as descriptive.

## Results and decision

**FACT:** On three locked audit seeds, applying cross-plane Givens in content space and then RoPE at read matched explicit re-encoded attention within 1.74e-7 relative output error. The incorrect position-independent query shortcut erred by 0.189–0.264. Actual NPZ payload was 84,570 B shared vs 198,434 B independent (42.62%). Native Givens coordinates exactly matched Mirror outputs. The incremental key-view transform proxy is 6,144 scalar FLOPs per decoded query; explicit cache construction costs 6,144 once per bank, so transform-only break-even is one generated token. CPU harness wall time was 8.0–9.2 ms and is not kernel throughput evidence.

**INTERPRETATION:** Deferred pre-RoPE View application exactly supports noncommuting positions, while the query-only shortcut does not. The frozen storage gate fails, eager materialization wins after one token under the proxy, and Mirror has no distinct attribution versus native Givens coordinates.

**HYPOTHESIS:** Position-free canonical caches can serve analytic logical views exactly, but this tested lazy route does not pass the combined useful storage/compute frontier.

**BOUNDARY:** Synthetic FP32 attention, D=16, T=128, 8 views. No neural training, language quality, GPU kernel, or production decode claim. Fresh seeds were opened only after protocol freeze; no audit tuning occurred.
