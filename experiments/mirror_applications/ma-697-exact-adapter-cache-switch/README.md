# MA-697 — Exact adapter switching with canonical prefix-cache reuse

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `af4020be`

## Hypothesis

H: Prefix-active logical adapter Views related by known K/V transforms can switch over one physical canonical prefix cache, with exact output equivalence and lower serialized cache bytes than independent specialist caches.

## Mirror insertion

> **Mirror insertion:** adds known per-specialist K/V transform coordinates at the read interface so specialists can differ over the prefix while reusing one physical cache.

- Native method before Mirror: independent standard-LoRA specialist caches or aLoRA's unchanged prefix.
- Insertion: transform query and attention output for each logical specialist.
- Cheapest ordinary equivalent: direct orthogonal/rotation coordinates; mandatory attribution control.

## Prior-art delta

PA154/aLoRA preserves cache reuse by delaying adapter activation. PA155 studies direct reuse across standard LoRA but does not establish exact quality or physical aliasing. MA-697 tests an analytically related, prefix-active specialist and charges view codes and complete serialized caches.

## Controls and gates

Frozen conditions are in [PROTOCOL.json](PROTOCOL.json). Independent materialized specialists are the upper control; direct untransformed reuse and aLoRA-style prefix behavior are negative/semantics controls.

## Storage and compute

Actual NPZ bytes include cache tensors and all view/metadata state. Transform-only break-even is reported over frozen dwell lengths; common attention compute is separated from query/output transforms. CPU wall time is descriptive.

## Results and decision

**FACT:** Across three locked audit seeds, lazy switched-view attention matched materialized specialists within 2.34e-7 relative error. The shared serialized cache plus eight view codes used 19,181 B versus 132,853 B for independent caches (14.44%). Direct untransformed cache reuse erred by .598–.679; aLoRA-style unchanged prefix has the same error against this prefix-active specialist. Native direct rotation coordinates matched lazy functions exactly. Query/output transform proxy is 512 MAC-equivalents per decoded token; eager cache transform costs 65,536 per bank, giving a 128-token break-even. At dwell 512, lazy transform work is 4x eager materialization.

**INTERPRETATION:** The known transform provides exact cache reuse for a prefix-active specialist and greatly reduces cache payload. The native rotation control is the same function and representation, so this does not establish Mirror-specific value. Transform work favors lazy mapping for shorter dwell and eager materialization for longer dwell.

**HYPOTHESIS:** Analytically related adapter Views can use one prefix cache without re-prefill, but the demonstrated mechanism is ordinary rotation algebra.

**BOUNDARY:** Synthetic single attention layer only; no trained adapter, language quality, GPU, TTFT, or production serving claim. Fresh seeds were locked before execution.
