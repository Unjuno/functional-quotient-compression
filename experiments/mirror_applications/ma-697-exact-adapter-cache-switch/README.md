# MA-697 — Exact adapter switching with canonical prefix-cache reuse

Status: SCREENING  
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

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic single-layer attention algebra only.
