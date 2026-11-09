# MA-699 — Top-k Mirror attention Views over one physical cache

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `81c702ad`

## Hypothesis

H: Multiple active logical attention Views can share one physical cache and retain materialized-view outputs, while their compute still scales with top-k.

## Mirror insertion

> **Mirror insertion:** applies selected per-view key/query and value/output transforms on reads from one canonical K/V cache; per-query selected indices are paid.

- Native baseline: independent per-view materialized caches.
- Single-view baseline: one attention pass.
- Cheapest equivalent: direct plane-phase coordinates, included as native control.

## Prior-art delta

PA156 and PA161 motivate lazy cache reads and latent K/V projections. MA-699 tests several simultaneous logical attention roles and explicitly measures active attention cost as k grows. View-combination counts are not treated as capacity.

## Results discipline

Frozen seeds, top-k values, and gates are in [PROTOCOL.json](PROTOCOL.json). All routing indices and view codes are serialized. Compare actual outputs and MAC proxies to materialized independent views.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic FP32 attention only.
