# MA-692 — RoPE-commuting Mirror KV views

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `bc1d9007`  
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`  
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## Hypothesis

H: Per-plane 2D rotations commute with RoPE, so one RoPE-encoded canonical cache plus small view-angle codes exactly reproduces materialized view attention; an equal-norm cross-plane transform cannot be represented by a position-independent query transform over all positions.

## Mirror insertion

> **Mirror insertion:** adds per-view angles at the canonical RoPE-encoded K/V read interface, making logical views without storing per-view cache tensors.

- Native method: independent materialized K/V for each view.
- Insertion: transform current query and attention output using view angles while reading canonical K/V.
- `m` is persistent per-view metadata.
- Cheapest ordinary equivalent: native complex phase/per-plane angle coordinates.

## Physical-to-logical claim

One FP32 K/V cache serves 8 logical views. Cache sharing may save storage; it does not remove each view's attention work. The result may be ordinary phase-coordinate algebra rather than Mirror-specific capacity.

## Prior-art delta

PA156 already supports deferred positional transforms; MA-691 established canonical-cache right-transform algebra. MA-692 isolates whether RoPE-encoded cache can be reused without stripping/reapplying RoPE, and measures the commuting boundary against a noncommuting transform.

## Comparisons and gates

See frozen [PROTOCOL.json](PROTOCOL.json): identity, independent materialized cache, native phase control, and noncommuting cross-plane transform. PASS requires <=1e-5 output error and <=25% of independent serialized payload on three locked seeds. A native exact alias disallows Mirror-specific attribution.

## Tuning boundary

One development seed validates implementation only. Three audit seeds are fixed before any run. No tuning from audit results.

## Storage and compute

Actual NPZ bytes include tensors, view angles, positions/frequencies, shape and dtype metadata. No optimizer or training examples; report zero updates and synthetic query positions. Report analytic MAC proxy and CPU wall time as descriptive only.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic FP32 attention mechanism only.
