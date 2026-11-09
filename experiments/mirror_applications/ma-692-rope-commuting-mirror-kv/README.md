# MA-692 — RoPE-commuting Mirror KV views

Status: FAIL  
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

## Results and decision

**FACT:** All three audit seeds produced lazy commuting output relative errors of 1.60e-7–2.05e-7. Native direct phase coordinates produced the same outputs and use the identical angle payload. The noncommuting cross-plane control had 0.194–0.258 relative output error. The actual NPZ payload was 17,846 B for one cache plus eight view codes and 59,578 B for eight materialized views (29.96%). The declared active transform proxy was 24,576 MAC-equivalents across 48 queries and 8 views; this corresponds to 512 per decoded query. Materializing the transforms is estimated at 24,576 MAC-equivalents per cache bank, so the simple transform-only break-even is 48 generated queries. CPU wall-clock was 5.3–6.1 ms per seed for this tiny NumPy harness and is not a kernel benchmark.

**INTERPRETATION:** The commuting algebra and noncommuting boundary are supported. The preregistered combined quality/storage gate fails because serialized state is 29.96%, above the 25% threshold. Mirror-specific attribution also fails: ordinary native phase coordinates exactly alias the functions and representation.

**HYPOTHESIS:** RoPE-plane commutation can preserve a shared cache without materialization, but this tested encoding is not Mirror-specific.

**BOUNDARY:** Synthetic FP32 attention, D=16, T=48, 8 views. No neural training, GPU kernel, language quality, or real decode throughput claim. Fresh seeds were opened only after the protocol was frozen; no audit tuning occurred.
