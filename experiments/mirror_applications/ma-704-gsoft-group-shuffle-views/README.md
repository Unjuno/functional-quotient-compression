# MA-704 — GSOFT group-shuffle Mirror task Views

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `161cd9d4`

## Hypothesis

H: Packed group-shuffle factors offer a lower-byte, matrix-free task View family over shared weights, but may be ordinary GSOFT parameterization.

## Mirror insertion

> **Mirror insertion:** adds per-task block-orthogonal factors and deterministic shuffle stages to one shared linear weight.

- Native baseline: independent full task matrices.
- Native structured control: same GSOFT group-shuffle coordinates and execution.
- Dense merged control: apply each full task matrix directly.

## Prior-art delta

PA172 directly establishes Group-and-Shuffle/GSOFT as a structured orthogonal family; PA21 and PA24 establish BOFT and Monarch controls. This experiment measures exact function replay, payload bytes, and matrix-free work over several stage counts. It does not claim orthogonality or group-shuffle as new.

## Gates

Frozen dimensions, seeds, gates, and packed payload format are in [PROTOCOL.json](PROTOCOL.json). Actual serialized bytes and CPU timing are both reported; CPU timing is not treated as fused-kernel performance.

## Results and decision

**FACT:** Across three audit seeds, group-shuffle output matched explicit dense matrices within 3.1e-8. One/two/four stages used 25.3%/37.8%/62.8% of independent bytes and 1.125x/1.25x/1.5x MAC proxy. Native GSOFT function and payload were identical. In this local NumPy implementation, matrix-free transform wall time was 3.8/6.3/10.9 ms per bank versus 0.20/0.22/0.25 ms for dense apply; these are not fused-kernel timings.

**INTERPRETATION:** GSOFT factors offer a compact exact transform family at one to two stages, while more stages exceed the storage gate and local CPU execution is much slower despite a low MAC proxy. Native GSOFT fully explains the parameterization, so no Mirror-specific value is established.

**HYPOTHESIS:** Group-shuffle views can retain dense logical functions with compact factors, but practical benefit depends on a fused kernel and is already native GSOFT.

**BOUNDARY:** Synthetic matrices generated from GSOFT factors; no learned task quality, GPU kernel, or actual serving speed claim.

## H/T/D/C/U

- **H:** Short group-shuffle codes can represent dense task maps over shared weights at reduced serialized bytes.
- **T:** D=64, 16 maps, three audit seeds, 1/2/4 stages; independent dense matrices, native GSOFT, and dense apply controls.
- **D:** FAIL for Mirror-specific attribution; stage 2 passes generic bytes/MAC gates, but exact native GSOFT alias and severe NumPy wall-time regression remain.
- **C:** This is the established GSOFT factorization itself, not additional Mirror freedom.
- **U:** Fused GPU/CPU kernel performance and trained-task quality are untested.
