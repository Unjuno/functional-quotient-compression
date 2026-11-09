# MA-704 — GSOFT group-shuffle Mirror task Views

Status: SCREENING  
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

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic linear maps only.
