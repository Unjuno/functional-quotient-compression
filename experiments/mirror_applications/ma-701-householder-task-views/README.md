# MA-701 — Householder Mirror task Views

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `a6c00329`

## Hypothesis

H: Short per-task Householder products can express multiple orthogonal task Views over one shared linear operator at less actual state than independent operators, while reflection count trades state against matrix-free compute.

## Mirror insertion

> **Mirror insertion:** adds per-task reflection-vector addresses around one shared operator; the View is applied matrix-free at inference.

- Native baseline: independent task matrices.
- Existing PEFT controls: dense OFT matrices and parameter-matched block-orthogonal transformations.
- Cheapest exact equivalent: native Householder parameterization, which is explicitly an attribution control.

## Prior-art delta

PA171 establishes Householder orthogonal parameterization; PA20/PA21/PA172 establish OFT/BOFT/GSOFT precedents. This experiment tests serialized task-view storage and matrix-free compute against those families without claiming orthogonality itself as new.

## Gates

Frozen dimensions, seeds, output gates, and reflection counts are in [PROTOCOL.json](PROTOCOL.json). All transforms, bases, and metadata are charged in payloads.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic linear maps only.
