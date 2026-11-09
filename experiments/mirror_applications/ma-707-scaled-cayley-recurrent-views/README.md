# MA-707 — Scaled-Cayley recurrent Mirror Views

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `322e8039`

## Hypothesis

H: A shared orthogonal recurrent transition plus small low-rank skew/Cayley task codes can provide stable logical dynamics at lower serialized bytes than independent matrices, with solve overhead amortized across the sequence.

## Mirror insertion

> **Mirror insertion:** adds per-task skew-symmetric coordinates at the recurrent transition interface and reconstructs the task view through a Cayley map.

- Native baseline: independent dense transitions.
- Hard tying: one shared Q0 for every task.
- Cheapest native equivalent: standard per-task low-rank Cayley parameterization, included exactly.

## Prior-art delta

PA171 establishes Householder and scaled-Cayley orthogonal parameterizations. MA-707 tests a shared physical transition plus per-task skew codes, including sequence stability and inversion/solve cost.

## Gates

Frozen dimensions, ranks, seeds, and gates are in [PROTOCOL.json](PROTOCOL.json). Actual payload includes all factors and metadata; reconstruction cost is charged/amortized separately.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic recurrence only.
