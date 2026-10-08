# MA-707 — Scaled-Cayley Mirror recurrent dynamics

## H — hypothesis

A shared orthogonal recurrence plus a small skew-Cayley task code may preserve stable multi-step dynamics with less inference state than full per-task matrices.

## T — planned test

Synthetic 4D linear recurrence with low-angle and high-angle task transitions. Compare tied shared transition, rank-2 skew-Cayley Mirror view, task-specific full orthogonal matrices and an unconstrained low-rank residual. Measure multi-step rollout MSE, spectral radius, serialized bytes and CPU solve/runtime cost. PA171 covers orthogonal recurrent parameterizations.

## D — status

SCREENING.

## C — strongest counter-hypothesis

The Cayley coordinate family may simply be a native structured orthogonal parameterization; independent task matrices may be needed outside its low-rank span.

## U — unconfirmed

No results yet. The teacher is synthetic and aligned to the Cayley family.

## Fact / Interpretation / Hypothesis

- Fact: PA171 establishes Householder and scaled-Cayley parameterizations for orthogonal recurrence.
- Interpretation: Mirror must add compact task multiplicity beyond this native parameterization.
- Hypothesis: shared generators plus task codes can preserve useful stable dynamics while reducing bytes.
