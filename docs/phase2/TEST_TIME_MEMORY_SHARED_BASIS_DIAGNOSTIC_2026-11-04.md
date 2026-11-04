# Test-time memory shared-basis code diagnostic — 2026-11-04

## Decision

Pause candidate variants whose complete Mirror proposal is only to store per-memory or per-program coefficients over a shared linear/PCA basis. Three distinct storage settings found that ordinary low-rank/PCA coding reproduces the Mirror state and functions exactly. A future experiment in this lane must add a native-distinct mechanism and retain the strongest native basis/memory control. This ruling does not pause temporal retention, retrieval/interference, surprise-driven allocation, or other test-time memory mechanisms that make a separate falsifiable claim.

## Evidence

| MA | Object | Observed benefit | Decisive counter-control |
|---|---|---|---|
| MA-653 | Writable long-term key/value values | Aligned rank-4 memory used about 43% of full complete state; private rank restored diverse values | Native low-rank value memory exactly matched Mirror state, output, and bytes |
| MA-655 | Stored linear controller programs | At zero residual, 16/64 functions fit in 16.66/17.47 KB versus 61.54/244.48 KB independent | Native PCA exactly matched Mirror; explicit NSPM was smaller and equally accurate. Three fresh seeds replicated this |
| MA-658 | Task functions retained across consolidation | Aligned 32-task bank used 16.84 KB versus 122.58 KB independent with zero reconstruction error | Native PCA exactly matched state and output. Fresh stayed sealed after the development alias gate |

## Interpretation

The byte reductions are real for these aligned synthetic function families. They support a low-rank shared-function storage result, not a Mirror-specific result. Increasing address count or adding a coordinate representation does not establish extra logical capacity unless task quality and bytes improve beyond ordinary bases, coefficient tables, or native low-rank memory.

The simplest next research target remains MA-660, whose task-context/test-time learning mechanism differs from a static program bank. MA-659 surprise-driven coordinate allocation and MA-689 episodic-to-semantic retirement also make separate mechanism claims and are not blocked by this diagnostic.

## Scope boundary

This is a queue-design ruling for direct shared-basis memory/program-code variants. It does not generalize to nonlinear memory networks, dynamic retrieval, or learned online consolidation. The experiments remain synthetic and do not establish language-model or deployment-level behavior.
