# MA-836 — NTK-metric task codebook for low-interference shared updates

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / NTK GEOMETRY  
Base commit: `6ff01124`  
Prior art: PA225, NTK-CL.

## Hypothesis

H: When task distinctions are low-rank in pooled tangent-kernel geometry, an NTK-weighted Mirror codebook preserves heldout quality with at least 25% fewer complete bytes than independent readouts and less task interference than Euclidean factorization, while strictly beating native NTK-weighted factorization.

## Mirror insertion

- Native method: support-fitted per-task linearized readouts.
- `m`: task coordinate in a rank-2 shared update basis learned after whitening by the pooled tangent metric.
- Cheapest equivalent: native NTK-weighted low-rank factorization.
- Measurements: per-task test error, NTK overlap, payload bytes, active MAC proxy, decode MACs.

## Physical-to-logical claim

- Shared physical object: a common low-rank update basis.
- View: one stored task code per task.
- Logical multiplicity: task-specific linearized functions with reduced cross-task tangent overlap.
- Failure mechanism: native metric factorization may be identical; rank truncation may harm retention; the metric may be task-specific or expensive to store.

## Protocol and boundaries

Eight synthetic 32D tasks, one development seed, closed-form fitting. Amendment 01 added full-rank Euclidean and NTK-orthogonal basis controls before they were measured. Rank-2 Mirror/native factorization aliases and misses the .02 quality gate (nRMSE .09086 at 1,117 B). Full-rank NTK-orthogonal basis reaches 3.27e-15 error but costs 2,065 B vs 1,614 B independent and leaves function overlap unchanged. No real continual-learning or trained-network claim. See [PROTOCOL.json](PROTOCOL.json) and [STATUS.md](STATUS.md).
