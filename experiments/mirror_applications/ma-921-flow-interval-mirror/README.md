# MA-921 — Factorized endpoint Views for reusable flow maps

Status: **FAIL at development gate; fresh seeds unopened**
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME
Branch: `research/ma-921-flow-interval-mirror-20261008`
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`
Prior art: PA260, Flow Map Matching.

## H — hypothesis

Factorized Mirror endpoint coordinates over one shared two-time flow-map basis can recover held-out interval maps and preserve semigroup composition with lower actual inference bytes than native Flow Map Matching time conditioning, while outperforming a byte-matched additive endpoint factor control.

> **Mirror insertion:** this experiment adds `m(s,t)` to the coefficient interface of one shared flow-map basis, so a distinct interval transport can be expressed without storing a separate map or free code for every interval pair.

## T — what ran

- Synthetic non-autonomous ODE: `dx/dt = (A0 + sin(2πt) A1)x`; noncommuting 2×2 matrices are drawn for each world. Reference interval maps use fixed RK4 integration.
- The time grid has 21 endpoints from 0 to 1. There are 210 ordered intervals, 198 training pairs and 12 exact held-out pairs. The same held-out pairs are tested through 12 two-segment compositions whose component intervals were seen during training.
- Development seeds 9211 and 9212; equal training budgets 200 and 500 full-batch Adam updates. Every method sees the same examples per world and budget. Fresh seeds 92101–92103 were never opened.
- Methods: direct native FMM MLP conditioned on state and both endpoints; native two-time coefficient network over the shared flow-map basis; additive endpoint factors; multiplicative Mirror endpoint factors; independent interval matrices as a seen-only upper control.
- Rank sweep: 2, 4, 8 for additive and Mirror. Direct map uses one learned map evaluation (NFE 1); two-segment composition uses NFE 2.
- Each condition trains on 6,336 examples per update over 198 interval pairs. The 3× forward MAC training proxy, actual `torch.save` payload size/hash, one-pass CPU runtime, held-out map MSE, composition-to-true error and semigroup discrepancy are in `RESULTS_CORE.csv`.
- CPU only: PyTorch 2.6.0, Linux x86_64, one thread. No diffusion image model or GPU run was part of this mechanism screen.

## Development results

Each row reports the mean held-out MSE across the two dev worlds. Ratios are per world, in seed order 9211 / 9212.

| Updates | Rank | Mirror held-out MSE | Mirror / additive MSE | Mirror / direct FMM MSE | Max normalized semigroup discrepancy | Full payload |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2 | 0.000124 | 8.25× / 2.49× | 0.48× / 0.05× | 0.00037 | 2704 B |
| 200 | 4 | 1.79e-05 | 0.81× / 1.55× | 0.05× / 0.03× | 6.7e-05 | 3152 B |
| 200 | 8 | 3.31e-05 | 0.59× / 6.37× | 0.03× / 0.12× | 7.5e-05 | 3856 B |
| 500 | 2 | 2.25e-05 | 1.05× / 1.87× | 0.19× / 0.09× | 0.00016 | 2704 B |
| 500 | 4 | 1.47e-05 | 0.61× / 1.49× | 0.11× / 0.07× | 3.6e-05 | 3152 B |
| 500 | 8 | 1.53e-05 | 0.33× / 3.24× | 0.06× / 0.12× | 2.8e-05 | 3856 B |

The preregistered selection rule required both dev worlds to meet all four gates: within 10% of direct FMM, at least 20% better than matched additive factors, normalized semigroup discrepancy ≤0.01, and full payload ≤70% of direct FMM. **0 of 6 rank-budget choices passed both worlds.** Rank 8 at 500 updates passed in world 9211 but was 3.24× worse than additive control in world 9212. No fresh run was authorized by the protocol.

The native shared-basis coefficient control had held-out MSE `1.31e-5` at 200 updates and `6.10e-6` at 500 updates averaged across worlds, with an 8,708 B payload. Mirror's 2,704–3,856 B payload is smaller, but its MSE did not reliably match this ordinary conditioned-basis control.

## Storage and compute

| Method | Actual full payload | Train MAC proxy, 500 updates | Mean training wall | Mean full-evaluation wall |
|---|---:|---:|---:|---:|
| Direct native FMM | 8,456 B | 12,773,376,000 | 1.496 s | 4.760 ms |
| Native conditioned basis | 8,708 B | 13,609,728,000 | 1.662 s | 4.589 ms |
| Additive endpoint, rank 2 | 2,704 B | 304,128,000 | 0.528 s | 0.644 ms |
| Mirror endpoint, rank 2 | 2,704 B | 323,136,000 | 0.536 s | 0.631 ms |
| Mirror endpoint, rank 8 | 3,856 B | 608,256,000 | 0.640 s | 0.774 ms |
| Independent interval matrices | 5,084 B | 38,016,000 | 0.252 s | 0.309 ms |

The independent control has no trained held-out interval map and is excluded from held-out quality comparisons. Its per-interval matrices have a normalized semigroup discrepancy around 0.02, while the factorized methods remain below `4e-4` in these dev runs. Timing is a single CPU pass over 13,440 interval-state examples; it is a small mechanism benchmark, not a general throughput claim.

## D — decision

**FAIL:** none of the six preregistered rank-budget candidates beat the matched additive endpoint control by 20% in both development worlds. The protocol therefore leaves fresh data unopened. Storage and semigroup gates passed, but the Mirror-specific quality gate did not.

## C — strongest counter-hypothesis

Ordinary additive endpoint factorization, or direct native conditioning of a shared basis, already captures this smooth low-dimensional ODE family. The multiplicative coordinate adds non-convex fitting sensitivity without a stable quality gain. The rank-8/500 result reversing across the two worlds supports this alternative.

## U — unconfirmed

- Performance on real flow-matching generative models, image sample quality, or FID.
- Generalization to arbitrary continuous endpoints; this screen uses a fixed discrete time grid.
- Whether other ODE families, initialization policies or private residuals can stabilize Mirror endpoints.
- Equal-NFE comparison against a native neural ODE solver; every learned direct map has one NFE here.

## Fact / interpretation / hypothesis

**Facts:** all six rank-budget choices miss at least one dev world on the additive-control quality gate. Full payloads are 2,704–3,856 B for Mirror versus 8,456 B for direct FMM. Semigroup discrepancy remains below `4e-4` for these Mirror runs. Fresh seeds were not opened.

**Interpretation:** endpoint factorization gives a compact, compositionally consistent representation on this synthetic task, but the multiplicative Mirror coordinate has no robust quality advantage over same-size additive factors. The large byte reduction versus direct FMM is not Mirror-specific because additive factors use the same bytes.

**Hypothesis:** structured endpoint coordinates may be useful when a learned flow family has identifiable multiplicative interval geometry; ordinary conditioning or additive factors are sufficient for many smooth flow families.
