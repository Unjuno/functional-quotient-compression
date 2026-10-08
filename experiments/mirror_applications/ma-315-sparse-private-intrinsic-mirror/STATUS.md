# MA-315 status

- Status: **FAIL** under the frozen storage promotion gate.
- Branch: `research/ma-315-sparse-private-intrinsic-mirror-20261008`
- Frozen protocol commit: `9bdcbdd`; fresh seeds were opened only afterward.
- Tests: 3 passed; deterministic replay: 45 summary rows and 1,280 task allocation events, exact metrics and payload hashes.

## Decision

**Fact:** Across fresh seeds 31511–31513, Mirror used 6,044–6,056B versus 5,850–5,862B for the matched shared-direct plus identical sparse-private control (+3.31–3.32%). Test normalized MSE was 9.58e-7–2.43e-6 for Mirror and 8.07e-8–1.13e-6 for direct, below the 1e-4 quality ceiling for both. Mirror fit-compute proxy was 41.9–42.2x direct; throughput was 0.61–0.78x.

**Interpretation:** Sparse shared/private allocation itself compresses the synthetic mixed task bank relative to fixed dense d=16. Replacing two direct common coefficients with an angle did not reduce total bytes after metadata and serialization overhead, and added fitting work.

**Hypothesis:** At this task count and FP16 code format, per-task angle plus shared radius does not beat directly stored FP16 common coefficients. The result does not rule out larger banks or amortized metadata under other payload formats.

**C:** The direct control is the exact low-cost representation for this fixed-radius orbit; Mirror's angle search also introduces avoidable compute, and ZIP/NPY headers amplify the extra arrays.

**U:** Natural learned task functions, optimizer updates, nonlinear models, address discovery, scaling beyond 128 tasks, and alternative packed serialization remain untested. No capacity claim.
