# MA-307 — Mirror code before PackNet physical allocation

Status: **FAIL for the preregistered storage/quality/runtime gate; storage frontier point retained**
Prior art: PA31 SupSup and PA32 Piggyback/PackNet.

## H — hypothesis

In a sequential task stream, try a compact View over the shared readout first and allocate a private readout only when the View misses a validation threshold. This should delay physical growth relative to binary-mask PackNet-style allocation while preserving prior tasks.

> **Mirror insertion:** this experiment adds a task angle `m_t` to one shared readout before allocating a task-private readout, so aligned tasks can reuse the physical weights through a compact Givens view.

## T — execution

Eight linear tasks use a fixed deterministic 16-to-32 ReLU feature encoder. The first five task readouts lie on one Givens orbit; the last three are independently sampled. Support/validation/test sizes are 512/256/512 per task. Development worlds 30710/30711 selected validation threshold 0.001 from {0.0001, 0.001, 0.01}; fresh worlds 30712–30714 used that fixed threshold.

The Mirror allocator grid-searches one angle, stores an int8 coordinate when validation normalized MSE is at most 0.001, and otherwise allocates a private float32 readout. The PackNet-style control greedily fits a packed binary mask, then allocates a private readout if the mask misses the same threshold. Independent readouts and hard tying are additional controls. Actual payload bytes include all arrays, task allocation records, encoder seed and transform metadata. Each allocation event re-evaluates every earlier test task.

This is a small linear PackNet allocation analogue. It is not a full PackNet neural-network reproduction.

## D — decision

All three fresh worlds accepted the four additional aligned tasks as Mirror views and allocated private weights only for the three unrelated tasks. The PackNet-style binary mask failed on tasks 1–7 and used seven private readouts. Mirror retained all earlier task functions with zero measured prior-task error change.

Mean fresh payload was 854 B for Mirror and 1,333 B for PackNet-style masks plus allocation, a 35.9% reduction. Mean normalized held-out MSE was 2.08e-5 for Mirror and approximately 9.2e-16 for PackNet/independent readouts. Thus the storage and allocation-delay signal is positive on this aligned stream, with a measurable quality gap versus private readouts.

The registered runtime gate also required allocation-search time at most 1.25× the PackNet control in each fresh world. Repeated isolated medians were 1.45×, 1.10× and 1.22×; one world missed the gate. Mean allocation-search operation proxy was 45.99M for Mirror and 26.74M for PackNet. End-to-end serialized-decode plus query throughput was also lower on average for Mirror. **Overall status is FAIL for the strict preregistered Pareto gate**, while the storage point remains a useful scoped frontier result.

## Fact / interpretation / hypothesis

- **Fact:** Four aligned tasks used one-byte View codes per task; three unrelated tasks used private readouts. Mirror payload was 35.9% below PackNet-style allocation in all fresh worlds. The maximum measured change to any prior task's normalized test MSE was zero. One of three repeated fresh runtime ratios exceeded 1.25×.
- **Interpretation:** View-first allocation delayed private growth by four tasks in this deliberately orbit-aligned stream. The effect trades extra search and view computation for storage; the unrelated-task boundary requires private parameters.
- **Hypothesis:** A fused or amortized View search may improve the compute side, but this experiment does not test such an implementation. A natural task stream may contain fewer tasks on a reusable orbit.

## C — strongest counter-hypothesis

The first five task functions were deliberately generated from the exact Givens family used by the Mirror allocator. The result establishes aligned mechanism feasibility, not that Mirror views will delay allocation on ordinary task streams.

## U — unresolved

No optimizer-driven continual learning, full PackNet model, learned task router, language model, or near-convergence capacity frontier was measured. Runtime measurements are small CPU mechanism timings; the three repeated medians are retained in `RUNTIME_CALIBRATION.json`.

Protocol: [PROTOCOL.json](PROTOCOL.json). Summary metrics: [RESULTS_CORE.csv](RESULTS_CORE.csv). Per-task allocations and retention: [ALLOCATION_EVENTS.csv](ALLOCATION_EVENTS.csv). Tests and replay: [VERIFICATION.json](VERIFICATION.json).
