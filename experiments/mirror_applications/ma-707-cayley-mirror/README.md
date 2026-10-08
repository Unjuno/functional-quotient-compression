# MA-707 — Scaled-Cayley Mirror recurrent dynamics

## H — hypothesis

A shared orthogonal recurrence plus a small skew-Cayley task code may preserve stable multi-step dynamics with less inference state than full per-task matrices. Falsifiable gates are frozen in `PROTOCOL.json`.

## T — execution

Pure NumPy CPU experiment; four development task matrices were used to estimate a rank-2 shared generator span. Fresh worlds 70711, 70712, 70713 each used 12 tasks (6 low-angle, 6 high-angle), 128 support transitions/task and 32 held-out initial states rolled for 32 steps. Controls: tied shared, rank-2 Cayley Mirror, task-specific orthogonal Procrustes, and shared plus rank-2 additive residual. Zero gradient optimizer updates; 1,536 support examples/world. Actual deployment packages were serialized to NPZ and their byte lengths measured.

## D — FAIL

Low-angle Mirror 32-step MSE was near numerical zero and stability radius was 1.0. But full orthogonal also fit at numerical precision. Combined Mirror package was 1,090 B versus 1,054 B for all per-task full matrices (ratio 1.034), so the <=60% storage gate failed on all three worlds. High-angle Mirror mean rollout MSE was 0.5248; full orthogonal was near 0. The preregistered Pareto/storage claim is therefore FAIL. This does not invalidate low-rank Cayley coordinates as stable parameterizations; the toy 4D task is too small for amortized shared-state savings.

## C — strongest counter-hypothesis

The apparent low-angle functional recovery follows directly from the synthetic teacher being generated from a low-rank Cayley family. A cheap native shared-generator parameterization explains the fit; no Mirror-specific quality advantage over rank-2 additive control was shown.

## U — unconfirmed

No natural sequence task, learned recurrent model, larger hidden dimension, GPU runtime, or trained-model capacity frontier was tested. Timing is a tiny CPU calibration and not throughput evidence.

## Fact / Interpretation / Hypothesis

- **Fact:** all three fresh worlds had low-angle Mirror MSE below 1e-29, spectral radius 1.0, and Mirror/full serialized bytes ratio 1.034. High-angle mean MSE was 0.5248.
- **Interpretation:** structured coordinates can recover aligned orthogonal dynamics but lose to storing tiny independent matrices at d=4.
- **Hypothesis:** at larger state size, a shared generator basis may become byte-efficient if real task dynamics remain in its low-dimensional span.

Full rows are in `RESULTS_CORE.csv`; seed summaries are in `source/metrics_summary.json`. Draw selection provenance is in `source/draw39_exclusions.json`.
