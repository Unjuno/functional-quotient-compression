# MA-1035 amendments

## A1 — Stabilize runtime measurements before audit (2026-11-15)

The initial development run showed large and inconsistent differences in single-shot encode/decode/scan times at this tiny 512-vector workload. No audit world had been accessed. The quality, model, training and storage conditions remain unchanged; timing now warms each operation and reports the median of five full database encodes and twenty full decode-at-load and query-scan repetitions. The initial development output is preserved as `runs/development_unwarmed_103500.json`. This amendment does not alter the frozen success gates or choose hyperparameters based on audit data.
