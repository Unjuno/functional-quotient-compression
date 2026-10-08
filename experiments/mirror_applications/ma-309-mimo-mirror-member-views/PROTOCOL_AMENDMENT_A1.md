# MA-309 protocol amendment A1

Frozen before the corrected fresh evaluation. The initial implementation serialized FP16 weights but measured quality on the in-memory FP32 training model. That implementation does not satisfy the experiment contract's serialized-inference-payload authority.

A1 changes only evaluation implementation: reload the actual serialized FP16 arrays into a fresh inference model before computing accuracy, NLL, calibration, diversity and throughput. Model, optimizer, update count, train/validation/test generation, thresholds and comparisons are unchanged. The initial fresh worlds 30911–30913 and their outputs are quarantined and excluded from the primary results. Development seeds 30901/30902 are rerun with the corrected runner; new fresh seeds 30921/30922/30923 remain sealed until corrected source and settings are committed.
