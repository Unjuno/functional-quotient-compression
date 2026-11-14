# A9R — frozen fresh-world replication (post-A9 exploratory follow-up)

2026-11-15. This secondary protocol was fixed **after seeing A9 main results** and before generating any A9R training checkpoint. It tests whether the success of a family-aware, teacher-threshold-blind initialization is stable across new world IDs; it is not statistically independent of the A8→A9 research selection process.

- New worlds 826,827,828,829,830,831; 2 initialization seeds 0,1; ramp teacher, K=16.
- Three kinds: mirror, pre, direct shift. Three methods: random_full4, scaffold_code4, scaffold_full4.
- 6*2*3*3=108 last-step models. Fixed 1200 AdamW updates, 256 batch, lr 0.012, same data generation/deterministic trainer/FP32 CPU and byte rules as A9; only omitted methods are random_code4, scaffold_noisy4, random_wide to isolate H1 and code-vs-full distinction.
- All 108 weight blobs/SHA are locked before any dev, IID or OOD evaluation. Joint quality means mean-over-2-seeds IID and OOD NRMSE <= 0.08 per world. Success target fixed at >=5/6 worlds for mirror scaffold_code4, >=5/6 worlds mirror scaffold_code4 IID-MSE improvement over random_full4. Compare the direct-shift scaffold_full4 to Mirror scaffold_full4 without optimizing on audit.
- Report every failure, and label the trial *after-discovery follow-up*. Results do not imply any general Miracle Mirror advantage since pre and shift share the scaffold model's initial and trained code behavior.
