# MA-1059 — Natural tourism-series View geometry

This experiment tests whether one shared rank-2 Chronos output adapter plus a per-series angle can recover useful monthly forecast behavior with much less adapter state than independent LoRA.

AM-01 invalidated the first development result after review found a pooled-hook leak and a residual normalization error. The corrected setting was frozen before the 24-series fresh audit. The corrected audit is a scoped FAIL: Mirror misses the native, pooled and diagonal quality gates, while seasonal naive performs substantially better. Exact replay from serialized method payloads passed. See `RESULTS.md` for H/T/D/C/U, facts, interpretation and limits.

Actual full inference payloads include the pinned Chronos checkpoint, method state, metadata and residuals. T1–T8 remain excluded due to an earlier metadata preview; no raw data is redistributed.
