# MA-1066 — TT-Rec tensor cores with role Mirror view

This experiment asks whether user and item roles can share one optimized TT-Rec table and recover useful role-specific behavior with a small Givens coordinate. It compares native independent TT tables, hard-shared TT, matched diagonal and FiLM controls, Givens Mirror, and full dense tables. The TT-EmbeddingBag upstream is pinned and vendored under `source/vendor/FBTT-Embedding` without modifications.

**Execution status:** SCREENING protocol/source preparation only. No model training or audit has run. The current container is CPU-only; GPU Control source review and an executable GPU profile are required before a result can be established. See [FEASIBILITY.md](FEASIBILITY.md).

The dataset is MovieLens 1M explicit ratings as binary preference, not the official Criteo/MLPerf-DLRM benchmark. GroupLens terms are in [DATASET_TERMS.md](DATASET_TERMS.md).
