# MA-1065 — QR complementary tables with Mirror field views

Tests whether user and item roles can share one complementary quotient/remainder embedding table pair and recover useful field-specific behavior with a small Givens code. The native QR implementation multiplies quotient and remainder table vectors elementwise, following PA339. Controls include independent QR per role, a shared hash table, matched diagonal/FiLM views, and full tables.

The data split excludes all pairs scored by MA-1063 and MA-1064. This uses MovieLens explicit preference classification as a compact recommender screen, not Criteo CTR or an official QR-Rec reproduction. GroupLens access terms are in `DATASET_TERMS.md`.

**Outcome:** scoped development-only FAIL. Independent QR, shared QR, and shared QR+Givens remained at AUC≈0.5 after a 15-epoch dev diagnostic; shared QR+FiLM reached 0.763 and full tables 0.808. Because the quality premise failed, the sealed fresh audit was not opened. Payload and audit comparisons are unestablished. See [STATUS.md](STATUS.md).
