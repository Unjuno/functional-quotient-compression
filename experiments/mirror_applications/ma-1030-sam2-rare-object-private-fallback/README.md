# MA-1030 — SAM2 rare-object memory private fallback

Status: **FAIL**, scoped to the synthetic CPU tracking world. A one-slot causal fallback halves object-memory bytes versus independent state, but chooses the wrong, persistently distinct object on every yielded frame. It trails both independent memory and the native rank-1 additive control; complete-session savings are 0.24% only.

- Frozen test and exact storage contract: `PROTOCOL.json`
- Development threshold selection and pre-audit finding: `DEVELOPMENT.md`
- Fresh measurements, counter-explanation and limits: `RESULTS.md`
- Per-world metrics: `RESULTS_CORE.csv`
- Source/cache/data/payload/prediction hashes and verification: `VERIFICATION.json`, `runs/`

The fallback rule uses only current model memory tensors; it never reads future masks or future prediction errors. This result does not establish performance on natural video or disprove temporal-change-aware private allocation.
