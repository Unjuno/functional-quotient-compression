# SRM003 execution record

Original plan commit:9f03af294dc43ca3c53fef23a429f321813d9438. The original PLAN.md remains a historical preregistration rather than being rewritten after outcomes.

- [x] Recover SRM001/002 artifacts and verify CRC/SHA; inspect repository.
- [x] Write failing core tests, implement causal models and compaction, pass tests.
- [x] Development-only LR comparison, then freeze training code and fresh seeds.
- [x] Run15 primary fresh models and9 pruning/small-from-start controls.
- [x] Run9 longer continuations and6 pair-only controls.
- [x] Run predicted-intermediate replay, no true intermediate state.
- [x] Audit182 inference payloads and52 final-model metric records, plus3 deterministic replays.
- [x] Publish runnable source/tests/counts/limits and update current repository navigation.

19 tests pass in the new experiment suite. Historical repository-wide tests were not rerun because those sources were unchanged. Scientific task-readiness and hybrid superiority gates failed. Numerical pruning tolerance passed at low composition quality; this is not useful-quality compression or recovered semantic private masks.
