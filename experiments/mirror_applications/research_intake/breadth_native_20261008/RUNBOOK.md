# Breadth intake — independent experiment runbook

**Status:** six new designs MA-1177..1182. No execution started, no trained model wins claimed. This runbook is sufficient to start a CPU mechanism test without first reading all literature papers. Publication-grade native comparison still requires verifying released source.

## Procedure

1. Open a specific `plans/MA-xxxx/PROTOCOL.json` and verify `plan_only:true`, `worker_claim:false`, native controls, source/dev/fresh masks and gate values.
2. Create an isolated worktree/branch from a pinned commit; never commit to current active worker branch. Keep original source benchmarks read-only.
3. Build a tiny local source-world fixture matching the plan's exact dimensions and train/test identity split; match the native function before inserting Mirror m.
4. Implement **native full** vs **native same-byte simple factorized** vs **native+Mirror** with matched data/updates/hardware. Run unit shape/oracle-leakage/identical-gauge tests, assert all save/load byte rounds.
5. Development seeds 11,12,13 only: select rank and LR; freeze code hash before auditing fresh 101–105. For an already used fresh set, preregister a *new* distinct one.
6. Materialize actual serializer payload. Include tokenizer/decoder, basis/role code, index, quantization scales, router, resident expert buffers, cache copies and metadata. Track physical RAM/HBM separately from file bytes.
7. Measure quality (CE, AUPRC, Pearson, etc in their native units), retention/OOD, actual P50/P95 runtime with warmups and synchronization, and active compute (MAC/FLOP proxy), not a speculative compression factor.
8. Report every fresh world and FAIL/UNCERTAIN honestly. New scientific statuses are never automatically written to `IDEA_REGISTRY.csv` by these plans.

**Suggested fast lane:** a CPU torch small synthetic fixture 32/64 dimensions as detailed by each MA; checkpoint hashes, exact data and code license mandatory only for natural source model lane. Biology tasks require molecule/protein/gene identity holdout to avoid data leakage.

## Accounting constraints

Quality, byte, and seconds are distinct axes. Source-only shared basis training time counts separately (amortization and onboarding). Native one-pass multi-head cannot be treated as a K-times-full-forward baseline. Condition code entropy may have no independent Shannon capacity; different output coordinates only count if useful output function changes.

## File contract

The design folder contains `README.md`, `PROTOCOL.json`, `STATUS.md`. Activated experiment directory must also contain `source/`, `tests/`, pinned native reference and full `RESULTS_CORE.csv` plus `VERIFICATION.json`, with fresh all-world rows and a separate replay report. Existing MA status immutable unless full evidence is explicitly integrated later.

## Branch separation

This is the research branch `research/mirror-breadth-method-sweep-20261008`, not main or the worker-ready canonical branch. MA-1175 is not a privileged next-candidate. Worker continues its own queue.
