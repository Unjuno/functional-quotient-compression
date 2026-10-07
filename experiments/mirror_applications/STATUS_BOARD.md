# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **690**
- P0: **273**
- P1: **315**
- P2: **102**
- Current MA statuses: **686 UNTESTED, 3 PROMISING, 1 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-247 — recursive shared block + Mirror depth modulation**

Why next:
- MA-241, MA-244 and MA-245 produced aligned feasibility signals;
- MA-253 showed a narrow Givens view fails on deliberately independent rank-2 private variation;
- MA-247 tests the same physical-to-logical principle on reused depth with strong controls from PA06;
- protocol should use the orbit/private frontier guidance in `docs/phase2/LATEST_WORKER_FINDINGS.md` rather than an aligned-only teacher if feasible.

If MA-247 is blocked, continue:
MA-248 -> MA-249 -> MA-250 -> MA-251 -> MA-003, then follow WORKER_QUEUE.md.

## Active experiments

No active experiment is registered on this baseline at the time of this update.

A worker may create a dedicated branch immediately after this commit. Before another worker claims the same ID, search existing `research/ma-*` branches.

## Recently completed

- **MA-245 — PROMISING**: aligned MLKV layer/role views; 50% cache-state saving vs two-group MLKV, 11.3% serialized model saving (missed 25% gate), eager CPU slowdown. Verified commit `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`.
- **MA-244 — PROMISING**: aligned K/V role/head views; 50% cache-state saving vs MQA, 10.6% serialized model saving (missed 20% gate), eager CPU slowdown. Verified commit `85de2fd65618d72bd0bf6a091b558a0dda57b741`.
- **MA-253 — FAIL** for Mirror expert replacement; cache-safe final-FFN placement PASS. Verified commit `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- **MA-241 — PROMISING**: aligned tied-expert layer views passed 3/3 synthetic quality/storage gates; runtime regressed. Verified commit `0ee183668285231d825e853c69c4791b9d252bf2`.

See `docs/phase2/LATEST_WORKER_FINDINGS.md` for the shared interpretation and mandatory next-design rules.

## Blocked

None.

## Concurrency rule

Before allocating a new MA ID, re-read the live registry and use max existing ID + 1. After editing, re-read and verify zero duplicate IDs.

## Status policy

The registry row is authoritative for scientific status. This board is an operational cache. If they disagree, fix the board from the registry.
