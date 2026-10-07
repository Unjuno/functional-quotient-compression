# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **250 UNTESTED, 3 PROMISING, 1 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-247 — recursive shared block + Mirror depth modulation**

Why next:
- MA-241, MA-244, MA-245 and MA-253 are now checked; MA-247 is the next untested cross-over P0 candidate.
- its closest prior-art controls are recursive tied blocks and generated per-step modulation (PA06).

If MA-247 is blocked, use this order:
MA-248 -> MA-249 -> MA-250 -> MA-251 -> MA-003.

## Active experiments

None.

When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

Remove from Active only after STATUS.md and VERIFICATION.json are committed.

## Recently completed

- MA-245 — PROMISING (aligned output/cache result; missed model-payload threshold; CPU slowdown); branch `research/ma-245-mlkv-layer-views-20261007`; report `experiments/mirror_applications/ma-245-mlkv-layer-views/README.md`; result commit `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`.
- MA-244 — PROMISING (aligned quality/cache mechanism; missed 20% model-payload gate; CPU slowdown); branch `research/ma-244-kv-role-view-20261007`; report `experiments/mirror_applications/ma-244-kv-role-view/README.md`; result commit `85de2fd65618d72bd0bf6a091b558a0dda57b741`.
- MA-253 — FAIL for Mirror expert replacement; cache-placement mechanics PASS; branch `research/ma-253-cache-safe-final-moe-20261007`; report `experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md`; result commit `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- MA-241 — PROMISING (3/3 synthetic quality/storage gate; CPU runtime regression); branch `research/ma-241-expert-tying-mirror-20261007`; report `experiments/mirror_applications/ma-241-expert-tying-mirror/README.md`; result commit `0ee183668285231d825e853c69c4791b9d252bf2`.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.
