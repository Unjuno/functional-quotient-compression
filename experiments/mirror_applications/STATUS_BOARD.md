# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **249 UNTESTED, 3 PROMISING, 2 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-248 — PTP random variable represented as packet Mirror code**

Why next:
- MA-241, MA-244, MA-245, MA-247 and MA-253 are now checked; MA-248 is the next untested cross-over P0 candidate.
- its closest prior-art control is packet-level stochastic conditioning in Parallel Token Prediction (PA10).

If MA-247 is blocked, use this order:
MA-249 -> MA-250 -> MA-251 -> MA-003.

## Active experiments



When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

- None. MA-247 is complete and verified.

## Recently completed

- MA-247 — FAIL at development screen; Mirror used fewer bytes than untied but had worse MSE than tied, scalar-gate, and static LoRA controls. Fresh worlds were not opened by the failure rule; branch `research/ma-247-recursive-depth-view-20261007`; report `experiments/mirror_applications/ma-247-recursive-depth-view/README.md`; result commit `62f0acf3f680ff3bbab8e0e20e194f4065f026e5`.

- MA-245 — PROMISING (aligned output/cache result; missed model-payload threshold; CPU slowdown); branch `research/ma-245-mlkv-layer-views-20261007`; report `experiments/mirror_applications/ma-245-mlkv-layer-views/README.md`; result commit `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`.
- MA-244 — PROMISING (aligned quality/cache mechanism; missed 20% model-payload gate; CPU slowdown); branch `research/ma-244-kv-role-view-20261007`; report `experiments/mirror_applications/ma-244-kv-role-view/README.md`; result commit `85de2fd65618d72bd0bf6a091b558a0dda57b741`.
- MA-253 — FAIL for Mirror expert replacement; cache-placement mechanics PASS; branch `research/ma-253-cache-safe-final-moe-20261007`; report `experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md`; result commit `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- MA-241 — PROMISING (3/3 synthetic quality/storage gate; CPU runtime regression); branch `research/ma-241-expert-tying-mirror-20261007`; report `experiments/mirror_applications/ma-241-expert-tying-mirror/README.md`; result commit `0ee183668285231d825e853c69c4791b9d252bf2`.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.
