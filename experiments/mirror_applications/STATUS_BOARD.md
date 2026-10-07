# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **246 UNTESTED, 5 PROMISING, 3 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-251 — factorized expert x depth Mirror coordinate**

Why next:
- MA-241, MA-244, MA-245, MA-247, MA-248, MA-249, MA-250 and MA-253 are now checked; MA-251 is the next untested cross-over P0 candidate.
- its closest prior-art controls are expert tying and depth-specific adapters (PA01/PA06).

If MA-251 is blocked, use this order:
MA-003.

## Active experiments





When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

- None. MA-247, MA-248, MA-249 and MA-250 are complete and verified.

## Recently completed

- MA-250 — PROMISING on aligned linear expert roles: Mirror matched untied quality 3/3 with 38.3% fewer payload bytes; independent roles required private/full weights; fixed MAP/Hadamard/HRR codes did not fit this Givens-aligned teacher. Branch `research/ma-250-vsa-expert-address-20261007`; report `experiments/mirror_applications/ma-250-vsa-expert-address/README.md`; result commit `05ad4f7efb92a3b7bbe8f4f0674377223d6bc768`.

- MA-249 — PROMISING on aligned synthetic head sharing: 3/3 fresh worlds matched MTP top-1 with 32.5% lower full model payload; CPU inference throughput 0.42x MTP; independent-head control required more private degrees of freedom. Branch `research/ma-249-future-head-views-20261007`; report `experiments/mirror_applications/ma-249-future-head-views/README.md`; result commit `20a0984965e879fed8c1b085f02fa5aefc68828b`.

- MA-248 — FAIL for Mirror-specific frontier: packet Givens views passed 2/3 correlated fresh worlds, while broadcast shared code passed 3/3 at 378 fewer serialized bytes; independent entropy boundary NOT ESTABLISHED. Branch `research/ma-248-packet-mirror-code-20261007`; report `experiments/mirror_applications/ma-248-packet-mirror-code/README.md`; result commit `433a2229362db23bcd43b3830358e674799c6247`.

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
