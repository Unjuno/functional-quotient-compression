# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **242 UNTESTED, 8 PROMISING, 4 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-019 — Mirror expert residual rank allocation**

Why next:
- MA-241, MA-244, MA-245, MA-247, MA-248, MA-249, MA-250, MA-251, MA-253, MA-003, MA-005 and MA-009 are checked; MA-019 is the next executable P0 candidate.
- its closest controls should compare Mirror expert residuals against shared bases and ordinary LoRA/low-rank controls (PA01–PA06).

If MA-019 is blocked, use this order:
MA-024.

## Active experiments

- MA-005 — branch `research/ma-005-signed-mirror-mixture-20261007`; directory `experiments/mirror_applications/ma-005-signed-mirror-mixture/`; worker/run `Codex session 2026-10-07`; start commit pending.





When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

- MA-003 — PROMISING: aligned synthetic top-1 Mirror experts passed routed-MSE gate in 2/3 fresh worlds with 35.8% fewer serialized bytes; independent experts needed private capacity; CPU inference was slower. Branch `research/ma-003-mirror-topk-expert-20261007`; report `experiments/mirror_applications/ma-003-mirror-topk-expert/README.md`; result commit `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`.
- MA-005 — PROMISING: signed Givens expert mixture passed aligned quality/storage 3/3 with 41.1% fewer serialized bytes; arbitrary experts required private capacity; current CPU runtime regressed sharply. Branch `research/ma-005-signed-mirror-mixture-20261007`; report `experiments/mirror_applications/ma-005-signed-mirror-mixture/README.md`; result commit `de68ec4c440e54fc4870734e9bbd1fdc99515628`.
- MA-009 — FAIL: one private rare-role expert plus common Mirror views used 0.774x full-MoE bytes and beat shared-only controls, but missed full-MoE relative MSE in 2/3 fresh worlds; arbitrary experts needed more capacity and CPU runtime regressed. Branch `research/ma-009-rare-private-mirror-expert-20261007`; report `experiments/mirror_applications/ma-009-rare-private-mirror-expert/README.md`; result commit `e2232f0382eb66721e90d7b295fb0b2c33eb9731`.
- MA-247 through MA-251 are complete and verified.

## Recently completed

- MA-251 — PROMISING: factorized Givens expert/depth coordinates matched untied quality 3/3 with 76.4% fewer bytes; independent pair functions required more capacity; CPU throughput regression recorded. Branch `research/ma-251-expert-depth-factorization-20261007`; report `experiments/mirror_applications/ma-251-expert-depth-factorization/README.md`; result commit `f494b8986a3807e512d0255811f62857758c7ca6`.

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
