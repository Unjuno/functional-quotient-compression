# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **239 UNTESTED, 9 PROMISING, 6 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-048 — physical-4 to logical-16 heads**

Why next:
- MA-241, MA-244, MA-245, MA-247, MA-248, MA-249, MA-250, MA-251, MA-253, MA-003, MA-005, MA-009, MA-019, MA-024 and MA-041 are checked; MA-048 is the next executable P0 candidate.
- its closest controls should compare ordinary MHA and learned/shared projection views, including generated adapters (PA04; PA07).

If MA-048 is blocked, use this order:
MA-061 -> MA-063.

## Active experiments

- None. MA-003, MA-005, MA-009, MA-019, MA-024 and MA-041 have completed; MA-048 is next.


When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

- MA-003 — PROMISING: aligned synthetic top-1 Mirror experts passed routed-MSE gate in 2/3 fresh worlds with 35.8% fewer serialized bytes; independent experts needed private capacity; CPU inference was slower. Branch `research/ma-003-mirror-topk-expert-20261007`; report `experiments/mirror_applications/ma-003-mirror-topk-expert/README.md`; result commit `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`.
- MA-005 — PROMISING: signed Givens expert mixture passed aligned quality/storage 3/3 with 41.1% fewer serialized bytes; arbitrary experts required private capacity; current CPU runtime regressed sharply. Branch `research/ma-005-signed-mirror-mixture-20261007`; report `experiments/mirror_applications/ma-005-signed-mirror-mixture/README.md`; result commit `de68ec4c440e54fc4870734e9bbd1fdc99515628`.
- MA-009 — FAIL: one private rare-role expert plus common Mirror views used 0.774x full-MoE bytes and beat shared-only controls, but missed full-MoE relative MSE in 2/3 fresh worlds; arbitrary experts needed more capacity and CPU runtime regressed. Branch `research/ma-009-rare-private-mirror-expert-20261007`; report `experiments/mirror_applications/ma-009-rare-private-mirror-expert/README.md`; result commit `41020ba0aa45d9337fec68f7f772d8d5076534a6`.
- MA-019 — FAIL at development: one-angle Mirror missed the full-MoE quality/storage gates; generic rank-2 basis used only 64B more and fit the aligned teacher much better at the same compute proxy. Fresh worlds were not opened. Branch `research/ma-019-mirror-coefficient-basis-20261007`; report `experiments/mirror_applications/ma-019-mirror-coefficient-basis/README.md`; result commit `b880e34edb8210ad318091be2e17e170ef09c5a0`.
- MA-024 — FAIL at development: one-angle virtual LoRA used 2,401B vs 2,657B generic shared basis, but generic basis had much lower MSE with the same compute proxy; fresh worlds not opened. Branch `research/ma-024-virtual-lora-mirror-20261007`; report `experiments/mirror_applications/ma-024-virtual-lora-mirror/README.md`; result commit `4a41b4af333d181bd873f2f79751177f8c1c9a95`.
- MA-041 — PROMISING: aligned attention output/payload gates passed 3/3 with 32.8% fewer bytes; head-level contribution quality and CPU runtime lagged, independent QKV required private weights. Branch `research/ma-041-mirror-attention-heads-20261007`; report `experiments/mirror_applications/ma-041-mirror-attention-heads/README.md`; result commit `359111bc9cc649fe29ad8793035c237dbf211cba`.
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
