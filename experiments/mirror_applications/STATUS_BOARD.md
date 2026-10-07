# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **750**
- Current MA statuses: **741 UNTESTED, 6 PROMISING, 3 FAIL**
- Historical SRM/TM lanes are evidence, not MA statuses.

## Next candidate

**MA-692 — RoPE-commuting Mirror KV Views**

Why next:
- MA-691 established exact lazy canonical-cache reuse for known K/V right-transforms in 3/3 fresh seeds;
- MA-692 tests the cleanest positional-encoding compatibility constraint;
- no existing `research/ma-692*` branch was found in the latest concurrency check;
- MA-693..700 already form a coherent preregisterable KV follow-up sequence.

Recommended existing KV order:
MA-692 -> MA-693 -> MA-694 -> MA-695 -> MA-696 -> MA-697 -> MA-698 -> MA-699 -> MA-700.

After that, use `KV_MOE_FOLLOWUP_QUEUE.md`.

## Active experiments

No MA-692..700 branch was found during the latest research-support concurrency check.

A worker may already be running from another checkout. Before claiming any ID:
1. search existing `research/ma-*` branches;
2. re-read the live registry;
3. use the existing ID if already registered;
4. never allocate an ID below the current maximum.

## Recently completed / verified

- **MA-691 — PROMISING**: exact lazy canonical-cache mechanism passed 3/3 fresh seeds; max lazy/materialized error 3.13e-7; RoPE-plane commutation and MLA absorption diagnostics passed. Branch `research/ma-691-lazy-kv-mirror-20261007`; verification `3643118351eb026c31fc00802e47381e8cdfba93`.
- **MA-250 — PROMISING**: aligned linear expert roles matched untied quality 3/3 with 38.3% lower payload; independent roles required much more private capacity. Branch `research/ma-250-vsa-expert-address-20261007`; verified result `05ad4f7efb92a3b7bbe8f4f0674377223d6bc768`.
- **MA-249 — PROMISING**: aligned future-head Views matched ordinary MTP top-1 3/3 with 32.5% lower payload; eager CPU was slower. Branch `research/ma-249-future-head-views-20261007`; verified result `20a0984965e879fed8c1b085f02fa5aefc68828b`.
- **MA-248 — FAIL**: Mirror packet View did not beat the simpler broadcast shared-code control; independent-entropy boundary remained unresolved. Branch `research/ma-248-packet-mirror-code-20261007`; verified result `433a2229362db23bcd43b3830358e674799c6247`.
- **MA-247 — FAIL**: recurrent depth Givens View lost to hard tying/scalar gate/static LoRA on development; fresh worlds correctly remained unopened. Branch `research/ma-247-recursive-depth-view-20261007`; verified result `62f0acf3f680ff3bbab8e0e20e194f4065f026e5`.
- **MA-245 — PROMISING**: aligned MLKV layer/role Views; cache-state saving passed mechanism check, strict model-payload threshold missed; CPU slowdown. Verified result `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`.
- **MA-244 — PROMISING**: aligned shared K/V role/head Views; physical cache halved versus MQA in synthetic fixture; strict whole-model payload gate missed. Verified result `85de2fd65618d72bd0bf6a091b558a0dda57b741`.
- **MA-253 — FAIL** for narrow Mirror replacement on independent rank-2 variation; final-FFN cache-safe placement itself passed. Verified result `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- **MA-241 — PROMISING**: tied-expert layer Views passed aligned quality/storage gate 3/3; eager CPU runtime regressed. Verified result `0ee183668285231d825e853c69c4791b9d252bf2`.

See:
- `docs/phase2/LATEST_WORKER_FINDINGS.md`
- `docs/phase2/MIRROR_KV_CACHE_REUSE.md`
- `docs/phase2/MIRROR_KV_MOE_GENERALIZATION_2026-10-07.md`

## Blocked

None recorded.

## Concurrency rule

Before starting or allocating:
- search live `research/ma-*` branches;
- registry row is authoritative for scientific status;
- dedicated verified branches are authoritative for measurements;
- research-support branches may consolidate findings but must not silently overwrite dedicated protocols.

## Status policy

If the board and registry disagree, repair the board from the registry and dedicated verified experiment branch.
