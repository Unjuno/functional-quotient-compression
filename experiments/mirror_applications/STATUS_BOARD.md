# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **740**
- P0: **308**
- P1: **330**
- P2: **102**
- Current MA statuses: **733 UNTESTED, 4 PROMISING, 3 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-249 — one physical future head + Mirror future-head views**

Why next:
- MA-247 and MA-248 are complete scoped FAIL results and should not be rerun under the same protocols;
- MA-249 is the next untested locked cross-over P0 candidate;
- MA-701..740 are newly researched KV/cache candidates and enter after currently locked worker order unless a blocker makes the temporal lane unavailable.

If MA-249 is blocked, continue:
MA-250 -> MA-251 -> MA-003, then follow WORKER_QUEUE.md. KV/cache expansion candidates begin at MA-701.

## Active experiments

No active experiment is registered on this baseline at the time of this update.

A worker may create a dedicated branch immediately after this commit. Before another worker claims the same ID, search existing `research/ma-*` branches.

## Recently completed

- **MA-248 — FAIL**: packet Mirror code did not establish a Mirror-specific frontier; broadcast code was cheaper and more consistent on correlated worlds. Verified commit `433a2229362db23bcd43b3830358e674799c6247`.
- **MA-247 — FAIL**: recurrent depth Givens View lost to scalar gating and static rank-1 LoRA on the development screen; fresh worlds were not opened by protocol. Verified commit `62f0acf3f680ff3bbab8e0e20e194f4065f026e5`.

- **MA-691 — PROMISING**: exact lazy canonical-cache algebra passed 3/3 fresh seeds; max lazy/materialized error 3.13e-7, RoPE-plane commutation and MLA latent absorption passed; CPU runtime exploratory. Branch `research/ma-691-lazy-kv-mirror-20261007`; verification commit `3643118351eb026c31fc00802e47381e8cdfba93`.

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
