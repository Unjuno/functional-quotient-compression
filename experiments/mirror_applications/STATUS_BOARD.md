# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **252 UNTESTED, 1 PROMISING, 1 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-244 — K=V projection sharing + Mirror role recovery**

Why next:
- MA-241 and MA-253 are now checked; MA-244 is the next untested cross-over P0 candidate.
- its closest prior-art controls are QKV/K=V projection sharing (PA07).

If MA-244 is blocked, use this order:
MA-245 -> MA-247 -> MA-248 -> MA-249 -> MA-250 -> MA-251 -> MA-003.

## Active experiments

- MA-244 — branch `research/ma-244-kv-role-view-20261007`; directory `experiments/mirror_applications/ma-244-kv-role-view/`; worker/run `Codex session 2026-10-07`; start commit `5b736a0f7cfca9c3f7794005dfb70954f155ea02`.

When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

Remove from Active only after STATUS.md and VERIFICATION.json are committed.

## Recently completed

- MA-253 — FAIL for Mirror expert replacement; cache-placement mechanics PASS; branch `research/ma-253-cache-safe-final-moe-20261007`; report `experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md`; result commit `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- MA-241 — PROMISING (3/3 synthetic quality/storage gate; CPU runtime regression); branch `research/ma-241-expert-tying-mirror-20261007`; report `experiments/mirror_applications/ma-241-expert-tying-mirror/README.md`; result commit `0ee183668285231d825e853c69c4791b9d252bf2`.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.
