# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **253 UNTESTED, 1 PROMISING**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-253 — cache-safe final-layer Mirror-MoE**

Why first:
- strong direct prior-art control exists (PA01);
- expert weights dominate MoE storage;
- it directly tests the project's "one physical object -> multiple logical functions" framing;
- failure cleanly bounds how much Mirror adds beyond ordinary expert tying.

If MA-253 is blocked, use this order:
MA-244 -> MA-245 -> MA-247 -> MA-248 -> MA-249 -> MA-250 -> MA-251 -> MA-003.

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

- MA-241 — PROMISING (3/3 synthetic quality/storage gate; CPU runtime regression); branch `research/ma-241-expert-tying-mirror-20261007`; report `experiments/mirror_applications/ma-241-expert-tying-mirror/README.md`; result commit `0ee183668285231d825e853c69c4791b9d252bf2`.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.
