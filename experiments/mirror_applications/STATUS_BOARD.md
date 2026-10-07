# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **300**
- P0: **55**
- P1: **151**
- P2: **94**
- Current MA statuses: **300 UNTESTED**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-241 — expert tying across depth + layer-specific Mirror expert views**

Why first:
- strong direct prior-art control exists (PA01);
- expert weights dominate MoE storage;
- it directly tests the project's "one physical object -> multiple logical functions" framing;
- failure cleanly bounds how much Mirror adds beyond ordinary expert tying.

If MA-241 is blocked, use this order:
MA-253 -> MA-244 -> MA-245 -> MA-247 -> MA-248 -> MA-249 -> MA-250 -> MA-251 -> MA-003.

## Active experiments

No active MA experiment is registered on this board in this baseline commit. A worker may already be running from an earlier checkout; it must claim its MA ID/branch here before another worker starts the same ID.

When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

Remove from Active only after STATUS.md and VERIFICATION.json are committed.

## Recently completed

None in MA namespace.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.
