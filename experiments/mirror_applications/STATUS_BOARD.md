# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 875-row registry)

- Registered candidates: **875**
- P0: **408** (35 completed; 373 UNTESTED)
- P1: **364** (12 completed; 352 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **828 UNTESTED, 29 PROMISING, 18 FAIL**
- 47 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-255 — Mirror context superposition for task models (P0; PA16)**

Reason:
- all previously locked MA-241/244/245/247–251/253 and subsequent old-chain P0 screens have completed;
- the original P0 cross-over queue is exhausted among checked candidates;
- MA-255 is the first untested P0 in the earlier high-information literature cross-over queue;
- direct Parameter Superposition prior art gives a strong nearest control for insertion of the extra Mirror parameter `m`.

Required controls: native Parameter Superposition, naive/shared task-code basis, matched byte-near low-rank or VeRA-style modulation, independent-model upper reference where practical. Do not claim the superposition concept itself as a Mirror invention.

If blocked by a documented reproducibility or harness issue, record it and resume at MA-260 (BatchEnsemble), then MA-261/265/268 in the registered literature priority sequence. Do not jump to P1/novelty-picked topics before P0.

## Active experiment

No active experiment was declared on either inspected baseline/status chain at reconciliation. Before claiming MA-255, check live research/ma-* branches again; this statement is not a realtime worker lock.

## Verified status index

- **PROMISING (29):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-691.
- **FAIL (18):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## Main scientific findings

- Aligned functional variation often admits a compact Mirror View, including experts, QKV, future heads, depth and structured codecs.
- Arbitrary unrelated functions are not made independent by cheap address combinatorics; private residuals are often necessary.
- In many cases a simple baseline (MQA, broadcast, FiLM, rank-1/2) wins or dominates the Mirror candidate.
- Many promising byte points have unfavorable eager runtime.
- MA-691 establishes an exact canonical KV-cache reuse algebra within specified transform conditions; optimized end-to-end cache switching is not yet established.

## Concurrency and truth policy

- The **875-row** `IDEA_REGISTRY.csv` is authoritative for status and candidate identity. Do not merge or overwrite it with an older 254-row experiment checkout.
- `CLAIM_LEDGER.csv` and each experiment's `VERIFICATION.json` are the evidence index; `STATUS_BOARD.md` is an operational cache.
- Before allocating an ID or starting work, re-read the live registry and search for experiment branches.
- Preserve old branches, failed results, exploratory data and locked protocols. No automatic merge to main.
