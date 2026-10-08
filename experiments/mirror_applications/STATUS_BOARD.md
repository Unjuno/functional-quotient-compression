# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 935-row registry)

- Registered candidates: **935**
- P0: **454** (36 completed; 418 UNTESTED)
- P1: **378** (12 completed; 366 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **887 UNTESTED, 30 PROMISING, 18 FAIL**
- 48 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). Prior sequential candidate recommendations are superseded by the randomized selection procedure in WORKER_QUEUE.md.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

Selection is randomized per the current worker instruction; no fixed next MA ID is preassigned. MA-276 was drawn from the second-expansion P0 pool and is PROMISING for an aligned serialized-state frontier. Its report separates a 2,048 B execution workspace from payload bytes.

## Active experiment

MA-276 completed on `research/ma-276-boft-depth-view-20261008`. Other workers may own overlapping IDs; this board is not a realtime lock.

## Verified status index

- **PROMISING (30):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-276, MA-691.
- **FAIL (18):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## 2026-10-08 research sweep

New directions are cross-model cache translators, neural graphics and 4D Gaussian fields, speaker-adaptive synthesis, generative flow/solver coordinates, and cross-model stitching. They are **UNTESTED**. See [current research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md). The new cross-model cache lane distinguishes exact MA-691 algebra from approximate learned transfer.

## Main scientific findings

- Aligned functional variation often admits a compact Mirror View, including experts, QKV, future heads, depth and structured codecs.
- Arbitrary unrelated functions are not made independent by cheap address combinatorics; private residuals are often necessary.
- In many cases a simple baseline (MQA, broadcast, FiLM, rank-1/2) wins or dominates the Mirror candidate.
- Many promising byte points have unfavorable eager runtime.
- MA-276 PROMISING, aligned operator scope: one tied block plus scalar BOFT depth views matched the four-step teacher in 396 B; deterministic prepared transforms consume an additional 2,048 B runtime workspace, so payload compression does not imply RAM compression.
- MA-691 establishes an exact canonical KV-cache reuse algebra within specified transform conditions; optimized end-to-end cache switching is not yet established.

## Concurrency and truth policy

- The **875-row** `IDEA_REGISTRY.csv` is authoritative for status and candidate identity. Do not merge or overwrite it with an older 254-row experiment checkout.
- `CLAIM_LEDGER.csv` and each experiment's `VERIFICATION.json` are the evidence index; `STATUS_BOARD.md` is an operational cache.
- Before allocating an ID or starting work, re-read the live registry and search for experiment branches.
- Preserve old branches, failed results, exploratory data and locked protocols. No automatic merge to main.
