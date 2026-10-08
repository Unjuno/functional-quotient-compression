# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 875-row registry)

- Registered candidates: **875**
- P0: **408** (43 completed; 365 UNTESTED)
- P1: **364** (12 completed; 352 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **820 UNTESTED, 32 PROMISING, 23 FAIL**
- 47 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-274 — BOFT logical expert views (P0; PA21)**

Required controls: independent BOFT experts, shared BOFT, ordinary rank-one angle factorization, and full expert upper reference. Count router and active expert compute.

## Recent experiments

MA-255 PROMISING (aligned synthetic orbit); MA-260 FAIL (linear ensemble byte gate missed); MA-261 PROMISING (aligned Givens experts, oracle router); MA-265 FAIL (4.4% bytes saved vs VeRA); MA-268 PROMISING (15.7% vs IA3, gate missed); MA-271 FAIL (identical rank-one angle control); MA-272 FAIL (scalar-skew control matched and input-side latency gate missed; corrected new-seed audit); MA-273 FAIL (BOFT task orbit compressed, but simple factorized-angle control identical; initial payload replay bug corrected using new seeds).

See dedicated 2026-10-08 research branches for protocols, results and verification. No automatic merge to main.

## Verified status index

- **PROMISING (32):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-255, MA-261, MA-268, MA-691.
- **FAIL (23):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-260, MA-265, MA-271, MA-272, MA-273.

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
