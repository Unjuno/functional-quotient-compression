# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 995-row registry)

- Registered candidates: **995**
- P0: **501** (36 completed; 465 UNTESTED)
- P1: **391** (12 completed; 379 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **947 UNTESTED, 29 PROMISING, 19 FAIL**
- 48 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-255 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-255 remains next.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

After the completed MA-299 draw, randomly select the next executable candidate from `[MA-274, MA-286, MA-296]` using a new OS-backed draw. MA-274 has a competing research branch and remains eligible for evidence adjudication if drawn. Record the frozen pool and draw result in the selected experiment protocol. Do not choose by list order.

## Active experiment

No active experiment was declared on either inspected baseline/status chain at reconciliation. Before claiming MA-255, check live research/ma-* branches again; this statement is not a realtime worker lock.

## Verified status index

- **PROMISING (29):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-691.
- **FAIL (19):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-299.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## 2026-10-08 research sweep

New directions are cross-model cache translators, neural graphics and 4D Gaussian fields, speaker-adaptive synthesis, generative flow/solver coordinates, and cross-model stitching. They are **UNTESTED**. See [current research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md). The new cross-model cache lane distinguishes exact MA-691 algebra from approximate learned transfer.

## Twelfth literature intake — 2026-10-08

Additional Mirror m insertion targets: neural video chunk sharing (NerVast/DCVC-UF), partial learned equivariance, spiking thresholds/time gains, physical photonic phase configurations, beamforming/CSI/RIS and personalized HRTF neural fields. All candidates are UNTESTED and require native efficient controls, full bytes and domain runtime/energy. See [twelfth sweep notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md) and PA266..295.

## 2026-10-08 MA-299 result

**MA-299 — FAIL for Mirror-specific advantage.** Mirror split-on-share used 2,337B and accepted four views plus one private task; the two-coefficient control made the same allocations and quality at 2,352B. The 15B total saving (0.64%) missed the preregistered 10% gate and fit compute was 1.74× higher. Both delayed private allocation versus 6,417B independent full weights on this deliberately aligned stream. Unrelated tasks still require private weights. See [MA-299 report](ma-299-split-on-share-mirror/README.md) and the [MA-297/299 family diagnostic](../../docs/phase2/MA297_MA299_SETA_FAMILY_DIAGNOSTIC_2026-10-08.md).

## Main scientific findings

- Aligned functional variation often admits a compact Mirror View, including experts, QKV, future heads, depth and structured codecs.
- Arbitrary unrelated functions are not made independent by cheap address combinatorics; private residuals are often necessary.
- In many cases a simple baseline (MQA, broadcast, FiLM, rank-1/2) wins or dominates the Mirror candidate.
- Many promising byte points have unfavorable eager runtime.
- MA-691 establishes an exact canonical KV-cache reuse algebra within specified transform conditions; optimized end-to-end cache switching is not yet established.

## Concurrency and truth policy

- The **995-row** `IDEA_REGISTRY.csv` is authoritative for status and candidate identity. Do not merge or overwrite it with an older 254-row experiment checkout.
- `CLAIM_LEDGER.csv` and each experiment's `VERIFICATION.json` are the evidence index; `STATUS_BOARD.md` is an operational cache.
- Before allocating an ID or starting work, re-read the live registry and search for experiment branches.
- Preserve old branches, failed results, exploratory data and locked protocols. No automatic merge to main.
