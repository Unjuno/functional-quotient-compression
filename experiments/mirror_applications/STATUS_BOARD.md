# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 995-row registry)

- Registered candidates: **995**
- P0: **501** (36 completed; 465 UNTESTED)
- P1: **391** (12 completed; 379 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **947 UNTESTED, 30 PROMISING, 18 FAIL**
- 48 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-255 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-255 remains next.
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

- **PROMISING (30):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-691, MA-282.
- **FAIL (18):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## 2026-10-08 research sweep

New directions are cross-model cache translators, neural graphics and 4D Gaussian fields, speaker-adaptive synthesis, generative flow/solver coordinates, and cross-model stitching. They are **UNTESTED**. See [current research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md). The new cross-model cache lane distinguishes exact MA-691 algebra from approximate learned transfer.

## Twelfth literature intake — 2026-10-08

Additional Mirror m insertion targets: neural video chunk sharing (NerVast/DCVC-UF), partial learned equivariance, spiking thresholds/time gains, physical photonic phase configurations, beamforming/CSI/RIS and personalized HRTF neural fields. All candidates are UNTESTED and require native efficient controls, full bytes and domain runtime/energy. See [twelfth sweep notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md) and PA266..295.

## 2026-10-08 MA-282 result

**MA-282 — PROMISING, narrow aligned synthetic FFN screen.** One shared 8x12x8 FFN plus one scalar Monarch view code per task reproduced four aligned functions at 925 serialized bytes vs 3,279B independent full FFNs (0.282×), exactly in 3/3 fresh worlds. Unrelated functions needed private task weights. Free Monarch factors matched aligned quality at 1,145B; simpler controls traded quality, bytes and runtime. Three invalid instrumentation runs are preserved and excluded. See the [MA-282 report](ma-282-monarch-mirror-ffn/README.md).

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
