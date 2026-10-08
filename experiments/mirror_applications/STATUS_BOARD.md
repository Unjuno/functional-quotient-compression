# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1115-row registry)

- Registered candidates: **1115**
- P0: **597** (38 completed; 559 UNTESTED)
- P1: **415** (12 completed; 403 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1065 UNTESTED, 30 PROMISING, 20 FAIL**
- 50 experiment directories, complete with status/protocol/results/verification files, are indexed in this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-265 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-265 remains next.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-265 remains next.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-265 is next.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-265 — VeRA Mirror scaling code bank (P0; PA18)**

MA-255 is reconciled as PROMISING only for its aligned post-fit representation screen; a distinct 1,200-update protocol variant failed at development and remains sealed on fresh worlds. See `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`.

MA-260 is complete FAIL: the one-layer linear screen missed its predeclared byte gate, while the independent-task stress case collapsed. It is not a deep BatchEnsemble result; see its report and verification.

MA-261 is complete FAIL after audit: its post-fit branch's raw MSE ratios meet the frozen <=1.10x independent quality criterion in only 1/4 fresh worlds, despite source wording claiming 4/4. A different fixed-update protocol also failed during development. Both variants are retained in its reconciliation report.

MA-265 is the next P0. Dedicated branches already exist; inspect their frozen protocols and verification records before starting or importing anything.

## Active experiment

No active experiment declared. MA-255–261 screened IDs are indexed. MA-265 branches exist and require protocol/verification reconciliation before status assignment.

## Verified status index

- **PROMISING (30):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-255, MA-691.
- **FAIL (20):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-260, MA-261.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## Newly reconciled result

- MA-255 — PROMISING only for an aligned post-fit representation screen: 734B Mirror payload achieved 6.68e-17 fresh MSE on seeds 101/211/307/401 versus 6,490B implemented PSP, 14,650B rank-2 task code, and 27,906B independent. A separate 1,200-update protocol variant failed its development quality gate (Mirror MSE 0.0922 vs 4.52e-5 independent; fresh sealed). These are different protocols, not a replication pair. See `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`. Source branches: `research/ma-255-mirror-context-superposition-20261008` and `research/ma-255-mirror-context-superposition-replication-20261008`.
- MA-260 — FAIL: on four fresh rotation-aligned worlds Mirror matched member accuracy but used 890B, only 27.4% below implemented BatchEnsemble (1,226B) and 3.9% below independent (926B), missing its <=25% byte-ratio gate. On independent task separators its accuracy fell to 0.6018 vs 0.8332 controls. Post-fit one-layer linear screen only; BatchEnsemble rank-one is a weak single-output control, and deep ensembles remain untested. Source result `33da88ba71cb053a8729a393ccc1ed1a2cc4fcf3`, report `experiments/mirror_applications/ma-260-batchensemble-mirror/README.md`.
- MA-261 — FAIL: post-fit aligned Givens views used 1,006B vs 1,562B BatchEnsemble and 1,794B independent, with tiny absolute error, but the frozen <=1.10x independent MSE ratio passed only 1/4 fresh worlds (raw ratios 1,390x, 302,057x, 11.7x, 0.227x). A separate 1,200-update expert task failed at development. The original source's “PASS 4/4” summary conflicts with raw rows and frozen gate; see `experiments/mirror_applications/ma-261-batchensemble-logical-experts/RECONCILIATION.md`.

## 2026-10-08 research sweep

New directions are cross-model cache translators, neural graphics and 4D Gaussian fields, speaker-adaptive synthesis, generative flow/solver coordinates, and cross-model stitching. They are **UNTESTED**. See [current research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md). The new cross-model cache lane distinguishes exact MA-691 algebra from approximate learned transfer.

## Twelfth literature intake — 2026-10-08

Additional Mirror m insertion targets: neural video chunk sharing (NerVast/DCVC-UF), partial learned equivariance, spiking thresholds/time gains, physical photonic phase configurations, beamforming/CSI/RIS and personalized HRTF neural fields. All candidates are UNTESTED and require native efficient controls, full bytes and domain runtime/energy. See [twelfth sweep notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md) and PA266..295.

## Thirteenth research intake — 2026-10-08

- Material interatomic potentials (MACE/NequIP/MatterSim): a structured m must preserve energy/force consistency and beat sparse/frozen transfer.
- MRI inverse problems (VarNet/MoDL/DUNE): constrain m to learned priors while enforcing k-space data consistency.
- Variational quantum circuits (TensorHyper-VQC): assess generated m at physically compiled gate depth, noise and shot costs.
- Visual prompting/streaming memory (CoCoOp/MaPLe/SAM2): compare native prompt generator and object-memory storage.
- ANN index/late interaction (RaBitQ/ColBERTv2/PLAID/QINCo): rank retrieval at real encoded index bytes and latency; gauge rotations are not new functions.

[Research note](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_THIRTEENTH_SWEEP.md) • PA296–325 • MA996–1045. New candidates remain UNTESTED.

## Fourteenth literature intake

Native methods Chronos/TimesFM/Moirai/PatchTST/TRACE, DHE/TT-Rec/QR/MMoE, and DOFA/AnySat/CROMA/TerraMind are **mandatory strong controls** for new MA-1046..1095 hypotheses. Evaluate new m beyond native input conditioning, table-free embeddings, wavelength-conditioned hypernetworks, and sensor fusion. See [fourteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_FOURTEENTH_SWEEP.md).

## Fifteenth direct-prior intake

Existing methods **Compress then Serve, CtM, EigenLoRAx, MetaTT, GLoRA, LRAgent and PReCache** already share adapter bases or canonical KV cache state. The distinct Mirror question is whether structured m gives incremental useful functionality, bytes and runtime gains beyond these methods on natural tasks. Compare only gauge-invariant task-delta geometry. See [fifteenth research note](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_FIFTEENTH_SWEEP.md) and [audit intake](research_intake/natural_lora_orbit_20261008/README.md).

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
