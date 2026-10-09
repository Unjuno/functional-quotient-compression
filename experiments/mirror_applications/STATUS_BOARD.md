# Mirror Application Status Board

Updated: 2026-10-09 UTC
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (55 completed; 575 UNTESTED)
- P1: **422** (13 completed; 409 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1087 UNTESTED, 36 PROMISING, 32 FAIL**
- 65 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-255 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-255 remains next.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-255 remains next.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-255 remains next.
- Research support PA372..381 and a **separate 48-row / two-seed real-digit shifted-task code pilot** have been added. Structured 8-value Mirror m improved average CE versus 6-value diagonal code but FAILED the preregistered full quality/bytes/runtime gate. This is an **exploratory negative pilot**, not an MA-1096/1099/1115 completed experiment. Existing 29 PROMISING / 18 FAIL and MA-255 next are unchanged. See [function-space review](../../docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md) and [pilot results](research_intake/natural_digit_function_20261008/RESULTS.md).
- Sixteenth literature sweep added **MA-1116..1155** (40 UNTESTED; 33 P0 / 7 P1) and **PA382..413** (32 primary sources): knowledge graph relation operators, camera ISP/lens optics, robot dynamics and room neural acoustic fields. No new MA measurements. See [sixteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md) and [40 per-ID test blueprints](../../docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md); next remains MA-255.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-519 — function-vector routing from context (P1; PA99)**

Reason:
- MA-470/471/473/475/476/478/481/482, MA-483, MA-484, MA-486, MA-487, MA-488, MA-492, MA-494, MA-498, MA-501, MA-502 and MA-516 are completed on dedicated branches and cross-linked in the claim ledger;
- MA-501 and MA-502 both found that VQ coordinates add substantial distortion over compact FP16 shared coordinates. The LoReFT representation-view family is temporarily deferred for redesign; skipped registry candidates remain UNTESTED. MA-516/517/518 now have checked scoped failures; proceed to MA-519 with native task-context routing as the control. Other completed candidates remain cross-linked in the claim ledger.

Required controls: explicit extracted function vectors, shared PCA/Mirror basis codes, randomized/signed coefficient control, and function-vector ablation; actual serialized vector bytes and held-out function accuracy are primary.

This temporary pointer skips the deferred LoReFT family; return to its remaining UNTESTED P0 candidates after a family redesign. Do not drop negative outcomes or treat synthetic PROMISING evidence as adoption.

## Active experiment

MA-516 completed on `research/ma-516-function-vector-mirror-compression-20261009`; next MA-517 is selected. CUDA was unavailable; pinned GPT-2 ran on CPU.

The LoReFT representation-view family remains temporarily deferred for redesign after two consecutive VQ-vs-FP16 distortion failures.

## Verified status index

- **PROMISING (29):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-691.
- **FAIL (18):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

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

## Sixteenth research intake — 2026-10-08

KG relation rotations (RotatE, TuckER, QuatE, PairRE), EXIF/device-aware ISPs (ParamISP, Uni-ISP), deployable dynamics adaptation (UP-OSI, RMA, CoRMA) and room/source acoustic neural fields (NAF, TA-RIR) are **existing directly conditioned-function baselines**. A new Mirror code m earns credit only for marginal useful function/byte improvement versus its corresponding native conditioner, with natural held-out relations/cameras/physics/rooms and true physical runtime/cost. New MA1116..1155 are all UNTESTED.

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


**MA-470 PROMISING (scoped, synthetic shared-basis storage):** at N64 Mirror+private reproduced edits at 3,793B vs MEND-style per-edit rank-two factors at 54,847B, but the generic independent four-atom basis was smaller (3,165B) at equal quality. No Mirror-specific gain; oracle least-squares coefficients, not learned MEND. **Next: MA-471.**

**MA-471 PROMISING (aligned analytic ROME coordinate screen):** at N64 the angle-coded shared-plane representation used 2,909B vs ROME factors 5,925B and generic Cartesian coefficients 3,421B; max edit efficacy NRMSE 1.07e-7 and specificity drift 4.43e-8. N20 missed the <=80% byte gate. Known fixed-norm orbit only; private residual control and natural factual edits remain untested. **Next: MA-473.**


**MA-473 PROMISING for aligned edit-bank storage only:** at N64, four-layer Mirror codes used 5,973B vs direct per-layer rank-one factors 35,105B and generic Cartesian shared-plane coefficients 8,021B. Standalone edit NRMSE stayed below 2e-7 and locality near zero; merged-bank NRMSE reached 4.80, identically across representations. This compresses a known synthetic orbit but does not establish useful simultaneous factual editing or learned MEMIT. **Next: MA-475 (P0); MA-474 remains P1.**


**MA-475 PROMISING for synthetic value-memory storage only:** at N64 Mirror used 7,069B vs SERAC explicit key/value 10,529B; generic shared Cartesian coefficients used 7,261B. Exact/paraphrase recall was 100%, unrelated false triggers 0. The incremental Mirror saving over generic was only 2.6%; known fixed-norm 2D value orbit, not a SERAC reproduction. **Next: MA-476 (GRACE value compression).**


**MA-476 PROMISING for aligned GRACE value compression:** at N64 Mirror used 7,449B vs explicit GRACE key/radius/value 10,973B, generic shared Cartesian coefficients 7,705B, and VQ K=8/16/32/64 at 7,641/8,153/9,177/11,225B. Recall was 100%, false triggers 0 and Mirror NRMSE <1.3e-7; VQ errors ranged 0.249–0.031. The incremental saving over generic coefficients is 3.3%; fixed-norm synthetic orbit only. **Next: MA-478 (P0); MA-477 remains P1.**


**MA-478 FAIL for the preregistered <=80% storage gate:** development selected tau=0.30; N64 retained exact outputs with 19/64 private values but used 9,485B vs 10,973B explicit (86.4%). View-only had 0.194 mean NRMSE. Generic adaptive coefficients used 9,741B, so the Mirror saving was only 2.6%. Full private vectors restore quality but erase much of the compression. **Next: MA-481 (P0); MA-479 remains P1.**


**MA-481 FAIL for the registered N64 address-quality gate:** Mirror used 4,505B vs explicit keys/values 11,997B (37.6%) with 100% exact recall and zero key collisions, but mean paraphrase correct-address recall was 98.35% (<99%) and output NRMSE 0.180. Full explicit storage had the same retrieval failure, so this is crowded-address geometry, not a Mirror-specific regression. Generic Cartesian codes were only 5.4% larger; VQ K=8–64 collided heavily and K=128 exceeded explicit bytes. **Next: MA-482 (P0).**


**MA-517 FAIL (scoped synthetic composition):** Direct ICL, query-only, raw vector sum, layer-factorized application and single-vector ablations all scored 0% in 9 fresh world-seed banks. Factorized placement lowered mean target NLL slightly (9.878 vs 9.967) at identical 99,945B payload, but did not recover exact behavior. Direct ICL itself missed the preregistered gate. **Next: MA-518 (P0).**


**MA-518 FAIL (scoped prompt-delta quality failure; factorization not established):** direct ICL was 31.25%, but explicit/generic/factorized interventions scored 0% across 9 fresh banks. The factorized decoder ignored stored function/domain codes and used a flat rank-2 projection, so its apparent byte advantage is invalid. See MA-518 report. **Next: MA-519 (P1).**
