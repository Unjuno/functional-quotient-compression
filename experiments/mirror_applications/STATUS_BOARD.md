# Mirror Application Status Board

Updated: 2026-10-09 UTC
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (87 completed; 543 UNTESTED)
- P1: **422** (13 completed; 409 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1053 UNTESTED, 44 PROMISING, 58 FAIL**
- 68 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-255 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-255 remains next.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-255 remains next.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-255 remains next.
- Research support PA372..381 and a **separate 48-row / two-seed real-digit shifted-task code pilot** have been added. Structured 8-value Mirror m improved average CE versus 6-value diagonal code but FAILED the preregistered full quality/bytes/runtime gate. This is an **exploratory negative pilot**, not an MA-1096/1099/1115 completed experiment. Existing 29 PROMISING / 18 FAIL and MA-255 next are unchanged. See [function-space review](../../docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md) and [pilot results](research_intake/natural_digit_function_20261008/RESULTS.md).
- Sixteenth literature sweep added **MA-1116..1155** (40 UNTESTED; 33 P0 / 7 P1) and **PA382..413** (32 primary sources): knowledge graph relation operators, camera ISP/lens optics, robot dynamics and room neural acoustic fields. No new MA measurements. See [sixteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md) and [40 per-ID test blueprints](../../docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md); next remains MA-255.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Completed: MA-268 — IA3 Mirror activation views

- **FAIL** against the registered ≤80% IA3 payload gate: aligned Givens view reached fresh NRMSE <5.3e-8 at 3,167 B vs IA3 3,283 B (3.5% reduction); IA3 remained nearly exact on independent diagonal task views.
- Scoped synthetic orbit feasibility only; no language-model or Mirror-specific novelty claim. Full report and verification are linked in the claim ledger.

## Completed: MA-271 — OFT Mirror task views

- **FAIL** against the registered storage gate. Aligned Mirror matched the generic fixed-plane scalar control exactly and saved only 7.5% versus support-fitted independent OFT; the 50% gate was missed. Independent oracle OFT is exact, but support-fit did not converge enough to establish the learning boundary.
- Synthetic CPU-only; no natural task or Mirror-specific claim. See claim ledger.

## Completed: MA-272 — input-centric OFTv2 Mirror views

- **FAIL** for Mirror-specific value. Input-centric and materialized OFT match within 2.13e-7; input-side path roughly doubles batch-1024 CPU latency but avoids one-time weight materialization. Aligned angle saves ~49.4% over full Q, but generic scalar is equivalent.
- Synthetic linear result only; see claim ledger.

## Completed: MA-273 — BOFT Mirror adapter bank

- **FAIL / broad capacity NOT ESTABLISHED:** bounded implementation missed aligned quality and storage gates; generic scalar matched Mirror, independent fitting remained near no-view. The butterfly schedule/fitting is not a validated canonical BOFT control. Fresh worlds are closed; see claim ledger.

## Completed: MA-274 — BOFT logical expert views

- **FAIL** for BOFT-specific adoption. A shared view bank used ~60% fewer bytes than untied MoE at similar synthetic quality, but one-plane Givens was better/slimmer and ~9× faster to fit. Router accuracy remained ≥0.997. Teacher is intentionally Givens-aligned; no natural expert-capacity claim.

## Completed: MA-276 — BOFT views for tied depth

- **FAIL** for Mirror-specific value: A1 on new worlds found generic scalar exactly matches the Mirror Givens result. Initial fresh showed an aligned depth-orbit compression signal, but independent tasks required more capacity. Original fresh row CSV was overwritten by A1; original payloads and recorded aggregates remain, and no viewed worlds were rerun.

## Completed: MA-278 — Compacter Mirror adapters

- **FAIL:** aligned shared atom codes saved only 5.4% bytes vs native Compacter and generic scalar matched Mirror exactly. Independent shared codes were near no-adapter; rank-2 LoRA recovered them at higher bytes.

## Completed: MA-282 — Monarch Mirror FFN

- **FAIL** for Mirror-specific value. Aligned scalar code reached NRMSE 4.8e-8 at ~1,971 B vs native per-task Monarch 3,202 B, but generic scalar was identical. Independent task codes stayed near baseline.

## Completed: MA-286 — Cheap-LoRA Mirror columns

- **FAIL** for Mirror-specific gain: aligned shared-subspace code used ~57% fewer bytes than Cheap-LoRA, but generic coefficient matched exactly. Independent tasks required private LoRA factors.

## Completed: MA-288 — fast-weight programmer context code

- **FAIL** for Mirror-specific value: dynamic rank-2 context code generalized to held-out identities at 264 B, but generic low-rank matched it. Fast outer update was cheaper but poor on this rule; static IDs failed.

## Completed: MA-292 — task-vector Mirror basis

- **FAIL** for Mirror-specific value. Shared rank-4 basis reconstructed held-out tasks/compositions, but PCA and generic coefficients matched or slightly beat Mirror with marginally lower bytes.

## Completed: MA-330 — tensorized KV reconstruction

- **PROMISING, narrowly:** fresh aligned synthetic cache views: Mirror 2,287 B/context NRMSE .000602 vs rank-2 PCA/Tucker 6,502 B/.000427 and full 32,909 B/.000312. Hard-shared MLKV used 2,188 B but context error was .999. Mirror materialization .846 ms vs PCA/Tucker .051 ms at the same 32,768 MAC proxy. Independent cache states fail (Mirror context error 1.366), requiring private state off-orbit. No autoregressive or LM evidence.

## Completed: MA-331 — Re-Basin-aligned Mirror task deltas

- **FAIL for Mirror-specific byte gate at development:** Re-Basin/direct used 2,574 B and Mirror phase 2,562 B, only 0.47% below the frozen 10% margin; both preserved synthetic task outputs at nMSE <1e-8. Alignment improved rank-2 output nMSE from 0.20–0.264 to <1e-8, but that gain is not Mirror-specific. Fresh seeds remained sealed because the payload byte structure already missed the gate.

## Completed: MA-332 — Mirror permutation-orbit audit

- **FAIL for symmetry-as-functional-multiplicity (audit gate confirmed):** fresh A1 exact permutation, positive scale plus compensation, and compensated Givens outputs all had NRMSE <1e-6; each paid View added bytes while representing the same function. Uncompensated Givens (NRMSE .183) and negative scale (.899) changed outputs. Synthetic fixed ReLU MLP only; no task utility. A0 rows were quarantined for a serializer dtype defect.

## Completed: MA-333 — Mirror sign/scale orbit audit

- **FAIL for general sign/scale functional multiplicity:** fresh permutation is exact across ReLU, GELU, tanh and LayerNorm+ReLU. Positive scaling is exact only for ReLU; sign flip is exact only for tanh. GELU and LayerNorm+ReLU break those nonmatching transforms. Exact gauge codes cost 83–147 B; changed functions have no task-utility evidence. Synthetic MLP only.

## Next candidate

**MA-334 — canonical-orbit storage + Mirror address (P1)**

MA-333 confirms scale/sign transformations are activation-specific gauges. MA-334 tests the narrow coordinate-storage use case by measuring canonical representative recovery, bytes, and runtime.

## Active experiment

MA-517, MA-257, MA-261, MA-265 and MA-266 are FAIL; MA-255 and MA-260 are NOT ESTABLISHED due implementation defects; MA-258 is PROMISING only on its aligned synthetic orbit. Next is MA-271.

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


## Completed: MA-296 — orthogonalized task-vector Mirror superposition

- **FAIL** on fresh synthetic linear tasks: Hadamard address and generic QR both reconstruct at ~2e-6 query NRMSE, but use 16,735/16,744 B versus 16,473 B raw task vectors. Unbound sum uses 2,131 B but query NRMSE is ~2.52. No Mirror-specific gain; random PSP unbinding control also has substantial crosstalk. See claim ledger.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-297 — SETA shared sparse subspace + Mirror views

- **FAIL for Mirror-specific value; shared-subspace compression signal observed.** Fresh synthetic sparse task vectors recovered shared support at 100%; Mirror code used 836 B and generic PCA 841 B at query NRMSE ~3.31e-7, versus SETA-style shared/private 1,055 B. Task deltas were given to the encoder; this is not a faithful SETA continual-learning reproduction.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-299 — Split-on-Share Mirror code allocation

- **FAIL for Mirror-specific value; adaptive allocation signal observed.** On a synthetic stream, threshold split all four novel tasks; adaptive storage used ~11.54 KB versus 20.59 KB independent sparse vectors with query NRMSE ~2.09e-7. Never-split was 3.30 KB but quality collapsed (NRMSE ~0.88). Generic PCA matched Mirror.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-301 — continuous Mirror supermask

- **PROMISING, narrowly scoped:** on intentionally rank-2-aligned synthetic masks, Mirror used 16,750 B vs packed binary masks 32,847 B (~49% less) at query NRMSE 0.00854. A byte-matched generic logistic factorization had NRMSE 0.04798; PCA used more bytes and had NRMSE 0.158. Since the teacher masks were generated from the same rank-2 basis Mirror stores, this is not general SupSup/Piggyback or learned-task evidence.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-307 — Mirror code before PackNet physical allocation

- **PROMISING, narrowly scoped synthetic screen:** Mirror with sparse fallback used 1,658 B and query NRMSE 0.000394 vs PackNet-style independent sparse storage at 9,358 B exact. It split four novel tasks; total physical values including shared basis/codes were ~482 vs 1,536. Robust generic shared-basis fit was close at 1,959 B / 0.000891. Teacher tasks were exactly generated from Mirror's basis with known codes; no trained PackNet or natural continual task evidence.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-311 — Mirror task code in random intrinsic subspace

- **FAIL for practical gain:** aligned fresh tasks: Mirror 4,314 B / NRMSE 1.97e-5; generic PCA 4,424 B / 2.90e-7; direct intrinsic codes 4,510 B / 2.80e-7. Mirror's task-state bytes fall 81%, but total payload only 4.3% after charging U; fitting costs ~0.915 s / 800 updates vs <0.5 ms direct. On independent tasks, Mirror NRMSE ~0.843 while direct remains near exact.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-312 — shared intrinsic basis + many Mirror task coordinates

- **FAIL:** on 64 aligned tasks, Mirror used 16,727 B / NRMSE 7.39e-6 vs direct intrinsic 20,624 B / 3.90e-7 and PCA 17,018 B / 2.89e-7. Total saving 18.9%, below the 20% gate; fitting cost ~0.463 s vs ~0.0036 s direct. Independent tasks require private coordinates (Mirror NRMSE ~0.971).
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-314 — adaptive intrinsic-dimension allocation

- **PROMISING narrowly:** fresh aligned variable-rank tasks: Mirror 8,557 B / NRMSE 3.78e-7 vs direct adaptive 9,110 B and PCA 9,352 B, with mean active dimension 15/32 and no private splits. Independent tasks triggered private fallback in 23/24 tasks; Mirror+private cost 12,978 B vs direct 9,926 B. Teacher is a constructed Givens orbit; random U dominates total bytes.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-319 — Tucker matrix-bank Mirror layer coefficients

- **FAIL:** aligned Mirror used 16,699 B / NRMSE 5.03e-4 vs free Tucker 17,559 B / 2.93e-4 (4.9% saving, below gate); generic PCA used 16,887 B / 3.56e-4. Independent coefficients required free Tucker; Mirror NRMSE was 1.036. Shared bank dominated storage.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-320 — Tucker logical experts

- **PROMISING, narrowly:** fresh aligned 128-expert bank: Mirror 4,555 B / routed NRMSE 4.95e-4 vs free Tucker 6,313 B / 2.97e-4 (~28% fewer bytes); generic PCA 4,867 B / 3.52e-4. Independent coefficients require free Tucker; Mirror NRMSE 1.037. Teacher is Givens-aligned; fixed router and no natural MoE evidence.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-322 — TT-core Mirror adapter bank

- **PROMISING narrowly:** aligned TT adapters: Mirror 509 B / NRMSE 5.90e-4 vs LoRETTA-style independent middle cores 2,355 B / 4.05e-4 and PCA 625 B / 4.05e-4. Independent cores are needed outside the Givens orbit; Mirror NRMSE 1.145. Synthetic only.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-325 — tensorized embedding domain views

- **PROMISING narrowly:** aligned synthetic domains: Mirror 636 B / lookup NRMSE 5.78e-4 vs generic PCA 981 B and independent TT cores 4,530 B; full tables 4.19 MB. Independent domains need private TT cores; Mirror NRMSE 1.288. No LM or real vocabulary evidence.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.


## Completed: MA-327 — factorized layer × expert Tucker address

- **FAIL:** aligned held-out NRMSE ~2.98e-4 for Mirror and generic A×B factorization; Mirror 1,552 B vs generic 1,745 B (11%, below 20% gate). Independent expert factors need free factorization; Mirror NRMSE ~1.073.
- Next P0 candidate: MA-330 — tensorized KV reconstruction.
