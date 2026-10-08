# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (83 completed; 547 UNTESTED)
- P1: **422** (12 completed; 410 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1060 UNTESTED, 39 PROMISING, 56 FAIL**
- 82 local experiment directories are complete; MA-301 and MA-307 are additional completed experiments linked to their dedicated research branches.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-276 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-276 is next; MA-275 remains UNTESTED (P1).
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-276 is next; MA-275 remains UNTESTED (P1).
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-276 is next; MA-275 remains UNTESTED (P1).
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-344 — PreLort nested-rank Mirror segments (P0)**

MA-325 was attempted on development seeds but is **NOT ESTABLISHED**: all methods, including independent full tables, remained at uniform NLL (~ln 16). Fresh seeds stayed sealed; a learnable task requires a separately versioned protocol. See its report on `research/ma-325-tt-embedding-domain-mirror-20261008`. MA-330 is PROMISING only for synthetic aligned cache views: 36,458B vs 131,752B independent, near-zero output nMSE, but the direct cos/sin control is only 30B larger and there is no decode-speed gain. One unrelated layer needs private cache. See report. MA-331 development failed its Mirror-specific 10% byte gate: Re-Basin/direct was 2,574B and phase Mirror 2,562B, both with nMSE <1e-8; unaligned low-rank deltas were poor. Fresh stayed sealed. MA-332 confirmed eight hidden-unit permutation states are one function (max output difference <=2.25e-7); the paid shared checkpoint is 81.6% smaller than eight duplicate archives, but adds zero functional multiplicity. MA-333 likewise confirmed coupled ReLU positive-scale and tanh sign symmetries preserve FP32 functions; FP16 quantization was separated. MA-332/333 pause pure gauge-orbit capacity proposals pending a function-changing extension.

MA-255 is reconciled as PROMISING only for its aligned post-fit representation screen; a distinct 1,200-update protocol variant failed at development and remains sealed on fresh worlds. See `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`.

MA-260 is complete FAIL: the one-layer linear screen missed its predeclared byte gate, while the independent-task stress case collapsed. It is not a deep BatchEnsemble result; see its report and verification.

MA-261 is complete FAIL after audit: its post-fit branch's raw MSE ratios meet the frozen <=1.10x independent quality criterion in only 1/4 fresh worlds, despite source wording claiming 4/4. A different fixed-update protocol also failed during development. Both variants are retained in its reconciliation report.

MA-265 is complete FAIL: a fresh aligned post-fit screen achieved numerical-zero task MSE but saved only 4.4% total payload versus VeRA, below the preregistered 20% gate; independent codes failed. A distinct 1,000-update diagonal-code protocol also failed at development against VeRA. Both protocols are retained separately.

MA-265 and MA-266 are consecutive FAILs in the VeRA Mirror family: native VeRA missed the strict margin in MA-265, and an ordinary coefficient-product control matched MA-266 within 2.1% bytes. Pause MA-267 until the family has a control/redesign that tests a distinct Mirror-specific contribution.

MA-268 is PROMISING on a narrow synthetic nonlinear task: a trained Givens-aligned screen passed its quality/byte gate 3/3 at 4,167B vs 4,273B IA3, with lower quality error but 1.5x MACs and substantially slower eager CPU throughput. A separate post-fit linear activation screen missed its strict 20% byte gate (15.7% saving). See its protocol reconciliation.

MA-271 is FAIL for Mirror-specific value. One post-fit screen found a low-byte Givens orbit, but an exact simple rank-one angle-factor control produced identical functions and bytes (1,086B); a separate trained dense-OFT screen omitted that control. See its reconciliation report.

MA-272 is FAIL for Mirror-specific/runtime gain: the corrected audit found that scalar-times-shared-skew exactly matches Mirror at 3,786B, while exact input-side execution averaged 0.335ms versus 0.138ms materialized. A separate trained screen retains its bounded storage/quality result but was slower than dense OFTv2.

MA-273 is complete FAIL for Mirror-specific value: corrected post-fit views exactly matched ordinary scalar-times-shared-angle factorization (same 1,150B payload); an independent neutral-initialization training screen failed at development and left fresh sealed. See its reconciliation report.

MA-274 is complete FAIL on a two-world fixed-update development screen: at 5,430B Mirror was 3.4% smaller than native BOFT but had worse MSE in both worlds; independent experts were much better and butterfly eager CPU throughput was 9–20x below IA3/rank-one. Fresh worlds stayed sealed. Together with MA-273, this pauses the shared scalar-angle task/expert BOFT insertion family. MA-275 remains UNTESTED (P1). MA-276 is PROMISING only for aligned serialized-state compression: 396B Mirror vs 1,193B untied (3/3 fresh), but 2,048B runtime workspace exceeds that payload and unrelated layers fail. No runtime-memory claim. MA-278 is FAIL on the unaligned task-factor screen: Mirror matched scalar control bytes at 2,525B but lost quality at both development worlds and both learning rates; native Compacter also dominated quality. Fresh worlds remained sealed. Aligned feasibility is untested. MA-282 is PROMISING only for a deliberately aligned synthetic ReLU FFN orbit: exact in 3/3 at 925B vs 1,145B free-angle Monarch and 3,279B independent full. Unrelated functions collapse to hard-tie quality; view workspace is 1,024B, so no runtime-RAM claim. MA-286 is FAIL for Mirror-specific margin: the shared-B index screen reached aligned quality with 4,412B vs task-local Cheap-LoRA 6,580B, but an identical one-hot shared-B gate was only 78B larger (1.74%, below the preregistered 10%). Two unrelated functions used private rank-4 factors. MA-288 is FAIL under its preregistered direct-path runtime gate: aligned Mirror state/quality was strong at 378B vs 1,203B independent and quarter the writes, but throughput was 0.178–0.456x rank-4 residual in all fresh seeds. A cached path reached 201M examples/s with 1,024B workspace and is only supplemental. MA-292 is FAIL for Mirror-specific value: one angle beat FP32 SVD code size but matched-byte FP16 SVD was 23B smaller overall, slightly more accurate in all 3 fresh worlds, and about 1.55x faster. Unrelated task vectors required richer/private state. MA-296 is FAIL for Mirror-specific value: the simpler direct shared-orbit generator matched Mirror exactly at 1,524B vs 1,561B, while independent task vectors were needed for unrelated maps (4,771B). Rademacher PSP interference was high; rank-8 SVD matched quality with larger payload. MA-297 and MA-299 are FAIL; this SETA shared/private Mirror-code family is paused for redesign under the two-consecutive-failure stop condition. MA-314 is PROMISING only for its separate aligned-only 48-task variant (1,996B vs 2,220B adaptive coefficients, 3/3 fresh); the broad 160-task private-fallback protocol FAILed at +5.2% bytes vs adaptive direct in 3/3. MA-322, MA-303 and MA-304 are complete FAILs (see reports below); MA-303/304 are consecutive P0 failures in the factorized/supermask subfamily against the same direct-code control. Pause MA-305 and related factorized-mask proposals; MA-309 is a distinct MIMO implicit-ensemble mechanism and remains next P0.

## Active experiment

MA-335 is complete FAIL for Mirror-specific value; see its result below. MA-337 is complete FAIL for Mirror-specific value; see its result below. MA-338 failed its frozen development byte margin; fresh remained sealed. MA-341 failed its frozen development byte margin; fresh remained sealed. MA-342 failed its frozen development storage gate; fresh remained sealed. MA-344 is PROMISING only for synthetic PreLort storage/communication, with no Mirror-specific gain. MA-346 is PROMISING only for a synthetic shared/private storage frontier; ordinary scalar phase matched exactly. MA-349 failed its development posterior storage gate; fresh remained sealed. MA-351 is next P0 by registry order. Pure permutation/sign/scale symmetry orbit proposals remain paused after MA-332/333. MA-312 is PROMISING only for the aligned 256-task storage/quality point; its fit compute proxy was over 1,000x the direct coefficient control and throughput lower. MA-258 has a narrow PROMISING aligned codec result; unrelated experts required private/richer state. MA-266 is FAIL for Mirror-specific value: factorized views generalized on an aligned task cross-product, but the ordinary coefficient-product control matched within 2.1% payload bytes. Together with MA-265 this pauses VeRA family follow-up MA-267 pending redesign. MA-255 through MA-299 now have verified status records. MA-257 failed against the exact native PA16 rotational-context control; MA-297/299 SETA remains paused. Next by the registered P0 queue: MA-330.

## Verified status index

- **PROMISING (37):** see registry; MA-335 recorded below as FAIL.
- **FAIL (56):** includes MA-335 and MA-351; see `IDEA_REGISTRY.csv` and `CLAIM_LEDGER.csv` for the full index.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## Newly reconciled result

- MA-255 — PROMISING only for an aligned post-fit representation screen: 734B Mirror payload achieved 6.68e-17 fresh MSE on seeds 101/211/307/401 versus 6,490B implemented PSP, 14,650B rank-2 task code, and 27,906B independent. A separate 1,200-update protocol variant failed its development quality gate (Mirror MSE 0.0922 vs 4.52e-5 independent; fresh sealed). These are different protocols, not a replication pair. See `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`. Source branches: `research/ma-255-mirror-context-superposition-20261008` and `research/ma-255-mirror-context-superposition-replication-20261008`.
- MA-260 — FAIL: on four fresh rotation-aligned worlds Mirror matched member accuracy but used 890B, only 27.4% below implemented BatchEnsemble (1,226B) and 3.9% below independent (926B), missing its <=25% byte-ratio gate. On independent task separators its accuracy fell to 0.6018 vs 0.8332 controls. Post-fit one-layer linear screen only; BatchEnsemble rank-one is a weak single-output control, and deep ensembles remain untested. Source result `33da88ba71cb053a8729a393ccc1ed1a2cc4fcf3`, report `experiments/mirror_applications/ma-260-batchensemble-mirror/README.md`.
- MA-261 — FAIL: post-fit aligned Givens views used 1,006B vs 1,562B BatchEnsemble and 1,794B independent, with tiny absolute error, but the frozen <=1.10x independent MSE ratio passed only 1/4 fresh worlds (raw ratios 1,390x, 302,057x, 11.7x, 0.227x). A separate 1,200-update expert task failed at development. The original source's “PASS 4/4” summary conflicts with raw rows and frozen gate; see `experiments/mirror_applications/ma-261-batchensemble-logical-experts/RECONCILIATION.md`.
- MA-265 — FAIL: four fresh aligned rotation worlds reached numerical-zero error, but payload was 3,494B vs 3,654B VeRA (only 4.4% savings, below the 20% gate); hard tying was 3,430B and unrelated scales failed. A separate trained diagonal-code protocol also failed against VeRA at development. Both protocols are retained at `experiments/mirror_applications/ma-265-vera-mirror-scaling/README.md`.
- MA-268 — PROMISING for the trained nonlinear aligned screen: 3/3 fresh worlds passed the frozen gate, with 4,167B Mirror vs 4,273B IA3 and 7,933B independent. MSE beat IA3; active MAC proxy was 1.5x and eager throughput ~0.29–0.34x IA3. The separate post-fit linear orbit screen saved only 15.7% vs IA3 and missed its 20% byte gate. No natural-language evidence; see the two protocols in `experiments/mirror_applications/ma-268-ia3-mirror-views/`.
- MA-271 — FAIL for Mirror-specific advantage: on four fresh worlds, Mirror and simple rank-one shared-angle control were identical in function and payload (1,086B); independent-plane stress also matched exactly while full OFT remained near-zero. A separate trained screen had 3,309B vs 7,253B dense OFT but lacked the exact rank-one control. See `experiments/mirror_applications/ma-271-oft-mirror-views/RECONCILIATION.md`.
- MA-272 — FAIL for Mirror-specific/runtime Pareto: corrected fresh audit found exact equivalence to scalar-times-shared-skew at 3,786B, with 0 aligned relative error, but exact input-side execution averaged 0.335ms vs 0.138ms materialized. A separate trained screen saved bytes vs dense OFTv2 but had slower eager CPU throughput. See `experiments/mirror_applications/ma-272-oftv2-mirror-views/RECONCILIATION.md`.

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


## MA-297 — FAIL

Corrected serialized-payload quality passed the frozen ratio gate in only one of three fresh aligned worlds. The angle-grid fit proxy was about 2,048x same-basis least squares and inference throughput was 0.43x; the simple coefficient control had better mean error and independent tasks required private capacity. Four tests and payload/metric/hash replay passed. See `experiments/mirror_applications/ma-297-seta-mirror-subspace/README.md`. Next: MA-299.


## MA-299 — FAIL; SETA family paused

**Fact:** On three amended fresh streams, Mirror split-on-share used 2,337B, mean normalized held-out MSE 0.0005970 and max 0.0040706; independent full used 6,417B at near-zero error. The two-coefficient split made the same four-view/one-private allocation at 2,352B and mean error 0.0005962. Mirror saved 15B (0.64%) but its fit-compute proxy was 1.74x the coefficient control. Five tests and all 21 corrected deterministic metric/byte/hash rows replayed.

**Interpretation:** share-before-private allocation delayed physical growth on aligned synthetic tasks; an ordinary two-coefficient address achieved the same allocation and quality.

**Family decision:** MA-297 and MA-299 are consecutive P0 FAILs in the SETA shared/private family, both showing ordinary coefficient representation removes Mirror-specific advantage. Per stop condition, pause MA-298/MA-300 family work pending redesign. This does not pause unrelated Mirror families.


## Sixteenth research intake — 2026-10-08

The latest canonical baseline added MA-1116..1155 (40 UNTESTED; 33 P0 / 7 P1) and PA382..413. These knowledge-graph, camera/optics, robot-dynamics and room-acoustics proposals remain behind the outstanding P0 queue. Their source sweeps do not change MA statuses. See `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md` and `docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md`.


## MA-257 — FAIL for Mirror-specific value

On three development worlds, the smallest selected support (64/512 factor combinations) composed the aligned teacher at 296B and mean test normalized MSE 6.58e-9. Native PA16 factorized rotational contexts produced the exact same functions, bytes and errors, so the registered Mirror-specific gate failed; fresh seeds remain sealed. The FP16 free-coefficient table used 348B, but it is a weaker control than native rotational context. On independent maps Mirror mean error was 8.64, while coefficient pairs reached 1.036 and independent full weights were near zero. Seven tests and 54 selected-support development metric/byte/compute/workspace/hash replays passed. Next P0: MA-319.


## MA-258 — PROMISING only for aligned post-fit expert-bank coding

On three fresh aligned worlds, an eight-expert Givens family used 1,514B Mirror payload at normalized MSE 8.99e-10–1.07e-9; rank-2 shared SVD reached near-zero error at 3,822B, and independent maps used 8,440B. Native binary-context PSP used 1,766B but had normalized MSE 4.62–5.46. This passed the preregistered development gate and replicated on 3/3 fresh worlds. On unrelated experts, Mirror normalized MSE was 0.848–0.889, essentially hard tying; independent weights were needed for near-zero error. Mirror had no measured throughput gain (11.16M examples/s fresh aligned vs 17.88M independent). Four tests and 60 metric/byte/hash replays passed. Oracle expert IDs and zero training updates; synthetic linear screen only. Next P0: MA-266.


## MA-266 — FAIL for Mirror-specific value; VeRA family paused

In three synthetic development worlds, Mirror composed 8 held-out factor pairs from 8 checkerboard support maps at normalized MSE 7.91e-9 and 1,276B. The direct sine/cosine coefficient-product control had normalized MSE 4.20e-8 at 1,304B, just 2.1% more bytes and within quality; this misses the frozen 10% byte and 10× quality margins. VeRA-style additive factor codes failed aligned composition (MSE 0.289). On unrelated maps Mirror and coefficient-product errors were about 0.401; independent full weights were near zero. The measured operation proxy favored direct Mirror rotations, but an optimized coefficient-product implementation can apply the same transform; no runtime claim. Four tests and 60 exact metric/byte/hash replays passed. Fresh 26611–26613 remain sealed. Alongside MA-265, this pauses MA-267 for redesign. MA-301 was completed on a separate research branch; next independent P0: MA-307.


## MA-301 — FAIL; branch-linked result

The amended 32-task random-feature screen saved 21.2% payload versus packed binary masks (334B vs 424B), but Mirror missed the independent-quality criterion in 2/3 fresh worlds (normalized MSE 2.14e-5, 0.412 and 0.256). Only one fresh world passed. Its initial four-task storage-gate miss is retained separately. Throughput used precomputed effective vectors, so no direct-view runtime claim. Three tests and 35 metric replays passed. Full evidence is on `research/ma-301-continuous-mirror-supermask-20261008`. Next P0: MA-307.


## MA-307 — FAIL for strict composite Pareto; storage frontier retained

On three fresh eight-task streams, four Givens-aligned tasks used one-byte View codes before allocation; three unrelated tasks received private readouts. Mirror payload averaged 854B vs 1,333B PackNet-style mask-plus-allocation (35.9% saving), and no prior-task test error changed. Mean normalized MSE was 2.08e-5 vs about 9.2e-16 for private PackNet readouts. Allocation-search runtime ratios were 1.45x, 1.10x and 1.22x; one exceeded the preregistered 1.25x ceiling. Four tests, 36 summary rows and 288 allocation-event rows replayed. Dedicated branch: `research/ma-307-packnet-mirror-allocation-20261008`. Next P0: MA-312.


## MA-311 — FAIL for both frozen protocol gates; one fresh d=64 follow-up

On three aligned development worlds, the one-angle Mirror code achieved 7.50e-7 normalized test MSE and 1,248B. It was 38.2% smaller than PA33 SAID-4 (2,018B), but 9.5% larger than the direct two-coefficient control (1,140B), failing the nearest-control payload gate. Its 322B tensor state expanded to 1,248B serialized NPZ; the coefficient state was 448B but serialized to 1,140B because the Mirror payload carries an additional array/header. Mirror angle search used 17.7M fit-operation proxy vs 16.9k for coefficient least squares. On unrelated codes Mirror MSE was 0.210; SAID-4 was near zero. Four tests and 30 exact metric/byte/hash replays passed. Fresh seeds stayed sealed. Two MA-311 protocol variants are retained on separate branches: the earlier d=4 screen missed the nearest-control byte gate with fresh sealed; a d=64 follow-up saved 14.1–21.1% vs ordinary intrinsic codes in 3/3 fresh worlds but missed its <=0.85x all-world promotion gate once and used one aligned private fallback. Both remain FAIL for their frozen gates.


## MA-312 — PROMISING only for aligned 256-task storage/quality

Across three fresh 256-task worlds, Mirror used 1,600B (normalized MSE 3.83e-7–5.30e-7), versus 2,100B two-coefficient codes, 10,034B SAID-8 and 33,024B independent. It saved 23.8% vs the nearest coefficient control and 84.1% vs SAID-8 while meeting the frozen quality gates. On unrelated tasks Mirror error was about 0.31; SAID-8/independent were near zero. Fit compute was 70.78M vs 67.6k coefficient ops, and Mirror throughput was lower than the coefficient/SAID controls. Four tests and all 60 dev/fresh metric/byte/hash replays passed. Synthetic post-fit screen; no capacity claim. Next P0: MA-319.


## MA-314 — PROMISING only for the aligned-only variant; broad screen FAIL

The broad mixed/private 160-task screen had a 24.5% adaptive-dimension reduction vs fixed d=16, but Mirror was 5.2% larger than adaptive direct in 3/3 fresh worlds, with ~103x fit proxy and 0.59–0.69x throughput; unrelated tasks used private fallbacks. A separate aligned-only 48-task variant passed its frozen storage/quality gate 3/3 fresh at 1,996B vs 2,220B adaptive coefficients (10.1% saving), but fit proxy was ~1,299x. These protocol variants are not pooled. Four tests for each variant and 72+36 exact replay rows passed. Dedicated branch: `research/ma-314-adaptive-intrinsic-mirror-20261008`. Next P0: MA-319.


## MA-315 — FAIL

On three fresh mixed common/private synthetic task banks, sparse private allocation met held-out quality (nMSE <=1e-4) but Mirror payload was 3.31–3.32% larger than the matched shared-direct two-coefficient plus identical sparse residual control (6,044–6,056B vs 5,850–5,862B). Mirror fit proxy was about 42x and throughput 0.61–0.78x. Three tests, 45 summary rows and 1,280 task allocation events replayed with exact metrics and payload hashes. Synthetic post-fit screen only; no capacity claim. Next P0: MA-319.


## MA-319 — FAIL

On three fresh four-layer nanoGPT seeds, circular rank-2 Mirror Tucker used 358,587B versus 356,435B for free Tucker coefficients (+0.60%) in all three seeds. Fresh NLL limits were missed in 2/3 seeds. Five tests passed; 25 serialized packages reloaded with exact hashes and metric replay. Amendment A1 quarantined the initial development audit-read bug; contaminated results were excluded. No capacity claim. Dedicated branch: `research/ma-319-tucker-matrix-bank-mirror-20261008`. Next P0: MA-320.


## MA-320 — FAIL on strict storage promotion

On three fresh synthetic MoE banks, a harmonic Mirror phase code represented 48 aligned experts and used private full matrices for all 16 unrelated experts. Total payload was 35,886B versus 37,074B for rank-16 Tucker with the same fallback, a 3.2% saving below the preregistered 10% gate. All test nMSE values remained <=1e-4; phase-search fit proxy was ~32x direct coefficients. Five tests and exact replay of 45 summaries/2,240 allocation events passed. Oracle task IDs and planted task orbit; no router or capacity claim. Dedicated branch: `research/ma-320-tucker-logical-experts-20261008`. Next P0: MA-322.

## MA-322 — FAIL

A one-phase address on a shared TT core met function quality on planted aligned tasks but saved only 0.51% bytes vs the free two-coefficient core control (5,048B vs 5,074B) in 3/3 fresh seeds. On mixed tasks Mirror was 0.15% larger (25,464B vs 25,426B); both methods stored 32 unrelated tasks privately. Phase fit proxy was ~1,536x direct. All max test nMSE <=6.44e-7. Thirty payloads and 1,536 allocation events replayed exactly. Synthetic post-fit function screen only; no capacity claim. Dedicated branch: `research/ma-322-tt-core-mirror-adapter-bank-20261008`. Next P0: MA-303.

## MA-303 — FAIL for Mirror-specific margin

Across three fresh seeds, factorized Mirror codes composed four held-out task/layer pairs with max nMSE 0 and no private fallback at 34,892B. This was 51.0% smaller than Piggyback masks (71,243B), but direct Cartesian factorization reconstructed the same masks and quality at 34,920B: only a 28B / 0.080% Mirror reduction, below the frozen 10% promotion margin. Mirror CPU throughput was similar to direct codes, and both were below pre-stored mask controls. Amendment A1 fixed validation fallback execution before fresh access. Fifteen serialized packages were byte/hash checked and all 15 summary metrics replayed exactly; four tests pass. Oracle planted task/layer codes, synthetic post-fit screen; no learned generalization or capacity claim. Dedicated branch `research/ma-303-factorized-layer-mask-mirror-20261008`. Next P0: MA-304.

## MA-304 — FAIL for strict storage / compute Pareto

Across three fresh seeds, Mirror phase views stayed at exactly 256 active edges per task and reached aligned max nMSE 4.28e-5–4.78e-5. Mirror payload was 21,288B vs 21,318B direct two-coefficient control (0.141% reduction, below the 10% gate); both used 16 private fallbacks for unrelated tasks. Phase fit proxy was ~3x direct, and eager CPU throughput 0.47–0.79x direct. The binary-mask heuristic used 77,008B but had aligned max nMSE 1.63–1.80; independent full matrices used 295,912B with near-zero error. Twenty-one fresh payloads and 3,024 task allocation rows replayed exactly; four tests passed. Synthetic post-fit task functions only; no learned capacity claim. Dedicated branch: `research/ma-304-supermask-active-mirror-transform-20261008`. MA-305 is P1; next P0 is MA-309.


## Masks/subnetworks family pause — 2026-10-08

MA-303 and MA-304 are consecutive P0 FAILs in the factorized/supermask subfamily. Both produced aligned synthetic quality, but the Mirror coordinate reduced actual payload by only 0.080% and 0.141% versus the closest ordinary direct-code control. Shared bases/backbone and serialization dominate the one-coordinate saving; direct coefficients reconstruct the same task functions. Pause MA-305 and related factorized-mask proposals pending redesign. MA-309 tests a distinct MIMO implicit-ensemble mechanism and remains next P0.


## MA-309 — FAIL for MIMO quality/diversity retention

After A1 reloaded the actual FP16 inference payload, three fresh seeds gave standard MIMO heads 0.986 mean macro accuracy and 0.497 member disagreement at 6,256B. Mirror views saved 25.6% payload (4,652B) but reached only 0.809 accuracy and 0.189 disagreement; each seed missed the 2pp accuracy and 0.05 diversity margins. Mirror throughput was 0.62–0.85x standard MIMO. Scalar gate had equal bytes but lower accuracy. Fifteen corrected payloads were hash/byte checked and 15 metric rows replayed exactly; five tests pass. Initial FP32-only runs were quarantined. Synthetic fixed-update screen only; no capacity claim. Dedicated branch `research/ma-309-mimo-mirror-member-views-20261008`. Next P0: MA-318.


## MA-318 — FAIL for Mirror storage/compute

Across three fresh 12-skill streams, validation-triggered basis growth added six directions: two for a second 2D skill orbit and one for each unrelated skill. No growth occurred for the first four aligned tasks. Direct evolving-basis coefficients retained all prior tasks at max test nMSE <=2.34e-7 and used 2,160B. Mirror retained all tasks at <=9.70e-6 with the same growth events but used 2,654B (+22.9% vs direct and +18.0% vs independent vectors at 2,250B). Mirror fit proxy was ~324x direct. All 144 fresh checkpoints and metrics replayed exactly; four tests pass. Synthetic post-fit stream; no online learning/capacity claim. Dedicated branch `research/ma-318-continual-coordinate-first-20261008`. Next P0: MA-330; MA-325 was attempted but NOT ESTABLISHED as summarized above.


## MA-327 — FAIL for Mirror-specific byte/quality value

On three fresh synthetic 4x4 layer-expert worlds, ordinary rank-2 coefficient product and Mirror product were hash-identical at 1,978B and had exactly equal test/held-out metrics in every seed. Both were 10.6% larger than flat pair codes (1,788B), so the frozen byte gates failed. Fifteen FP16 payloads and metric rows replayed exactly; five tests passed. This is a synthetic linear Tucker screen, not trained MoE evidence. Dedicated branch `research/ma-327-factorized-layer-expert-tucker-20261008`. Next P0: MA-330.


## MA-330 — PROMISING for aligned synthetic cache views, no runtime gain established

Across three fresh seeds, a shared rank-4 temporal K/V basis plus phase views used 36,458B (72.3% below 131,752B independent caches) at attention-output nMSE 5.15e-8–6.39e-8. Direct cos/sin coefficients used 36,488B at comparable quality, only 30B more; the phase code halves view-code storage but barely changes total payload. No-private Mirror was 3,238B but the unrelated fourth layer had nMSE ~1.0; private cache restored quality. Eager CPU throughput showed no consistent speed improvement. Fifteen payloads/hash/metric rows replayed exactly; four tests passed. Synthetic causal-attention screen with oracle-known phases only. Dedicated branch `research/ma-330-tensorized-kv-cache-mirror-20261008`. Next P0: MA-335.


## MA-331 — FAIL for the frozen Mirror-specific storage gate

On two development seeds, Re-Basin/direct task codes used 2,574B and Mirror phase used 2,562B, only 0.47% smaller versus a preregistered 10% threshold. Both had function-output nMSE below 1e-8. Unaligned rank-2 task deltas had mean nMSE 0.20–0.264, showing that alignment helped but did not make that gain Mirror-specific. Eight payloads/hash/metric rows replayed exactly; four tests passed. Fresh remained sealed because byte structure is fixed and the development gate missed. Synthetic post-fit ReLU screen only. Dedicated branch `research/ma-331-rebasin-mirror-task-deltas-20261008`. Next P0: MA-335.


## MA-332 — FAIL as additional function capacity; permutation audit passed

Across five trained synthetic MLP seeds, eight consistent hidden-unit permutation states preserved FP16-reloaded outputs within max difference 1.1e-7–2.3e-7. One shared checkpoint plus all paid permutation indices used 1,740B versus 9,440B for eight separate archives, an 81.6% reduction in duplicate storage. These are one function in eight parameter coordinates, not eight logical functions. Incoming-only permutation changed outputs (max difference 10.0–18.8). Ten payload/metric rows replayed exactly; four tests passed. Dedicated branch `research/ma-332-permutation-orbit-audit-20261008`. Next P0: MA-335.


## MA-333 — FAIL as added functional capacity; exact sign/scale audit passed

On three fresh seeds, coupled ReLU positive hidden scaling preserved FP32 outputs within 2.98e-6, while tanh sign flips were exact. FP16 rounding error was reported separately and reached 0.0108 for ReLU. The symmetry codes add bytes but no new function. Twenty symmetry payload/hash/metric rows replayed; four tests passed. MA-332 and MA-333 are consecutive symmetry-orbit results with the same gauge-only cause; pause pure gauge-orbit capacity candidates pending a function-changing extension. Dedicated branch `research/ma-333-sign-scale-orbit-audit-20261008`. Next P0: MA-335.


## MA-335 — FAIL for Mirror-specific advantage

Across three confirmatory synthetic C4 worlds, group-action Mirror payload averaged 717.3B with exact related-view and private-task outputs; five independent experts averaged 1,091.3B. However, the ordinary irreducible-coefficient control averaged 722.0B at equal quality, only 0.65% larger, so this is not a Mirror-specific gain. Four C4 addresses yielded two distinct functions. Removing the private fifth matrix produced mean unrelated-task nMSE 4.6976. The exact equivariant projection retained one function and failed the non-equivariant target quality. The corrected direct-control protocol was validated on development seeds before confirmatory seeds 33520–33522. Three earlier screens are retained but excluded from the confirmatory verdict. Four tests passed and byte/hash/metric replay was exact. Synthetic analytic 2x2 functions only; no trained MoE or LM evidence. Dedicated branch: `research/ma-335-group-action-mirror-experts-20261008`. Next P0: MA-337.


## MA-337 — FAIL for Mirror-specific advantage

Across three fresh synthetic C4×C2 worlds, factorized group coordinates reconstructed both held-out compositions at nMSE 0 using 974.3B, 72.9% below eight independent operators (3,598.3B). An ordinary direct position/role coordinate control also had zero error at 985.3B, only 1.1% larger, so the result is not Mirror-specific. Eight group compositions yielded eight distinct functions. The off-orbit ninth task had nMSE 1.0958 without private state; a private operator restored zero error at 1,411.3B versus 4,036.3B for nine independent operators. Flat six-entry table did not encode the held-out compositions; exact equivariant projection and hard tying collapsed to one function. 27 fresh payload/hash/metric rows replayed exactly; three tests passed. Analytic operators only, known generators, no trained attention or LM evidence. Dedicated branch: `research/ma-337-group-factorized-position-role-20261008`. Next P0: MA-341.


## MA-338 — FAIL for frozen Mirror-specific development gate

Across two development worlds, activation-based hidden permutation/sign canonicalization reduced the task delta subspace to rank 2; raw checkpoint PCA rank 2 had held-out function nMSE 0.92–0.93 and raw parameter variance required rank 44 for 99% coverage. Normalized direct and phase codes both retained held-out aligned functions at nMSE <1e-14, but actual Mirror phase payload was 3,033B/3,037B versus 2,987B/3,012B direct coefficients, larger in both seeds and short of the preregistered 10% margin. Fresh seeds 33811–33813 remained sealed. Off-orbit task quality required paid private residual. 12 development payload/hash/metric rows replayed exactly; three tests passed. Synthetic post-fit MLPs only. MA-335/337/338 now show the same family-level weakness: native direct group/functional coordinates equal or beat Mirror bytes at matched quality. Pause this group/symmetry insertion family for redesign. Dedicated branch: `research/ma-338-symmetry-normalized-mirror-code-20261008`. Next P0 in the federated personalization family: MA-342.


## MA-341 — FAIL for frozen Mirror-specific development gate

On two synthetic linear-regression development worlds, a shared linear pFedHN-style generator and Mirror phase both predicted 16 unseen clients at mean nMSE <5e-15. With private off-orbit fallback, Mirror payload was 1,527–1,528B vs pFedHN 1,721–1,730B, an 11.2–11.7% reduction that missed the frozen 20% gate. The ordinary scalar-phase control was byte/hash identical to Mirror. Client code communication fell from 8B to 4B; the off-orbit client needed private residual state. Fourteen development payload/hash/metric rows replayed exactly; three tests passed. Fresh seeds 34111–34113 remained sealed. Synthetic task attributes and linear hypernetwork only, not full pFedHN training. Dedicated branch: `research/ma-341-federated-mirror-vs-pfedhn-20261008`. Next P0: MA-351.


## MA-342 — FAIL for frozen Mirror-specific development gate

Across two synthetic LoRA worlds, both Mirror and the linear HyperLoRA full-factor generator recovered 16 unseen clients at nMSE <7e-15. With an off-orbit private factor, Mirror payload was 2,980B/2,976B while HyperLoRA was 2,861B/2,874B; Mirror was larger in both worlds and missed the frozen 20% margin. The native scalar-phase control had an identical payload/hash, and direct coefficients remained a simpler control. Mirror did halve client-code communication (8B to 4B) and used fewer generation MACs, but those did not improve total storage. The off-orbit client required private rank-2 state. Sixteen development payload/hash/metric rows replayed exactly; three tests passed. Fresh seeds 34211–34213 remain sealed. Dedicated branch: `research/ma-342-hyperlora-outputs-mirror-code-20261008`. Next P0: MA-351.


## MA-344 — PROMISING for the synthetic PreLort storage/communication point

Across three fresh synthetic rank-1/2/4 client worlds, shared prefix factors plus per-active-segment phase retained each rank group at nMSE 1.21e-14–1.39e-14. Payload averaged 4,713B vs 11,412B PreLort-style active client A factors (58.7% lower), and communication proxy 608B vs 8,768B. The direct two-coefficient control was 5,234B (9.95% higher); ordinary scalar phase was byte/hash identical, so no Mirror-specific advantage is claimed. The off-orbit client had nMSE 0.651 without private rank-4 factors; private state restored near-zero error. 35 payload/hash/metric rows replayed exactly; three tests passed. Synthetic post-fit tasks only. Dedicated branch: `research/ma-344-prelort-nested-rank-mirror-20261008`. Next P0: MA-351.


## MA-346 — PROMISING for a synthetic shared/private frontier

Across three fresh client worlds, at 25%/50% outliers a shared phase plus rank-1 private residual retained exact function quality and used 6,739B/8,639B vs 20,243B/35,696B FedRep dense-private (66.7%/75.8% lower). Communication proxy was 2,336B/4,416B vs 16,928B/33,344B. Without private factors, outlier cohort nMSE was approximately 0.0035. The native scalar-phase control was byte/hash identical, so no Mirror-specific gain is claimed. At 0% heterogeneity the dense-private margin was only 2.1%. Full 0–100% sweep, 175 exact payload/hash/metric replays, three tests passed. Synthetic post-fit linear clients only. Dedicated branch: `research/ma-346-federated-shared-private-mirror-20261008`. Next P0: MA-351.


## MA-349 — FAIL for frozen posterior storage gate

In two development worlds, Mirror exactly matched the planted two-mode posterior (NLL ~0.658, Brier ~0.233, ECE ~0.012, probability calibration MSE 0), but used 1,196B versus 1,195B for the rank-1 BNN and 651–652B for two independent posterior weights. The direct sparse scalar control had identical bytes/hash. Fresh seeds 34911–34913 stayed sealed. Ten development payload/hash/metric rows replayed exactly; three tests passed. Analytic sparse logistic posterior only. Dedicated branch: `research/ma-349-rank1-bayesian-mirror-posterior-20261008` (commit `52d734a`). Next P0: MA-351.

## MA-351 — FAIL for MIMO member diversity

Across two development worlds, rank-1 Mirror and ordinary direct member coefficients collapsed to identical member functions (pairwise JSD 0, correctness correlation 1.0), equal ensemble NLL, and actual payloads 6,767–6,790B differing by at most 1B. Native MIMO preserved lower correctness correlation (~0.82/0.73) at similar ensemble NLL with ~8.7KB. Fresh seeds 35111–35113 stayed sealed. An invalid initial dev attempt was excluded under a documented pre-fresh amendment. Ten amended payload/hash/metric rows replay exactly; three tests passed. Synthetic fixed-budget screen only. Dedicated branch: `research/ma-351-mimo-mirror-diversity-20261008`. Next P0: recheck registry and queue.
