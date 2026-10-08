# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (58 completed; 572 UNTESTED)
- P1: **422** (12 completed; 410 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1085 UNTESTED, 34 PROMISING, 36 FAIL**
- 68 local experiment directories are complete; MA-301 and MA-307 are completed experiments linked to their dedicated research branches.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-276 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-276 is next; MA-275 remains UNTESTED (P1).
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-276 is next; MA-275 remains UNTESTED (P1).
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-276 is next; MA-275 remains UNTESTED (P1).
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-311 — Mirror task code in a random intrinsic subspace (P0; PA33)**

MA-255 is reconciled as PROMISING only for its aligned post-fit representation screen; a distinct 1,200-update protocol variant failed at development and remains sealed on fresh worlds. See `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`.

MA-260 is complete FAIL: the one-layer linear screen missed its predeclared byte gate, while the independent-task stress case collapsed. It is not a deep BatchEnsemble result; see its report and verification.

MA-261 is complete FAIL after audit: its post-fit branch's raw MSE ratios meet the frozen <=1.10x independent quality criterion in only 1/4 fresh worlds, despite source wording claiming 4/4. A different fixed-update protocol also failed during development. Both variants are retained in its reconciliation report.

MA-265 is complete FAIL: a fresh aligned post-fit screen achieved numerical-zero task MSE but saved only 4.4% total payload versus VeRA, below the preregistered 20% gate; independent codes failed. A distinct 1,000-update diagonal-code protocol also failed at development against VeRA. Both protocols are retained separately.

MA-265 and MA-266 are consecutive FAILs in the VeRA Mirror family: native VeRA missed the strict margin in MA-265, and an ordinary coefficient-product control matched MA-266 within 2.1% bytes. Pause MA-267 until the family has a control/redesign that tests a distinct Mirror-specific contribution.

MA-268 is PROMISING on a narrow synthetic nonlinear task: a trained Givens-aligned screen passed its quality/byte gate 3/3 at 4,167B vs 4,273B IA3, with lower quality error but 1.5x MACs and substantially slower eager CPU throughput. A separate post-fit linear activation screen missed its strict 20% byte gate (15.7% saving). See its protocol reconciliation.

MA-271 is FAIL for Mirror-specific value. One post-fit screen found a low-byte Givens orbit, but an exact simple rank-one angle-factor control produced identical functions and bytes (1,086B); a separate trained dense-OFT screen omitted that control. See its reconciliation report.

MA-272 is FAIL for Mirror-specific/runtime gain: the corrected audit found that scalar-times-shared-skew exactly matches Mirror at 3,786B, while exact input-side execution averaged 0.335ms versus 0.138ms materialized. A separate trained screen retains its bounded storage/quality result but was slower than dense OFTv2.

MA-273 is complete FAIL for Mirror-specific value: corrected post-fit views exactly matched ordinary scalar-times-shared-angle factorization (same 1,150B payload); an independent neutral-initialization training screen failed at development and left fresh sealed. See its reconciliation report.

MA-274 is complete FAIL on a two-world fixed-update development screen: at 5,430B Mirror was 3.4% smaller than native BOFT but had worse MSE in both worlds; independent experts were much better and butterfly eager CPU throughput was 9–20x below IA3/rank-one. Fresh worlds stayed sealed. Together with MA-273, this pauses the shared scalar-angle task/expert BOFT insertion family. MA-275 remains UNTESTED (P1). MA-276 is PROMISING only for aligned serialized-state compression: 396B Mirror vs 1,193B untied (3/3 fresh), but 2,048B runtime workspace exceeds that payload and unrelated layers fail. No runtime-memory claim. MA-278 is FAIL on the unaligned task-factor screen: Mirror matched scalar control bytes at 2,525B but lost quality at both development worlds and both learning rates; native Compacter also dominated quality. Fresh worlds remained sealed. Aligned feasibility is untested. MA-282 is PROMISING only for a deliberately aligned synthetic ReLU FFN orbit: exact in 3/3 at 925B vs 1,145B free-angle Monarch and 3,279B independent full. Unrelated functions collapse to hard-tie quality; view workspace is 1,024B, so no runtime-RAM claim. MA-286 is FAIL for Mirror-specific margin: the shared-B index screen reached aligned quality with 4,412B vs task-local Cheap-LoRA 6,580B, but an identical one-hot shared-B gate was only 78B larger (1.74%, below the preregistered 10%). Two unrelated functions used private rank-4 factors. MA-288 is FAIL under its preregistered direct-path runtime gate: aligned Mirror state/quality was strong at 378B vs 1,203B independent and quarter the writes, but throughput was 0.178–0.456x rank-4 residual in all fresh seeds. A cached path reached 201M examples/s with 1,024B workspace and is only supplemental. MA-292 is FAIL for Mirror-specific value: one angle beat FP32 SVD code size but matched-byte FP16 SVD was 23B smaller overall, slightly more accurate in all 3 fresh worlds, and about 1.55x faster. Unrelated task vectors required richer/private state. MA-296 is FAIL for Mirror-specific value: the simpler direct shared-orbit generator matched Mirror exactly at 1,524B vs 1,561B, while independent task vectors were needed for unrelated maps (4,771B). Rademacher PSP interference was high; rank-8 SVD matched quality with larger payload. MA-297 and MA-299 are FAIL; this SETA shared/private Mirror-code family is paused for redesign under the two-consecutive-failure stop condition. The next independent P0 screen is MA-257.

## Active experiment

No active experiment declared. MA-258 has a narrow PROMISING aligned codec result; unrelated experts required private/richer state. MA-266 is FAIL for Mirror-specific value: factorized views generalized on an aligned task cross-product, but the ordinary coefficient-product control matched within 2.1% payload bytes. Together with MA-265 this pauses VeRA family follow-up MA-267 pending redesign. MA-255 through MA-299 now have verified status records. MA-257 failed against the exact native PA16 rotational-context control; MA-297/299 SETA remains paused. Next by the registered P0 queue: MA-311.

## Verified status index

- **PROMISING (34):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-255, MA-258, MA-268, MA-276, MA-282, MA-691.
- **FAIL (36):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-257, MA-260, MA-261, MA-265, MA-266, MA-271, MA-272, MA-273, MA-274, MA-278, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299, MA-301, MA-307.

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

On three development worlds, the smallest selected support (64/512 factor combinations) composed the aligned teacher at 296B and mean test normalized MSE 6.58e-9. Native PA16 factorized rotational contexts produced the exact same functions, bytes and errors, so the registered Mirror-specific gate failed; fresh seeds remain sealed. The FP16 free-coefficient table used 348B, but it is a weaker control than native rotational context. On independent maps Mirror mean error was 8.64, while coefficient pairs reached 1.036 and independent full weights were near zero. Seven tests and 54 selected-support development metric/byte/compute/workspace/hash replays passed. Next P0: MA-311.


## MA-258 — PROMISING only for aligned post-fit expert-bank coding

On three fresh aligned worlds, an eight-expert Givens family used 1,514B Mirror payload at normalized MSE 8.99e-10–1.07e-9; rank-2 shared SVD reached near-zero error at 3,822B, and independent maps used 8,440B. Native binary-context PSP used 1,766B but had normalized MSE 4.62–5.46. This passed the preregistered development gate and replicated on 3/3 fresh worlds. On unrelated experts, Mirror normalized MSE was 0.848–0.889, essentially hard tying; independent weights were needed for near-zero error. Mirror had no measured throughput gain (11.16M examples/s fresh aligned vs 17.88M independent). Four tests and 60 metric/byte/hash replays passed. Oracle expert IDs and zero training updates; synthetic linear screen only. Next P0: MA-266.


## MA-266 — FAIL for Mirror-specific value; VeRA family paused

In three synthetic development worlds, Mirror composed 8 held-out factor pairs from 8 checkerboard support maps at normalized MSE 7.91e-9 and 1,276B. The direct sine/cosine coefficient-product control had normalized MSE 4.20e-8 at 1,304B, just 2.1% more bytes and within quality; this misses the frozen 10% byte and 10× quality margins. VeRA-style additive factor codes failed aligned composition (MSE 0.289). On unrelated maps Mirror and coefficient-product errors were about 0.401; independent full weights were near zero. The measured operation proxy favored direct Mirror rotations, but an optimized coefficient-product implementation can apply the same transform; no runtime claim. Four tests and 60 exact metric/byte/hash replays passed. Fresh 26611–26613 remain sealed. Alongside MA-265, this pauses MA-267 for redesign. MA-301 was completed on a separate research branch; next independent P0: MA-307.


## MA-301 — FAIL; branch-linked result

The amended 32-task random-feature screen saved 21.2% payload versus packed binary masks (334B vs 424B), but Mirror missed the independent-quality criterion in 2/3 fresh worlds (normalized MSE 2.14e-5, 0.412 and 0.256). Only one fresh world passed. Its initial four-task storage-gate miss is retained separately. Throughput used precomputed effective vectors, so no direct-view runtime claim. Three tests and 35 metric replays passed. Full evidence is on `research/ma-301-continuous-mirror-supermask-20261008`. Next P0: MA-307.


## MA-307 — FAIL for strict composite Pareto; storage frontier retained

On three fresh eight-task streams, four Givens-aligned tasks used one-byte View codes before allocation; three unrelated tasks received private readouts. Mirror payload averaged 854B vs 1,333B PackNet-style mask-plus-allocation (35.9% saving), and no prior-task test error changed. Mean normalized MSE was 2.08e-5 vs about 9.2e-16 for private PackNet readouts. Allocation-search runtime ratios were 1.45x, 1.10x and 1.22x; one exceeded the preregistered 1.25x ceiling. Four tests, 36 summary rows and 288 allocation-event rows replayed. Dedicated branch: `research/ma-307-packnet-mirror-allocation-20261008`. Next P0: MA-311.
