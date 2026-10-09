# Mirror Application Status Board

Updated: 2026-10-09 UTC (MA-545 result integrated)
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (208 completed; 421 UNTESTED; 1 SCREENING)
- P1: **422** (13 completed; 409 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **162 FAIL, 14 NOT ESTABLISHED, 45 PROMISING, 933 UNTESTED, 1 SCREENING**
- 47 baseline experiment directories remain present; 171 additional per-ID outcomes are linked to their dedicated research branches in `LIVE_BRANCH_RECONCILIATION.csv`.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This was the intake-time queue; current selection follows the live-branch reconciliation at the top of this board.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed at intake time; later live-branch outcomes are indexed above.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results were included at intake; live queue selection is recorded above.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. At intake, the queue pointed to MA-255; the live branch reconciled queue is at the top of this board.
- Research support PA372..381 and a **separate 48-row / two-seed real-digit shifted-task code pilot** have been added. Structured 8-value Mirror m improved average CE versus 6-value diagonal code but FAILED the preregistered full quality/bytes/runtime gate. This is an **exploratory negative pilot**, not an MA-1096/1099/1115 completed experiment. At pilot time the board held 29 PROMISING / 18 FAIL and MA-255 next; see current counts above. See [function-space review](../../docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md) and [pilot results](research_intake/natural_digit_function_20261008/RESULTS.md).
- Sixteenth literature sweep added **MA-1116..1155** (40 UNTESTED; 33 P0 / 7 P1) and **PA382..413** (32 primary sources): knowledge graph relation operators, camera ISP/lens optics, robot dynamics and room neural acoustic fields. No MA results were included at intake time; later live-branch outcomes are indexed above. See [sixteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md) and [40 per-ID test blueprints](../../docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md); the live queue is recorded at the top of this board.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

MA-545 is a verified **FAIL** on `research/ma-545-fv-residual-moe-20261009`: few-shot route accuracy and oracle FV quality pass, and routed FVs improve gold likelihood, but shared-mean FV has higher held-out candidate accuracy in all five seeds at 8.27x lower fresh payload. MA-546 representation-space symmetry audit is now SCREENING on `research/ma-546-representation-symmetry-20261009`.

## Active experiment

MA-486 FAIL: sparse int8 reduced bytes 11.8% and decode operations 90.6% versus dense int8, but missed the frozen 20% byte gate and exactly aliased native OMP. Fresh stayed sealed. MA-487 FAIL: LISTA beats OMP compute but direct top-3 is more accurate at similar/lower bytes and ops. MA-486/487 family diagnostic is recorded; MA-488 FAIL for Mirror-specific attribution: at 12.5% private residuals, shared/private is 33.6% smaller than full int8 at zero RMSE; at 25% it exceeds int8 bytes. Native shared/private exactly aliases it. Fresh sealed; MA-494 FAIL for Mirror-specific attribution: Hamming(7,4) cuts p=.05 route errors to 4.5% at 17,322B versus raw IDs 19% at 17,307B, but exactly matches native ECOC. Fresh sealed; MA-498 FAIL: at sigma .2 the distance-regularized code reduced wrong-route rate 60–75% versus random orthogonal addresses, but 9,384 B was 1.0613x raw-ID bytes (frozen max 1.05x) and native metric learning exactly matched its payload/output. Fresh stayed sealed. MA-501 FAIL: at rho=0, shared rank-four intervention is lossless at 2,138 B vs 13,668 B independent rank-four, but exactly aliases native shared coefficients. At rho=.25 heldout RMSE .173–.226 exceeds the .10 gate and independent rank-four is more accurate; fresh sealed. MA-502 completes the shared-LoReFT code-count sweep: 256 aligned functions occupy 6,005 B vs 201,837 B independent rank-four with relative RMSE <=3.6e-6, but exactly alias native shared coefficients. MA-501/502 family diagnostic pauses unchanged static shared-subspace coefficient variants; MA-503 layer×task factorization is complete and FAIL: the held-out quality varied from .002 to 6.754 across seeds, its factor payload was 95.6% of the dense code table, and native bilinear coefficients exactly matched it. The diagnostic now pauses unchanged static and factorized shared-subspace code variants. MA-508 activation-addition bank FAIL: rank-8 reproduces rho=0 outputs at 8,324 B vs 13,194 B explicit vectors (0.631x), but exact native PCA alias; at rho=.1 RMSE .063/.080 and decode compute is 9x explicit. Fresh sealed. MA-510 conditional activation FAIL: rank4/rank4 uses 4,334 B vs 7,532 B CAST and 640 vs 1,088 operation proxy, but event recall .145/.071, behavior accuracy .944/.830 and false-trigger .0173/.0105 miss gates; exact native PCA alias, fresh sealed. Synthetic mechanism only. MA-511 FAIL: rank4 on 16 held-out pairs RMSE .024/.045 at 4,136 B vs 16,932 B full table; native additive/PCA exactly matches. CAST additive is exact at 5,414 B and lower operation count; rho=.1/.25 errors rise. Fresh sealed; synthetic only. MA-516 FAIL: explicit FVs improved held-out logprob by 2.62–2.97 nats vs no intervention, but rank4 lost .535/.606 nats and native PCA exactly matched; payload 12,688 B vs 34,454 B. Fresh sealed; single small Pythia model only. MA-517 is a verified development-screen FAIL: no-intervention accuracy was .938/.979; product ranks reached .854–.958 but did not clear the frozen +.10 gate over both raw sum and difference in both worlds; 8,456–29,576 B vs 34,446 B explicit bank, with exact native PCA product aliases. Fresh sealed; deterministic replay exact. MA-520 is a verified development FAIL: learned rank4 uses 12,692 B but loses .607/.859 gold-logprob nats versus explicit FVs and ties native PCA within .0002 nats; int8 FVs use 10,188 B with <.0017 nat loss and same accuracy. Fresh sealed; deterministic replay exact. MA-516/517/520 share a static PCA/linear-code failure cause; unchanged variants are paused under [family diagnostic](../../docs/phase2/FUNCTION_VECTOR_CODE_FAMILY_DIAGNOSTIC_2026-10-09.md). MA-521 is a verified FAIL: tanh compiler saves 89.7%/90.1% query tokens but loses 1.301/.729 gold-logprob nats vs explicit FVs, is .120/.026 nats worse than linear, and misses its byte cap by 23 B. Direct ICL has better gold likelihood and smaller token-bank bytes; fresh sealed. MA-526 is a verified FAIL: shared-pool code loses 1.463/1.567 gold-logprob nats versus explicit FV and misses the .05 accuracy tolerance on seed one. It improves over global OMP by only .187/.048 nats and costs 266 B more incremental bytes. Standalone Pythia+SAE+code is 4,172,903 B larger than Pythia+explicit FV; fresh sealed. MA-527 is a verified FAIL: Givens beat same-size pairwise gains by .204/.225 nats but lost 2.333/2.607 nats to explicit FVs; incremental code is 3,432 B, while required SAE makes standalone storage 4,172,997 B worse. Fresh sealed; deterministic replay exact. MA-528 is a verified FAIL: residual-selected pool beats activation-pool selection by .359/.670 nats, but loses 1.237/1.157 nats to explicit FV and .099/.090 to global OMP16; payload 3,990 B vs 3,840 B. Fresh sealed. Static task-FV SAE feature-bank variants MA-526/527/528 are paused pending a changed dictionary/target or private-residual design; see the [family diagnostic](../../docs/phase2/SAE_FEATURE_VIEW_FAMILY_DIAGNOSTIC_2026-10-09.md). MA-530 is a verified FAIL: query-only task routing is .477/.516; shared SAE experts lose 1.013/1.162 nats to same-router FV and cost 150 B more than global OMP16. Fresh sealed; exact replay. MA-533 is a verified FAIL: trained transcoder FVU .0159/.0157 did not transfer to task FVs (−2.400/−2.289 nats; 36,294 B vs 34,214 B). Fresh sealed; corrected deterministic replay. MA-534 FAIL: role Givens stayed within explicit mean quality tolerance on both seeds, but the 70,902 B view was 2.11x the explicit 33,580 B bank; Givens lost to pairwise by .187/.068 nats and failed the .10 nat margin against elementwise in both seeds. Replay exact; fresh sealed. MA-539 FAIL: support-extracted FV yields 0/96 exact packets in both arbitrary-permutation worlds, as did generic/slot-code controls. Exact function-table upper gets 100% at 1,052 B; FV payload is 23,784 B vs 21,094 B generic. Selected NLL improves, but packet quality and byte gates fail; fresh sealed, replay exact. MA-540 FAIL: tied FV reaches .9795/.9920 composition, saving 46.9% bytes vs untied, but misses .99 in world1 and exactly aliases native tied-ID/external two-call (which use 14 fewer bytes). Endpoint sum scores .19/.15; affine table reaches 1.0 at 1,286 B. Fresh sealed; exact replay. Next P0: MA-545 function-vector MoE without weight experts.

MA-461 and MA-466 are verified development-screen FAILs; MA463 remains paused with MA-461/462. MA-466 shows useful factorized gates but no Mirror-specific gain over native CP and misses the strict byte gate.

MA-457 is a verified development-screen FAIL with fresh data sealed. It reduced births but not actual storage or compute, and exactly aliased native low-rank adaptation.

MA-453 and MA-455 are verified development-screen FAILs; fresh data remains sealed. MA-455 confirms ordered Views preserve order sensitivity on its aligned toy task but are larger than independent blocks and exactly alias native Givens. See the dedicated branch report and verification.

MA-416..419, MA-424, MA-434, MA-436, MA-442, MA-444 and MA-451/452/453 are verified development-screen FAILs with fresh data sealed. MA-451/452 both exactly alias native Givens conditioning, and MA-452's withheld path-role result is recorded as aligned compositional generalization only. Pause the PathNet path × role family; see [MA-451/452 diagnostic](../../docs/phase2/PATHNET_MIRROR_PATH_ROLE_FAMILY_DIAGNOSTIC_2026-10-08.md). State-space role/expert candidates remain paused under their separate diagnostic. See the corresponding per-ID reports and verifications.

## Verified status index

- **PROMISING (45):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-257, MA-258, MA-268, MA-276, MA-282, MA-312, MA-314, MA-330, MA-344, MA-346, MA-374, MA-381, MA-603, MA-691, MA-771, MA-841, MA-879.
- **NOT ESTABLISHED (14):** MA-325, MA-468, MA-469, MA-470, MA-643, MA-899, MA-971, MA-998, MA-1010, MA-1066, MA-1069, MA-1091, MA-1138, MA-1141.
- **FAIL (162):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-255, MA-260, MA-261, MA-265, MA-266, MA-271, MA-272, MA-273, MA-274, MA-278, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299, MA-301, MA-303, MA-304, MA-307, MA-309, MA-311, MA-315, MA-318, MA-319, MA-320, MA-322, MA-327, MA-331, MA-332, MA-333, MA-335, MA-337, MA-338, MA-341, MA-342, MA-349, MA-351, MA-353, MA-355, MA-356, MA-357, MA-359, MA-360, MA-361, MA-364, MA-366, MA-367, MA-368, MA-375, MA-379, MA-383, MA-385, MA-389, MA-391, MA-392, MA-393, MA-395, MA-397, MA-399, MA-401, MA-403, MA-405, MA-416, MA-417, MA-418, MA-419, MA-424, MA-427, MA-434, MA-436, MA-442, MA-444, MA-446, MA-451, MA-452, MA-453, MA-455, MA-457, MA-461, MA-462, MA-464, MA-466, MA-471, MA-473, MA-475, MA-476, MA-478, MA-481, MA-482, MA-486, MA-487, MA-488, MA-492, MA-494, MA-498, MA-501, MA-502, MA-503, MA-504, MA-508, MA-510, MA-511, MA-516, MA-517, MA-520, MA-521, MA-526, MA-527, MA-528, MA-530, MA-533, MA-534, MA-539, MA-540, MA-545, MA-669, MA-674, MA-707, MA-715, MA-721, MA-732, MA-742, MA-753, MA-760, MA-767, MA-783, MA-784, MA-790, MA-818, MA-824, MA-840, MA-921, MA-932, MA-945, MA-951, MA-962, MA-981, MA-990, MA-1018, MA-1064, MA-1075, MA-1097, MA-1120.

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

- The **1155-row** `IDEA_REGISTRY.csv` is authoritative for status and candidate identity. Do not merge or overwrite it with an older 254-row experiment checkout.
- `CLAIM_LEDGER.csv` and each experiment's `VERIFICATION.json` are the evidence index; `STATUS_BOARD.md` is an operational cache.
- Before allocating an ID or starting work, re-read the live registry and search for experiment branches.
- Preserve old branches, failed results, exploratory data and locked protocols. No automatic merge to main.
