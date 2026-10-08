# Mirror Application Status Board

Updated: 2026-10-08 JST (live branch reconciliation)
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (170 completed; 460 UNTESTED; 0 SCREENING)
- P1: **422** (13 completed; 409 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **127 FAIL, 11 NOT ESTABLISHED, 45 PROMISING, 972 UNTESTED, 0 SCREENING**
- 47 baseline experiment directories remain present; 138 additional per-ID outcomes are linked to their dedicated research branches in `LIVE_BRANCH_RECONCILIATION.csv`.
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

**MA-455 — sequential Mirror program over one block (P0; PA81).** MA-453 is a verified development-screen FAIL: it improved routed-only RMSE but exactly aliased native Givens, lost quality to role-vector residuals, and was dominated in quality/bytes by independent blocks. Fresh seeds stayed sealed. MA-453 result branch is pushed; continue with MA-455.

MA-434 and MA-436 found exact native SSM parameterization aliases; MA-442 and MA-453 likewise alias native Givens conditioning. MA-453’s fixed-router synthetic screen showed no Mirror-specific value against role-vector or independent-block controls. Fresh seeds remain sealed.

## Active experiment

**MA-455** is in frozen-protocol screening on `research/ma-455-sequential-mirror-program-20261008`; the ordered non-commutative block program is being compared with tied, rank-1, direct Givens and independent-block controls.

MA-416..419, MA-424, MA-434, MA-436, MA-442, MA-444 and MA-451/452/453 are verified development-screen FAILs with fresh data sealed. MA-451/452 both exactly alias native Givens conditioning, and MA-452's withheld path-role result is recorded as aligned compositional generalization only. Pause the PathNet path × role family; see [MA-451/452 diagnostic](../../docs/phase2/PATHNET_MIRROR_PATH_ROLE_FAMILY_DIAGNOSTIC_2026-10-08.md). State-space role/expert candidates remain paused under their separate diagnostic. See the corresponding per-ID reports and verifications.

## Verified status index

- **PROMISING (45):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-257, MA-258, MA-268, MA-276, MA-282, MA-312, MA-314, MA-330, MA-344, MA-346, MA-374, MA-381, MA-603, MA-691, MA-771, MA-841, MA-879.
- **NOT ESTABLISHED (11):** MA-325, MA-643, MA-899, MA-971, MA-998, MA-1010, MA-1066, MA-1069, MA-1091, MA-1138, MA-1141.
- **FAIL (127):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-255, MA-260, MA-261, MA-265, MA-266, MA-271, MA-272, MA-273, MA-274, MA-278, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299, MA-301, MA-303, MA-304, MA-307, MA-309, MA-311, MA-315, MA-318, MA-319, MA-320, MA-322, MA-327, MA-331, MA-332, MA-333, MA-335, MA-337, MA-338, MA-341, MA-342, MA-349, MA-351, MA-353, MA-355, MA-356, MA-357, MA-359, MA-360, MA-361, MA-364, MA-366, MA-367, MA-368, MA-375, MA-379, MA-383, MA-385, MA-389, MA-391, MA-392, MA-393, MA-395, MA-397, MA-399, MA-401, MA-403, MA-405, MA-416, MA-417, MA-418, MA-419, MA-424, MA-427, MA-434, MA-436, MA-442, MA-444, MA-446, MA-451, MA-452, MA-453, MA-462, MA-464, MA-492, MA-504, MA-669, MA-674, MA-707, MA-715, MA-721, MA-732, MA-742, MA-753, MA-760, MA-767, MA-783, MA-784, MA-790, MA-818, MA-824, MA-840, MA-921, MA-932, MA-945, MA-951, MA-962, MA-981, MA-990, MA-1018, MA-1064, MA-1075, MA-1097, MA-1120.

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
