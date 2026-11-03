# Mirror Application Status Board

Updated: 2026-11-03 UTC
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (220 completed; 409 UNTESTED; 1 SCREENING; 1 NOT ESTABLISHED)
- P1: **422** (15 completed; 407 UNTESTED)
- P2: **103** (0 UNTESTED)
- Current MA statuses: **175 FAIL, 1 NOT ESTABLISHED, 59 PROMISING, 1 SCREENING, 919 UNTESTED**
- 47 baseline experiment directories remain present; verified external outcomes are linked to their dedicated branches in the registry and claim ledger.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). See the current Next candidate section for the live queue.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. see the current Next candidate section.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; see the current Next candidate section.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. see the current Next candidate section.
- Research support PA372..381 and a **separate 48-row / two-seed real-digit shifted-task code pilot** have been added. Structured 8-value Mirror m improved average CE versus 6-value diagonal code but FAILED the preregistered full quality/bytes/runtime gate. This is an **exploratory negative pilot**, not an MA-1096/1099/1115 completed experiment. The pilot remains exploratory and does not change registry statuses. See [function-space review](../../docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md) and [pilot results](research_intake/natural_digit_function_20261008/RESULTS.md).
- Sixteenth literature sweep added **MA-1116..1155** (40 UNTESTED; 33 P0 / 7 P1) and **PA382..413** (32 primary sources): knowledge graph relation operators, camera ISP/lens optics, robot dynamics and room neural acoustic fields. No new MA measurements. See [sixteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md) and [40 per-ID test blueprints](../../docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md); see the current Next candidate section.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-622 — Factorized CAVIA context (P0; PA129)**

MA-574, MA-575, MA-576, MA-578, MA-581, MA-582, MA-589 and MA-591 are reconciled from their dedicated branches. MA-574 is NOT ESTABLISHED because its frozen source digest mismatches the committed source; the other seven are scoped FAILs. MA-581/582 triggered the documented MLA latent family pause for MA-583/585/586. MA-545 is reconciled as FAIL from its canonical few-shot routed FV branch; an alternative activation-only MA-545 protocol is preserved separately and also fails attribution, without pooling. MA-592 fails its top-2 prompt composition quality/native-L2P gates. MA-621 is a verified development FAIL: direct-angle exactly aliases Mirror and CAVIA fits the aligned tasks with fewer adaptation evaluations. MA-622 remains the next executable P0.

## Active experiment

MA-622 is next for a dedicated research branch and frozen development screen. The cumulative evidence branch is `research/ma-evidence-reconciliation-1155-20261009`; main remains untouched.

## Verified status index

- **PROMISING (59):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-255, MA-258, MA-268, MA-276, MA-282, MA-312, MA-314, MA-325, MA-330, MA-344, MA-346, MA-381, MA-408, MA-411, MA-431, MA-453, MA-470, MA-471, MA-473, MA-475, MA-476, MA-482, MA-494, MA-503, MA-504, MA-511, MA-561, MA-608, MA-612, MA-613, MA-691.
- **NOT ESTABLISHED (1):** MA-574.
- **FAIL (175):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-257, MA-260, MA-261, MA-265, MA-266, MA-271, MA-272, MA-273, MA-274, MA-278, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299, MA-301, MA-303, MA-304, MA-307, MA-309, MA-311, MA-315, MA-318, MA-319, MA-320, MA-322, MA-327, MA-331, MA-332, MA-333, MA-335, MA-337, MA-338, MA-341, MA-342, MA-349, MA-351, MA-353, MA-355, MA-356, MA-357, MA-359, MA-360, MA-361, MA-364, MA-366, MA-367, MA-368, MA-369, MA-371, MA-372, MA-374, MA-375, MA-379, MA-383, MA-385, MA-389, MA-391, MA-392, MA-393, MA-395, MA-397, MA-399, MA-401, MA-403, MA-405, MA-407, MA-413, MA-416, MA-417, MA-418, MA-419, MA-424, MA-425, MA-427, MA-429, MA-434, MA-436, MA-437, MA-442, MA-444, MA-446, MA-451, MA-452, MA-455, MA-457, MA-461, MA-462, MA-463, MA-464, MA-466, MA-468, MA-469, MA-478, MA-481, MA-484, MA-486, MA-487, MA-488, MA-492, MA-498, MA-501, MA-502, MA-508, MA-510, MA-516, MA-517, MA-518, MA-520, MA-521, MA-527, MA-539, MA-540, MA-545, MA-546, MA-547, MA-550, MA-551, MA-553, MA-554, MA-556, MA-557, MA-558, MA-560, MA-562, MA-565, MA-567, MA-569, MA-571, MA-573, MA-575, MA-576, MA-578, MA-581, MA-582, MA-589, MA-591, MA-592, MA-596, MA-597, MA-598, MA-599, MA-600, MA-601, MA-602, MA-603, MA-604, MA-606, MA-607, MA-610, MA-611, MA-614, MA-616, MA-617, MA-618, MA-621.

All per-ID evidence is retained in experiment directories and `CLAIM_LEDGER.csv`. The MA-545 alternative protocol remains on its dedicated branch and is cross-referenced in its reconciliation note.
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

## Registry-driven continuation

MA-516 is audited and integrated as FAIL: PCA rank-4 reduced explicit intervention bytes to 34.6% with low vector NRMSE, but explicit and compressed interventions all had 0% held-out task accuracy. This is scoped to the tested GPT-2 support-delta extraction/injection. The registry is authoritative; the next untested P0 is MA-517, function-vector Mirror composition. Earlier cached pointers to MA-483 are stale.

MA-517 is audited and integrated as FAIL: on pinned GPT-2, direct ICL, query-only, raw-sum, layer-factorized, and single-vector conditions all had 0% exact accuracy across 9 fresh world-seed banks. Raw and factorized payloads were equal at 99,945B. This is scoped to the tested synthetic composition and support-delta vectors.

MA-518 is audited as a task-quality FAIL; its factorization comparison is NOT ESTABLISHED because inference ignored the stored procedure/domain codes. Direct ICL was valid at 31.25%, but all interventions scored 0% across nine fresh banks. Do not treat the smaller 11,993B payload as a quality-matched Mirror compression result.

MA-520 is FAIL: K8 product quantization reduced vector bytes to 27.7% of explicit but explicit, PCA and PQ interventions all had 0% task accuracy; vector NRMSE was .1755 and NLL was worse than explicit. Together MA-518/520 show repeated failure of prompt-delta interventions to produce useful task behavior; MA-518 also had a factor-decoder defect. This FV compression subfamily is paused for redesign of the causal extraction/measurement interface. Resume only with head-specific causal vectors and a valid task-native quality gate; continue the registry queue in other families.

MA-521 is FAIL: a demo encoder recovered task identity only 25% of the time while direct ICL reached 37.5%; oracle and compiled function-vector executions had 0% task accuracy, and compiled bytes exceeded the explicit FV+ID bank. Across MA-518/520/521, no prompt-delta FV execution produced task utility; MA-518 additionally had a factor decoder defect. This meets the family pause criterion: defer further FV variants until a causal extraction/execution interface and valid matched generic encoder are established.

MA-526 remains SCREENING / NOT ESTABLISHED: the CPU fresh run was interrupted without metrics or serialized payloads. Development showed no Mirror-specific uplift over independent/generic codes, and the attempted fresh row IDs may be contaminated. A future amended run needs new row IDs, batched inference, SAE semantic validation, full sequence scoring, and interrupt-safe artifacts.

MA-527 FAILs on the frozen development causal-quality gate in two distinct protocol branches (reported separately; no fresh result). Both save incremental intervention bytes but lose materially to explicit FVs, and the complete SAE deployment payload can exceed the explicit baseline. MA-526/527 show the current SAE feature span is poorly aligned or the scoring/implementation path is not yet trustworthy. Pause the shared SAE-feature family before MA-528/530 until feature semantics and causal measurement are redesigned; current MA-526/527 results do not establish any fresh-data benefit. Next P0 outside this held family: MA-539.

MA-539 is a scoped FAIL for function-vector-specific value. Fact: the exact finite-state synthetic packet task achieved 100% packet accuracy at 468B shared address versus 1,245B repeated slot addresses. The FV, PTP, generic latent, and oracle rule-ID payloads were identical in bytes/hash/output, so ordinary shared-address factorization explains the savings. Protocol amendment timing was imperfect; no neural or language-model claim follows.

MA-540 FAILs the registered Mirror-specific comparison in two separately preserved development protocols; fresh remained sealed. In the first, native tied operator-ID control is exact, 14B smaller and far cheaper per packet than FV. In the second, FV misses the .99 quality gate in one world and native tied-ID outputs match at 14B fewer bytes. Together with MA-539 (generic shared latent equals FV), the same structural cause is repeated: function identity/state passing is captured by ordinary codes, with no FV-specific benefit. Pause the sequential function-vector family; defer further variants until a design can beat matched native/generic controls.

MA-546 FAILs as a useful multiplicity/compression application but confirms an important boundary: compensated hidden-coordinate views can be function-invariant gauge choices, while unpaired transforms change outputs only with larger serialized state. The Pythia branch has verified weights, fresh worlds, and exact replay; a separate fixed linear block reproduces the symmetry mechanism but lacks model-hash provenance. Address count is not function count.

MA-547 FAIL: the synthetic locality edit missed its frozen efficacy gate in one development seed. The simpler gated scalar bias exactly matched edit efficacy/locality at 3,026B versus Mirror at 35,990B; matched gated RoseLoRA was also equivalent. Fresh stayed sealed.

MA-550's 2-dev + 3-fresh synthetic allocation screen passes reconstruction but FAILs Mirror-specific attribution: all selected functions reduce to native sparse output biases plus ReFT/shared output-basis codes. Actual state is 19,682B and decode proxy 412.1M FLOPs/bank. Verification JSON has a contradictory nested fresh flag; the audit addendum preserves and discloses this while retaining fresh rows/hashes.

MA-551 FAIL: Givens views recover an aligned synthetic function bank in fewer bytes than independent FiLM, but miss the frozen fused-throughput gate in both development worlds and exactly alias a native Givens adapter. Fresh stayed sealed.

MA-553 FAIL: task×layer rank-4 FiLM factorization and direct coefficients are the same function at the same 5,204B, but miss held-out output quality badly (nMSE 4.93). A smaller HyperFormer-like generator also misses quality; fresh stayed sealed.

MA-554 FAILs Mirror-specific attribution: on an aligned synthetic rotation orbit, the direct two-basis coefficient control is byte- and function-identical to Mirror (973B, zero held-out nMSE). Fresh stayed sealed.

MA-556 FAILs the registered storage frontier and Mirror-specificity test. Sparse private residuals restore oracle synthetic operator quality only after their bytes erase the savings (at rho=.1, 2,548B versus 2,539B independent; at higher heterogeneity, roughly half the entries become private). Fresh stayed sealed.

MA-557 FAILs Mirror-specificity: across three widths, direct channel gains exactly equal the Mirror functions and bytes (1,016B); ordinary independent-width storage is 1,249B. Plain nested prefix is smaller but loses quality (mean nMSE .06485). Fresh stayed sealed.

MA-558 FAILs Mirror-specificity but shows strong aligned factorization: six held-out width/layer mixes reconstruct exactly at 1,364B versus a 27,564B flat table. Native prefixes are smaller but inaccurate; ordinary direct factorized coefficients exactly match Mirror. Fresh stayed sealed.
