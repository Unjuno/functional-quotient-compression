# Mirror Application Status Board

Updated: 2026-10-08 JST
Integration branch: `research/mirror-application-current-evidence-20261008`; canonical baseline: `research/mirror-application-worker-ready-20261007`

## Program totals (reconciled from authoritative 1155-row registry)

- Registered candidates: **1155**
- P0: **630** (128 completed; 502 UNTESTED)
- P1: **422** (17 completed; 405 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1010 UNTESTED, 44 PROMISING, 101 FAIL**
- 48 experiment directories, complete with status/protocol/results/verification files, are represented in the current evidence set.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This does not change prior verified results or the current MA-255 next-candidate decision.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-268 is next.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-268 is next.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-268 is next.
- Research support PA372..381 and a **separate 48-row / two-seed real-digit shifted-task code pilot** have been added. Structured 8-value Mirror m improved average CE versus 6-value diagonal code but FAILED the preregistered full quality/bytes/runtime gate. This is an **exploratory negative pilot**, not an MA-1096/1099/1115 completed experiment. Existing statuses are reconciled below; MA-268 is next. See [function-space review](../../docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md) and [pilot results](research_intake/natural_digit_function_20261008/RESULTS.md).
- Sixteenth literature sweep added **MA-1116..1155** (40 UNTESTED; 33 P0 / 7 P1) and **PA382..413** (32 primary sources): knowledge graph relation operators, camera ISP/lens optics, robot dynamics and room neural acoustic fields. No new MA measurements. See [sixteenth research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_SIXTEENTH_SWEEP.md) and [40 per-ID test blueprints](../../docs/phase2/MIRROR_APPLICATION_EXPERIMENT_BLUEPRINTS_MA1116_1155.md); next remains MA-255.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

**MA-405 — StyleGAN2-like FFN weight modulation (P0; PA65)**

**MA-405 FAIL:** on a synthetic context-permutation MLP, four-angle Mirror scored 57.64% vs native modulation 63.28%, FiLM 72.37%, and independent FFNs 89.83%. Mirror used 10,081 actual bytes; the quality gain over shared was only 0.52pp. Payload hashes and serialization replay verified. **Next: MA-407, demodulated Mirror-MoE.**

**MA-407 FAIL:** raw and row-norm-demodulated Givens Mirror both scored 66.45% in the fresh synthetic expert screen; activation RMS CV matched within 1e-7. The orthogonal input rotations preserve the weight norms being normalized, making demodulation functionally null here. **Next: MA-408, CondConv-style synthesized FFN.**

**MA-408 PROMISING (scoped native control):** CondConv on a fixed-context synthetic output-mixture task stayed 0.50pp behind output-mixture quality at equal actual bytes and 50.1% of the MAC proxy. Training wall clock was slower; no Mirror-specific benefit is claimed. **Next: MA-411.**

**MA-411 PROMISING (aligned feasibility only):** sparse Mirror on a shared-support Hadamard operator achieved normalized RMSE 0.000005 using 6,369B (28.2% of dense residual bytes), with maximum condition number 1.09. Natural support discovery and neural-model transfer remain untested. **Next: MA-413.**

**MA-413 FAIL:** in the aligned two-attribute Givens task, held-out combination accuracy was Mirror 70.72%, FiLM 68.45%, direct table 69.97%. The Mirror-specific margin missed the preregistered 5pp gate. **Next: MA-416/417.**

**MA-416 FAIL:** on 16 synthetic Fourier functions, Mirror2 code normalized RMSE was 0.8884 vs latent2 0.7264. Bytes were 2,905B vs 3,161B, but the <=10% error gate failed. Latent1 also beat Mirror at similar bytes. **Next: MA-417.**

**MA-417 FAIL:** Mirror codebook K=16 normalized RMSE 0.9032 vs ordinary latent K=16 0.7994 and continuous latent2 0.7099; only 9.6 distinct codes were used on average. The small byte saving missed the quality gate. **Next: MA-418/419.**

**MA-418 FAIL:** on held-out object/style pairs, factorized Mirror normalized RMSE was 0.0740 vs latent factors 0.0327 and FiLM 0.0149. Its payload was smaller than latent by 14.5%, but quality missed the registered gate. **Next: MA-419.**

**MA-419 FAIL:** Mirror3 sinusoid error was 0.3476 vs latent3 0.2653 at nearly equal bytes. The per-world gate vs modulation network failed on two of three fresh worlds. **Next: MA-424/425.**

**MA-424 FAIL:** aligned vector-field Mirror reached RK4-16 endpoint NRMSE 0.00193 but serialized to 1,829B vs independent 1,641B and slightly exceeded its stiffness ratio. **Next: MA-425.**

**MA-425 FAIL by strict byte gate:** aligned t=2 extrapolation error was Mirror 0.0171 vs simple linear embedding 0.3133 at equal 1,829B; discrete depth was smaller (1,641B) but weaker. **Next: MA-427.**

**MA-427 FAIL:** all fixed-point codes converged, but Mirror equilibrium NRMSE 0.0219, payload 2,081B, and 13.25 iterations were all worse than independent on quality/bytes/iterations. **Next: MA-429/431.**

**MA-429 FAIL:** aligned recurrent depth5–8 error was Mirror 0.0133 vs linear time 0.1279, but actual bytes were equal at 1,829B; the registered <=90% byte threshold failed. MA-431 deferred pending depth-family redesign after repeated 2x2 serialization overhead failures. **Next: MA-434.**

**MA-434 FAIL:** synthetic selective SSM Mirror quality improved (NRMSE 0.02235 vs independent 0.03894), but actual bytes were larger (2,837B vs 2,649B) and unfused training wall time was 9.026s vs 0.873s. **Next: MA-436/437.**

**MA-436 PROMISING (synthetic quality/storage Pareto only):** token-wise logical SSM roles achieved fresh NRMSE 0.00047 vs independent 0.00074 at 2,529B vs 3,045B (-17.0%). It uses 50% more recurrent MAC and 1.56× batch inference time; strict compute gate missed. Oracle roles and Givens-aligned transitions limit scope. **Next: MA-437, S4-native structured Views.**

**MA-437 FAIL:** after amendment A1 corrected the teacher to rotate only rank-one S4 factors, Mirror achieved length-64 NRMSE 0.000035 vs independent 0.000106, but payload was 2,591B vs 2,691B (96.3%, failing <=60%); rank-one residual was more accurate at 0.000009 and 2,973B. Stability passed. **Next: MA-438, frequency-band S4 Views.**

**MA-438 PROMISING, scoped:** 8-pole S4D timescale Mirror reached horizon-128 NRMSE 0.000041 vs independent 0.000075 using 2,061B vs 2,293B (-10.1%). Rank-two residual was 8B smaller but ~6.7x worse error. Stable and at the 1.25x MAC limit; strict <=60% byte gate missed. **Next: MA-439.**

**MA-439 FAIL:** packet-plan SSM Mirror reached NRMSE 0.000225 vs independent 0.000068 at 2,125B vs 2,485B (85.5%); it beat shared recurrence and ordinary latent controls but missed independent quality and <=60% byte gates. Stability passed; this was sequential rollout, not parallel decode evidence. **Next: MA-440.**

**MA-440 FAIL:** on four sequential synthetic dynamics skills, Mirror final NRMSE was 0.000092 vs independent 0.000058, forgetting 0.256 vs 0.052, and payload 2,125B vs 2,485B (85.5%). It only slightly improved forgetting over shared-only and missed the retention/byte gates. Mask/SETA and natural continual tasks remain untested. **Next: MA-441.**

**MA-441 FAIL:** recurrent Mirror address retained 100% hard recall through 512 distractors, but probability NRMSE drifted from 0.00419 to 0.01559 and payload was 2,920B vs 1,938B external exact route. Equal-width ordinary register had lower error; A1/A2 initialization/metric amendments are logged, with initial fresh runs excluded. **Next: MA-442.**

**MA-442 FAIL:** in fresh aligned 8D linear tasks, Mirror NRMSE was 0.5948/2,430B; shared no-adaptation was 0.5888/2,161B; full adaptation 0.7937/2,348B; LoRA 0.6923/2,476B. Mirror loses to shared quality/bytes and misses the registered byte gate. Full MAML control was weak, limiting generalization. **Next: MA-443.**

**MA-443 FAIL (A1 disjoint task split):** support-encoder Mirror NRMSE 0.2851 matched full adaptation 0.2758 and beat zero-init Mirror 0.3932, but N=20 amortized bytes were 2,764B vs full 2,435B; query wall was ~3.2x. LEO was more accurate (0.1491) at 2,657B. Encoder helps initialization, but registered storage/compute gate fails. Initial overlapping-task results are excluded. **Next: MA-444.**

MA-366 is reconciled as FAIL: direct pair coefficients match Mirror outputs and bytes, and PA02 factorization is smaller. The runner accidentally generated the registered fresh IDs before the gate; those rows are excluded and fresh integrity is invalid. MA-367 and MA-368 are also recorded FAIL. MA-369 is completed FAIL on its dedicated branch; next executable P0 is MA-371.

**MA-371 — MatFormer granularity Mirror views (P0; PA53)**

MA-369 FAIL: on fresh digits worlds the OFA-style supernet + four-angle View improved mean accuracy only 0.09pp over shared weights and was matched by equal-byte FiLM; shared-bank bytes were 40,191 vs 98,185 independent. The storage reduction came from weight sharing, not Mirror. Complete report and replay verification are in its dedicated branch. MA-371 is next.

**MA-371 — MatFormer granularity Mirror views (P0; PA53): FAIL.** Fresh mean accuracy was 95.28% vs 94.85% nested baseline (+0.43pp), below the +1pp gate; 40,063B was 65.4% of the independent bank, above the 50% limit. Equal-byte FiLM was close. The screen is a nested MLP, not a Transformer reproduction. **Next: MA-372**, a separate held-out Mix'n'Match question.

**MA-372 — MatFormer Mix'n'Match factorized Mirror codes (P0; PA53): FAIL.** Across 24 held-out mixed-width configurations and three fresh worlds, factorized Mirror gained 0.16pp over nested sharing, below the 1pp gate; equal-byte FiLM was more accurate and factor codes were 77.5% of direct-bank bytes versus the 60% limit. This is a digits MLP screen. **Next: MA-374.**

**Supernet/depth family ruling:** MA-369/371/372/374 all failed Mirror-specific gates against FiLM/partial sharing or native shared baselines. Per WORKER_QUEUE stop condition, defer MA-375 until the family is redesigned; retain it UNTESTED. Continue with a different family.

**MA-374 — ALBERT shared layers + depth Mirror (P0; PA61): FAIL.** On a digits residual-block proxy, tied+Mirror scored 96.39% vs tied 96.48%, and equal-byte FiLM 96.67%; partial FFN sharing scored 97.50%. Storage reduction vs untied came from shared parameters. Not ALBERT language-model evidence. **Next: MA-375.**

**Prior branches reconciled:** MA-375 FAIL (ranking codes did not improve child ranking); MA-379 FAIL (independent AdapterFusion quality not recovered; 46× decode MAC proxy); MA-381 PROMISING only for aligned synthetic rank-2 LoRA orbit, while strict byte/quality gates failed and fresh stayed sealed. These were validated from their dedicated branches and are now counted in the registry.

**Supernet/depth family ruling:** MA-369/371/372/374/375 all fail Mirror-specific quality or compute against native/FiLM/partial-sharing controls. Defer additional variants until the structure is redesigned.

**MA-383/385/389 reconciled from dedicated branches:** L2P prompt Mirror (383) failed quality and 80%-of-explicit byte limits with perfect retrieval; DualPrompt Mirror (385) failed quality, bytes and forgetting; Hash Embedding Mirror (389) saved bytes and beat scalar but missed native quality and collision-bin limits. Each branch has replayed serialized payloads and passing tests; fresh remained sealed.

**MA-391/392/393/395/397/399 reconciled from their branches:** multiplicative composition matched or beat Mirror at lower bytes; Mirror failed held-out token-domain quality despite storage savings; the adaptive Zipf pilot had no rare-token undercoverage because every token appeared during training. All six have serialized replay and passing tests; fresh remains sealed when development gates missed.

MA-255 is reconciled PROMISING only for its aligned post-fit screen; its separate fixed-update variant failed. Dedicated branch evidence for all new results is cited in `CLAIM_LEDGER.csv`.

MA-327 FAIL: ordinary rank-2 coefficient products match Mirror exactly and both exceed flat-pair bytes. MA-330 PROMISING only for aligned synthetic KV views: 72.3% below independent caches, but only 30B below direct cos/sin control; unrelated layers need private cache and runtime/LM gains remain unestablished. MA-331 FAIL (fresh sealed after development byte miss); MA-332/333 FAIL as pure function-preserving gauge orbits; MA-335 FAIL for Mirror-specific margin (direct irreducible coefficients within 0.65%). MA-337/338/341/342 also fail their registered Mirror-specific gates; MA-341 has a separate Digits screen whose overall Mirror advantage vs FiLM is NOT ESTABLISHED. MA-344/346 are PROMISING only for synthetic storage/communication frontiers against native methods; scalar phase controls are exact equivalents. MA-349 FAILs against rank-1 BNN and independent posterior bytes. MA-257 FAILs because native PA16 rotational contexts exactly alias its Mirror representation. MA-258 is narrowly PROMISING on an oracle-routed aligned synthetic expert bank: 60.4% fewer bytes than quality-passing rank-2 SVD, but unrelated experts collapse and no runtime/learned routing gain is established. MA-265 and MA-266 are consecutive VeRA-family FAILs against simple controls; MA-267 is paused until redesign. MA-318 failed its storage/compute gates despite correctly locating basis growth points. MA-325 A0 was NOT ESTABLISHED due unlearnable near-uniform task; A3 increased teacher logit scale to make it learnable, then failed the frozen Mirror/direct byte gate (2631B/2637B), with fresh sealed. Next P0 is MA-351.

## Active experiment

MA-434 is completed FAIL and committed on its dedicated branch. MA-436 is PROMISING only for synthetic quality/storage (Mirror 2,529B vs independent 3,045B, but 192 vs 128 MAC/token and 1.56× measured inference time). MA-437 failed its strict byte gate; MA-438 is narrowly PROMISING for a synthetic S4 timescale quality/storage Pareto point (10.1% fewer actual bytes than independent, but missed <=60% gate). MA-439 failed both independent quality and byte gates; MA-440 failed retention and byte gates; MA-441 also failed storage/drift gates despite 100% hard recall. MA-442 failed the Mirror-specific adaptation/storage gate; MA-443 also failed amortized storage/compute despite improving over zero-init Mirror. Next is MA-444.

## Verified status index

- **PROMISING (39):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-255, MA-268, MA-276, MA-282, MA-312, MA-258, MA-314, MA-330, MA-344, MA-346, MA-691.
- **FAIL (66):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253, MA-257, MA-260, MA-261, MA-265, MA-271, MA-272, MA-273, MA-274, MA-278, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299, MA-301, MA-303, MA-304, MA-307, MA-309, MA-311, MA-315, MA-319, MA-320, MA-322, MA-327, MA-331, MA-332, MA-333, MA-335, MA-337, MA-338, MA-341, MA-266, MA-318, MA-325, MA-342, MA-349, MA-351, MA-353, MA-355, MA-357, MA-359, MA-356, MA-360, MA-361, MA-364, MA-366, MA-369, MA-371, MA-372, MA-374, MA-375, MA-379, MA-383, MA-385, MA-389, MA-391, MA-392, MA-393, MA-395, MA-397, MA-399, MA-401, MA-403.

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
