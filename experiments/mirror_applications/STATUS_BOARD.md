# Mirror Application Status Board

Updated: 2026-10-08 JST
Canonical branch: `research/mirror-application-worker-ready-20261007`

## Program totals (1115-row baseline registry; MA-998 blocker recorded branch-locally)

- Registered candidates: **1115**
- P0: **597** (35 completed; 562 UNTESTED)
- P1: **415** (12 completed; 403 UNTESTED)
- P2: **103** (0 completed; 103 UNTESTED)
- Current MA statuses: **1068 UNTESTED, 29 PROMISING, 18 FAIL**
- 47 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.
- New research: MA-876..935 (60 UNTESTED), PA236..PA265 (30 primary sources). This is prior context; candidate selection is randomized.
- Twelfth literature sweep added MA-936..995 (60 UNTESTED; 47 P0/13 P1) and PA266..PA295. No new experiment results were claimed. MA-255 is a legacy queue pointer only; active selection is randomized.
- Thirteenth literature sweep added MA-996..1045 (50 UNTESTED; 40 P0/10 P1) and PA296..PA325. IDs MA-1000+ use four digits; consult `check_registry_integrity.py`. No new experiment results.
- Fourteenth sweep added **MA-1046..1095** (50 UNTESTED; 40 P0 / 10 P1) and **PA326..PA350**. Focus: time-series foundation forecasting, recommender embedding tables, and Earth-observation multi-sensor networks. No new experiment results; MA-255 is a legacy queue pointer only; active selection is randomized.
- Fifteenth direct-prior sweep added **MA-1096..1115** (20 UNTESTED, 16 P0 / 4 P1) and **PA351..371**. A gauge-invariant LoRA audit harness is present; its unit tests are **not** trained-model evidence. MA-255 is a legacy queue pointer only; active selection is randomized.
- PROMISING is **not** ADOPTED. Treat reports with strict-gate misses or exploratory protocol deviations at their documented scope.
- SRM/TM and prior Phase I results are not MA statuses.

## Next candidate

No deterministic next candidate is authoritative. Refresh the live registry and status board, inspect live research branches and exclusions, then draw uniformly from eligible P0/UNTESTED IDs.

## Active experiment

MA-998 was selected randomly and is blocked on this branch; its scientific registry status remains UNTESTED. Other live worker branches have independent outcomes and remain separate.

## Verified status index

- **PROMISING (29):** MA-001, MA-002, MA-003, MA-004, MA-005, MA-006, MA-007, MA-008, MA-012, MA-014, MA-015, MA-041, MA-076, MA-079, MA-111, MA-121, MA-156, MA-160, MA-171, MA-173, MA-181, MA-189, MA-241, MA-244, MA-245, MA-249, MA-250, MA-251, MA-691.
- **FAIL (18):** MA-009, MA-010, MA-011, MA-013, MA-019, MA-024, MA-048, MA-061, MA-063, MA-086, MA-116, MA-129, MA-186, MA-199, MA-208, MA-247, MA-248, MA-253.

All per-ID evidence is retained in the local experiment directories and in `CLAIM_LEDGER.csv`. Consult [the evidence integration audit](../../docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md) for linked reports, verifications, provenance and claim boundaries.

## 2026-10-08 research sweep

New directions are cross-model cache translators, neural graphics and 4D Gaussian fields, speaker-adaptive synthesis, generative flow/solver coordinates, and cross-model stitching. They are **UNTESTED**. See [current research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md). The new cross-model cache lane distinguishes exact MA-691 algebra from approximate learned transfer.

## Twelfth literature intake — 2026-10-08

Additional Mirror m insertion targets: neural video chunk sharing (NerVast/DCVC-UF), partial learned equivariance, spiking thresholds/time gains, physical photonic phase configurations, beamforming/CSI/RIS and personalized HRTF neural fields. Candidates remain UNTESTED except individually screened IDs; they require native efficient controls, full bytes and domain runtime/energy. See [twelfth sweep notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md) and PA266..295.

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

## Random worker update — MA-998 blocked

MA-998 was selected at draw 14 and assigned to `research/ma-998-mace-element-env-mirror-20261008`. Native MACE/NequIP work is blocked because PyTorch/MACE/e3nn/ASE and GPU/CUDA are absent. Registry status remains UNTESTED; see the experiment branch. MA-990 and MA-643 results are on independent branches and are not folded into these baseline counts.
