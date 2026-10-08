# MA-1123 — Cross-architecture functional descriptor Mirror code

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P1; **Evidence lane:** MECHANISM first; **Prior art:** PA390;PA375;PA380;PA241.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-292 task-vector Mirror basis; MA-885 cross-model cache, PA390 Platonic Task Arithmetic, PA375 Task Vector Bases.

## H — Falsifiable hypothesis

For unseen task-model cross-products, a learned factorized m bank retains >=95% of native UTD transfer improvement on task metric, reduces serialized inference edit-bank bytes >=15% at K>=8 tasks and >=2 architectures, and does not increase inference P95 >10%.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **one shared functional task descriptor basis and architecture-specific small fixed readout heads** at **Encode each source-task descriptor into a shared small m_task; separately store small m_model conditioning the target decoder** so that **heldout task x model combinations without storing full per-pair task edits** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Run the source paper's Universal Task Descriptor (UTD) baseline or a faithful independently reproduced functional affinity-shift descriptor, trained on identical unlabeled probe embeddings. Compare native folded linear edit and full LoRA edit, plus a flat low-rank descriptor compression control.
- **What is stored physically:** one shared functional task descriptor basis and architecture-specific small fixed readout heads.
- **What changes causally when m changes:** factorized task m plus architecture m in a common probe-defined functional descriptor space.
- **Nearest non-Mirror control:** native Universal Task Descriptors / per-model least-squares edits / LoRA task edits / generic factorized descriptor.
- **Native prior vs proposed delta:** Compress UTD-style functional descriptors with factorized Mirror coordinates and test held-out task-model cross-products; UTD cross-architecture transfer itself is prior art.

## T — Executable minimal test

**Implementation recipe.** Stage 1 create two fixed frozen random encoders with different output widths (32 and 48) on one shared latent teacher, and eight source label/affinity tasks. Fit task descriptors only on shared permitted unlabeled probes, factorize via shared U/V basis plus task/model codes, and fold into output map. Stage 2 uses two verified licensed pretrained image-text encoders with source-only task splits. Compare UTD descriptors, flat factorized regression and m composition.

**Dataset/harness.** Eight source task IDs, four heldout task identities and a separate set of heldout task x model combinations. Shared unlabeled calibration probes disjoint from supervised task audit images and class prompts. At least 2 encoder architectures; checkpoint identities and tokenizer prompts frozen.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** native Universal Task Descriptors / per-model least-squares edits / LoRA task edits / generic factorized descriptor; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU NumPy/PyTorch; Stage 2 optionally one GPU for two licensed encoder checkpoints; if checkpoints/code unavailable label blocked and keep only synthetic mechanism evidence.

**Measured quantities:** Top-1 accuracy, target-relative gain retained, pairwise edit interference, true serialized task/architecture basis/decoder bytes, optimization/probe FLOPs, P95 inference and per-adapter calibration.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** Five fresh seeds hold >=95% of native UTD quality gain (when denominator robust; report absolute scores when near zero), >=15% total bank bytes saved at fixed task count, and P95 <=1.10 full UTD folded-head baseline; positive requires no target-pair oracle labels.

**FAIL:** Cross-arch quality below control, native UTD already factorizes at equal/lower bytes, or unlabeled probe/target labels leak across splits.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Architecture-specific residual can be large; one common m may not retain tasks; compiled linear UTD edits have virtually no extra runtime so Mirror can only win on stored edit bank size.

## U — Uncertainty and error budget

Base-encoder heterogeneity, task similarity, descriptor normalization, output width, language prompts, source-calibration size; source-only calibration bootstrap and five task seeds.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Replicate UTD descriptor calculation for two frozen architectures on source-only probes.
2. Verify exact probe/task split, prompts and target checkpoint hashes.
3. Implement native UTD and ordinary descriptor SVD/compressed baseline.
4. Fit task+architecture m without heldout pairing oracle.
5. Sweep number of tasks and serialize all banks/readout state.
6. Gate on actual task utility and throughput, not descriptor Frobenius similarity alone.

**Hard stop:** Cross-model descriptors and their arithmetic are established UTD prior; benefit claimed is additional code-bank compression after native UTD.

**Required experiment outputs when later promoted:** `README.md`, frozen `PROTOCOL.json`, `STATUS.md`, implementation under `source/`, tests, `RESULTS_CORE.csv`, `VERIFICATION.json` containing dataset/hash/seed/byte/compute provenance. None of these runs have been executed by this design document.

## Variable table / dimension check

| Symbol | Meaning (Japanese) | Unit (SI / practical) | Domain / type |
|---|---|---|---|
| `m` | Mirror機能座標 | 1 | real vector of length k |
| `k` | 機能座標の数 | 1 | positive integer scalar |
| `W`, `B_i` | 対象重み・共有基底 | 1 | same-shaped real matrices, finite entries |
| `Q,K,V` | 注意機構の活性値 | 1 | real token x channel matrices, compatible attention shape |
| `S` | 追加推論状態の容量 | byte (non-SI byte unit) | nonnegative integer; actual serialized payload |
| `L` | 正規化交差エントロピー | nat/token (dimensionless) | finite nonnegative scalar |
| `t` | 推論実時間 | second (s) | nonnegative scalar |

Dimension check: `m_i * B_i` has the same dimensionless learned-weight units as `W`; `W + sum_i m_i B_i` is conformable by matrix shape, and logit error/CE is not a byte or second. A byte ratio `S_m/S_control` is dimensionless; compare wall-clock and bytes on distinct axes.

## Primary literature and verified venue

- **PA390:** Platonic Task Arithmetic — https://arxiv.org/abs/2610.00929 (arXiv:2610.00929 (2026 preprint)); Proposes architecture-agnostic task descriptors and cross-model arithmetic/transfer via functional probes. Multi-model descriptor sharing is native; the Mirror delta is extra code-bank compression or cheaper factorized task x model state.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
