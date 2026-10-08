# MA-1119 — Symmetry-equivariant functional Mirror code generator

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P0; **Evidence lane:** MECHANISM first; **Prior art:** PA386;PA387;PA363;PA368.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-288 context fast-weight Mirror; MA-446 learned optimizer; MA-1108 task descriptors; PA386 NFN / PA387 UNF.

## H — Falsifiable hypothesis

With an amortized generator counted as inference state, a symmetry-equivariant m generator achieves <=+0.01 held-out CE vs a byte-near generic m-generator, improves performance after unseen hidden-unit permutations, and saves >=10% total multi-task inference bytes versus full-adapter generation.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **shared frozen task backbone and a single amortized meta-code generator** at **Replace the generator final head with a k-dimensional m head and a shared fixed task-update basis; feed permutable gradients/weights through an equivariant encoder** so that **unseen-task functional adapters without one stored dense adapter per task** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Train a generic support-set encoder -> low-rank adapter and a native permutation-equivariant functional network/UNF-style generator -> full adapter. Count the amortized generator size; do not weaken native controls.
- **What is stored physically:** shared frozen task backbone and a single amortized meta-code generator.
- **What changes causally when m changes:** task-specific m generated from support data and symmetry-equivariant weight/gradient features.
- **Nearest non-Mirror control:** ordinary MLP hypernetwork / native UNF or NFN / direct learned LoRA task code.
- **Native prior vs proposed delta:** Use equivariant neural functionals to emit only low-description m, not full weights, and test byte/transfer gains beyond native UNF and generic hypernetworks.

## T — Executable minimal test

**Implementation recipe.** For each support set, compute pooled activation and gradient features for a frozen d=32 MLP (2 hidden layers) with explicit neuron-permutation twins. Build small NFN with shared intra-/inter-layer aggregation, plus generic MLP and simple DeepSets baselines; use k={4,8,16}, common base/basis fixed across task families. Independently test exact permutation transport, label-independent feature pipeline and output consistency.

**Dataset/harness.** Ten source task transformations and six unseen task transformations over class-balanced synthetic classification; each unseen task has 32 support examples, disjoint 128 dev and 256 audit examples; 4 unique permutations of each model for invariance checks.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** ordinary MLP hypernetwork / native UNF or NFN / direct learned LoRA task code; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU PyTorch float32; optional stage 2 CUDA 1 GPU with transformer if source-only NFN architecture and licenses can be verified.

**Measured quantities:** Few-shot accuracy, CE, m invariance and output equivalence across permutations, actual generator plus basis plus codes bytes, adaptation wall-clock/updates and OOD task identity transfer.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** All five fresh seeds preserve prediction invariance within 1e-4 float32 relative after valid neuron permutation; median held-out loss within +0.01 of best byte-matched m baseline, >=10% full-system byte saving versus UNF-generated full adapters, and no added test-only oracle information.

**FAIL:** Generic DeepSets/MLP equal or better under matched bytes, broken permutation invariance, or generator cost erases task-code savings.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

NFN/UNF parameter sharing is already an effective inductive bias; specialized m may be an information bottleneck and meta-network overhead dominates until many tasks.

## U — Uncertainty and error budget

Meta-training task diversity, support set draw, permutation group scope, optimizer and model scale; report five seed variation and calibration/test disjointness.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Freeze source tasks, held-out task identities and permutation twins.
2. Implement exact base MLP permutation tests; derive output equivariance targets.
3. Build native generic hypernetwork, UNF/NFN, and shared-basis m head.
4. Sweep k and number of tasks on source/dev only.
5. Lock generator, evaluate unseen tasks and permutations under five fresh seeds.
6. Report break-even task count including all generator and basis bytes.

**Hard stop:** Permutation equivariance of an encoder is not itself novel and is not functional multiplicity.

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

- **PA386:** Permutation Equivariant Neural Functionals — https://arxiv.org/abs/2302.14040 (NeurIPS 2023; arXiv:2302.14040); Neural functionals processing weights/gradients equivariantly under neuron permutations. This symmetry-aware encoding is established; compare functional m output to native weight-space models.
- **PA387:** Universal Neural Functionals — https://proceedings.neurips.cc/paper_files/paper/2024/hash/bd20595c8e5802ba40ed418f4ec116f0-Abstract-Conference.html (NeurIPS 2024; DOI 10.52202/079017-3326); Builds permutation-equivariant neural functionals for general weight spaces, including learned optimizers. m code generation must beat ordinary UNF + cheap native adapter at counted generator bytes.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
