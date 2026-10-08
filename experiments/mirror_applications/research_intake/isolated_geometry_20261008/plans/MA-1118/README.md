# MA-1118 — Invariant fingerprint selected Mirror bank

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P1; **Evidence lane:** MECHANISM first; **Prior art:** PA384;PA382;PA386.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-1096 source-task gauge audit; MA-1108 canonical task descriptors; PA384 curvature-bispectrum equivalence tests.

## H — Falsifiable hypothesis

Source-only invariant fingerprints select code families that lower total stored task-bank bytes >=10% versus best output-probe or SVD clustering at held-out loss difference <=0.01 nats/token, with zero false equivalences under prescribed counterexamples.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **shared pretrained model plus a bank of candidate adaptation bases** at **Select basis family using a gauge-invariant signature estimated on source/calibration data; train per-task m on support data and store basis index** so that **many naturally trained tasks mapped into fewer task-code/basis families** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Reproduce a task-delta SVD + nearest-codebook and a function-logit probe clustering baseline; if canonical bispectral computation from PA384 is unavailable, label any simplified spectral signature as a PROXY, not native replication.
- **What is stored physically:** shared pretrained model plus a bank of candidate adaptation bases.
- **What changes causally when m changes:** fingerprint-selected basis index plus small m and optional paid private residual.
- **Nearest non-Mirror control:** output-probe clustering / weight-SVD clustering / bispectral-only invariant test / independent adapters.
- **Native prior vs proposed delta:** Use hybrid function/gauge invariants to choose the cheapest Mirror code bank that preserves task quality; equivalence signatures alone are native prior art.

## T — Executable minimal test

**Implementation recipe.** Generate source task delta functions and multiple exact Q/K, V/O gauge copies. Make feature sets: raw Frobenius/SVD, held-out calibration logits, curvature/Fisher sketch, and full bispectral signature only if independently implemented/validated. Cluster on source; fit m with frozen source basis; include a negative pair with identical low-order moments but different outputs.

**Dataset/harness.** Eight source task families x 3 gauge twins, six truly new held-out tasks, and two adversarial near-collision families using tiny attention/MLP blocks; target adapters are learned independently but share the identical frozen base revision.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** output-probe clustering / weight-SVD clustering / bispectral-only invariant test / independent adapters; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU float64 invariance checks and torch float32 training; stage 2 GPU natural independently learned adapter bank only if checkpoint provenance identical.

**Measured quantities:** False same-function/equivalence decisions, held-out NLL/accuracy and logits, bank basis count, actual compressed bytes including signature/selector indices, inference CPU latency and source-training time.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** Across five fresh seeds, no confirmed functionally different pair merged as exact equivalent, held-out NLL within +0.01 of best byte-near control, >=10% bank byte saving and P95 inference <=1.10 control; otherwise at most PROMISING for screening.

**FAIL:** Any false-equivalence decision using ground-truth synthetic exact functions; audit-task delta used to form source basis; simple output clustering equal/better at matched bytes.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Gauge invariant fingerprint may be too coarse, expensive or non-identifying; output-space clustering can dominate.

## U — Uncertainty and error budget

Calibration coverage and fingerprint collisions, condition number, signature truncation, oracle-equivalence testing limitations; bootstrap by task identity and report collision confidence interval.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Define native bispectral reproduction criterion versus optional cheap invariant proxy.
2. Generate exact-gauge twins and non-equivalent counterexamples.
3. Train source family fingerprints; never inspect heldout task deltas.
4. Freeze clustering/selector and train target m only on target support.
5. Count offline fingerprint and online descriptor bytes separately.
6. Report false-equivalence table first; no compression claim if invariance validation fails.

**Hard stop:** No claim that signatures completely characterize global transformer equivalence; diagnostic finite-probe score only.

**Required experiment outputs when later promoted:** `README.md`, frozen `PROTOCOL.json`, `STATUS.md`, implementation under `source/`, tests, `RESULTS_CORE.csv`, `VERIFICATION.json` containing dataset/hash/seed/byte/compute provenance. None of these runs have been executed by this design document.

### Quantified uncertainty and sequential stopping

For quality observable `Q`, combine same-unit standard uncertainties `u_seed` (independent seed/world mean standard error), `u_eval` (task-cluster bootstrap sample error), and `u_num` (numerical replay effect expressed in Q units). The independence model is `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)`, expanded `U = k_cov * u_c` with indicative `k_cov = 2`. For only five fresh worlds, this **does not guarantee 95% coverage**; report paired per-seed outcomes and a task-bootstrap 95% interval. If terms correlate, use full covariance. Runtime has separate `u_t = sqrt(u_repeat^2 + u_clock^2)` (s); storage S is measured exactly as serialized bytes. Never combine uncertainties of different dimensions.

| Symbol | Meaning (Japanese) | Unit (SI/practical) | Definition, domain, type |
|---|---|---|---|
| `Q` | 主評価量 | nat/token or 1 | finite real scalar, preselected |
| `u_seed` | 独立試行による標準不確かさ | Qと同単位 | nonnegative real scalar, paired std/sqrt(n), n=5 |
| `u_eval` | 評価標本の標準不確かさ | Qと同単位 | nonnegative real scalar, task bootstrap |
| `u_num` | 数値誤差成分 | Qと同単位 | nonnegative real scalar, dtype/replay |
| `u_c` | 合成標準不確かさ | Qと同単位 | nonnegative scalar, covariance-aware |
| `k_cov` | 包含係数 | 1 | dimensionless positive scalar; indicative 2 |
| `U` | 拡張不確かさ | Qと同単位 | nonnegative scalar, `k_cov * u_c` |
| `u_t,u_repeat,u_clock` | 実行時間の標準不確かさ | s | real nonnegative scalars, runtime only |

**Dimension check:** every squared term in `u_c` has unit Q², so square-root returns unit Q; `k_cov` has unit 1. Runtime uncertainty remains seconds, not bytes. **Sequential stop:** tune only on 3 dev seeds, run all 5 frozen fresh worlds without selective early stopping, and never use fresh outputs to retune. Separate independent replication is mandatory for ADOPTED.

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

- **PA384:** Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants — https://proceedings.mlr.press/v282/wang26c.html (PMLR 282 (2026)); Relates gauge-aware geometry and bispectral invariant signatures for equivalence diagnosis. Native invariant comparison is a strong control, not a new Mirror operator.
- **PA382:** Complete Characterization of Gauge Symmetries in Transformer Architectures — https://proceedings.mlr.press/v282/wang26a.html (PMLR 282 (2026)); Determines canonical MHA Q/K and V/output head-wise gauge groups, including RoPE commutant restrictions and head permutations. Exact gauge actions preserve function. The proposed Mirror insertion must not claim these invariances as new logical functions.
- **PA386:** Permutation Equivariant Neural Functionals — https://arxiv.org/abs/2302.14040 (NeurIPS 2023; arXiv:2302.14040); Neural functionals processing weights/gradients equivariantly under neuron permutations. This symmetry-aware encoding is established; compare functional m output to native weight-space models.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
