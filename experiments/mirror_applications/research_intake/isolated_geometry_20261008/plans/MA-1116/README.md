# MA-1116 — Horizontal-gauge Mirror task codes

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P0; **Evidence lane:** MECHANISM first; **Prior art:** PA382;PA383;PA373;PA381.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-338 symmetry-normalized codes; MA-1115 LoRA gauge audit; PA382 full MHA gauge characterization; PA383 gauge quotient.

## H — Falsifiable hypothesis

On held-out tasks with the same frozen attention backbone, a gauge-horizontal Mirror code reaches no worse than +0.01 nats/token relative to the best byte-matched ordinary task code, while reducing serialized code-bank bytes by at least 10% and not increasing P95 forward latency by over 10%.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **shared frozen attention Q/K/V/O maps and a source-task residual basis** at **Compute attention-layer gauge tangent directions on the SOURCE models only (Q/K inverse coordinate transforms and V/O inverse output transforms)** so that **multiple genuinely different task attention functions from shared maps** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Freeze a 2-head 32-wide attention block. Fit a regular shared task residual basis with unrestricted small task codes; separately instantiate unconstrained low-rank and rank-one codes. Preserve identical task split, optimizer and shared basis.
- **What is stored physically:** shared frozen attention Q/K/V/O maps and a source-task residual basis.
- **What changes causally when m changes:** small task m in the numerically estimated horizontal (non-gauge) complement.
- **Nearest non-Mirror control:** unprojected Mirror code / direct low-rank code / native gauge-invariant optimization.
- **Native prior vs proposed delta:** Test whether allocating m only to observable function directions improves capacity-per-byte beyond unconstrained codes; do not treat gauge orbit as a new expert.

## T — Executable minimal test

**Implementation recipe.** For each source model, sample invertible G for Q'=QG and K'=KG^{-T}, plus H for V'=VH and output map O'=H^{-1}O. Confirm attention/model outputs unchanged before estimating a gauge tangent. Construct finite-difference tangent columns and orthonormalize in float64. Compare m-basis before/after removing their tangent components; train 4 code sizes {2,4,8,16} and retain best dev. Do not assume a Frobenius-horizontal direction automatically gives lower end-to-end loss.

**Dataset/harness.** Synthetic teacher attention: d_model=32, heads=2, sequence=16; eight train/source tasks and four task-identity heldouts, each with disjoint support/dev/audit examples. A teacher generates coherent classification/next-token labels; no target oracle weight deltas allowed for fitting m. Repeat with random and structured task families.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** unprojected Mirror code / direct low-rank code / native gauge-invariant optimization; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU PyTorch float64 for gauge diagnostics, float32 for training; no GPU required. Stage 2 single CUDA GPU if available, otherwise record Stage-2 BLOCKED instead of inventing language-model results.

**Measured quantities:** Teacher/logit MSE, held-out cross entropy, output Jacobian singular-rank diagnostic on fixed probe inputs, per-task accuracy, serial bytes including QR/basis metadata, training FLOPs, CPU/GPU P50/P95 inference.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** All five fresh task-world seeds satisfy non-inferiority margin 0.01 nats/token vs best byte-near non-Mirror code, median additional serialized bytes <=0.90 control, median P95 <=1.10 control, and functional Jacobian rank is nonzero for each claimed logical role.

**FAIL:** Any exact gauge transform is counted as a distinct logical function; or matched control reaches quality with <= candidate bytes and latency; or fresh quality margin is exceeded in >=2 of 5 worlds.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Gauge complement can rotate with task/data and a training-friendly horizontal projection can discard useful nonlinear directions; simpler low-rank basis may win.

## U — Uncertainty and error budget

Finite-probe tangent nullspace, near-degenerate rank, learned basis randomness, optimizer choice, float32 vs float64. Treat near-zero singular values as uncertain; bootstrap by task identity and report u_c from seeds plus numerical precision.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Freeze model and source/audit task identity manifests before training.
2. Build exact gauge-transform unit checks in float64; reject incorrect RoPE assumptions.
3. Implement unprojected, low-rank, and projected m in one harness.
4. Select code rank by source/dev only; lock fresh configuration.
5. Run five fresh worlds; serialize actual inference payload and benchmark isolated inference.
6. Report M0 if ordinary code matches, even if the projection is mathematically valid.

**Hard stop:** A local/probe-space gauge complement is not a proof of global quotient geometry or independent expert capacity.

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

- **PA382:** Complete Characterization of Gauge Symmetries in Transformer Architectures — https://proceedings.mlr.press/v282/wang26a.html (PMLR 282 (2026)); Determines canonical MHA Q/K and V/output head-wise gauge groups, including RoPE commutant restrictions and head permutations. Exact gauge actions preserve function. The proposed Mirror insertion must not claim these invariances as new logical functions.
- **PA383:** Gauge Fiber Bundle Geometry of Transformers — https://proceedings.mlr.press/v282/wang26b.html (PMLR 282 (2026)); Studies quotient geometry, Ehresmann connection, gauge gradient split, curvature and holonomy. Prior art for horizontal and path-dependent diagnostics; Mirror-specific claim must be incremental m utility.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
