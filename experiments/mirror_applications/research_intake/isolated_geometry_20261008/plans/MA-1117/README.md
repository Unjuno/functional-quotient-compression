# MA-1117 — Holonomy-robust Mirror code transport

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P1; **Evidence lane:** MECHANISM first; **Prior art:** PA383;PA382;PA367.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-332 permutation-orbit audit; MA-338 static canonicalization; MA-1115 factor gauge; PA383 holonomy diagnostics.

## H — Falsifiable hypothesis

On a frozen sequence of chart changes, transported Mirror codes reduce same-function code drift by >=25% median versus direct code carryover and native Procrustes, without worsening target function loss by >0.01 nats/token or raising serialized inference state >10%.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **one pretrained attention model and shared Mirror basis transported between equivalent parameter charts** at **At each optimizer/chart transition, calculate a source-only transport matrix for the small m code or its tangent basis** so that **stable reusable task m across equivalent optimization/gauge histories** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Implement standard code re-fitting or Procrustes alignment between weight-basis charts using a fixed calibration split and equal adaptation steps.
- **What is stored physically:** one pretrained attention model and shared Mirror basis transported between equivalent parameter charts.
- **What changes causally when m changes:** chart-specific m corrected by a cheap transport address.
- **Nearest non-Mirror control:** no transport / orthogonal Procrustes / direct native re-fit / gauge-aware optimizer.
- **Native prior vs proposed delta:** Test whether cheap m transport suppresses spurious gauge drift during repeated reuse, beyond static gauge audits; holonomy itself is not new functional multiplicity.

## T — Executable minimal test

**Implementation recipe.** First use a closed exact change-of-basis loop G1,G2,G2^{-1},G1^{-1} and verify final parameter product is identity; this is a gauge-safety unit test, NOT a holonomy demonstration. Then use a small rectangular horizontal parameter-update loop and estimate connection-based transport plus a separate gauge-normalization step. Log path dependence and whether the induced code correction preserves predictions; report connection definition and discretization error.

**Dataset/harness.** Two-head d=16 attention, 6 source chart trajectories and 4 unseen loop trajectories of 4 or 8 small update segments, lengths fixed before audit. Use one frozen teacher objective with separate probe inputs and held-out examples.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** no transport / orthogonal Procrustes / direct native re-fit / gauge-aware optimizer; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU float64 NumPy/PyTorch; no pretrained large language model required. Stage 2 CUDA only for a separately approved small-transformer transfer test.

**Measured quantities:** Relative code drift after cycle, maximum function/logit discrepancy, NLL retention, fit steps to restore code, code/transport bytes, matrix conditioning and update-loop area.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** At least 4/5 fresh seeds show >=25% lower code drift than best native alignment, no excess NLL >0.01, and all persistent transport bytes <=1.10 best control; any claimed m reuse also meets equal-input prediction check.

**FAIL:** Correct exact closed gauge loop alone appears to create new function; transport loses accuracy or uses more bytes without sample-efficiency gain; Procrustes alone matches.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Nontrivial holonomy lives in coordinate transport, not newly distinguishable functions; computation may overwhelm storage savings.

## U — Uncertainty and error budget

Discrete connection approximation, floating point, ill-conditioned charts, path length and loop area; estimate errors across 5 seeds and two step-size grids.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Verify exact gauge group action and loop closure in float64.
2. Freeze target/probe and path families without seeing audit loops.
3. Implement no-transport, Procrustes, and horizontal transport.
4. Sweep two step sizes on dev only and lock.
5. Report both prediction identity and distinct-code drift on five fresh paths.
6. Fail if the only benefit is coordinate cosmetics without reduced retraining/storage.

**Hard stop:** Does not claim holonomy itself represents a new task or that gauge loops guarantee useful memory savings.

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

- **PA383:** Gauge Fiber Bundle Geometry of Transformers — https://proceedings.mlr.press/v282/wang26b.html (PMLR 282 (2026)); Studies quotient geometry, Ehresmann connection, gauge gradient split, curvature and holonomy. Prior art for horizontal and path-dependent diagnostics; Mirror-specific claim must be incremental m utility.
- **PA382:** Complete Characterization of Gauge Symmetries in Transformer Architectures — https://proceedings.mlr.press/v282/wang26a.html (PMLR 282 (2026)); Determines canonical MHA Q/K and V/output head-wise gauge groups, including RoPE commutant restrictions and head permutations. Exact gauge actions preserve function. The proposed Mirror insertion must not claim these invariances as new logical functions.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
