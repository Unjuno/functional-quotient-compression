# MA-1120 — Lie-bracket controlled noncommuting Mirror composition

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P0; **Evidence lane:** MECHANISM first; **Prior art:** PA388;PA389;PA375;PA16.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-257 group-factorized context tasks, MA-266 VeRA composition, PA388 and PA389 for order effects.

## H — Falsifiable hypothesis

For unseen ordered pairs (A,B) and (B,A), a commutator-aware m composition lowers mean held-out pair loss by >=2% relative to native additive composition at matched bytes and by >=1% vs a byte-near ordinary ordered adapter, with at most 10% extra P95 latency.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **one shared task-update matrix basis or function block** at **Before a shared block, decode two small task codes into structured operators and compose in order; optional single rank-corrected Lie bracket term is stored in a shared basis** so that **ordered composite behaviors from factor codes without a per-ordered-pair trained adapter** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Train native additive task-vector composition and ordinary sequential A->B versus B->A updates. Include full small-matrix product, independent ordered-pair adapter as upper bound (not for hyperparameter search), and MA-257 factorized group-code comparison.
- **What is stored physically:** one shared task-update matrix basis or function block.
- **What changes causally when m changes:** two cheap task codes m_A and m_B with an explicit order bit or Lie-bracket correction.
- **Nearest non-Mirror control:** additive Task Arithmetic / native sequential gradient updates / full matrix products / MA-257.
- **Native prior vs proposed delta:** Predict or correct order-dependent functional interaction using small Mirror commutator information, not mere Cartesian code combination already in MA-257.

## T — Executable minimal test

**Implementation recipe.** Start with bounded noncommuting 2D/4D rotations/shears as a synthetic teacher, then learned tasks. Compute AB, BA and additive controls with identical active matmul count when possible; on fixed task-gradient fields estimate bracket via paired Hessian-vector products only from training support. Lock a heldout ordered-pair split; forbid fitting pair-specific m.

**Dataset/harness.** Four source primitive skills and six held-out ordered 2-skill combinations, each with disjoint support/dev/audit sequences; include commuting counterexamples where expected advantage is zero; d=32 shared two-layer MLP/attention.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** additive Task Arithmetic / native sequential gradient updates / full matrix products / MA-257; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU float64 algebra tests and float32 PyTorch training; stage 2 1 GPU for language-style tasks only after mechanism screen.

**Measured quantities:** Heldout pair CE, direction/order confusion rate, primitive-task retention, bytes incl ordering metadata, active multiplies and P95 CPU/GPU latency.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** Across five fresh seeds, median held-out ordered-pair CE improves by >=2% vs additive and >=1% vs best cheap ordered non-Mirror control, persistent bytes <= control, P95 <=1.10 control, and no held-out pair task training leaked.

**FAIL:** Improvements vanish against native ordered product, any pair-specific trained model is secretly present, or commuting tasks show artificial order benefit indicating leakage.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Order effects may be too weak or native matrix multiplication already optimal; extra bracket correction may overfit and cost more than a simple ordered adapter.

## U — Uncertainty and error budget

HVP finite precision, optimizer step size, higher-order truncation, order-label correlation, task-seed variance; evaluate four step scales on dev and bootstrap by ordered pair.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Verify AB != BA analytically/numerically on noncommuting synthetic teacher and AB=BA on commuting control.
2. Construct source-only primitive task codes and freeze pair-identity splits.
3. Implement additive, native ordered, factorized group and bracket-corrected m.
4. Select correction rank/step scale using source/dev only.
5. Measure quality, interference, active compute, bytes on fresh five seeds.
6. Mark no-specific-value if native ordered method wins or ties.

**Hard stop:** An order bit does not create independent expert capacity; report unique target functions and the true number of independently trained primitive tasks.

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

- **PA388:** The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order — https://proceedings.mlr.press/v306/sweeney26a.html (ICML 2026; PMLR 306); Relates order-dependent sequential learning to computable gradient Lie brackets. Bracket-based sequencing is prior art; proposed Mirror insertion tests factorized code composition rather than claiming the Lie bracket.
- **PA389:** First-Order Predictable but Pairwise Fragile: Local Task Adaptation in Trained Transformers — https://arxiv.org/abs/2607.16821 (arXiv:2607.16821 (2026 preprint)); Reports order-sensitive pairs and fragile local task arithmetic in trained Transformers. Requires heldout pair, step-size and gauge audits; an optimistic global composition radius is not assumed.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
