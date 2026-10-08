# MA-1161 — GVA head-reconstruction Mirror code bank

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P0; **Evidence lane:** MECHANISM first; **Prior art:** PA417;PA115;PA156.  
**Frozen research baseline:** `c935a903daca5c7d1d48aa50d05b5bd50f239cba` (the cited canonical branch as of staging).  
**Closest registry:** MA-581 MLA latent head reconstruction, MA-691 lazy KV reads, MA-694 Mirror MLA; PA417 is direct GVA prior.

## H — Falsifiable hypothesis

At fixed training tokens and matched native GVA architecture, a structured head-m bank reduces all inference reconstruction state by >=15% while keeping NLL within +0.02 nats/token and P95 decode latency <=1.10 native GVA.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **one GVA grouped-V and decoupled positional-key cache with shared reconstruction basis** at **Parameterize content key reconstruction M_h(m)=M_shared+sum_i m_hi B_i with small head m; use q'=q M_h(m)^T at inference instead of materializing K** so that **logical query-head key functions over the same grouped cached V stream** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Reproduce a 2-group 4-head tiny GVA block with separate position key and full per-head matrix M_h, plus GQA/MLA and low-rank GVA map controls. If GVA author kernels/checkpoint unavailable, report proxy reproduction and do not claim paper speedups.
- **What is stored physically:** one GVA grouped-V and decoupled positional-key cache with shared reconstruction basis.
- **What changes causally when m changes:** head/role m selecting a compact content-key reconstruction map.
- **Nearest non-Mirror control:** native GVA full M_h / GVA low-rank M_h / GQA / MLA.
- **Native prior vs proposed delta:** Compress multiple GVA head-specific M_h maps with small Mirror codes and quantify incremental gain beyond native GVA; query-side absorption itself belongs to GVA.

## T — Executable minimal test

**Implementation recipe.** Use shape-conformant tensors V_cache[b,t,g,dv], q[b,t,h,dk], M_h[dv,dk]. Compute scores both explicitly with reconstructed content keys and via absorbed query; their max difference in float64 must be <1e-9 on a deterministic toy fixture. Sweep k=2,4,8 against full M and LoRA rank-matched maps. Verify GVA content/position channels never get incorrectly fused.

**Dataset/harness.** Synthetic attention retrieval (copy and multi-query matching) with length 64/128, d_model=64, H=4, grouped V G=2, position channel=16, fixed optimizer steps=2000 per model and identical training examples; a tiny word-level LM lane is secondary and must be labeled.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** native GVA full M_h / GVA low-rank M_h / GQA / MLA; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU float64 exact algebra and PyTorch float32 8M-style tiny LM; Stage 2 optional CUDA GPU with batch 1 and 8, context length 128 and 512, warmup and 100 repeated isolated steps, report GPU type/clock.

**Measured quantities:** NLL/retrieval accuracy, serialized basis+head m bytes, physical cache scalars/bytes and allocator stats, attention score error, FLOPs/step, isolated CPU P50/P95 and optional GPU decode.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** All 5 fresh seeds <=+0.02 nats/token vs matched native full-GVA, >=15% reconstruction-state bytes saved (not double count baseline cache), and P95 decode <=1.10 GVA. End-to-end claims require a real decode kernel, not eager PyTorch alone.

**FAIL:** Byte-near native GVA low-rank control matches or beats quality/runtime, nonmatching attention scores, or basis+code overhead erases reconstruction savings.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Native full M can be too cheap relative to basis metadata; GVA baseline already eliminates persistent content-key cache, leaving little marginal Mirror opportunity.

## U — Uncertainty and error budget

Kernel and hardware dependence, eager-vs-fused query transform, RoPE position provenance, model training variance. Report CPU and GPU separately, seeds=5.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Implement exact synthetic GVA reconstructed-key versus query-absorbed equality tests.
2. Freeze architecture, token counts and train/dev/audit data split.
3. Train native GVA then byte-near low-rank and structured m under same budget.
4. Measure both physical persistent cache and map storage independently.
5. Record eager CPU plus any actual fused decode GPU path, with hardware clocks if available.
6. Publish no speedup claim without real end-to-end measured decode.

**Hard stop:** GVA's query absorption and value caching are established prior art; Mirror claim is only the marginal head-map parameterization.

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

- **PA417:** Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction — https://arxiv.org/abs/2609.13285 (arXiv:2609.13285 (2026 preprint)); Reconstructs head content keys from grouped values with an absorbable query-side map and separately cached position key. Mirror can only claim extra gain from compact per-head/readout m or safe role sharing; paper itself does not establish end-to-end kernel throughput gains.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
