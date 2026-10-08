# SFM001 / SFM002 — single-forward multi-Mirror + predictive block offload (2026-10-08)

**FROZEN mechanism-screen report.** This is CPU synthetic algebra, one toy trained linear readout, CPU eager timing and a deterministic **hypothetical transfer scheduler**. No pretrained natural-language checkpoint, real GPU, measured PCIe/VRAM transfer, full MoE training or established Mirror-specific Pareto win. Canonical worker branch and queue were not touched. Full MA-1175/1176 remain **UNTESTED**; the mechanism study does not change their scientific status.

## 1. Research question and exact factorization theorem

The question is **not** whether an arbitrary full model has several numerical Mirror coordinates. It is whether a *single evaluated statistic* suffices to obtain every View's output without executing its expensive part again. Ordinary Parameter Superposition (PA16), MIMO (PA42), GVA (PA417) and predictive expert prefetch (PA434–436) are mandatory prior/controls.

### Variable table (symbols precede formulas)

| symbol | meaning (日本語) | SI unit | definition, domain, type |
|---|---|---|---|
| x | 入力 | 1 | finite real row vector in R^d |
| U | 共通の入力射影 | 1 | real d×h matrix |
| D | 共通の出力射影 | 1 | real h×o matrix |
| phi | 非線形GELU | 1 | component-wise R^h→R^h map |
| T_m | 隠れ状態用Mirror行列 | 1 | invertible real h×h, pairwise orthogonal in tests |
| R_m | 最終出力用Mirror行列 | 1 | real orthogonal o×o; fixed address m |
| m | Mirrorコード | 1 | pairwise-rotation angles (radian, SI dimensionless), real vector |
| r(x) | 再利用する十分表現 | 1 | arbitrary finite-vector statistic of input |
| f_m, g_m | 論理関数と変換decoder | 1 | functions with compatible finite output dimensions |
| M | 使用するView数 | 1 | positive integer in {1,2,4,5,8} |
| Q,K,V | Query/Key/Value | 1 | real compatible matrices, Q: queries×head dim, K: tokens×head dim, V: tokens×value dim |
| A_m,B_m | KV変換行列 | 1 | real invertible head/value channel matrices |
| C_shared,C_view,C_full | 共有処理/各View/独立Expert計算時間 | s | finite nonnegative scalar, per physical block or View as declared |
| T_io | 一つの物理ブロックの転送時間 | s | setup latency + bytes/bandwidth |
| S | 推論用の保存容量 | byte (practical, non-SI) | nonnegative integer, actual NPZ file length |
| L | 評価損失 | dimensionless (nat/sample) | nonnegative scalar for given fixture |
| u_seed,u_eval,u_num | 品質不確かさ成分 | L unit | nonnegative standard uncertainties, covariance if correlated |
| u_c,k_cov, U_exp | 合成標準不確かさ/包含係数/拡張不確かさ | L,1,L | u_c>=0, k_cov=2 indicative, U_exp=k_cov u_c |

**Theorem (necessary and sufficient exact reuse).** There exists a deterministic decoder `g_m` satisfying `f_m(x)=g_m(r(x))` over a declared input domain **if and only if** every pair with identical cached statistic `r(x)=r(x')` has identical View output `f_m(x)=f_m(x')`. Proof: Necessity follows from applying a function to equal arguments. For sufficiency define `g_m(s)=f_m(x)` for any `x` with `r(x)=s`; the condition makes it well defined. No architectural model assumption is needed. It does **not** bound the computation/bytes required by g_m.

For the tested nonlinear hidden View in **row-vector convention**:

```
z=x @ U
f_m(x) = GELU(z @ T_m) @ T_m^(-1) @ D
f_0(x) = GELU(z) @ D
```

`z` can be computed **once**. The activation and final GEMM remain View-specific. In contrast, if the model *was trained/defined* as `f_m(x)=f_0(x)@R_m`, all roles can be formed from one full forward followed by cheap output transforms. This is a strictly constrained family and can also be expressed by conventional output heads/gates.

**Impossibility witness:** for two input vectors obtained by swapping coordinates, a symmetric base `f_0(x)=sum_i GELU(x_i)` gives exactly equal base output, but a hidden Mirror rotation around GELU separates the two by **0.141630**. Any decoder of `f_0(x)` alone necessarily commits absolute error >= **0.070815** on at least one of them, by the triangle inequality. No special router can fix this without extra information.

**Attention reuse boundary:** `K_m=K A_m`, `V_m=V B_m`, `q_m=q A_m^(-T)` preserves `q_m K_m^T=q K^T`; therefore `softmax` weights and value read can be shared and only `output @ B_m` changes. Value-only role changes also share attention scores. With `K_m` changed but `q` fixed, attention logits typically change, so new QK scores and softmax are needed even when K/V **storage** is shared. The maximum changed-key score difference was > 1.7954 on every fresh world.

**Backprop distinction:** Parallel multi-View execution is optional, not mathematically required. For the sum of per-View losses, the derivative through shared `z=xU` is the sum of per-View derivatives; accumulating role gradients sequentially yields the same shared-parameter gradient (under identical arithmetic). The measured float64 maximum full-versus-cached shared gradient difference is **6.94e-18**.

**Unit/shape check:** `x(1×d) @ U(d×h) @ T_m(h×h)` is `1×h`; `GELU` preserves shape, then `T_m^(-1)(h×h) @ D(h×o)` returns `1×o`. Every entry is dimensionless under normalized activations. `q A_m^(-T) (1×dh) @ (K A_m)^T(dh×T)` yields `1×T` scores. Bytes and seconds are independent objective axes; dividing two measured wall times yields a dimensionless speedup.

## 2. Freeze and audited numerical outcomes

- Primary frozen protocol committed **before experiments** at `8579b4368bc8a39f7ea9f15861963c2319fd973c`.
- Stage-1 pre-folded alternative was **separately pre-frozen before its new 201–205 seeds** at `758a59c28b510bd9a9081b59c1e2383e20b38665`.
- Development 11–13, fresh 101–105; pre-folded dev 21–23, separate fresh 201–205. No hyperparameter selection on fresh task outputs. Shape/broadcasting defect was fixed in deterministic unit tests before final CPU benchmark; the initial run timed out after development and was resumed for fresh without changing method choices. Therefore this is a **same-session synthetic screen**, not a blinded independent lab replication.
- Hardware: AMD EPYC 9V74 (CPU), PyTorch 2.10.0+cpu, NumPy 2.3.5, float64 mathematics/readout fitting and float32 timing, **one thread**; clock not locked. Batch 16 and 128; runtime block dimensions d_in=64/h=256/d_out=64; 20 warmups and **100 measured calls** for each method/seed/shape. CUDA not available.

### Stage-0 algebra and learned readout

| Check | Fresh five-world outcome | Decision |
|---|---:|---|
| Shared preactivation -> complete nonlinear Mirror outputs | max abs 6.66e-16 | PASS exact math |
| Output-only Mirror -> independently repeated output-only full forwards | max abs 0.00e+00 | PASS exact math |
| Value-only and coupled QK Attention reuse | max abs 1.78e-15 | PASS exact math |
| Cached/shared vs repeated backward gradients | max abs 6.94e-18 | PASS exact math |
| Output-only transport for arbitrary hidden Mirror | explicit base-output collision, View gap 0.14163 | **FAIL general claim** |
| Trained affine/ridge base-output -> hidden-Mirror output | fresh mean-NMSE median **0.1431**, range **0.1257–0.1579** | Non-exact, only approximation |

Ridge used 1,024 support examples and regularization 1e-3, no target model delta oracle. It is ordinary **non-Mirror** output decoding. The sample mean of five world-mean NMSE values is 0.141777; independent-world SE is 0.006610, indicative `2SE`=0.013219. This omits evaluation-sample and dtype contributions, hence is **not a complete u_c or a reliable 95% coverage interval**.

### Stage-0 CPU P95 (median over fresh five worlds, milliseconds)

| Active views | Fastest full recompute control | Shared-z packed Mirror | Exact output-only orbit | Shared-z / fastest-full ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.494 | 0.560 | 0.321 | 1.112 |
| 2 | 0.927 | 0.843 | 0.343 | 0.869 |
| 4 | 1.524 | 1.573 | 0.391 | 1.040 |
| 5 | 1.890 | 1.682 | 0.382 | 0.862 |
| 8 | 3.537 | 2.573 | 0.574 | 0.717 |

At **K=4, batch128**, the preregistered <=0.90 ratio gate had **0/5** successes, median ratio 1.040: **FAIL**. At **K=5**, it had **5/5**, median ratio 0.862: conditional **PASS for this single setting**. Since the combined gate requested **both** K4 and K5 in >=4/5, its overall verdict is **FAIL**. A theoretical MAC reduction does not guarantee wall-clock speed because of GEMM packing and per-View nonlinear/rotation overhead.

The output-only route is much faster but **does not implement the same hidden-nonlinear target**. It should be trained and validated as a separate constrained architecture. Standard native linear heads can produce equally cheap output transforms; output-only runtime is not a Mirror-specific advantage.

### Stage-1 native pre-folded competitor (separately fresh)

| Batch | Views | Fastest pre-folded ordinary expert P95 ms | Shared-z Mirror P95 ms | Mirror/folded median ratio | Pre-folded/Shared NPZ bytes |
|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 0.141 | 0.376 | 2.770 | 524,814 / 133,866 |
| 16 | 5 | 0.131 | 0.403 | 2.622 | 655,886 / 134,378 |
| 128 | 4 | 0.637 | 1.435 | 2.261 | 524,814 / 133,866 |
| 128 | 5 | 0.753 | 1.737 | 2.350 | 655,886 / 134,378 |

**Fact:** pre-folding the View into ordinary per-View `U_m,D_m` is algebraically exact (float32 maximum error 2.03e-06), and was faster in **20/20 paired seed/size/View settings**. It uses about 4.88× the actual NPZ state for K5. For batch128 K5, its P95 was around **0.753 ms** versus **1.737 ms** for cached-preactivation Mirror. **This is a RAM/latency Pareto tradeoff, not full-model compression.** Prefolded expert matrices are tied derived copies, not independently trained arbitrary experts.

## 3. Hypothetical CPU→GPU transfer scheduling — NOT hardware measurement

Two-buffer schedule model (fixed durations in milliseconds for reporting): group compute `C(K)=1.0+0.18 K` ms, independent full-expert compute `K*1.2` ms; per-group synthetic transfer `T_io=0.25ms + block_bytes / virtual_bandwidth`, bandwidth 4/12/24 decimal GB/s, block 16/64/256 MiB, group count 12, predictor hit probability 0.5/0.9/1.0. These values are **illustrative assumptions, not measured CPU/GPU throughput**. The exact same predictor success/failure draws are applied to every method.

For a correct prefetch, exposed transfer before the next group is `max(0,T_io-C(K))`; for a missed prediction, the entire `T_io` stalls. An ordinary native **factorized** group of the same bytes and compute gets the **same result** (explicit negative control).

Case: 16 MiB group, 12 GB/s, K=4, 12 groups: `T_io` = 1.648 ms, `C(K)`=1.720 ms. With **perfect** next-block prediction, the entire transfer after the initial group is hidden, virtual total 22.288 ms. With expected predictor hit 0.9, the five-world median virtual total is **25.584 ms** versus **40.417 ms** without prefetch (the exact realized 5-seed interval is 22.288–25.584 ms). The native same-byte factorized schedule equals Mirror in every case.

A high K increases the available overlap window but also increases compute, memory traffic, and likely kernel overhead. It does **not** make router/transfer cost monotonically vanish in end-to-end latency. Knowing the current Mirror address does not reveal the next address unless a predictor/pregate has sufficient causal information; q=1 is an oracle-bound scenario. Published native Pre-gated MoE, SpecPrefetch and SPICE already address predictive transfer.

## 4. H/T/D/C/U verdict

- **H exact-reuse:** PASS for output-only family, value-only and coupled-QK compatible attention, and cached preactivation; FAIL for arbitrary hidden nonlinear Mirror decoded from final base output alone.
- **T:** Mechanism exact checks all 5/5 fresh. 100-call P95 CPU timing across 2 batches × 5 K × 5 worlds. Separately frozen folded control 2 batches × 2 K × 5 worlds. Virtual transfer sweep 675 rows (not physical GPU).
- **D:** Primary shared-z *combined K4+K5 speed* gate FAIL despite K5 success; pre-folded native ordinary experts beat cached preactivation in timing (5/5 for K4/5 with both batches). Offload overlapping mechanism works for some virtual bandwidth/compute budgets but has **zero Mirror-specific gain against same-cost ordinary factorization**. No FULL MA scientific status change.
- **C:** Cheap output transforms can be expressively limited, and nonlinearity prevents universal output transport. Folding trades GPU RAM for runtime. Native prefetch achieves the overlap gain; if m cannot represent independent functions, extra logical experts are only different labels.
- **U:** Timing depends on CPU frequency (unlocked), eager kernels and workload; no GPU transfer. Five worlds are insufficient for a production speed guarantee. The exact algebra tests demonstrate invariants under controlled tensor shapes but not full Transformer equivariance. Quality u_c cannot be fully estimated without class/task-distributed real audit examples. No reported speed is portable without hardware matching.

## 5. Next architecture test (independent protocol needed)

1. Train **multi-view, same-physical-block** models from data rather than relying only on random/aligned teacher views. Compare output-only `T_m(y0)`, shared-preactivation `z`, and full internal Mirror layers at identical task quality and total bytes. Include ordinary linear/gate heads and MIMO/BatchEnsemble controls.
2. Run native predictor (not oracle) to predict physical group before current group's execution completes; measure hit@1/4, bytes prefetched and discarded, new-group router dependencies.
3. Use a two-tier GPU cache: *cold* physical shared block + cheap views, selectively hot pre-folded expert variants in RAM if repeated use justifies expansion. Account for both resident and transferred bytes, and evictions.
4. Evaluate real overlapped CUDA streams and host pinned-memory / NVMe DMA on hardware with 8 GB/12 GB/24 GB GPU VRAM; log real prefill TTFT, TPOT, synchronization, workload, hit rate and clocks. No claims until this lane exists.
5. Audit KV cache provenance independently from FFN output sharing; exact attention-read reuse does not imply all downstream KV states are identical.

## 6. Provenance and files

- [Original preregistration](FROZEN_PROTOCOL.json) and [follow-up pre-folded preregistration](FROZEN_FOLDED_CONTROL_AMENDMENT.json) (paths relative to study root in repository; see top directory).
- Original full source/test scripts are in the linked conversation ZIP; an independent subset algebra preflight is in [pilots/sfm001/source/algebra_reference.py](pilots/sfm001/source/algebra_reference.py). No external checkpoint.
- All original dev/fresh algebra, function-fit, CPU timing, virtual prefetch and folded control CSVs are in the reproducibility ZIP. The Git research directory preserves the [summary RESULTS_CORE.csv](RESULTS_CORE.csv) and [verification summary](VERIFICATION_SUMMARY.json).
- SHA-256 hashes and backend details for every original CSV/source are inside ZIP's `results/VERIFICATION.json`, `results/FOLDED_VERIFICATION.json` and `results/RESULTS_CORE_VERIFICATION.json`; Git has `VERIFICATION_SUMMARY.json`.
- Original required tests: 10/10 SFM001, 2/2 SFM002. Post-hoc independent formula/checker tests replay selected untouched numerical rows; CPU wall-clock microbenchmarks are not byte-identical across executions and are not described as such.

**Prior art**: [Parameter Superposition](https://arxiv.org/abs/1902.05522), [MIMO](https://arxiv.org/abs/2010.06610), [GVA](https://arxiv.org/abs/2609.13285), [Pre-gated MoE (ISCA 2024)](https://doi.org/10.1109/ISCA59077.2024.00078), [SpecPrefetch](https://arxiv.org/abs/2607.24787), [SPICE](https://arxiv.org/abs/2608.21240).