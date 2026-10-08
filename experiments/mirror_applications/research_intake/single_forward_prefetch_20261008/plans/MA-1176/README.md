# MA-1176 — Grouped physical Mirror blocks + predictive prefetch

**Status: UNTESTED (full real-GPU system hypothesis).** Native mandatory: PA434 Pre-gated MoE; PA435 SpecPrefetch; PA436 SPICE. Nearby: MA-231 pre-folded hot states, MA-691 lazy KV, MA-697 switching, MA-253 cache-safe final FFN, MA-1175 reuse proof.

| 記号 | 意味（日本語） | 単位(SI) | 定義・範囲・型 |
|---|---|---|---|
| x | トークン/hidden入力 | 1 | 実行ベクトル、幅d |
| z | 共有preactivation | 1 | z=xU、長さhの実行ベクトル |
| U,D | FFN共有射影 | 1 | U∈R^{d×h}, D∈R^{h×o} 実行列 |
| m | Mirror機能座標 | 1 | 有限次元実ベクトル、回転角はrad |
| T_m,R_m | hidden / output変換 | 1 | h×h または o×o、可逆 / 直交行列 |
| φ | GELU非線形 | 1 | 成分ごとR→R |
| f_m | 第m論理出力 | 1 | R^d→R^o の関数 |
| M | 論理View数 | 1 | 正整数、1/2/4/5/8 |
| S | 直列化容量 | byte (実用単位) | 0以上整数、base/code/decoder込み |
| C, T_io | compute/転送時間 | s | 非負スカラー、デバイス/帯域依存 |
| L | 教師との予測誤差またはNLL | 1 (nat/token) | 有限非負スカラー |
| u_seed,u_eval,u_num,u_c,U_exp | 標準/拡張不確かさ | Lと同単位 | 非負実数、相関考慮 |
| k_cov | 包含係数 | 1 | 正実数、暫定2 |

## H — falsifiable

When four or five logical specialists can be served from one resident physical block with acceptable task quality, a two-buffer predictive offloader overlaps fetching the **next physical block** with current multi-View work to improve measured tokens/s and P95 TPOT at a fixed GPU resident-byte cap against a native pre-gated standard MoE, an equally packed ordinary factorized expert bank, and a hot-prefolded cache. Falsify if either ordinary prefetch/low-rank factorization equals the gain or Mirror decode latency consumes it.

## Mirror insertion

> **Mirror insertion:** one CPU-resident physical block W plus short View coordinates m_1...m_M supplies M logical specialists in a GPU compute window; predict the *next physical block* and asynchronously stage its bytes while currently selected Views run. Group factorization and IO overlap are separate mechanisms.

Do not assume a router's choice can always be predicted from W or previous m. A learned pre-gate must achieve actual hit@1/hit@K on heldout token sequences. If all logical Views depend on different nonlinear hidden states, run their remaining nonshared operations; only retain one common intermediate where exact factorization permits.

## T — full GPU/CPU worker-free protocol

1. Stage-0 virtual two-buffer simulator: N=12 group accesses, capacity two physical buffers, staged group size 16/64/256 MiB, nominal PCIe bandwidth 4/12/24 decimal GB/s and setup latency 0.25 ms. Compute window C(M)=1.0+0.18M ms is an **assumption**, not hardware measurement; hit probabilities 0.5/0.9/1.0. See frozen [protocol](../../FROZEN_PROTOCOL.json).
2. Stage-1 physically pre-fold each known View into plain FFN weights and measure CPU RAM/NPZ bytes vs shared runtime Mirror from independent seeds 201–205. Folded weights are tied derived copies, not arbitrary independent experts. See [amendment](../../FROZEN_FOLDED_CONTROL_AMENDMENT.json).
3. GPU Stage-2 only on actual accessible CUDA hardware: allocate pinned host memory, a separate transfer stream, two GPU staging slots, and synchronize via CUDA events. Populate realistic expert groups by natural router traces, not generated oracle next addresses. Precompute a small predictor from source-only routing traces; freeze predictor and all ranks before fresh prompt sequences.
4. Exact native controls: Pre-gated MoE with independent experts and the same predictor budget, native ordinary packed/linear LoRA expert group + same two buffers, SPICE low-rank surrogate fallback, output-only linear code and selective pre-folded hot-cache policy. Cache misses use exact native fallback and pay all bytes/time; no hidden CPU recovery.
5. Sweep 2/4/5/8 logical Views per block, resident cap 8/12/24 GiB if hardware supports, batch=1/8, context length 128/512/2048, hit@1/hit@4, block dwell length, and feature precision FP16/BF16/INT4 (zero-point and scales paid).
6. Record real GPU allocated/max-reserved VRAM, time to first token, per output token, P50/P95, real host-device DMA bytes and bandwidth, prefetch waste, misprediction, synchronization, cache copy/alias, and task NLL. The synthetic CPU time windows cannot be mapped directly to GPU latency.

## Mathematical transfer / unit check

With n physical groups, per-group transfer time T_io=launch latency + S_block/bandwidth, a correct next-group prefetch hides at most min(T_io,C_group) while incorrect prefetch exposes the transfer and may pollute buffers. All compute and transfer times use SI seconds; S_block uses bytes, bandwidth bytes/s, so S_block/bandwidth is seconds. Per-group compute scales with remaining per-View operations. For q=1 prediction and C_group>=T_io, transfers after the initial group may be fully overlapped, but total compute continues to grow with M.

## D — decisions

**PASS:** On >=4/5 fresh whole-sequence task worlds a confirmed native-equivalent quality threshold (+<=0.02 nat/token), >=15% actual TPOT improvement over the **fastest** pre-gated + ordinary factorized + prefolded competitor under identical VRAM cap, predictor model bytes and task NLL, and measured physical buffer alias/transfer. Must demonstrate native predictor accuracy, not q=1 oracle.

**FAIL:** Same-byte native factorized/pre-gated pipeline matches, inaccurate next-block prediction induces more IO than useful work, hot pre-folded expert bank wins within permitted RAM, or Mirror's remaining activation and down-projection computation makes total latency higher. Stage-0 virtual simulator achieved parity with native factorized, so no Mirror-specific claim is currently established.

**UNCERTAIN:** No CUDA hardware, true PCIe transfer timeline, pinned memory, natural language benchmark or trustworthy predictor. Stage-0 is UNCERTAIN for deployment, even if virtual overlap is mathematically valid.

## C — failure modes

More Views lengthen current compute but may also reduce throughput; prefetch is an existing native scheduling optimization. All View branches cannot be produced from the same final output if the model's hidden nonlinear dynamics differ. Prefix KV reuse requires exact compatible token/hidden provenance or explicit approximate target-quality evaluation; pointer equality and allocator accounting mandatory.

## U — uncertainty

Paired seed/task bootstrap; predictor accuracy and miss cost variation; clock frequency and GPU stream contention; memory allocator pooling, NPZ/weight format metadata; precision errors. Model u_c from seed/evaluation/dtype in the same loss unit with covariance; k_cov=2 indicative, not exact 95% at n=5. Keep milliseconds and bytes separate, and publish hit/miss trace.

## Evidence and next action

[CPU exactness and virtual scheduler report](../../REPORT.md). A truthful deployment experiment cannot be marked PASS without a GPU. A hot-fold cache may be worth further testing, but its memory residency must be charged.

## Dependency on the primary 1-forward output hypothesis

This is a *secondary deployment application*, not the primary Mirror research claim. Start only after MA-1175 has K4/K5 **independently useful learned outputs** with realistic native controls. Include PA439 SpecMD (native cache/eviction benchmark) alongside PA434 Pre-gated MoE, PA435 SpecPrefetch and PA436 SPICE. Compare true native proactive prefetch at equal router/predictor cost, model quality, PCIe bandwidth, hot/cold VRAM and TTFT/TPOT; virtual DMA simulator alone is not evidence of GPU throughput.

