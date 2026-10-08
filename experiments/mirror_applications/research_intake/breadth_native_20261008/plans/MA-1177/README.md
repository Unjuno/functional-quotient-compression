# MA-1177 — UniSparse composite-token Mirror role-readout code

> **ISOLATED PLAN / NOT ACTIVE.** Status UNTESTED. Existing canonical worker MA-255 remains next. This plan is a new cross-over candidate rather than proof of a Mirror improvement. Native source must be checked for a publication-grade reproduction; the CPU mechanism is self-contained here.

## Scientific identity and direct paper

- Family: **Sparse attention / composite token banks**; priority: P0; sources: PA440, PA441.
- Closest existing MA (non-duplication check): MA-691, MA-696, MA-699, MA-1172.
- Original native method: UniSparse native composite-token multi-granularity summarization and block selection, not replaced by a different attention approximation.
- Original source: https://proceedings.mlr.press/v306/liu26h.html.
- Success means **incremental m-specific utility beyond the native method**, not that Sparse attention / composite token banks was invented here.

## Symbols, units, and shapes

| symbol | 日本語の意味 | SI 単位 | 形状・型・定義域 |
|---|---|---|---|
| x | モデル入力 | 1 | 正規化された実ベクトル/系列、有限 |
| theta | 既存方式の共有パラメータ | 1 | 実数重み行列・テンソル |
| m | 追加Mirror機能コード | 1 | 有限実kベクトル、角度はrad(1) |
| k | Mirrorコードの次元 | 1 | 正整数 |
| y | 予測出力 | 1 | 実ベクトル/確率分布、実装で明示 |
| K | 論理タスク/role数 | 1 | 正整数 |
| S | 全推論保存容量 | byte (非SI実用単位) | 0以上の整数、実Serializerで測定 |
| t | 一回の推論時間 | s | 単調時計、非負実数 |
| L | 平均タスク交差エントロピー | nat/token (SIでは1) | 非負実数、タスクごとに正規化 |
| u_seed,u_task,u_num | 誤差源別の標準不確かさ | Lと同単位 | 非負実数 |
| u_c | 合成標準不確かさ | Lと同単位 | 交差項を含める |
| k_cov | 包含係数 | 1 | 正実数、参考値2 |
| U_exp | 拡張不確かさ | Lと同単位 | k_cov * u_c |

Additional experiment-specific symbols:

| symbol | 日本語の意味 | 単位 | 形状・型・定義域 |
|---|---|---|---|
| `T` | token sequence length | 1 | positive integer 512 pilot |
| `g` | summary granularity | token (1) | integer {8,16,32,64} |
| `S` | physical resident+serialized state | byte | integer >=0 incl index and role codes |
| `E` | attention-output relative error | 1 | finite nonnegative scalar |

**Dimensional check:** Mirror coefficients are dimensionless. Additive updates or matrix composition are allowed only when tensors have shape-compatible dimensionless weights. `S` is physically counted in bytes and `t` in seconds; they are independent Pareto dimensions and must not be added to task loss.

## H — falsifiable hypothesis

Given fixed UniSparse composite-token construction and exactly the native sparse attention selector, m yields 4 useful query-role outputs from one physical index with total readout/index/code state <=0.95x native separate-per-role state at heldout attention output error <=1.05x native and P95 <=1.10 native, and cannot be explained solely by a byte-matched ordinary linear projection code.

## Mirror insertion — minimal, precise native interface

> **Mirror insertion:** introduce small per-role m at **short query-role m selecting task-conditioned transforms of composite-token readout coefficients and block score correction, without independently materializing per-role indices**, keeping **one native multi-granularity composite-token index and KV bank produced for the document once** as the native shared physical object. Do not recreate/retrain the native method under a new name.

- Native implementation to preserve: UniSparse native composite-token multi-granularity summarization and block selection, not replaced by a different attention approximation.
- Shared physical state paid: one native multi-granularity composite-token index and KV bank produced for the document once.
- New functional coordinate: short query-role m selecting task-conditioned transforms of composite-token readout coefficients and block score correction, without independently materializing per-role indices.
- Logical outputs: UniSparse composite-token Mirror role-readout code evaluated for K independently useful tasks; gauge-equivalent or identical logits count as zero new functions.

## T — worker-independent method and runbook

**Self-contained synthetic mechanism:** Create 512-token x32-dimensional stationary and shift/nonstationary source sequences, four composite granularities g={8,16,32,64}, 4 query-role labels. Train 6 source query-family/sequence worlds; hold out entirely new query-role covariance and sequence families with seeds 101-105. Use 8 selected blocks per query after a fixed native composite-token score ranking. Causal masks/prefix index provenance must match native.

**Exact coding procedure:** (1) Build native per-granularity composite tokens as blockwise pooled K/V in fixed source-only strategy. (2) Query-side selector ranks native block summaries and retrieves top-8 blocks exactly, with no use of target labels. (3) Fit a shared role correction dictionary on source-role query-output pairs only; m dimension k={2,4,8}, try diagonal, signed/butterfly and short Givens. (4) Evaluate no-view/native UniSparse, native UniSparse per-role trained readouts, native single shared readout, same-byte linear factor bank and Mirror on the same frozen query outputs. (5) Reconstruct full attention only as an accuracy ceiling and separately benchmark a CUDA block-sparse kernel if available. Never call a dense Python selector faster solely from fewer reported FLOPs.

**Strong controls** (same seed/data/examples/training budget and same physical serializer):

1. Exact native UniSparse compressed-token block selector and its per-role ordinary readout
2. UniSparse single shared readout (no role)
3. same-byte native rank-k linear/diagonal task code on the same composite tokens
4. full FlashAttention or PyTorch dense attention quality ceiling
5. equal-byte INT4 cache plus selected-block native path
6. MInference/XAttention/FlexPrefill when official kernels/checkpoints available

**Splits and firewall:** dev seeds **11,12,13** may choose rank k, LR and early stopping; fresh full-task seeds **101,102,103,104,105** are locked and may not be inspected/tuned on. Hold out complete task families plus inputs; do not merely random-split correlated tokens/cells/sequences. Once fresh starts, execute every preregistered world. If the actual public dataset requires different splits, preregister a separate named natural stage before access. Native paper reproduction and arbitrary synthetic teachers are different evidence lanes.

**Measurements:** per-role attention output relative RMSE and heldout NLL where model available; selected-block overlap/recall; bytes for all token summaries, selector, weights, code, metadata; CPU/GPU P50/P95, active MACs and memory footprint. Count all shared dictionaries, decoder parameters, per-role code, routing, caches/indices, resident copies, serial file headers and optional private residual. Track offline source-model training work separately. Report each model's optimizer steps/examples, actual FLOPs/MAC estimate, CPU/GPU identity, dtype, batch shape, P50/P95, warmup and synchronization, logit functionality not only weight-distance.

**Hardware:** CPU NumPy/torch for 512-token mechanism, one GPU with actual block-sparse kernels for any GPU runtime claim; record kernel choice and source commit.. No GPU clock/real NLL claim may be inferred from CPU fixture.

## D — frozen PASS / FAIL / UNCERTAIN

**PASS (preliminary screen only):** At least 4/5 fresh full-role worlds satisfy <=1.05x native error (or NLL<=+0.02 nat/token), combined resident summary+selector+readout bytes <=0.95 native, P95<=1.10 native; and strict utility/bytes benefit over ordinary code (otherwise M0).

**FAIL:** Ordinary linear readout bank Pareto-dominates, native composite index must be duplicated per role, selector uses heldout attention as oracle, or any role falls below allowed NLL/error in >=2 fresh worlds.

**UNCERTAIN:** Native source, task/preprocessing, real bytes or latency is missing; confidence interval contains both margins; datasets have contamination or native comparators fail. Mark BLOCKED where code/checkpoints unavailable, do not invent metrics.

**Adoption:** >=10 independent new task/seed worlds, natural realistic benchmark and quality/bytes/runtime improvement over direct native and cheap matched controls. An aligned synthetic PASS only certifies the mechanism.

## C — counter-hypothesis / falsification

Native UniSparse already shares all composite tokens; code only saves a small readout payload, while different query roles may require different token selection and thus attention work remains. This is not a claim of 4x sparse-attention compute savings.

## U — statistical and numerical uncertainty

Query distribution shift and selection near-ties, correlation of heldout tokens, granularity/batch kernel effects, decoder cost, stochastic retrieval; bootstrap by entire sequence-role world, not token. Confidence from five worlds is only preliminary.
With loss measured in nat/token, `u_c² = u_seed² + u_task² + u_num²` only under independent components; otherwise include twice the relevant covariances. Expanded `U_exp = k_cov * u_c` with `k_cov=2` is indicative at n=5, not a certified 95% interval. Use paired entire-task/bootstrap intervals and report outliers without cherry-picking. Benchmark timing uncertainty in seconds separately.

## Preservation / evidence products

Keep this directory design-only: `README.md`, `PROTOCOL.json`, `STATUS.md`. Do not emit an MA-scoped `RESULTS_CORE.csv` or `VERIFICATION.json` before executing frozen code. If activated, create a new non-worker experiment branch with immutable dataset/checkpoint/license hashes, recorded native versions, source unit tests, actual serializer output, all fresh rows and an independent replay. Report negative and M0 results explicitly.
