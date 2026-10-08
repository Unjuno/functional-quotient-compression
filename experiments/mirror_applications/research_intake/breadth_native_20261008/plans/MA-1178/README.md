# MA-1178 — LoGo instance-level Mirror-compressed adapter bank

> **ISOLATED PLAN / NOT ACTIVE.** Status UNTESTED. Existing canonical worker MA-255 remains next. This plan is a new cross-over candidate rather than proof of a Mirror improvement. Native source must be checked for a publication-grade reproduction; the CPU mechanism is self-contained here.

## Scientific identity and direct paper

- Family: **Multi-adapter serving / instance-specific mergers**; priority: P0; sources: PA442, PA427, PA351.
- Closest existing MA (non-duplication check): MA-670, MA-714, MA-1169, MA-1114.
- Original native method: Training-free LoRA on the Go (LoGo) selection/merging of an existing adapter library from instance-specific model signals, without new target labels.
- Original source: https://aclanthology.org/2026.acl-long.1837/.
- Success means **incremental m-specific utility beyond the native method**, not that Multi-adapter serving / instance-specific mergers was invented here.

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
| `K` | adapter library size | 1 | integer>=4 |
| `m_j` | address of adapter j | 1 | real k-vector |
| `Delta_j` | native adapter update | 1 | real d_in x d_out matrix |
| `B` | resident serialized adapter pool | byte | integer>=0 including shared dictionary |

**Dimensional check:** Mirror coefficients are dimensionless. Additive updates or matrix composition are allowed only when tensors have shape-compatible dimensionless weights. `S` is physically counted in bytes and `t` in seconds; they are independent Pareto dimensions and must not be added to task loss.

## H — falsifiable hypothesis

For mixed-domain heldout instances using an unchanged source adapter library and the ORIGINAL LoGo algorithm, a Mirror-coded adapter bank saves >=15% complete adapter-bank serialized bytes, preserves native LoGo NLL within +0.02 nat/token and P95 <=1.10x, and improves the quality/bytes Pareto frontier beyond plain same-byte SVD/compressed adapter banks.

## Mirror insertion — minimal, precise native interface

> **Mirror insertion:** introduce small per-role m at **small role/adapter Mirror code m reconstructing each adapter update before native LoGo computes its input-dependent relevance weights**, keeping **one source-learned shared LoRA delta dictionary plus the original frozen base LLM and native LoGo instance router/merger** as the native shared physical object. Do not recreate/retrain the native method under a new name.

- Native implementation to preserve: Training-free LoRA on the Go (LoGo) selection/merging of an existing adapter library from instance-specific model signals, without new target labels.
- Shared physical state paid: one source-learned shared LoRA delta dictionary plus the original frozen base LLM and native LoGo instance router/merger.
- New functional coordinate: small role/adapter Mirror code m reconstructing each adapter update before native LoGo computes its input-dependent relevance weights.
- Logical outputs: LoGo instance-level Mirror-compressed adapter bank evaluated for K independently useful tasks; gauge-equivalent or identical logits count as zero new functions.

## T — worker-independent method and runbook

**Self-contained synthetic mechanism:** Frozen 64x64 2-layer small teacher with six independently trained rank-4 adapter updates for 6 domain shifts; 3 aligned tasks, 3 independent off-orbit; 256 source training examples per domain, 128 valid, 512 fresh. Generate heldout mixed-domain instances and task-combination sequences; do NOT fit codes to the full target audit LoRA deltas or target test labels. Native LoGo router observes exactly same permissible token/activation signals for every method.

**Exact coding procedure:** (1) Freeze same pretrained base and train six native LoRAs independently. (2) Reproduce LoGo's training-free relevance extraction and merge on frozen native LoRAs; validate with reference outputs. (3) Learn a source-only shared delta basis and per-adapter m; optional sparse private exception when evidence of off-orbit mismatch. (4) Run two bank controls: common SVD delta dictionary+linear coefficients and independent byte-near quantized LoRAs. (5) Pass reconstructed LoRAs to the *same native LoGo routing/merge*. (6) Probe new combinations, 4 vs 8 vs 16 adapter libraries; count offline fitting separately from online signals. (7) Natural lane: author LoGo evaluation dataset/model with license, tokenizer/hash fixed before fresh.

**Strong controls** (same seed/data/examples/training budget and same physical serializer):

1. Official/native LoGo on uncompressed, independently fitted LoRAs
2. LoGo on ordinary source-bank SVD coefficients at equal bytes
3. LoGo on per-adapter quantized LoRA at equal bytes
4. LoGo on ALoRA shared-B when valid
5. LoRAHub/native instance composer where equivalent
6. Single chosen LoRA and dense merged delta quality floor

**Splits and firewall:** dev seeds **11,12,13** may choose rank k, LR and early stopping; fresh full-task seeds **101,102,103,104,105** are locked and may not be inspected/tuned on. Hold out complete task families plus inputs; do not merely random-split correlated tokens/cells/sequences. Once fresh starts, execute every preregistered world. If the actual public dataset requires different splits, preregister a separate named natural stage before access. Native paper reproduction and arbitrary synthetic teachers are different evidence lanes.

**Measurements:** per-domain and joint instance-level NLL, worst-domain retention, negative transfer, instance router agreement/entropy, serial bank bytes, per-instance merge overhead, B1/B8 P95 and library scalability. Count all shared dictionaries, decoder parameters, per-role code, routing, caches/indices, resident copies, serial file headers and optional private residual. Track offline source-model training work separately. Report each model's optimizer steps/examples, actual FLOPs/MAC estimate, CPU/GPU identity, dtype, batch shape, P50/P95, warmup and synchronization, logit functionality not only weight-distance.

**Hardware:** CPU torch 2-layer test; standard LLM/LoGo released code is GPU lane; record optimizer steps for source adapter fitting, native router overhead, and actual copy-on-write state.. No GPU clock/real NLL claim may be inferred from CPU fixture.

## D — frozen PASS / FAIL / UNCERTAIN

**PASS (preliminary screen only):** At least four of five fresh instance-domain worlds NLL <= native LoGo+0.02 nat/token; complete bank bytes <=0.85 native; P95 <=1.10; and some measurable Pareto gain over ordinary SVD/quant LoRA bank. Native LoGo itself is never attributed to Mirror.

**FAIL:** Native LoGo or same-byte source delta factorization dominates, native per-instance merging requires hidden K full passes, task labels/target oracle leak into code fitting, or poorly aligned tasks suffer systematic negative transfer.

**UNCERTAIN:** Native source, task/preprocessing, real bytes or latency is missing; confidence interval contains both margins; datasets have contamination or native comparators fail. Mark BLOCKED where code/checkpoints unavailable, do not invent metrics.

**Adoption:** >=10 independent new task/seed worlds, natural realistic benchmark and quality/bytes/runtime improvement over direct native and cheap matched controls. An aligned synthetic PASS only certifies the mechanism.

## C — counter-hypothesis / falsification

LoGo already obtains useful per-instance combinations from one evaluation of adapters; its selection signal and merge computation could dominate any savings from a small m. Random independently trained adapters may not share a cheap function orbit.

## U — statistical and numerical uncertainty

Identical initialization vs independent LoRA gauge ambiguity; cross-task identity leakage; router per-instance cost; token-level class prevalence and interference; group CI by task/world, not token.
With loss measured in nat/token, `u_c² = u_seed² + u_task² + u_num²` only under independent components; otherwise include twice the relevant covariances. Expanded `U_exp = k_cov * u_c` with `k_cov=2` is indicative at n=5, not a certified 95% interval. Use paired entire-task/bootstrap intervals and report outliers without cherry-picking. Benchmark timing uncertainty in seconds separately.

## Preservation / evidence products

Keep this directory design-only: `README.md`, `PROTOCOL.json`, `STATUS.md`. Do not emit an MA-scoped `RESULTS_CORE.csv` or `VERIFICATION.json` before executing frozen code. If activated, create a new non-worker experiment branch with immutable dataset/checkpoint/license hashes, recorded native versions, source unit tests, actual serializer output, all fresh rows and an independent replay. Report negative and M0 results explicitly.
