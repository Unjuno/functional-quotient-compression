# MA-1188 — Aurora shared latent multi-field Mirror decoder-bank compression

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://www.nature.com/articles/s41586-025-09005-y
- Official code/model: https://github.com/microsoft/aurora
- Native method as published: Nature2025 Aurora pretrained heterogeneous geophysical 3D encoder + Swin processor + task-specific Perceiver decoder/fine-tune; Aurora already supports multiple forecast domains and autoregressive rollouts
- Native literature references: PA460, PA459, PA461.
- Nearest pre-existing candidate IDs: MA-1053, MA-1081, MA-1139, MA-1175.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

For multiple simultaneously available Earth-system fields at the same forecast lead step, Mirror-coded decoder bank reduces task-specific adaptation bytes while retaining native forecast quality at P95<=1.10; separate forecast horizons requiring autoregression must still pay their actual steps and cannot be counted as single-forward outputs.

## Mathematical interface and unit contract

| 記号 | 日本語 | 単位（SIまたは実用単位） | 形状/定義域 |
|---|---|---|---|
| x | 観測入力/系列 | 1（物理場は別途SI単位） | 有限実ベクトルまたはトークン列 |
| θ | native共有重み | 1 | 実行列/テンソル、元モデルの型 |
| m | 追加Mirror機能コード | 1 | 次元rの有限実ベクトル、角度はrad |
| r | Code次元 | 1 | 1以上の整数 |
| K | 役割/論理出力数 | 1 | 2以上の整数 |
| L | タスク損失 | nat/sample（SIでは1） | 有限実数、タスク固有指標を併記 |
| S | 真の直列化容量 | byte（非SI実用単位） | 0以上整数、共有基底/metadata含む |
| τ | 1回の処理時間 | s | 非負実数、P50/P95など |
| u_seed,u_eval,u_num | 不確かさ成分 | 損失と同単位 | 非負実数、相関あり得る |
| u_c | 合成標準不確かさ | 損失と同単位 | 共分散項あり |
| k_cov | 拡張係数 | 1 | 正数、仮値2 |
| U_exp | 拡張不確かさ | 損失と同単位 | k_cov u_c |

**Native shared object:** one native Aurora pretrained encoder+processor with field-specific decoders for atmosphere/waves/air quality and calibrated physical variable masks.

**Mirror insertion:** small field/pressure-level/horizon data-source role m on a shared, source-learned decoder basis; preserve per-variable physical unit scaling and missing-data masks.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** CPU 16×16 shallow-water/advection diffusion fields (u,v,temperature,synthetic chemical tracer) over 4 regimes and K=4 target variables, 256 source patches, 128 dev and 256 fresh per independent weather regime; spatial train/dev/fresh and calendar-year holdout. Native stage uses AuroraSmallPretrained for API/memory debugging only; publication-grade lane pinned AuroraPretrained with ERA5/CAMS/HRES-WAM licensed data.

**Implementation/measurements:** 1. Reproduce native Aurora variable/pressure-level input channel masks and one-step output with author code; record no hidden rollout. 2. Train native per-variable original Perceiver decoder heads or official pretrained fine-tunes from identical source parent. 3. Fit a shared decoder basis and low-description m for each field (optionally grid cell class), with genuinely paid output projection, calibrations and physical unit conversions. 4. Compare native Aurora individual task heads/fine-tuned adapters, same-byte ordinary linear/FiLM/spectral shared heads, source-only low-rank LoRA, and structured m plus private correction. 5. Stress heldout years, source/data domain shifts, extreme events and missing variables. 6. Count full autoregressive calls for lead-time trajectories, real model HBM/offload and IO bottlenecks, one physical shared one-step processing path for simultaneous fields.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. Official native AuroraPretrained and original task fine-tuned domain-specific decoder
2. AuroraSmallPretrained only as debug, never strong full-model native comparison
3. Single unmodified pretrained Aurora processor + K ordinary native task-specific Perceiver heads
4. Same-byte ordinary linear/spectral shared decoder dictionary with FiLM/IA3
5. Simple climatology/persistence or numerical PDE solver per variable for skill score
6. Separate full per-field fine-tunes quality upper bound and quantized native adapter bank

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** temperature RMSE (K), wind RMSE (m/s), significant-wave-height RMSE (m), NO2/PM forecast errors (µg/m³), extreme event skill, lead-specific rollout error, total resident/serialized decoder+model bytes, GPU/CPU seconds and IO bytes. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** CPU small PDE smoke only; official Aurora pretrain/fine-tune released with package microsoft-aurora; full 0.25° models require substantial accelerator RAM; honest BLOCKED if unavailable.

## D — decision thresholds

- **PASS preliminary mechanism:** In >=4/5 independent heldout physical regimes, Mirror joint decoder meets native per-variable quality within 5% relative native-metric tolerance on all primary fields, saves >=20% paid decoder bank bytes at <=1.10 P95 and improves over same-byte linear shared decoders; no misleading reuse of autoregressive steps.
- **FAIL / reject this code family on tested task:** Native Perceiver multi-field decoding already saturates quality-byte Pareto, converting field units artificially improves aggregate score, distinct horizons wrongly counted as same forward, or GPU IO dominates and Mirror introduces instability.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

Aurora already shares most expensive encoder/processor and has heterogeneous decoders. Mirror may compress a small adapter only, with negligible whole checkpoint reduction and no forecast speed gain; strong atmospheric physics may require separate output channels.

## U — measurement uncertainties

Temporal autocorrelation and cross-field covariance, ERA5 assimilation leakage, climatological nonstationarity, pressure levels, missing masks, target physical units and autoregressive error accumulation; bootstrap whole yearly regime events.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
