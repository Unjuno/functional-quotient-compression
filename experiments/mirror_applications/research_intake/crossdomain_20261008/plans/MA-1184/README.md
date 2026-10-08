# MA-1184 — TabPFN dataset-context functional Mirror post-calibration code

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://www.nature.com/articles/s41586-024-08328-6
- Official code/model: https://github.com/PriorLabs/TabPFN
- Native method as published: Nature 2025 TabPFN original prior-data fitted transformer, in-context dataset posterior predictive inference and default native ensemble/preprocessing; native TabPFN can already predict across unseen datasets without retraining
- Native literature references: PA454, PA455, PA453.
- Nearest pre-existing candidate IDs: MA-416, MA-624, MA-1159, MA-1175.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

A learned low-description context m can improve calibrated heldout tabular predictions without per-dataset fine-tuning or extra backbone copies, relative to native TabPFN preprocessing/ensembling and equally compact temperature/vector/Dirichlet calibration. A posterior-only View cannot recover dataset information erased in the native logits.

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

**Native shared object:** one already trained TabPFN checkpoint with context-dependent in-context posterior and readout (model from pinned release; no copying the checkpoint per dataset).

**Mirror insertion:** tiny dataset/support-conditioned Mirror calibration/readout code m applied to native posterior logits or permitted intermediate features; no unaccounted second TabPFN context forward.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** Source-conditioned 24-feature binary tasks with 10 distinct synthetic likelihood/shift families, 256 labeled support, 256 development and 1024 fresh samples per task; permute features and inject unobserved confounders. Train support-to-code generator only on allowed source tasks; freeze before five fresh entire tasks. For real tasks, use OpenML IDs with group/time splits and mandatory CatBoost/XGBoost.

**Implementation/measurements:** 1. Lock TabPFN model checkpoint, native preprocessing and in-context classifier/regressor API. 2. Run official native context prediction once per task; cache logits and optionally legally exposed feature summaries. 3. Train dataset-level m from support-only features/labels and original outputs, with tiny rank/diagonal/orthogonal charts. 4. Compare to native TabPFN native ensembles and repeated context fit, temperature scaling, Platt, vector/Dirichlet calibration, isotonic, and fine-tuned low-rank readout where authorized. 5. Measure ECE/Brier/NLL/accuracy on entirely heldout datasets and task variants and count all dataset embeddings, calibration generator, predictor workload.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. Official native TabPFN default classifier/ensemble and preprocessing
2. Native TabPFN per-dataset temperature, Platt, vector and Dirichlet calibration
3. Same-byte linear or low-rank logit calibration and feature-conditioned gate
4. CatBoost/XGBoost strong tabular predictor without the foundation checkpoint
5. Native TabM efficient tabular ensemble with chronological holdout
6. TabPFN identical logits with shuffled task context; target-label leakage null

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** dataset-weighted NLL, ECE, Brier, AUROC and worst-dataset shift; support count, amortized permanent task code bytes, paid generator size, input/context forward count, P95 and OOD. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** TabPFN official release/version licensed checkpoint. CPU only tiny tasks, real TabPFN checkpoint may need GPU; mark BLOCKED when checkpoint/access incompatible. Record any model version drift.

## D — decision thresholds

- **PASS preliminary mechanism:** On >=4/5 unseen datasets, Mirror m beats strongest native TabPFN calibration at same allowed support, improves paired NLL >=0.01 nat/example without >0.005 AUROC regression and source-model plus generator+K codes remains smaller than K independently tuned native adapters; no in-context extra forward.
- **FAIL / reject this code family on tested task:** Temperature/vector/Dirichlet or ordinary linear head matches or dominates, m trained using heldout task labels or dataset ID leak, TabPFN native in-context output already optimal or raw posterior erases needed information.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

TabPFN is already an amortized conditional predictor. A new 'Mirror' temperature parameter is just ordinary calibration, not extra independent function. Its heavy in-context attention may depend on entire support and cannot generally be reused across different datasets.

## U — measurement uncertainties

Small tabular dataset sampling error, label prevalence, support/test contamination and model calibration, hyperparameter search/posterior logit degeneracy; confidence group by whole datasets.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
