# MA-1183 — TabM native rank-one ensemble with additional structured Mirror fast-weight chart

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html
- Official code/model: https://github.com/yandex-research/tabm
- Native method as published: TabM (ICLR 2025) original parameter-efficient tabular ensemble based on shared MLP+BatchEnsemble input/output rank-one fast weights, k-member loss and joint optimization; official yandex-research/tabm PyTorch code
- Native literature references: PA453, PA455, PA454.
- Nearest pre-existing candidate IDs: MA-260, MA-351, MA-690, MA-1175.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

Under original TabM optimized member forward, a short shared structured m-bank can reduce paid native fast-weight member state without losing ensemble gain on temporally heldout tabular tasks, outperforming a same-byte plain linear coefficient bank; single matrix shared does not mean one actual GEMM because input-side member fast weights differ.

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

**Native shared object:** one native TabM MLP trunk with native rank-one per-member fast-weight vectors r_kl/s_kl; do NOT replace TabM with plain shared-head MLP.

**Mirror insertion:** per-member, per-dataset small structured m_j in a shared chart for r_kl/s_kl (block-Givens/Householder/sign, or an additional residual structured output chart after original TabM); keep member-dependent input-side transforms and ensemble predictions.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** 16 informative + 16 nuisance numerical variables, binary and 3-class labels with independent task functions, 4 dataset families and 5 member heads; 4096 examples/family; seeded source-only drifting feature/label correlations; dev 11–13 and fresh 101–105 with entire unseen data-generating families. Separate synthetic linear/nonlinear and off-orbit controls; negative null from MCX Stage0 input-code collision.

**Implementation/measurements:** 1. Reproduce the native official TabM parameter-efficient ensemble source including member-specific gradient/loss, shared MLP and native fast weights. 2. Under same native base, fit short m per member and reconstruct fast weights through a structured dictionary; m dim 2,4,8,16. 3. Train original TabM, original TabM-mini, ordinary BatchEnsemble, equal-byte SVD/linear code factorization of fast weight vectors, output-only Givens, and the proposed structured m from same parent/update schedule. 4. Verify member diversity, cal/NLL, ensemble mean and worst subnetwork, true MAC/GEMM invocations, file and active RAM bytes. 5. Stress whole heldout tabular datasets and TabReD chronological splits; source-only preprocessing and feature selection.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. Native official TabM and TabM-mini, full r/s member fast weights
2. Native BatchEnsemble with rank-one fast weights
3. Ordinary shared-trunk K independent linear heads and MIMO for task identity fairness
4. Same-byte PCA/SVD/low-rank linear dictionary of TabM fast weights
5. CatBoost/XGBoost/LightGBM tabular baselines, especially TabReD temporal shift
6. Native TabM ensemble member selection/pruning without m

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** per-dataset ROC AUC/accuracy, NLL, ECE, Brier; independent ensemble-member variance; actual NPZ/state-dict bytes and RAM, true CPU/GPU MAC, latency p50/p95, training steps at fixed work and near convergence. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** CPU 32-64 dim standalone pilot under numpy/torch; official TabM package and TabReD datasets for natural stage. Pin pyproject version, dataset licenses, BLAS threads and CPU/GPU benchmark code.

## D — decision thresholds

- **PASS preliminary mechanism:** At K>=5, >=4/5 fresh heldout independent tabular families match native TabM mean NLL within +0.02 nat/example and ROC AUC within 0.01, cut native member-specific state >=15% and whole model persisted bytes >=3%, P95<=1.10x native, with strictly better quality-byte frontier than same-byte ordinary SVD code.
- **FAIL / reject this code family on tested task:** Native TabM, TabM-mini or same-byte ordinary linear fast-weight bank dominates; apparent speedup only compares K independent whole models rather than packed native TabM; heterogeneous distributions collapse member diversity.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

TabM already has the compressed rank-one member parameterization and packed multi-member training. A Mirror chart that merely renames r/s has no added degrees of freedom; input-fast-weight operations are not generally derivable from xW alone.

## U — measurement uncertainties

Chronological drift and correlated rows, categorical leakage, TabM initialization/member training variance, choice of tabular dataset; bootstrap entire dataset/temporal blocks, not individual table rows.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
