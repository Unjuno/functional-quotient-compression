# MA-1181: CellFM-RegFormer perturbation Mirror function and combo codes

**Research-only, UNTESTED, NOT AUTO-CLAIMED.** Independent breadth-intake branch: `research/mirror-breadth-method-sweep-20261008`. No special queue priority; MA priority P1 remains a register label and current worker MA-255 is untouched. These are design hypotheses, not results.

## Source classification / deduplication

- New native method: native CellFM/RegFormer expression encoder with official perturbation prediction head and simple calibrated single/double perturbation regression; no asserted superiority from published task scores alone.
- Direct paper: https://www.nature.com/articles/s41592-025-02772-6; primary references: PA446; PA447; PA448; PA449; PA452.
- Nearby earlier candidate IDs: MA-835; MA-851; MA-1081; MA-1173. Current proposal differs by its explicitly identified physical object and per-biological-task code.
- Mirror novelty boundary: the foundation model/transcoder/efficient encoder belongs to prior art. Only incremental useful functions or saved *total paid state* attributable to m could be project-specific.

## Symbols, SI units and shape assumptions

| variable | 日本語 | SI単位 | 形状・型・許容範囲 |
|---|---|---|---|
| x | 生体系列／特徴入力 | 1 | 有限実系列（生物学的配列は別途符号化） |
| theta | 共有ネイティブ重み | 1 | 正規化された実行列・テンソル |
| m | 追加Mirror機能座標 | 1 | k成分の有限実ベクトル |
| K | 論理タスク／目的の数 | 1 | 整数>=2 |
| L | タスク損失 | nat/token（SIでは1） | 有限非負実数、元タスク単位も併記 |
| S | 全推論用保存容量 | byte（実用単位） | 整数>=0 |
| t | 実行遅延 | s | 実数>=0、同期単調時間 |
| u_seed,u_task,u_num | 不確かさ成分 | 損失と同単位 | 実数>=0、相関を記録 |
| u_c | 合成標準不確かさ | 損失と同単位 | 実数>=0、共分散を含む |
| k_cov | 包含係数 | 1 | 実数>0、仮に2 |
| U | 拡張不確かさ | 損失と同単位 | k_cov * u_c |

Specific terms:

| symbol | 日本語 | SI単位 | 型・範囲 |
|---|---|---|---|
| `G` | number of modeled genes | 1 | integer >=1 |
| `c` | cell context | 1 | categorical cell-line/tissue identifier from allowed covariates |
| `m(g1,g2,c)` | perturbation functional code | 1 | finite k-vector; ordered pair only when biologically justified |
| `r_RMSE` | relative expression error | 1 | nonnegative scalar |

**Dimensional validation:** input/feature vectors and learned factors are dimensionless model tensors unless original physical target defines its own units. A Mirror m may multiply only shape-compatible normalized weights or activations; classification AUPRC, correlation and NLL are **not interchangeable**. Raw biochemical temperature, gene-expression count, and sequence length must retain their original units/provenance. Never combine S (bytes) and t (seconds) into task loss.

## H — falsifiable hypothesis

On heldout entire gene perturbations/pairs and cell lines, Mirror factors generate useful perturbation-response functions at >=15% less per-perturbation trained state than a native individually fitted response bank, without worse than +0.03 calibrated relative RMSE or -0.02 calibrated Pearson on novel gene pairs, beating an ordinary linear factor and simple mean/linear baselines.

## Mirror insertion — precise minimal native change

> **Mirror insertion:** insert small m into **compact factorized gene1 × gene2 × cell-context Mirror m encoding perturbation-specific response corrections, with optional measured out-of-orbit private residual** around **one frozen cell-state embedding and one source-trainable perturbation-response predictor shared across cell types and perturbation roles**. Preserve native task/data processing and model's shared physical state; candidate outputs should differ usefully without running an independent expensive encoder per role.

- Native operator not invented here: native CellFM/RegFormer expression encoder with official perturbation prediction head and simple calibrated single/double perturbation regression; no asserted superiority from published task scores alone.
- Shared paid object: one frozen cell-state embedding and one source-trainable perturbation-response predictor shared across cell types and perturbation roles.
- Physical count includes tokenizer/gene masks, embedding, adapters, code decoder, task heads, router and any fold/materialized copies.
- A different feature coordinate that keeps predictions identical contributes no new functional multiplicity.

## T — self-contained implementation contract

**Mechanism fixture:** CPU stage: synthetic 64-gene regulatory stable dynamical system W (spectral radius<0.9), 4 cell types, 12 perturbation genes and 16 heldout ordered/symmetric 2-gene combinations, source-only no future conditions; generate paired unperturbed/perturbed expression from stable linear dynamics plus mild nonlinear saturation and independent private interactions. Natural stage: Norman/Adamson/Replogle perturb-seq benchmark data with perturbation-identity and cell-type held-out splits, consistent gene universe/normalization; no donor or sequencing-batch leakage.

**Implementation stages:** (1) On source splits train ordinary mean response, standard ridge/linear gene-embedding regression and native GEARS/CellFM/RegFormer adapters where their official checkpoints/code are available. (2) Freeze native cell embedding pre/post processing with source training only. (3) Decode perturbation m(g1,g2,c)=u_g1+u_g2+pair_delta or tensor-factor product, and attach low-description signed/diagonal/short Givens View to the original native predictor; make order/symmetry assumptions explicit and verify them. (4) Compare plain sum/product of perturbation codes, mean change, ridge, tensor factor, full private tasks and Mirror+private residual. (5) Freeze normalization, nonzero gene masks, dropout and allowed source examples; audit truly unseen gene pairs and different cell lines. (6) Report per-gene differential response, sign accuracy, calibrated positive controls and uncertainty; do not claim biological causality from random splits.

**Mandatory native/simpler controls:**

1. Strong mean expression change, OLS/ridge and calibration-aware empirical simple baselines from PA448/PA449
2. native CellFM/RegFormer official perturbation setup or BLOCKED with reason
3. GEARS/CPA standard perturbation model
4. same-byte ordinary gene-factor/tensor-factor and linear-MLP predictor
5. per-perturbation private head (quality upper reference)
6. identity-only and shuffled-gene negative controls

**Data firewall and preregistration:** source/dev worlds seeds 11,12,13 may select code k, learning rate and target normalization. Fresh 101,102,103,104,105 are disjoint **whole task/protein family/perturbation identity** units, not just random labeled rows. Freeze source/model revision, tokenizer/gene set and label pipeline before accessing any fresh outcomes; run all five fresh worlds and no cherry-picked early stop. Real public benchmark stage needs its **own separately frozen** species/line/family partition and may not reuse synthetic fresh IDs to tune.

**Measurements:** calibrated relative error vs positive control, per-differential-expression-gene Pearson, sign agreement, dose/line heldout, top perturbation ranking, cell-level and gene-level RMSE, actual trained state and inference time. Report source-training compute separately from runtime, native model/adapter checkpoint SHA, absolute/pairwise quality, worst task, count of real shared backbone forwards, actual serialized payload bytes including metadata, P50/P95 on CPU/GPU with synchronization and batch/context shapes, active MAC/FLOPs, and any cache/copy state. If official native model cannot run, label native replication BLOCKED; synthetic feasibility cannot substitute.

**Hardware:** CPU 64-gene synthetic; official CellFM MindSpore/RegFormer Mamba real model needs released pretrained weights, authorized data and compatible NPU/GPU. Mark native reproduction BLOCKED if unavailable..

## D — pass/fail/uncertain and clear stopping rule

**PASS (initial screen):** In >=4/5 independent fresh unseen-perturbation-group worlds, meet quality Δrelative RMSE<=+0.03 and Δcalibrated Pearson>=-0.02 versus strongest available native method, total task-specific bytes <=0.85 reference, P95<=1.10, and a strict advantage over simple mean/ridge and matched-byte linear code.

**FAIL:** Any simple mean/linear method matches or beats the Mirror frontier, leakage of heldout perturbation identity/target expression into m, target data normalization mismatch, or calibrated positive control shows benchmark metric insensitive to meaningful output quality.

**UNCERTAIN:** native source/checkpoint unavailable, data contamination, missing paid bytes/runtime, calibration masks violated, or task-level uncertainty spans the threshold. A code that just changes logits but not utility is not a success.

A fresh five-world PASS is exploratory; **>=10 new independent task/seed units** plus natural-data reproduction and native matched-byte Pareto proof are required before any ADOPTED recommendation.

## C — counter-hypothesis

Gene perturbation response may be mostly captured by simple cell-line mean effects and gene embedding factors. A structural code may simply interpolate non-causal correlations; strong baselines can dominate claimed foundation-model benefits.

## U — uncertainty and numerical error

Large sample dependence within same perturbation/cell type, RNA dropout/normalization, compositional perturbation asymmetry, metric calibration (PA448 vs PA449), donor/tissue batch and OOD noise; bootstrap whole perturbation identities, not cells.
For loss in nat/token and independent terms only, `u_c^2=u_seed^2+u_task^2+u_num^2`; include double covariance terms where independence fails. `U=k_cov*u_c` with k_cov=2 is a preliminary expanded interval, **not a guaranteed 95% bound** for five task worlds. Bootstrap by whole protein family/gene perturbation, not highly correlated observations. Runtime uncertainty in seconds and actual bytes are separate.

## Reproducibility deliverables

Run smoke unit tests for task-code shape invariance, invalid role/oracle leakage, decoder-byte accounting, and zero-role native parity before training. After frozen execution, emit full source, immutable protocol, development and all fresh raw rows, measured `RESULTS_CORE.csv`, `VERIFICATION.json` and checkpoint/dataset hashes. Do not change current worker files or original scientific MA status based on a design document.
