# MA-1180: ESME protein multi-property Mirror readout library

**Research-only, UNTESTED, NOT AUTO-CLAIMED.** Independent breadth-intake branch: `research/mirror-breadth-method-sweep-20261008`. No special queue priority; MA priority P1 remains a register label and current worker MA-255 is untouched. These are design hypotheses, not results.

## Source classification / deduplication

- New native method: ESME efficiency-optimized native ESM2 inference/fine-tuning (packing, FlashAttention and per-property parameter-efficient heads), plus a standard frozen ESM2 encoder.
- Direct paper: https://doi.org/10.1016/j.isci.2025.113495; primary references: PA444; PA445.
- Nearby earlier candidate IDs: MA-351; MA-416; MA-1175. Current proposal differs by its explicitly identified physical object and per-biological-task code.
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
| `N` | number of proteins | 1 | integer>=1 |
| `L` | protein residues per input | residue (1) | positive integer length |
| `m` | property code | 1 | k-dimensional finite real vector |
| `AUPRC` | precision-recall area | 1 | real [0,1] |

**Dimensional validation:** input/feature vectors and learned factors are dimensionless model tensors unless original physical target defines its own units. A Mirror m may multiply only shape-compatible normalized weights or activations; classification AUPRC, correlation and NLL are **not interchangeable**. Raw biochemical temperature, gene-expression count, and sequence length must retain their original units/provenance. Never combine S (bytes) and t (seconds) into task loss.

## H — falsifiable hypothesis

Across naturally independent protein-property prediction tasks with identical encoder/checkpoint and matched labeled support, a compact Mirror property code bank reduces the entire multi-task inference head state >=15% versus native separate PEFT heads while preserving mean/worst AUROC/AUPRC or correlation within 0.02 absolute at <=1.10 P95 of a strong one-pass shared-encoder multihead.

## Mirror insertion — precise minimal native change

> **Mirror insertion:** insert small m into **short property/motif/assay m that changes the readout of a single encoder output without running the full protein encoder per property** around **one native pretrained or frozen ESM2-8M/ESME protein sequence encoder and one shared property-readout basis**. Preserve native task/data processing and model's shared physical state; candidate outputs should differ usefully without running an independent expensive encoder per role.

- Native operator not invented here: ESME efficiency-optimized native ESM2 inference/fine-tuning (packing, FlashAttention and per-property parameter-efficient heads), plus a standard frozen ESM2 encoder.
- Shared paid object: one native pretrained or frozen ESM2-8M/ESME protein sequence encoder and one shared property-readout basis.
- Physical count includes tokenizer/gene masks, embedding, adapters, code decoder, task heads, router and any fold/materialized copies.
- A different feature coordinate that keeps predictions identical contributes no new functional multiplicity.

## T — self-contained implementation contract

**Mechanism fixture:** CPU first: 20-symbol amino-acid alphabet, length 64 sequences, 5 planted motifs and 3 non-motif private functions; generate 4 source assays and 4 heldout assay/motif combinations, with truly independent off-orbit rules. IMPORTANT: synthetic motif classification is not evidence on real proteins. Natural stage: frozen ESM2-8M/ESME original source (pin revision) and published protein family/thermostability/missense effect datasets with shared protein IDs; isolate assay-specific labels, keep biochemical measures in native units and report missing label masking.

**Implementation stages:** (1) Freeze source ESM2/ESME encoder/tokenizer and record sequence pooling and input length. (2) Fit native property-specific linear/MLP and official LoRA/fine-tuned heads using their labeled source support only. (3) Fit a shared output/readout dictionary with k={2,4,8,16} property codes m, compare signed/permutation/Givens code to plain linear diagonal codes and multihead. (4) Evaluate identical protein encodings cached once and a true single-forward sequence processor, not different forward calls hidden in a wrapper. (5) Split real sequences by protein family and <=30% sequence identity where feasible; derive identity clusters only from training/source and freeze splits. (6) Test all property tasks and reporting units separately; no conversion of Pearson/AUPRC into one artificial score.

**Mandatory native/simpler controls:**

1. Original ESME/ESM2 property-wise fine-tuned LoRA/head (native, with its optimized encoder)
2. one frozen ESM2/ESME trunk + ordinary independent property linear/MLP heads computed in one forward
3. same byte-size shared low-rank linear/head dictionary, FiLM and IA3
4. multi-input MIMO/MIMMO where comparable function target
5. independent encoder per property upper quality cost
6. one classifier with task one-hot code, same parameter bytes

**Data firewall and preregistration:** source/dev worlds seeds 11,12,13 may select code k, learning rate and target normalization. Fresh 101,102,103,104,105 are disjoint **whole task/protein family/perturbation identity** units, not just random labeled rows. Freeze source/model revision, tokenizer/gene set and label pipeline before accessing any fresh outcomes; run all five fresh worlds and no cherry-picked early stop. Real public benchmark stage needs its **own separately frozen** species/line/family partition and may not reuse synthetic fresh IDs to tune.

**Measurements:** per-property AUROC/AUPRC, Matthews correlation for imbalanced function labels, Spearman or Pearson for continuous stability, worst-property drop, identity leakage, all serialized heads+encoder bytes, CPU/GPU P50/P95, encoder call count, source training operations. Report source-training compute separately from runtime, native model/adapter checkpoint SHA, absolute/pairwise quality, worst task, count of real shared backbone forwards, actual serialized payload bytes including metadata, P50/P95 on CPU/GPU with synchronization and batch/context shapes, active MAC/FLOPs, and any cache/copy state. If official native model cannot run, label native replication BLOCKED; synthetic feasibility cannot substitute.

**Hardware:** CPU torch ESM2 8M inference where supported; ESME original source uses GPU optimized kernels for true speed measurement; document amino acid tokenizer, model revision, ESM2/ESME differences..

## D — pass/fail/uncertain and clear stopping rule

**PASS (initial screen):** At least 4/5 fresh full-protein/assay worlds match native one-pass shared encoder+linear heads within 0.02 absolute relevant task metric, save >=15% per-property *paid* readout state and >=3% whole total model state, P95 <=1.10, and strict Pareto advantage over byte-matched ordinary linear output bank; otherwise M0.

**FAIL:** One-pass standard multi-head is as good as Mirror at equal bytes/quality, protein family or homology leaks across splits, native ESME packing/FlashAttention disabled in runtime comparison, or private assay labels require an independent feature extractor.

**UNCERTAIN:** native source/checkpoint unavailable, data contamination, missing paid bytes/runtime, calibration masks violated, or task-level uncertainty spans the threshold. A code that just changes logits but not utility is not a success.

A fresh five-world PASS is exploratory; **>=10 new independent task/seed units** plus natural-data reproduction and native matched-byte Pareto proof are required before any ADOPTED recommendation.

## C — counter-hypothesis

ESM2 pretrained representation is itself the expensive state, and ordinary task-specific linear classifiers already have cheap one-forward multiplicity. Mirror may only compress tiny output heads with negligible whole-model rate gain.

## U — uncertainty and numerical error

Protein-family identity sharing, imbalance, assay/lab batch confounders, uncensored vs censored melting point, tokenizer special tokens, inverse correlation signs; report clustered bootstrap by sequence family and independent task worlds.
For loss in nat/token and independent terms only, `u_c^2=u_seed^2+u_task^2+u_num^2`; include double covariance terms where independence fails. `U=k_cov*u_c` with k_cov=2 is a preliminary expanded interval, **not a guaranteed 95% bound** for five task worlds. Bootstrap by whole protein family/gene perturbation, not highly correlated observations. Runtime uncertainty in seconds and actual bytes are separate.

## Reproducibility deliverables

Run smoke unit tests for task-code shape invariance, invalid role/oracle leakage, decoder-byte accounting, and zero-role native parity before training. After frozen execution, emit full source, immutable protocol, development and all fresh raw rows, measured `RESULTS_CORE.csv`, `VERIFICATION.json` and checkpoint/dataset hashes. Do not change current worker files or original scientific MA status based on a design document.
