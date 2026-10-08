# MA-1182: ProtoMech cross-layer protein-circuit Mirror address bank

**Research-only, UNTESTED, NOT AUTO-CLAIMED.** Independent breadth-intake branch: `research/mirror-breadth-method-sweep-20261008`. No special queue priority; MA priority P1 remains a register label and current worker MA-255 is untouched. These are design hypotheses, not results.

## Source classification / deduplication

- New native method: ProtoMech native top-k cross-layer transcoders (CLT), windowed CLT and separate per-layer transcoders with sparse circuit discovery, starting from ESM2-8M/35M.
- Direct paper: https://proceedings.mlr.press/v306/tsui26a.html; primary references: PA445; PA444.
- Nearby earlier candidate IDs: MA-526; MA-533; MA-537; MA-1165. Current proposal differs by its explicitly identified physical object and per-biological-task code.
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
| `ell` | protein model layer | 1 | integer 1..L |
| `k` | number of active sparse latent neurons | 1 | integer >=1 |
| `m` | function-circuit address | 1 | real k_code vector |
| `R` | cross-layer circuit response fidelity | 1 | fraction [0,1] |

**Dimensional validation:** input/feature vectors and learned factors are dimensionless model tensors unless original physical target defines its own units. A Mirror m may multiply only shape-compatible normalized weights or activations; classification AUPRC, correlation and NLL are **not interchangeable**. Raw biochemical temperature, gene-expression count, and sequence length must retain their original units/provenance. Never combine S (bytes) and t (seconds) into task loss.

## H — falsifiable hypothesis

For multiple heldout protein function/family predictions from the same ESM2/ProtoMech cross-layer latent state, a tiny m circuit selector compresses the bank of per-function sparse circuit coefficients >=15% without reducing probe AUPRC by >0.02 absolute or CLT feature-fidelity by >5% versus native per-task circuit selection, and beats native same-byte sparse linear factor bank at <=1.10 P95.

## Mirror insertion — precise minimal native change

> **Mirror insertion:** insert small m into **small motif/protein-family task m selecting a signed/structured subspace or low-rank coefficient chart over already identified ProtoMech circuit nodes** around **one original trained cross-layer sparse latent/circuit decoder bank, explicitly retaining its cross-layer graph dependencies**. Preserve native task/data processing and model's shared physical state; candidate outputs should differ usefully without running an independent expensive encoder per role.

- Native operator not invented here: ProtoMech native top-k cross-layer transcoders (CLT), windowed CLT and separate per-layer transcoders with sparse circuit discovery, starting from ESM2-8M/35M.
- Shared paid object: one original trained cross-layer sparse latent/circuit decoder bank, explicitly retaining its cross-layer graph dependencies.
- Physical count includes tokenizer/gene masks, embedding, adapters, code decoder, task heads, router and any fold/materialized copies.
- A different feature coordinate that keeps predictions identical contributes no new functional multiplicity.

## T — self-contained implementation contract

**Mechanism fixture:** Synthetic 4-layer width-32 causal feature network, 64 sparse latents/layer with 4 source motif tasks and 4 heldout motif/family labels drawn from reused atom families plus independent private motifs, no pretrained protein claims. Natural lane: ESM2-8M ProtoMech official weights/data and family classification/fitness benchmarks, frozen labels and protein-family-disjoint test, not natural ESM2-35M by default.

**Implementation stages:** (1) Verify released ProtoMech CLT top-k activation, cross-layer decoder and AuxK configuration at one checked code revision. (2) Train/reuse a source CLT on ESM2 activations or synthetic source features; freeze it. (3) Obtain native function circuits and their sparse latent coefficients from source task support only; target readouts require *only support labels*, not fresh test labels. (4) Fit shared signed/linear/Givens low-code bank over function circuit coefficient vectors m, keeping sparse top-k routing and layer dependencies unchanged; compare native per-function sparse circuit, PLT, windowed CLT and shared linear code. (5) Score all heldout task probes, feature reconstruction, OOD protein family and per-latent code bytes/FLOPs. (6) For perturbation/steering, require independently measured sequence fitness and never treat the paper's recovered accuracy or design-rate as Mirror reproduction.

**Mandatory native/simpler controls:**

1. Original ProtoMech cross-layer transcoder and native per-function circuit selection
2. Windowed CLT and independent per-layer PLT
3. plain shared sparse dictionary with per-function linear coefficients, byte matched
4. ordinary ridge/linear probe on frozen ESM2 embeddings
5. random sparse circuit with identical number of active latents
6. full ESM2 task head quality reference

**Data firewall and preregistration:** source/dev worlds seeds 11,12,13 may select code k, learning rate and target normalization. Fresh 101,102,103,104,105 are disjoint **whole task/protein family/perturbation identity** units, not just random labeled rows. Freeze source/model revision, tokenizer/gene set and label pipeline before accessing any fresh outcomes; run all five fresh worlds and no cherry-picked early stop. Real public benchmark stage needs its **own separately frozen** species/line/family partition and may not reuse synthetic fresh IDs to tune.

**Measurements:** family AUROC/AUPRC, fitness Spearman (if measured), per-layer latent activation overlap/faithfulness, sparse top-k error, actual serialized circuit IDs+dictionary/code bytes, active decoding FLOPs and P95. Report source-training compute separately from runtime, native model/adapter checkpoint SHA, absolute/pairwise quality, worst task, count of real shared backbone forwards, actual serialized payload bytes including metadata, P50/P95 on CPU/GPU with synchronization and batch/context shapes, active MAC/FLOPs, and any cache/copy state. If official native model cannot run, label native replication BLOCKED; synthetic feasibility cannot substitute.

**Hardware:** CPU 4-layer sparse synthetic first. Official ProtoMech supports ESM2-8M/35M; full natural reproduction requires GPU, released code/data, checkpoint SHA and model license..

## D — pass/fail/uncertain and clear stopping rule

**PASS (initial screen):** At least four of five fresh entire protein-family worlds within -0.02 AUPRC native and <=1.05x feature reconstruct error, per-function circuit trained state <=0.85 of native, total CLT+code bytes decrease at least 3%, P95<=1.10 native and gain beyond same-byte simple sparse linear code.

**FAIL:** PLT/native sparse CLT or plain linear task probes Pareto-dominate; low-description m destroys functional circuit fidelity, or circuit interpretation is asserted from feature activation correlation without task intervention.

**UNCERTAIN:** native source/checkpoint unavailable, data contamination, missing paid bytes/runtime, calibration masks violated, or task-level uncertainty spans the threshold. A code that just changes logits but not utility is not a success.

A fresh five-world PASS is exploratory; **>=10 new independent task/seed units** plus natural-data reproduction and native matched-byte Pareto proof are required before any ADOPTED recommendation.

## C — counter-hypothesis

ProtoMech already builds compressed sparse cross-layer circuits; Mirror transformation of circuit coefficients may merely rename task-specific sparse selection. Different protein motifs may require disjoint atoms and expensive exceptions.

## U — uncertainty and numerical error

Protein sequence homology, latent top-k ties, sparse support instability, ESM2 layer normalization, binding-site localization vs family labels, true model fidelity vs feature proxy; bootstrap by protein family.
For loss in nat/token and independent terms only, `u_c^2=u_seed^2+u_task^2+u_num^2`; include double covariance terms where independence fails. `U=k_cov*u_c` with k_cov=2 is a preliminary expanded interval, **not a guaranteed 95% bound** for five task worlds. Bootstrap by whole protein family/gene perturbation, not highly correlated observations. Runtime uncertainty in seconds and actual bytes are separate.

## Reproducibility deliverables

Run smoke unit tests for task-code shape invariance, invalid role/oracle leakage, decoder-byte accounting, and zero-role native parity before training. After frozen execution, emit full source, immutable protocol, development and all fresh raw rows, measured `RESULTS_CORE.csv`, `VERIFICATION.json` and checkpoint/dataset hashes. Do not change current worker files or original scientific MA status based on a design document.
