# MA-1186 — Caduceus parity-aware task Mirror codes beyond RC symmetry orbit

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://proceedings.mlr.press/v235/schiff24a.html
- Official code/model: https://github.com/kuleshov-group/caduceus
- Native method as published: Caduceus ICML2024 bidirectional reverse-complement equivariant MambaDNA; native RC parameter sharing and strand-consistency already built into model, so orientation transform is NOT new functional multiplicity
- Native literature references: PA457, PA453.
- Nearest pre-existing candidate IDs: MA-335, MA-332, MA-1175, MA-546.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

By separating group-symmetry-equivalent orientation Views from task-different signed even/odd functional codes, one RC equivariant representation can support multiple biological outputs with smaller task-specific bank than native independent task heads without breaking required RC parity. Generic unsigned global rotation may violate DNA orientation labels.

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

**Native shared object:** one frozen native Caduceus/biMambaDNA feature state with RC symmetry representation rho(C) and separate permitted functional readout task dimensions.

**Mirror insertion:** short assay/task m restricted to RC-even/RC-odd feature blocks to generate genuinely different motif/variant outputs; symmetry-only m_rc is explicit M0.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** DNA length 256 over A/C/G/T (16-base RC pilot tests already completed). Generate 4 motif presence assays (RC-invariant), 2 strand-specific orientation assays (RC-odd sign/change), 2 independent off-orbit motif interaction assays. 3 development genome families and 5 fresh entirely unseen motif/genome backgrounds. Distinct chromosome/family split for real ENCODE/GTEx sequence tasks; reverse complement doublets remain within same split.

**Implementation/measurements:** 1. Unit-test RC map R²=I, native equivariant feature rho(R), parity of each task label. 2. Create trainable tiny bidirectional conv/SSM source, then apply official Caduceus frozen encoder when authorized. 3. Learn source-only shared even/odd circuit bank and small task m per motif/assay; compare exact RC orbit only, random latent m, same-byte linear task readouts, native RC-equivalent classifier, post-hoc reverse-complement augmentation/conjoining and independent per-task heads. 4. Audit both forward and RC test sequences, wrong-strand negative controls, heldout chromosome accuracy, NLL, true physical bytes and actual forward count. 5. Any RC-equivalent output is not counted as a distinct useful task.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. Native Caduceus equivariant model and task-specific classifier
2. Native RCPS or reverse-complement augmentation plus post-hoc conjoining
3. One RC equivariant shared encoder with ordinary K linear assay heads
4. Same-byte source-only task-feature PCA/SVD factorization, diagonal/FiLM m
5. Non-equivariant BiMamba/ConvDNA classifier with RC test-time averaging
6. Shuffled antisymmetric label parity and false RC input reference as negative controls

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** assay AUROC/AUPRC, motif/variant effect rank, per-assay worst, orientation equivariance error (FP64 and deploy dtype), correctly distinct task Jacobian rank, bytes and P95 per DNA sequence. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** CPU numpy preliminary parity Stage-0, then official Caduceus code/pretrained checkpoints. No checkpoint duplication in storage numerator; input length/tokenizer/model size pinned.

## D — decision thresholds

- **PASS preliminary mechanism:** In >=4/5 independent fresh sequence families, exact native symmetry error <=1e-5 (trained float32) and assay AUPRC not worse than native single-encoder multihead by >0.02, task-code bytes <=0.85 native and measured gain over same-byte ordinary linear parity head. RC orbit-only is always M0, never enough.
- **FAIL / reject this code family on tested task:** All apparent extra outputs are just RC permutations/sign reindexing, native single-head/RCPS controls are strictly better, strand-specific assay parity violated, source motif identity leaked into fresh or task-specific private paths are uncounted.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

RC equivariance naturally gives cheap transformed output but no new independent model, exactly the naive Single-Forward Multi-Mirror trap. Non-palindromic strands have orientation-specific function labels that require new information outside the invariant quotient.

## U — measurement uncertainties

Genome sequence homology, reverse-complement duplicate leakage, strand orientation assay convention, class imbalance, extreme 1kb dependencies and SSM runtime. Count independently sampled genomes not sliding-window tokens.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
