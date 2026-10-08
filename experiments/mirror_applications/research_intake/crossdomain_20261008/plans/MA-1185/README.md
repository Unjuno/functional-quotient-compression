# MA-1185 — DINOv3 multi-dense-task compressed Mirror head bank

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://arxiv.org/abs/2508.10104
- Official code/model: https://github.com/facebookresearch/dinov3
- Native method as published: DINOv3 (Meta 2025) frozen high-quality image patch features, Gram anchoring and native linear or Mask2Former segmentation decoder; official segmentation and depth heads are meaningful controls
- Native literature references: PA456, PA455.
- Nearest pre-existing candidate IDs: MA-351, MA-1100, MA-1024, MA-1081.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

After source-family learning of one shared dense readout basis over frozen DINOv3 features, learned m can replace much of the stored per-task decoder state while preserving useful independent segmentation/depth/normal outputs; cheaper simple native linear decoder basis is mandatory null.

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

**Native shared object:** one unmodified frozen DINOv3 ConvNeXt-Tiny or ViT-small feature pyramid evaluated once for each image; task decoder family for semantic classes, depth bins, surface normals or cloud masks.

**Mirror insertion:** tiny task/scale-specific Mirror m operating in shared dense feature/readout basis with separate **paid** output-shape projection per task; not a rotation of incompatible semantic outputs.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** CPU 16×16 synthetic images with geometric shape segmentation labels, depth ramps and oriented normal regression; 4/5 independent target tasks with overlap alpha in {0,.5,1}; 128 source scenes, 256 dev, 512 fresh, whole scene-grammar split. Use one shared 32-channel 4-level feature encoder; no pretrained-image claim. Natural stage pinned DINOv3 ConvNeXt Tiny and four real dense-label domains with scene/site-disjoint splits.

**Implementation/measurements:** 1. Verify official frozen DINOv3 multi-layer features and original linear/Mask2Former decoder exactly. 2. Train native per-task decoder and DPT-style depth head under allowed source/task budget, no Mirror. 3. Learn a source-only common feature decoder dictionary and tiny task m (sign/Householder/Givens/diagonal) with task-specific final linear map and units declared. 4. Compare full native outputs, ordinary same-byte low-rank/tensorized heads, one shared trunk K independent native linear heads, task-conditioning FiLM, and proposed m+private residual. 5. Evaluate spatial heldout scenes, segmentation boundaries, depth ranking and normal angles; count all image encoder/decoder bytes and actual single-backbone calls.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. DINOv3 original frozen backbone with independently trained native linear segmentation/depth heads
2. Native DINOv3 Mask2Former and dense decoder when official runnable
3. Shared feature pyramid with ordinary K linear readouts (single encoder forward)
4. Same-byte rank-k/tensorized low-rank native head dictionary, FiLM and IA3
5. Independent task decoders and a full per-task fine-tuned backbone as upper reference
6. Random feature code, shuffled task labels and segmentation-only ablation

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** segmentation mIoU/Dice, depth AbsRel/RMSE, normal cosine/angle (rad), worst task, genuine extra function diversity, total model+head bytes, GPU/CPU P50/P95 and actual backbone calls. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** CPU synthetic first; official DINOv3 code and licensed pretrained checkpoints required for natural; e.g. ConvNeXt Tiny to fit limited GPU; record license and pretrained hash.

## D — decision thresholds

- **PASS preliminary mechanism:** In >=4/5 fresh disjoint scene worlds, preserve task quality within <=0.02 mIoU or +0.02 relative depth RMSE of best native one-backbone multihead, reduce paid task-decoder payload >=20% with <=1.10x P95 and a Pareto win over ordinary equal-byte decoder basis.
- **FAIL / reject this code family on tested task:** Native shared-backbone independent heads are faster/more accurate at same bytes, m only moves a semantic class color/coordinate without improving heldout prediction, feature normalization/preprocessing differs across methods, or dense native decoder GPU dominates.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

DINOv3 already supports frozen single-backbone multipurpose predictions, so one heavy vision forward itself is not new. Task-specific output geometry varies; code may only compress a small head while baseline encoder dominates physical storage.

## U — measurement uncertainties

Class imbalance, spatially correlated pixels and scenes, geometry-specific target units, image patch resolution and augmentation leakage. Bootstrap by whole scene and sensor domain.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
