# MA-1191: FAAR task-spectrum View code under native rank shrinking

**RESEARCH INTAKE / UNTESTED / NOT WORKER CLAIMED**. Branch: `research/mirror-lime-output-view-20261009`. Interest P1 but **no special queue rank**, and the canonical active worker continues its preexisting candidate MA-255. Native family FAAR adaptive multi-task decoder / frequency-aware ranks. Related already-registered plans: MA-1185;MA-1081;MA-1172;MA-314. Prior-art IDs: PA466;PA470;PA468.

## 1. Native method and novelty firewall

**Actual native physical object:** native FAAR pretrained Swin backbone, native Performance-Driven Rank Shrinking (PDRS) adapter ranks and Task-Spectral Pyramidal Decoder (TS-PD)

**Mirror insertion:** Add the low-description address `m` at **task × frequency-band × feature-scale small m used only in TS-PD spatial residual readout after native PDRS allocation**, so that multiple dense image tasks or scales with fewer stored task spectral head residuals. The native model and its prior specialization/routing mechanisms remain intact. An exact-gauge or ordinary Givens/diagonal reparameterization is **M0** (no incremental Mirror-specific benefit).

## Symbols, units and dimensional check

| Symbol | 意味（日本語） | SI単位（実用単位） | 型と範囲 |
|---|---|---|---|
| x | 入力 / token / 画像 | 1 | 正規化された有限実ベクトルまたは配列 |
| theta | 共有のネイティブ学習重み | 1 | shape整合の実行列/テンソル |
| m | 追加Mirror機能コード | 1 (角度rad) | R^k、kは正整数 |
| K | 有用な論理role/Expert数 | 1 | 正整数 |
| L | タスク固有損失/NLL | nat/token または nat/例（SI=1） | 非負実数、native分母固定 |
| S | 推論時の物理直列化容量 | byte (非SI実用単位) | 非負整数、basis/decoder/metadata込み |
| t | 実測推論時間 | s | 非負実数、同期・固定batch |
| u_seed,u_eval,u_num | seed/標本/浮動小数点の標準不確かさ | Lと同じ単位 | 非負実数 |
| u_c | 合成標準不確かさ | Lと同じ単位 | 相関を計算する実数 |
| k_cov | 被覆係数 | 1 | 正数（仮に2） |
| U_exp | 拡張不確かさ | Lと同じ単位 | k_cov*u_c |

**Shape check:** m is allowed to multiply only a compatible feature tensor, matrix or native coefficient core; under a full output-orbit f_m(x)=g_m(r(x)), r must be sufficient on all inputs (fiber condition). Runtime and physical bytes are separate Pareto axes; never sum NLL, seconds, bytes into one unscaled scalar.

## H — falsifiable hypothesis

Replacing only FAAR's task-by-band residual decoder coefficients with a low-description m preserves task-specific dense prediction quality at fewer total bytes than FAAR's already dynamic-rank native optimizer and a same-byte simple spectral code.

## T — standalone reproducible execution

First synthetic FFT mechanism: 32x32 images of mixed smooth/edge/textures, four tasks with distinct segmentation/normal/depth targets, 3 train-worlds dev and 5 fresh unseen image-generator worlds. Keep native fixed PDRS-like rank masks under source-only calibration, then learn a paid common FFT-band decoder, per-task Mirror m (block/signed/Givens) and same-parameter plain diagonal/FiLM/spectral-linear code; measure all task-native metrics separately, not one pooled RMSE. Stage natural: reproduce official FAAR TS-PD and PDRS using PASCAL-Context fixed train/val splits, same batch/rank budget and kernel; freeze feature pyramid, native task-specific adaptation and full decoder. Benchmark complete actual saved checkpoint bytes (decoder, rank lists, m, positional features, FFT basis if paid), activation bytes and GPU wall. Different tasks must use same input image; no oracle target spatial map to choose m.

**Native and simpler comparators, all mandatory:** full native FAAR PDRS+TS-PD, native FAAR with smaller rank, byte-matched ordinary FiLM/linear spectral readout, shared Swin independent task decoders, adaptive rank LoRA, no frequency View.

**Data firewall**: development seeds 11,12,13 only for support/basis and hparam choices. Before any fresh evaluation commit a separate machine-readable frozen source+dataset manifest. Evaluate **all fresh seeds 101..105** without outcome-based selection or early stopping and disjoint whole-task/domain identities, not just random tokens. Existing public-data repeated partitions do not constitute independent world replication; require separate natural holdouts. If these seed identities have been audited in a related study, preregister new disjoint seeds (e.g., 601..605) instead of claiming first-seen replication.

**Hardware**: CPU torch/NumPy first to check exact k/rank/gradient/alias conditions; natural method under its released official implementation and licenses/weights on GPU separately. Record source Git SHA and pinned checkpoint/dataset/hash. If native method cannot be faithfully reproduced, label **BLOCKED**; do not replace the original with a weaker caricature and call it reproduced.

**Measurements**: PASCAL-Context semantic/human segmentation mIoU, depth RMSE, normals angular error, actual decoder+basis bytes, GPU P95, old-task interference. Count trained decoder, shared-basis construction/offline work, role codes, per-task private residual, actual serializer metadata, router/index and KV/activation state. Measure real P50/P95 after warmup/synchronization and active MAC/FLOP instead of using nominal model-size ratio as speed.

## D — acceptance and rejection

**PASS (small mechanism only)**: On >=4/5 fresh task/data-generating worlds all task-native metrics within 2% relative native FAAR, total model+decoder bytes <=0.95 native, P95<=1.10 native, and strict gain over byte-matched native frequency/FiLM code.

**FAIL/M0**: FAAR rank shrinking or ordinary spectral task adapter outperforms Mirror, task-specific dense decoder capacity/geometry absent, or quality claims conflate mIoU/RMSE and reduce task fidelity.

**UNCERTAIN**: Native control unavailable, audit/source leakage, ambiguous task split, no actual bytes or native throughput, rank/solver numerical instability, or paired interval spans the declared gain. Full MA remains UNTESTED unless registered and independently replayed.

## C — adversarial counterexample

FAAR already compresses ranks and learns spatially aware fusion; additional m may only encode the remaining tiny task head, giving negligible whole-model saving.

## U — uncertainty, limits, reproduction

Different target units, class imbalance, spatial correlation and benchmark preprocessing, GPU fused kernels; stratify by independent image family. If error components independent only, `u_c²=u_seed²+u_eval²+u_num²`; else include all covariance terms. `U_exp=k_cov*u_c` with k_cov=2 is merely indicative at n=5, not guaranteed 95%. Reproducibility requires code, exact source/checkpoint hashes, frozen split/preprocessing and every raw fresh row. Same-batch heavy-trunk recomputation of K outputs is **not** a strong baseline against native single-forward multihead. A toy aligned task success does not imply general independent functionality or 46×/64× practical compression.

**Scope**: This directory is a design. It contains no completed experiment results. MA scientific status and WORKER_QUEUE must not change automatically.
