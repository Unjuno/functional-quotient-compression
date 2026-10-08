# MA-1190: M3LoRA task-mixer Mirror coordinate compression

**RESEARCH INTAKE / UNTESTED / NOT WORKER CLAIMED**. Branch: `research/mirror-lime-output-view-20261009`. Interest P0 but **no special queue rank**, and the canonical active worker continues its preexisting candidate MA-255. Native family Multi-subspace PEFT / M3LoRA. Related already-registered plans: MA-1099;MA-1167;MA-1179;MA-1105. Prior-art IDs: PA464;PA465;PA443;PA428.

## 1. Native method and novelty firewall

**Actual native physical object:** Native M3LoRA multiple low-rank A_i/B_j subspace factors initialized in minor singular components of pretrained W

**Mirror insertion:** Add the low-description address `m` at **short task-specific m on native learnable M3LoRA subspace mixing matrix L, with optional rank1 genuinely private residual**, so that multiple useful task deltas from a shared paid M3LoRA low-SV bank. The native model and its prior specialization/routing mechanisms remain intact. An exact-gauge or ordinary Givens/diagonal reparameterization is **M0** (no incremental Mirror-specific benefit).

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

After M3LoRA's genuine multi-subspace and minor-SV initialization are retained, low-description structured Mirror mixing codes compress multiple L_t mixers more efficiently than native low-rank PCA coefficient banks, without removing independently useful task directions.

## T — standalone reproducible execution

Native M3LoRA stage: freeze same pretrained 64x64 W, construct source-only SVD and minor singular bases, 4 pairs rank4 each, native mixed update ΔW_t=Σ_ij L_t,ij B_i A_j with source-bank train; hold out entire new tasks and source-covariate shifts. Train native independent L_t, source-trained plain tensor/SVD mixing code, 4-angle Givens Q(m) L0 Q(m)^{-1}, low-rank/simple diagonal code and rank1/2 private residual; use exactly same original W/base, labels, optimizer steps and initialization. Sweep learned task count 4/8/16 with actual shared-bank amortization, true float32 end-to-end task NLL and generalization and function-space projection; prohibit using target test LoRA deltas for m fitting. Compare to M3LoRA official released natural benchmark separately, or label native BLOCKED. Gauge-invariant task function values instead of raw factor differences; count all singular basis storage and source construction time.

**Native and simpler comparators, all mandatory:** full native M3LoRA task-specific mixing L, equal-byte ordinary SVD/linear task mixers, native minor-SV LoRA, RanLoRA/VeRA, LoDA shared/private, independent LoRA.

**Data firewall**: development seeds 11,12,13 only for support/basis and hparam choices. Before any fresh evaluation commit a separate machine-readable frozen source+dataset manifest. Evaluate **all fresh seeds 101..105** without outcome-based selection or early stopping and disjoint whole-task/domain identities, not just random tokens. Existing public-data repeated partitions do not constitute independent world replication; require separate natural holdouts. If these seed identities have been audited in a related study, preregister new disjoint seeds (e.g., 601..605) instead of claiming first-seen replication.

**Hardware**: CPU torch/NumPy first to check exact k/rank/gradient/alias conditions; natural method under its released official implementation and licenses/weights on GPU separately. Record source Git SHA and pinned checkpoint/dataset/hash. If native method cannot be faithfully reproduced, label **BLOCKED**; do not replace the original with a weaker caricature and call it reproduced.

**Measurements**: per-task OOD NLL/CE, old-task retention, task gradient conflict and subspace angles, total source-trained A/B/L/code and private bytes, actual P95. Count trained decoder, shared-basis construction/offline work, role codes, per-task private residual, actual serializer metadata, router/index and KV/activation state. Measure real P50/P95 after warmup/synchronization and active MAC/FLOP instead of using nominal model-size ratio as speed.

## D — acceptance and rejection

**PASS (small mechanism only)**: For >=4/5 fresh whole task worlds, task NLL <= native M3LoRA+0.02 nat/label, whole adapter-bank bytes <=0.85 native, P95<=1.10 native and paired strictly better quality/byte Pareto than ordinary same-byte SVD/linear L code and native minor-SV LoRA.

**FAIL/M0**: Native learned mixing L or compressed simple tensor-code beats Mirror, interference/old-task loss worsens, basis source/target leaks, source-bank build outweighs actual saved state at tested K.

**UNCERTAIN**: Native control unavailable, audit/source leakage, ambiguous task split, no actual bytes or native throughput, rank/solver numerical instability, or paired interval spans the declared gain. Full MA remains UNTESTED unless registered and independently replayed.

## C — adversarial counterexample

The learnable subspace mixture and low-SV adaptation are M3LoRA native. Mirror may merely constrain the same mixer and lose function directions, while minor singular modes can be task-useful despite low magnitude.

## U — uncertainty, limits, reproduction

Singular-value degeneracy/gauge, optimizer/old-task interference, correlated tasks; source-only eigenspaces, paired whole task-world bootstrap; report off-orbit failures. If error components independent only, `u_c²=u_seed²+u_eval²+u_num²`; else include all covariance terms. `U_exp=k_cov*u_c` with k_cov=2 is merely indicative at n=5, not guaranteed 95%. Reproducibility requires code, exact source/checkpoint hashes, frozen split/preprocessing and every raw fresh row. Same-batch heavy-trunk recomputation of K outputs is **not** a strong baseline against native single-forward multihead. A toy aligned task success does not imply general independent functionality or 46×/64× practical compression.

**Scope**: This directory is a design. It contains no completed experiment results. MA scientific status and WORKER_QUEUE must not change automatically.
