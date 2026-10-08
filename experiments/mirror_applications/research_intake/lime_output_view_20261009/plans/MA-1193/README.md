# MA-1193: DR-MGF task-route Mirror compression under gradient-deconflict controls

**RESEARCH INTAKE / UNTESTED / NOT WORKER CLAIMED**. Branch: `research/mirror-lime-output-view-20261009`. Interest P1 but **no special queue rank**, and the canonical active worker continues its preexisting candidate MA-255. Native family Multi-output gradient interference / task-preferred routes. Related already-registered plans: MA-1175;MA-1173;MA-299;MA-844. Prior-art IDs: PA468;PA469;PA470;PA463.

## 1. Native method and novelty firewall

**Actual native physical object:** native DR-MGF multi-output backbone with task-preferred filter routes/importance variables

**Mirror insertion:** Add the low-description address `m` at **short per-task m compressing trained filter-importance/path coordinates on the native route selection, not a new gradient optimizer**, so that K conflict-aware useful output functions from one physical network with compact task-specific path control. The native model and its prior specialization/routing mechanisms remain intact. An exact-gauge or ordinary Givens/diagonal reparameterization is **M0** (no incremental Mirror-specific benefit).

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

A compact structured m for native DR-MGF task-preferred inference routes preserves multi-task learning conflict reduction while using fewer paid route coordinates and faster one-shared-forward outputs than native full per-filter routing and equal-byte lowrank gates.

## T — standalone reproducible execution

Small d64 shared multitask model with 5 independent binary targets on same x, plus deliberately contradictory label/missing shared feature worlds. Native DR-MGF task-filter importance variables and its meta-weighted gradient fusion must be retained and verified before insertion. Fit source-only route dictionary, per-task m dim 4/8 with lowrank/Givens structured path factors and an ordinary same-byte linear gate; compare native full DR-MGF, PCGrad projection and CAGrad, static task heads, fixed-route and random equal-byte gates. Hold out entire label-task/data-generation families, not merely random samples, and measure loss, route saliency, signed per-task shared-parameter gradient dot product and backprop time. For DR-MGF natural stage verify original TPAMI source and CIFAR/NYUv2 benchmark; record task gradient fusion overhead and CUDA kernel performance. Do not call reduced gradient conflicts independent model capacity.

**Native and simpler comparators, all mandatory:** native DR-MGF per-filter route variables and native gradient fusion, PCGrad, CAGrad, naive equal-weight MTL with ordinary multihead, same-byte low-rank task gates, Dense task heads.

**Data firewall**: development seeds 11,12,13 only for support/basis and hparam choices. Before any fresh evaluation commit a separate machine-readable frozen source+dataset manifest. Evaluate **all fresh seeds 101..105** without outcome-based selection or early stopping and disjoint whole-task/domain identities, not just random tokens. Existing public-data repeated partitions do not constitute independent world replication; require separate natural holdouts. If these seed identities have been audited in a related study, preregister new disjoint seeds (e.g., 601..605) instead of claiming first-seen replication.

**Hardware**: CPU torch/NumPy first to check exact k/rank/gradient/alias conditions; natural method under its released official implementation and licenses/weights on GPU separately. Record source Git SHA and pinned checkpoint/dataset/hash. If native method cannot be faithfully reproduced, label **BLOCKED**; do not replace the original with a weaker caricature and call it reproduced.

**Measurements**: per-task NLL/worst task, signed gradient-cosine conflict fraction, number of active filter paths and route diversity, retention, paid bytes, active per-task CPU/GPU P95. Count trained decoder, shared-basis construction/offline work, role codes, per-task private residual, actual serializer metadata, router/index and KV/activation state. Measure real P50/P95 after warmup/synchronization and active MAC/FLOP instead of using nominal model-size ratio as speed.

## D — acceptance and rejection

**PASS (small mechanism only)**: At least 4/5 independent fresh task-worlds within +0.02 nat/sample and <=2 percentage point worst-task accuracy of native DR-MGF, route-state bytes <=0.85 native, whole model <=0.97 native, P95<=1.10, and a strict quality-byte Pareto improvement vs PCGrad/CAGrad+ordinary rank4 route code.

**FAIL/M0**: PCGrad, CAGrad or native full DR-MGF/PFN task gate performs as well or better; m causes worse task interference and forgotten tasks or route code basis leaks target task audit labels.

**UNCERTAIN**: Native control unavailable, audit/source leakage, ambiguous task split, no actual bytes or native throughput, rank/solver numerical instability, or paired interval spans the declared gain. Full MA remains UNTESTED unless registered and independently replayed.

## C — adversarial counterexample

Task-preferred paths and meta-weighted gradient fusion are DR-MGF's established contributions. The remaining route code may contain little redundancy and task interference is an optimization issue, not a Mirror coordinate deficiency.

## U — uncertainty, limits, reproduction

Gradient norms/rank, correlation of task labels, different task classes and optimization steps; group by independently generated entire tasks/datasets and report loss/bytes/runtime separately. If error components independent only, `u_c²=u_seed²+u_eval²+u_num²`; else include all covariance terms. `U_exp=k_cov*u_c` with k_cov=2 is merely indicative at n=5, not guaranteed 95%. Reproducibility requires code, exact source/checkpoint hashes, frozen split/preprocessing and every raw fresh row. Same-batch heavy-trunk recomputation of K outputs is **not** a strong baseline against native single-forward multihead. A toy aligned task success does not imply general independent functionality or 46×/64× practical compression.

**Scope**: This directory is a design. It contains no completed experiment results. MA scientific status and WORKER_QUEUE must not change automatically.
