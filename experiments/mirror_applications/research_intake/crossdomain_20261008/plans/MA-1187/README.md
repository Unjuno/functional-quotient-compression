# MA-1187 — DISCO neural-operator splitting with low-description Mirror composition codes

> **ISOLATED RESEARCH DESIGN / UNTESTED.** No data or trained-model output from this MA has been evaluated. Native paper baselines must be reproduced honestly. This is one candidate in a breadth portfolio, never the next worker instruction.

## Source and uniqueness

- Native paper: https://proceedings.mlr.press/v306/serrano26a.html
- Official code/model: https://github.com/LouisSerrano/neural-operator-splitting
- Native method as published: DISCO ICML2025 learns small evolution operators emitted from trajectory using a hypernetwork; ICML2026 neural operator splitting already searches test-time operator compositions without weight updates. 'Compose learned physics rules' is native; Mirror must improve code storage/execution at equal dynamics fidelity
- Native literature references: PA458, PA462, PA459, PA461.
- Nearest pre-existing candidate IDs: MA-731, MA-732, MA-818, MA-1160, MA-1173.
- Minimal new unit: Mirror m is not the native model, symmetry, member rank-1 parameter, shared decoder, test-time composition, or pretraining. Its marginal functional utility must beat **already cheap native alternatives**.

## H — preregisterable hypothesis

A Mirror-structured compact operator dictionary could reduce paid per-equation evolution/operator code without losing native DISCО/test-time-splitting PDE rollout fidelity, and beat ordinary same-byte low-rank operator codes and Strang splitting under noncommuting physics; exact group symmetries alone cannot create unseen dynamics.

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

**Native shared object:** native DISCO source-trained dictionary of small learned physical update operators plus its original test-time search/splitting method.

**Mirror insertion:** small compositional m encoding source operator coefficients or operator-pair condition and schedule in a shared decoder, explicitly preserving operator *order* and nonlinear solver steps.

Example generic interface: (z=N_θ(x), y_j=D_{mathrm{native}}(z;,m_j)), or, where native accepts task code, (D_{mathrm{native}}(z;,c_j)=D_{mathrm{native}}(z;,C_m m_j)). Every learned matrix in (C_m), output head, temporary factor, routing index and m requires paid serialization and computation; comparing only code bytes is invalid. For non-commuting operators an explicitly ordered product is required. Only when the native intermediate is sufficient for the desired output does a second heavy forward become unnecessary.

**Shape check:** All m transforms must act on a named finite-dimensional native feature/core and have matching in/out widths; any task-specific final projection from the feature space to a different output dimension remains paid. Physical targets such as atmospheric temperatures, velocities and concentrations use their original units; no global sum of heterogeneous raw RMSE is allowed.

## T — exact independent worker recipe

**Synthetic fixture (does not imply native reproduction):** 1D periodic spatial state n=16/64, source advection A, diffusion B and spatial heterogeneous reaction C, operators with explicit units 1/s; 8 source coefficient combinations and heldout unseen A+B+C regimes, dt={0.05,0.1,0.2}s and 1/10/100 steps. Start with MCX Stage0's exact expm Lie/Strang/BCH controls and add learned low-rank operator decoder code (K={2,4,8}); freeze train PDE trajectories and fresh equations by coefficient family. Natural stage official DISCO dataset/code, ERA5 not conflated.

**Implementation/measurements:** 1. Reproduce native DISCO operator hypernetwork output and 2026 test-time beam/ODE operator split with same operator source and beam width; source trajectory identities separated from future test dynamics. 2. Compress native learned dictionary with source-trained low-rank/structured Mirror m and identical native search. 3. Compare full native model, ordinary low-rank operator dictionary, direct operator-code quantization, parallel additive mixing, exact native Strang/Lie splitting, learned task-conditioned FNO/GEPS/MPP/ZEBRA where released. 4. Evaluate unseen coefficients, complete physical rollouts, conservation/stability (when appropriate), task error versus true solver and active evaluation steps, actual serialized operator dictionary+hypernet+code bytes. 5. Prohibit claiming single forward when K sequential nonlinear physics steps remain.

**Strong baselines to implement BEFORE calling a Mirror gain:**

1. Original DISCO native hypernetwork operator and dictionary
2. Native 2026 neural-operator test-time splitting method incl beam search
3. Lie and stronger Strang/symmetric splitting and BCH/commutator corrected reference where solvable
4. Ordinary same-byte tensor/SVD linear code dictionary and grouped decoder
5. FNO/GEPS/MPP/Zebra native PDE prediction with comparable task/prompt data
6. Independent trained per-PDE operator and full numerical solver upper fidelity reference

**Data firewall:** use source/dev seeds [11,12,13] for code rank/LR/early stopping only and **fresh [101,102,103,104,105]** as unseen independent whole task/data-generating regime sets. Do not use heldout role deltas/labels when constructing m basis. Freeze preprocessing, target feature order, tokenizer and code hash before opening fresh. Natural benchmark is a separate lane with preregistered family/time/station/site/group split and model/license/checkpoint hash; do not recycle already opened synthetic fresh worlds.

**Measured endpoints:** relative L2 state/rollout error vs true PDE, conserved mass/energy residual with original SI physical units, extrapolation at heldout PDE coefficients, actual dictionary/model/code bytes and beam search compute, latency P50/P95 and stability through 100 steps. Also record per-task output difference and worst-task utility, training examples/tokens and optimizer steps, actual serializer file bytes, resident CPU/GPU memory, active MAC/FLOP estimates, wall P50/P95 after warmup/sync, and source-training vs inference amortization. A repeated native heavy inference is never the sole baseline for a native one-pass multihead/ensemble.

**Compute resources:** CPU numpy/scipy exact linear n16 stage0 done; CPU small torch learned operator stage1; full DISCO neural operator GPU and released datasets stage2. Benchmark GPU synchronously if claiming acceleration.

## D — decision thresholds

- **PASS preliminary mechanism:** At >=4/5 heldout PDE families, <=1.05x native test-time splitting true solver relative error, total model+operator bank bytes <=0.85 native (or >=20% operator-bank saving with honestly small whole-model improvement), <=1.10x P95, and strict superiority over Strang+ordinary low-rank code at matched accuracy and compute.
- **FAIL / reject this code family on tested task:** Native symmetric splitting or ordinary low-rank dictionary dominates, order effects ignored or new dynamics inferred from target audit trajectory, conservation fails, step count/memory hidden; synthetic commutator correction alone is not a PASS.
- **UNCERTAIN/BLOCKED:** missing official native weights/code, license/data restrictions, uncalibrated native baseline, fresh leakage, insufficient independent worlds, undefined physical units or real GPU missing for runtime claim. Report missingness explicitly.
- **Adoption:** Five fresh synthetic seeds are merely a mechanism screen. Require >=10 *new independent* task/world units, public real task benchmark, native-equivalent function quality, strictly Pareto-superior real bytes/latency, and qualified uncertainty before adopting.

## C — strongest counterargument

Operator splitting and dictionary composition are mature numerical/learning methods; a Mirror code may be only an inconvenient coordinate of the original dictionary, while noncommuting dynamics demand more active operator evaluations than a compact code suggests.

## U — measurement uncertainties

PDE solver discretization error, stability, spatial boundary conditions, learned operator inductive bias, choice of time step, noncommuting residual, extrapolation outside training coefficients; statistical unit is a PDE family, not grid cells.

For quality L, if independent only, `u_c²=u_seed²+u_eval²+u_num²`; otherwise add twice covariances. Indicative `U_exp=k_cov u_c` with k_cov=2 **is not** automatically a true 95% interval with five worlds. Report paired whole-family bootstrap ranges, unit/numerical roundoff separately. Serialized S in byte and measured τ in SI seconds are separate Pareto axes, not dimensionless loss terms.

## Claim firewall and expected evidence files

This design creates only `README.md`, `PROTOCOL.json`, `STATUS.md`. No `RESULTS_CORE.csv`/MA-`VERIFICATION.json` exists yet. If activated, freeze separate source/test/checkpoint hashes on a new research-only experiment branch before any fresh test, run all world IDs, preserve negative results, benchmark strongest native and equal-byte controls, and never auto-edit canonical worker queue/main.

## Related stage-0 algebraic audit

- `../../pilots/mcx_stage0/PROTOCOL.json` — frozen before numeric checks.
- `../../pilots/mcx_stage0/REPORT.md` — RC, input-fast-weight and noncommutative splitting algebra only, **not** a trained native benchmark.
