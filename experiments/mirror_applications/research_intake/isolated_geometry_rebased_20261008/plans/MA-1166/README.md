# MA-1166 — KQ-SVD multi-context Mirror query-conditioned cache View
**Status: UNTESTED / DESIGN ONLY.** Quarantined on `research/mirror-isolated-protocols-rebased-20261008`, worker queue not changed. Native method is a required direct baseline; no paper result may be called a Mirror result.

## Variables and units
| Symbol | 意味（日本語） | SI単位 | 定義・条件 | 型 |
|---|---|---|---|---|
| m | Mirror機能コード | 1 | 有限な長さkの係数 | 実ベクトル |
| k | コード次元 | 1 | 正の整数 | スカラー |
| W | 共有線形重み | 1 | d_out×d_inの正規化重み | 実行列 |
| B_i | View残差基底 | 1 | Wと同じ形状・無次元 | 実行列 |
| K | Keyキャッシュ | 1 | T×d実行列 | 実行列 |
| Q | Query行列 | 1 | n_query×d実行列 | 実行列 |
| S | 全直列化容量 | byte（実用単位） | 全basis/code/header/index込み非負整数 | 整数スカラー |
| t | 推論遅延 | s | 非負の同期測定値 | 実スカラー |
| L | token当たり交差エントロピー | nat/token（無次元） | 監査集合上の非負実数 | スカラー |
| u_seed,u_eval,u_num | 不確かさ各成分 | Lと同単位 | 非負、独立性を検証 | 実スカラー |
| u_c | 合成標準不確かさ | Lと同単位 | 共分散込み | 実スカラー |
| k_cov | 包含係数 | 1 | 正の実数、説明用2 | スカラー |
| U | 拡張不確かさ | Lと同単位 | k_cov×u_c | スカラー |
Dimensional check: any residual View uses `W + sum(m_i B_i)`, valid only for same-shaped dimensionless matrices. Score `Q @ K.T` is n_query×T dimensionless. Memory S is bytes (not SI fundamental); runtime t is seconds. Quality and runtime errors cannot be combined numerically.

## H — Falsifiable hypothesis
For heldout query roles, a shared source-KQ-SVD dictionary and small m attains attention-score RMSE <=1.05x per-role native KQ-SVD at complete cache+projector bytes <=0.90 native, with <=1.10 P95 latency and <=0.02 nat/token target NLL regression when tested on a causal LM.

## Mirror insertion
> **Mirror insertion:** Add small `m` to **canonical physical K/V cache and source-calibrated query-aware KQ-SVD projector bank** through **query-family k-dimensional m selecting a structured query-side readout without copying physical cache**, so native physical objects need not be independently repeated.
- **Existing/native:** per-role native KQ-SVD query-key low-rank approximation with source-only calibration.
- **Incremental objective:** measure benefit of m beyond native low-rank sharing. Related registered: MA-691, MA-692, MA-879.
- **Not an invention:** xKV and KQ-SVD already perform the described native compression.

## T — Self-contained experiment
**Fixture:** CPU synthetic K:128x32, 4 source query-role covariance families and 2 whole heldout query roles with distinct spectra; optional frozen tiny LM calibration with role and prompt identities disjoint from audit.
**Implementation:** Native control minimizes query-key score approximation, not raw key weight error. First calculate exact rank-r SVD of source score matrix Q@K.T as an oracle bound, then source-only implementable KQ-SVD projection, then K-only SVD. Fit task m to source projectors; contrast ordinary linear code versus structured rotation. Score on heldout Q before/after softmax and on attention output. Validate bad RoPE-noncommuting transforms and counterfactual long-tail queries.
**Controls:** 1. native per-role KQ-SVD with full paid projector bytes 2. one shared native KQ-SVD projector 3. K-only SVD 4. same-byte linear/diagonal task-code projector 5. INT4/INT8 full-dimensional KV at equal physical bytes
**Training split:** Seeds 11, 12, 13 select hyperparameters on source tasks only. Fresh seeds 101, 102, 103, 104, 105 and whole role/task identities are locked and never used for hyperparameter selection. Fix source-model revision, permutation/order, rank budget and eval sampler before audit. Run all five fresh worlds if the dev gate passes; no favorable early stop.
**Hardware & backend:** NumPy float64 algebra and torch float32 CPU first; CUDA causal LM only if available and recorded.
**Measurements:** score RMSE; attention output/logit error; heldout NLL if LM; physical cache aliases and bytes; actual projector bank bytes incl metadata; P95 attention time and active FLOPs. Log total paid inference state, not a raw coefficient count. Serializers must count headers and indices; CPU/GPU latency, active MAC/FLOP and training updates are separate.

## D — PASS / FAIL / UNCERTAIN
**PASS (exploratory only):** 4/5 fresh query roles score RMSE <=1.05 per-role native at <=0.90 total bytes and <=1.10 P95, and a strict Mirror-specific advantage over same-byte plain linear code on an independent axis; conditional NLL <=+0.02 nat/token.
**FAIL:** Native linear code matches/surpasses, audit Q used to tune projection, RoPE-invalid change claimed exact, or resident KV actually cloned.
**UNCERTAIN:** native method or true memory/latency cannot be reproduced, underpowered task-identity holdout or uncertain numerical rank. Never call capacity or ADOPTED at n=5; require a separate >=10 fresh-world replication and natural workload.

## C — Falsification mechanism
Per-role KQ-SVD is a strong rank-constrained optimum and cannot be beaten on the same fitting objective without more information, rank or storage; multi-role amortization is the only defensible incremental hypothesis.

## U — Uncertainty
For a quality metric in nat/token, use `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)` only if independent; include covariance otherwise; `U = k_cov*u_c`, k_cov=2 is indicative, not guaranteed 95% coverage at n=5. Bootstrap by task/world (not by correlated token only); report all paired seeds, FP64/FP32 max error, and wall-clock repeat/clock uncertainty in seconds separately. Freeze before fresh.

## Source-ready run order
1. Reproduce native score/kernel or post-hoc source-only compressor on the specified tensor shapes.
2. Verify score equality/invariance and invalid counterexamples in float64.
3. Fit a simple linear/native control then structured m on source/dev only.
4. Fix ranks, step counts, selection masks and evaluation manifest; never leak target oracle factors.
5. Run fresh 5 worlds, serialize all paid inference objects, measure accuracy and active runtime.
6. Record FAIL even if the synthetic aligned mechanism works but natural tasks or native controls fail.

## Mandatory primary papers
- PA426
- PA425
- PA432
- xKV: https://proceedings.mlr.press/v306/chang26d.html (official code https://github.com/abdelfattah-lab/xKV)
- KQ-SVD: https://proceedings.mlr.press/v300/lesens26a.html

**Output contract:** only upon activation create RESULTS_CORE.csv, VERIFICATION.json, source and tests; this directory is currently a protocol, not a performed experiment.
