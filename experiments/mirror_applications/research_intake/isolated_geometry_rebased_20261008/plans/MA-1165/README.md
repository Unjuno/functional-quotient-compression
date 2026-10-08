# MA-1165 — xKV cross-layer shared-basis Mirror reconstruction codes
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
At fixed native xKV rank and selector, Mirror layer m reduces complete physically resident compressed KV state >=10% relative to xKV-SR, with <=0.02 nats/token NLL increase and P95 <=1.10x native, while beating a same-byte ordinary linear coefficient bank.

## Mirror insertion
> **Mirror insertion:** Add small `m` to **native xKV post-hoc shared token basis and per-layer KV reconstruction coefficients** through **layer/head m encoding additional shared reconstruction dictionary**, so native physical objects need not be independently repeated.
- **Existing/native:** xKV-SR full native cross-layer SVD, selective token reconstruction.
- **Incremental objective:** measure benefit of m beyond native low-rank sharing. Related registered: MA-582, MA-695, MA-691.
- **Not an invention:** xKV and KQ-SVD already perform the described native compression.

## T — Self-contained experiment
**Fixture:** CPU source-only synthetic 8-layer, 128-token x 32-channel KV with low-rank signal plus independent perturbations; 4 unseen entire layer-role families. Natural follow-up: frozen tiny LM with heldout prompts and matched checkpoint.
**Implementation:** Compute native grouped-layer xKV shared token basis with thin SVD; keep it physically identical for all conditions. Factorize only native per-layer reconstruction coefficient bank using source-only least squares/SVD. Train m (k=2,4,8) as diagonal/short Givens coefficient changes. Reconstruct the exact same selected token positions as native xKV-SR. Do not compare a dense xKV to sparse Mirror.
**Controls:** 1. native xKV-SR matched group/rank and sparse token selector 2. ordinary truncated linear factorization of xKV reconstruction coefficients 3. native xKV dense readout 4. independent per-layer SVD 5. bitpacked INT4 native KV at same whole-cache bytes
**Training split:** Seeds 11, 12, 13 select hyperparameters on source tasks only. Fresh seeds 101, 102, 103, 104, 105 and whole role/task identities are locked and never used for hyperparameter selection. Fix source-model revision, permutation/order, rank budget and eval sampler before audit. Run all five fresh worlds if the dev gate passes; no favorable early stop.
**Hardware & backend:** CPU torch float32, NumPy float64 SVD; production xKV-SR CUDA kernel optional. Report GPU type, clock, batch, context and fused-kernel status; CPU eager benchmarks are not xKV-SR throughput proof.
**Measurements:** heldout NLL; KV reconstruction/logit error; physical cache bytes; actual NPZ/safetensors total bytes incl token basis, dictionary and m; P50/P95 decode seconds; SVD and selection FLOPs. Log total paid inference state, not a raw coefficient count. Serializers must count headers and indices; CPU/GPU latency, active MAC/FLOP and training updates are separate.

## D — PASS / FAIL / UNCERTAIN
**PASS (exploratory only):** 4/5 fresh worlds <=+0.02 nat/token NLL, median complete KV bytes <=0.90 native xKV-SR, median P95 <=1.10 native; simple native linear coefficient code not Pareto-dominant.
**FAIL:** Source-only linear coefficient factorization ties or beats m; lost native selector advantage, miscounted basis/metadata, or NLL fails >=2/5 worlds.
**UNCERTAIN:** native method or true memory/latency cannot be reproduced, underpowered task-identity holdout or uncertain numerical rank. Never call capacity or ADOPTED at n=5; require a separate >=10 fresh-world replication and natural workload.

## C — Falsification mechanism
xKV already shares most dominant token state, so remaining layer reconstruction bank may be a small fraction; arbitrary codes add decode work.

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
- PA425
- PA426
- PA432
- xKV: https://proceedings.mlr.press/v306/chang26d.html (official code https://github.com/abdelfattah-lab/xKV)
- KQ-SVD: https://proceedings.mlr.press/v300/lesens26a.html

**Output contract:** only upon activation create RESULTS_CORE.csv, VERIFICATION.json, source and tests; this directory is currently a protocol, not a performed experiment.
