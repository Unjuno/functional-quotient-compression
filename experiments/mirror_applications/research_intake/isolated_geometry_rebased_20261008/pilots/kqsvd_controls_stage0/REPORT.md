# KQ-SVD Stage-0: strong query-aware score control (2026-10-08)

**Result scope: CPU synthetic algebra and serialization only.** Frozen design lives in [PROTOCOL.json](PROTOCOL.json), committed before any development/fresh data were generated. This study is NOT the MA-1166 full native model, is not a trained LLM and does not change any MA status or canonical worker queue.

## H / statement being checked

For a rank-4 low-rank projector with source-calibrated query distribution, approximate Query–Key score interactions rather than independently truncating Keys. The audit-calculated rank-4 SVD serves only as a **non-deployable mathematical lower bound**. A source-only learned bank with ordinary linear code is the direct same-byte non-Mirror control; a structured Mirror View is **not tested in this pilot**.

## T / actual run

- CPU Linux x86-64, NumPy 2.3.5, OpenBLAS threads 1, float64 SVD/algebra, float32 NPZ inference serialization.
- Key cache: 128 × 16, 6 calibrated source query-role families, 16-dimensional head, rank 4, two code coefficients, source queries 96 per role, unseen target role support 12 queries, audit 96 disjoint queries.
- Three development seeds: 11, 12, 13. Frozen five fresh seeds: 101–105. Source dictionary is derived exclusively from source role calibration. Target code is fit from permitted target **support** full scores, which is an expensive calibration procedure; its cost is **not free**.
- The full audit Query–Key score SVD is inspected **only after** code fitting as an oracle lower bound. It is not an implementable projector selected using heldout data.
- Reproduce: `OPENBLAS_NUM_THREADS=1 python source/run_stage0.py --output results`. From the source directory, `python test_stage0.py` runs four tests.

## D / fresh measurements

| Quantity (five fresh worlds) | Mean | Min–max | Interpretation |
|---|---:|---:|---|
| Query–Key audit score rank-4 SVD bound, relative Frobenius | 0.415758 | 0.294663–0.591427 | Oracle bound, no deployable use |
| Key-only rank-4 SVD score error | 0.901884 | 0.791251–0.956977 | Native weak comparison |
| Source-mean projector score error | 0.675945 | 0.611737–0.765391 | Simple shared baseline |
| Source-trained target-support native low-rank projector | 0.563795 | 0.386627–0.812151 | Uses target support only |
| Ordinary *linear* 2-value task code | 0.534069 | 0.476358–0.595508 | Exactly the proposed linear-m form; **NOT Mirror-specific** |
| Real named NPZ, physical K alone | 8,448 B | — | Same fixed serial format |
| Real named NPZ, native seven-projector bank + K | 15,886 B | — | Strong reference state |
| Real named NPZ, linear dictionary + task code bank + K | 12,580 B | — | Simple factorization, NOT Mirror |

**Algebra gate:** PASS. Across all five fresh worlds, the source-calibrated rank-4 projection factor reproduces its source score-space rank-4 SVD target to at most 7.3e-15 relative, and the heldout oracle bound never exceeds Key-only rank-4 score error. Four local unit tests PASS. Result quality / actual byte superiority of structured Mirror: **NOT EVALUATED** (no Mirror-specific PASS).

**Capacity caution:** The linear-code model and a Mirror described only as linear coordinates have the **same model class** in this pilot. The 20.8% serialization saving relative to a naive seven-projector bank cannot be attributed to a new Mirror geometry, and byte counts are for this particular synthetic bank and serializer, not a complete production LM/cache stack.

## C / counterargument

An attention-score optimal rank-4 oracle uses heldout audit queries and is unrealistically strong if treated as a deployable control. Conversely, the proper native query-aware deployable KQ-SVD projector is fitted on permitted source/support queries; it can have more heldout score error than an audit oracle. Attention-score Frobenius approximation does not guarantee downstream softmax/NLL ordering. A shared dictionary may compress trivially because target query families were generated from common structured covariances.

## U / uncertainty and limitations

Exactness maximum 7.3e-15 in float64 for source calibration; five fresh worlds are insufficient for a confident population effect. We publish every paired result with seeds rather than extrapolate significance. Runtime/kernel benefits are **not measured** and must not be inferred. Tokens are synthetic, not a pretrained LLM; no inter-token causal network, no physical cache alias test. `np.savez` counts named array headers but not trained-model metadata not used in this fixture.

### Variables and SI dimensional check

| Symbol | 日本語の意味 | 単位 | 定義・範囲・型 |
|---|---|---|---|
| Q | Query集合 | 1 | 実 n_query×16 行列 |
| K | Keyキャッシュ | 1 | 実 128×16 行列 |
| P | 射影作用素 | 1 | 実16×16行列、rank<=4 |
| m | 共有基底のtask係数 | 1 | 実2成分ベクトル |
| r | 低ランク近似のrank | 1 | 整数4 |
| E | 相対スコア誤差 | 1 | 有限非負実数（Frobenius比） |
| S | NPZ推論補助容量 | byte（非SI実用単位） | 非負整数、実ファイルサイズ |

Dimension check: `Q P K.T` has dimension `n_query × 128`, equal to native score `Q K.T`; both are dimensionless from normalized activations. `E` is a ratio of same-unit Frobenius norms; byte S is independent and cannot be combined arithmetically with E or latency.

## Exact provenance and reconstruction

- Full source: [run_stage0.py](source/run_stage0.py).
- Unit tests: [test_stage0.py](source/test_stage0.py).
- [Development rows](results/dev_raw.csv), [all five fresh rows](results/fresh_raw.csv), [hashes/runtime](results/VERIFICATION.json).
- Verification SHA-256: source `e98dcddbe38a681d247d189ec42bdc2ca5f6f540b0d2edc453a9138a603983ff`; dev CSV `015a1db0020ae71989dff0ac5573c7e9d42839fb57521622c23e47f62f9a79ba`; fresh CSV `b5983c135a16f9741d03dde7648d2b87eaa86f02bd275214570050eafa8241f5`.
- Local machine/run environment is in verification; the reproducible source is also packaged as a downloadable conversation ZIP.
- Original native paper: https://proceedings.mlr.press/v300/lesens26a.html .

No newer Mirror MA-1166 status is claimed; the next valid experiment requires adding and evaluating a genuinely *different structured m*, matched-byte ordinary factorization, and a natural trained-model task.
