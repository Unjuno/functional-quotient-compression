# MA-1156 Stage-0: Gauge-only Mirror directions provide zero functional multiplicity

**Status:** math/finite-difference audit PASS on 5/5 fresh worlds, **no Mirror-specific performance result**. Canonical MA-1156 remains UNTESTED because this is only an independent Stage-0 scalar score Jacobian diagnostic. No task model was trained.

## Hypothesis and scope

For Q and K of shape 4×2, both full column rank, define the pre-RoPE score coefficient C0=QK^T; adding a nontrivial 2D RoPE rotation R defines C1=QRK^T. A change of parameters along a gauge direction is not a distinct logical task. The parameter-to-score Jacobian has generically four redundant directions without position rotation, but only two common gauge directions when including C1, because R restricts the commutant.

## Results: fresh worlds 101..105, frozen thresholds

| Test | Measured range | Verdict |
|---|---|---|
| rank of Jacobian C0 (16 × 16) | 12/16 in each of 5 seeds | PASS |
| rank of Jacobian (C0,C1) (32 × 16) | 14/16 in each of 5 seeds | PASS |
| redundant/gauge parameter dimension | 4 (bare), 2 (RoPE) | matches exact derivation |
| exact GL(2) pre-RoPE C0 residual | max 5.5511e-16 | PASS |
| exact RoPE-valid commuting transform residual | max 1.5543e-15 | PASS |
| invalid noncommuting RoPE transform discrepancy | min 0.507161 | detected |
| finite-difference Jacobian valid-null residual | max 1.2324e-10 | precision-limited |
| min singular-value separation (retained vs discarded) | 9.9869e-01 vs at most 1.9797e-11 | threshold is stable |

CPU NumPy float64, Python 3.13.5, scalar 2D rotation θ=0.63 radians; Jacobian central finite differences with h=1e-5 and rank relative cutoff 1e-5. 3 dev seeds 11..13 and five unseen fresh seeds 101..105 were predeclared before fresh. No hyperparameter adjustment to audit.

## Derivation (valid on the specified linear score interface)

Let δQ and δK be infinitesimal weight changes. The differential of C0=QK^T is δC0=δQ K^T+QδK^T. Full-column-rank K implies that the left-orthogonal projection of δQ vanishes whenever δC0=0, so δQ=QX for a unique 2×2 matrix X. Substituting gives δK=-KX^T. Thus every X∈R^(2×2) is a score-preserving infinitesimal Q/K gauge direction: nullspace dimension 4 and Jacobian rank 16−4=12.

For the rotated score C1=QRK^T, the same tangent direction changes the score by δC1=Q(XR−RX)K^T. Full column rank of Q and K makes δC1=0 equivalent to XR=RX. With R=cosθ I + sinθ J, sinθ≠0 and J=[[0,−1],[1,0]], its real commutant comprises exactly X=a I + b J. This is two-dimensional, so the joint Jacobian rank is 16−2=14. Conversely, an arbitrary noncommuting shear changes the score and is not an admissible RoPE gauge. This derivation does not establish non-gauge directions are useful learned tasks.

## H / T / D / C / U

- **H:** quotient out gauge-only parameter perturbations rather than counting them as distinct logical m coordinates; RoPE reduces valid gauge dimension.
- **T:** exact analytic tangents and centered finite-difference 16→16 and 16→32 Jacobians, source-only random matrices, five frozen independent seeds; null directions validated against finite GL(2) transforms.
- **D:** rank and exact invariance checks 5/5 PASS, no compression or capacity PASS. This is a mandatory negative-control / geometry audit for downstream Mirror claims.
- **C:** low-dimensional m may still encode only parameter reparametrizations that cancel under a function-level observable; counting different weights as new experts is invalid.
- **U:** near-singular Q/K would reduce generic rank; central differences and SVD thresholds produce numerical errors. Full model RoPE implementations, attention softmax degeneracy, high-dimensional head symmetries and OOD predictions remain untested.

## Variable table and SI dimensional validation

| Symbol | Meaning (Japanese) | Unit | Definition / range / type |
|---|---|---|---|
| Q, K | Query/Key射影行列 | 1 | real 4×2 matrices, rank 2 |
| R | 位置回転行列 | 1 | real 2×2 orthogonal, θ=0.63 |
| G | ゲージ変換 | 1 | invertible real 2×2 matrix |
| C0, C1 | 注意スコア係数行列 | 1 | real 4×4 matrices |
| X | 無限小ゲージ生成子 | 1 | arbitrary real 2×2 matrix; commutant under RoPE |
| δQ, δK | 重みの微小変位 | 1 | real 4×2 matrix tangents |
| δC | 機能出力の微分 | 1 | real 4×4 matrix |
| θ | RoPE回転角 | rad (SI derived dimensionless) | real scalar, sinθ≠0 |
| h | 数値微分刻み | 1 | positive real, 1e-5 |
| ε | 数値誤差の最大絶対値 | 1 | nonnegative scalar |
| rank | ヤコビアンの階数 | 1 | integer 0..16 |

**Unit check:** δQK^T, QδK^T and δC all share dimensionless normalized score units; XR−RX multiplies two dimensionless 2×2 matrices. Jacobian rank is dimensionless; h is dimensionless. Numerical error in normalized score cannot be interpreted directly as training loss, memory bytes or seconds.

## Provenance

Recorded source SHA256: `acb4ee733ea1665d26814b1fd2830939425cb3fe5fe07c0b6d5bd2ef75e864c8`; fresh CSV SHA256: `eacf728b01bdcffe297c5c1a48230513618d1848041d55f15ce06c14a439172a`. See `run_gauge.py`, `dev_raw.csv`, `fresh_raw.csv`, and `VERIFICATION.json`. Audit result remains scoped; no global Mirror novelty or compression theorem has been proven.
