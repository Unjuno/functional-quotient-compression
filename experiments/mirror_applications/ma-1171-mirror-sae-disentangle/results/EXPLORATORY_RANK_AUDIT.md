# Post-hoc mechanism audit — feature-detector matrix rank

**Exploratory: not part of the frozen quality gate.** This audit uses *only the 30 already frozen model checkpoints*; no retraining, fresh-guided hyperparameter change, or cherry-picking.

## Definitions and unit/type ledger

| Symbol | Japanese meaning | SI units | Definition | Range/assumption | Type |
|---|---|---|---|---|---|
| A | 特徴ごとのページ混合係数 | 1 | softmax(address @ fixed_keys.T * 2.5), shape 64×8 | rowwise nonnegative and sums 1 | matrix |
| P | コアから生成される固定8ページ行列 | 1 | 8 signed permutations of the shared core, shape 8×32 | real | matrix |
| G | 各特徴ごとの行正規化・ゲイン | 1 | diagonal 64×64 | strictly positive on trained checkpoints | diagonal matrix |
| H | Householder直交行列 | 1 | I - 2 vv.T, shape 32×32, v unit | H H.T = I | matrix |
| φ | 要素ごとのMirror非線形変換 | 1 | tanh(2.5u)/2.5 | real → (-0.4,+0.4) | function |
| E | 仮想特徴検出器群 | 1 | shape 64×32 | real | matrix |
| σ_i | Eの特異値 | 1 | descending nonnegative singular values | i=1..32 | scalar |
| p_i | 特異値二乗の正規化比 | 1 | σ_i² / sum_j σ_j² | [0,1], sum 1 | scalar |
| r_eff | エネルギー型実効ランク | 1 | exp(-sum_i p_i log p_i) | [1,32] | scalar |

The matched non-Mirror decoder is **E_affine = G A P**, therefore rank(E_affine) ≤ min(rank(G),rank(A),rank(P)) ≤ 8. This is a strict linear algebra result, regardless of optimizer convergence. The nonlinear Mirror detector is **E_mirror = G φ(A P H) H.T**. Elementwise φ generally does not commute with matrix factorization; the rank≤8 proof does not apply. Both outputs remain shape 64×32 (unit consistency: all activations/weights dimensionless).

## Observed checkpoint spectra

The numerical rank counts singular values satisfying σ_i/σ_max > 0.001. Across the same six paired world/seed checkpoints:

| Encoder | Numerical rank, 6 checkpoints | Median effective rank |
|---|---|---:|
| Dense one pass | 32/32 for every checkpoint | 27.65 |
| Dense Loop3 | 32/32 for every checkpoint | 26.80 |
| Learned rank8 Loop3 | 8/32 for every checkpoint | 7.28 |
| Shared affine 8-page Loop3 | 8/32 for every checkpoint | 4.83 |
| Mirror nonlinear 8-page Loop3 | **32/32 for every checkpoint** | **5.33** |

Thus the Mirror operator breaks the *linear rank upper bound* of the matched affine generated detector on this test. However, the energy is still concentrated in a small number of directions. Higher algebraic rank alone did not achieve accurate recovery: the Mirror's held-out OOD factor F1 median was only 0.1431 versus dense one-pass 0.5683. This is a scoped mechanism finding, not a claim of interpretability or general function-space capacity.

Raw numerical singular value summaries per checkpoint are saved in `EXPLORATORY_ENCODER_RANK.csv`.
