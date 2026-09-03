# TinyStories 8M: actual WQ joint-coupling results

This note records high-value **actual-model** joint-distortion experiments run locally in the container on the user-provided TinyStories 8M GPT-Neo checkpoint. It is not a general theorem and it is not a whole-model compression result.

## Setup

- Model: TinyStories GPT-Neo 8M, 8 Transformer layers.
- Intervention family: every layer's `q_proj.weight` reconstructed from one shared block-VQ family.
- VQ: 32-scalar blocks, K=128, 3-bit quantized shared codebook.
- Task witness: KL to the uncompressed model on fixed held-out text probes; top-1/NLL were also recorded in the raw local experiment files.
- All individual layers, all 28 pairs, all 56 triplets, all 70 quadruples, and all subsets of size 5–8 were actually replayed.

## 1. Individual distortion is strongly non-additive

On the first six-passage probe, pair distortion was systematically super-additive:

- mean `KL(i,j) / (KL(i)+KL(j))`: **1.0866**
- maximum ratio: **1.1909**
- 18/28 pairs were more than 5% super-additive
- no pair was more than 5% sub-additive

On a completely different six-passage probe:

- mean ratio: **1.0979**
- maximum: **1.2463**
- 22/28 pairs were more than 5% super-additive
- all 28 ratios were above 1.01

Therefore summing individual task costs is systematically optimistic for this intervention family.

## 2. Pair-specific coupling is stable across probes

Across the two independent held-out probe sets:

- pair-ratio Spearman: **0.806**
- absolute-interaction Spearman: **0.731**
- pair-KL Spearman: **0.984**

Layer distance contains useful structure. Mean pair ratio was about `1.126` for adjacent layers and about `1.016` for the pair separated by seven layers.

Predicting second-probe pair KL from second-probe individual KL:

| surrogate | MAPE |
| --- | ---: |
| additive | 8.67% |
| one global coupling factor | 4.24% |
| layer-distance coupling | 3.84% |
| pair-specific factor learned on probe 1 | **2.76%** |

This supports a sparse coupling matrix as a screening surrogate, with exact replay still required for commit.

## 3. Pairwise coupling explains most 3-way and 4-way distortion

For each subset S, define the second-order predictor

`D2(S) = sum_i D(i) + sum_{i<j in S} [D(i,j) - D(i) - D(j)]`.

All subsets were evaluated by actual forward replay.

| simultaneous WQ layers | additive MAPE | pairwise-corrected MAPE |
| ---: | ---: | ---: |
| 3 | 13.2% | **0.50%** |
| 4 | 17.7% | **1.10%** |
| 5 | 21.6% | **1.95%** |
| 6 | 25.0% | **3.11%** |
| 7 | 27.9% | **4.51%** |
| 8 | 30.5% | **6.12%** |

The 3-way result replicated on an independent probe: additive MAPE `15.1%`, pairwise-corrected MAPE `0.499%`.

Thus second-order coupling captures the dominant joint distortion in this tested family, but begins to over-correct as subset size grows.

## 4. Size-dependent pair shrinkage captures the remaining saturation

Least-squares shrinkage on the pair-interaction sum gives approximately:

| subset size | pair-interaction multiplier gamma |
| ---: | ---: |
| 3 | 0.966 |
| 4 | 0.935 |
| 5 | 0.907 |
| 6 | 0.880 |
| 7 | 0.856 |
| 8 | 0.833 |

Using only the size-3 and size-4 values to fit a linear decline of about `-0.031` per added layer, then predicting previously unused size-5 through size-8 subsets, gives MAPE:

- size 5: **0.82%**
- size 6: **0.92%**
- size 7: **0.79%**
- size 8: **0.81%**

An independent probe gave gamma3 `0.987`, close but not identical. The shrink factor therefore requires calibration; it is not a universal constant.

## 5. QCO implication

For this intervention family, a practical distortion model is much closer to

`individual costs + calibrated pair couplings + small subset-size saturation correction`

than to a purely additive block cost.

This materially changes optimizer design:

- candidate screening can use individual costs;
- selected candidates should pay pair-coupling penalties;
- nearby-layer coupling deserves priority measurement;
- a small calibration set may estimate pair scaling;
- exact full replay remains the commit authority.

## Boundaries

Do not infer that:

- all Transformer intervention families are pairwise-dominated;
- the same coupling matrix transfers to other roles, precisions, models, or tasks;
- pairwise correction is exact at large subset size;
- these KL values are codec-bit evidence or a 64× certificate.

The narrow result is strong: **for shared-K128 WQ compression on this actual 8-layer GPT-Neo model, additive distortion is systematically wrong, while measured pairwise coupling predicts higher-order joint KL surprisingly well.**
