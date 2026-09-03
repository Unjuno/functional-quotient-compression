# Actual codebook-sharing scaling result

This note isolates one scaling experiment from the locally executed TinyStories GPT-Neo 1M / 3M / 8M / 28M checkpoint series.

## Experiment

For the WQ role in each model:

1. use 32-scalar block VQ with `K=128`;
2. measure calibration task cost for all 28 possible two-layer shared-codebook pairs;
3. enumerate all 105 perfect matchings of eight layers into four pair-codebooks;
4. choose the minimum calibration-task-cost matching;
5. evaluate the selected four-codebook partition on six unseen TinyStories-style passages;
6. compare against eight layer-local K128 codebooks.

The four-codebook solution pays **half the codebook count** of the local solution. Centroid precision/headers are not finalized here, so this is a shared-representation quality/count result, not exact emitted-byte evidence.

## Holdout results

| checkpoint | hidden size | priced-4 mean KL | local-8 mean KL | priced/local |
|---|---:|---:|---:|---:|
| 1M | 64 | `0.004889` | `0.002763` | **1.769×** |
| 3M | 128 | `0.006680` | `0.005236` | **1.276×** |
| 8M | 256 | `0.009606` | `0.010366` | **0.927×** |
| 28M | 512 | `0.017244` | `0.018974` | **0.909×** |

The ratio improves monotonically across this four-checkpoint family and crosses below 1 between the 3M and 8M checkpoints: at 8M and 28M, task-priced four-codebook sharing is better on the six-passage holdout than eight local codebooks while using half as many codebooks.

The best pairings are not simple adjacent-layer groupings. For example:

- 8M Q: `(0,2), (1,6), (3,5), (4,7)`;
- 28M Q: `(0,4), (1,2), (3,5), (6,7)`.

## Interpretation

This is the first actual-model result in this project that points in the direction of the positive-scaling hypothesis for shared representation: **as model width grows, task-priced codebook sharing becomes substantially less costly and eventually beneficial relative to local codebooks in the tested family**.

Several mechanisms can contribute:

- more training blocks per shared codeword as hidden size grows;
- better sample pooling for codebook estimation;
- changing weight geometry with scale;
- amortization of shared representation state.

This experiment does not separate those mechanisms.

## Boundaries

Do not infer that:

- all model families will exhibit the same crossover;
- sharing benefit is monotone beyond these four checkpoints;
- codebook count reduction alone equals a specific serialized compression ratio;
- the same result holds for V/O/MLP roles;
- this establishes 64× feasibility.

The narrow empirical claim is: **within this four-model GPT-Neo family, task-priced WQ pair-codebook sharing shows a monotone improvement in holdout quality relative to local codebooks as width increases, and becomes quality-positive at 8M and 28M.**