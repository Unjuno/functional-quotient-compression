# TinyStories Q/K same-layer coupling across model scale

This note records actual-model follow-up experiments on the uploaded TinyStories GPT-Neo checkpoints (`1M`, `3M`, `8M`, `28M`). Experiments were run locally in the container; GitHub is used only to preserve the result.

## Setup

- GPT-Neo family, 8 layers in all four checkpoints;
- hidden sizes: 64 / 128 / 256 / 512;
- fixed four-text next-token probe;
- WQ and WK quantized with 32-scalar block VQ;
- K=128 shared layer-family codebooks for the scale comparison;
- metric: `KL(Q_l + K_l) / [KL(Q_l) + KL(K_l)]` against the unmodified checkpoint.

A value below 1 means same-layer Q/K is functionally sub-additive relative to independent action costs.

## Results

| checkpoint | hidden size | same-layer ratio median | range | layers < 1 |
| --- | ---: | ---: | ---: | ---: |
| 1M | 64 | **0.750** | 0.410–0.909 | 8/8 |
| 3M | 128 | **0.696** | 0.301–0.848 | 8/8 |
| 8M | 256 | **0.655** | 0.616–0.723 | 8/8 |
| 28M | 512 | **0.602** | 0.551–0.730 | 8/8 |

Across all 32 layer/model cases, the same-layer Q/K joint cost was below the sum of independent Q-only and K-only costs.

The median ratio appears to decrease with model size in this family. This is **descriptive only**: K128 reconstruction NMSE also changes substantially with scale, so the experiment does not isolate model size as the causal variable.

## Perturbation-severity control on 8M

The same 8M experiment was repeated at three codebook sizes.

| K | mean Q NMSE | mean K NMSE | same-layer ratio median |
| ---: | ---: | ---: | ---: |
| 64 | ~0.820 | ~0.739 | **0.648** |
| 128 | ~0.777 | ~0.699 | **0.655** |
| 256 | ~0.736 | ~0.662 | **0.707** |

All 24 layer/K cases remained sub-additive. The effect weakens somewhat as reconstruction improves, but it does not disappear over the tested range.

## Interpretation

Combined with the score-level decomposition in the 8M experiment, this supports treating same-layer Q/K as an architecture-coupled candidate bundle rather than two independent scalar-cost actions. The evidence is stronger than a single-checkpoint observation because the sign is stable across four checkpoints and three K settings on 8M.

Do not infer that:

- every Q/K perturbation family is sub-additive;
- the magnitude follows a universal scaling law;
- bundling removes the need for inter-layer interaction modeling;
- these checkpoints demonstrate a 64x compressed model.
