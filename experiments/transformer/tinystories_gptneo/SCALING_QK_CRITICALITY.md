# Scaling of layer-0 Q/K criticality in TinyStories GPT-Neo

This note records one high-value cross-scale result from local-container actual-checkpoint experiments on the user-provided `1M`, `3M`, `8M`, and `28M` GPT-Neo-family checkpoints. It is fixed-probe evidence, not a universal scaling theorem.

## Setup

For each of the four 8-layer models, all Q/K matrices were first compressed with K=2 block VQ using 32-scalar blocks (`1/32 = 0.03125` index bits/scalar). Then only layer 0 Q/K was restored, either exactly or with row-wise 2-bit scalar quantization. The same four held-out TinyStories-style passages were used for the scale comparison.

## Criticality is strongly scale-dependent

Fraction of the all-Q/K K2 damage recovered by restoring only layer 0 exactly:

- 1M: ~0.4%
- 3M: ~-0.8% (no benefit within this probe / interaction)
- 8M: ~9.2%
- 28M: ~74.3%

Row-wise 2-bit restoration gives the same qualitative transition:

- 1M: ~1.1%
- 3M: ~-1.0%
- 8M: ~11.2%
- 28M: ~71.8%

Thus the 28M result is not merely a codec-precision effect. In this checkpoint family, layer-0 Q/K becomes much more functionally critical at the largest tested scale.

## A simple attention-sharpness explanation fails

Baseline layer-0 normalized attention entropy remained high and similar across sizes (~0.978–0.994), and QK-score standard deviation for 8M and 28M was nearly identical (~0.296). The 28M layer is not simply distinguished by much sharper baseline attention probabilities.

## Perturbation transfer changes with scale

Zeroing only layer-0 Q/K gives approximate mean values:

| model | layer-0 post-attention-state relative delta | final/decoder transfer | KL |
| --- | ---: | ---: | ---: |
| 1M | 0.0328 | 0.523 | 0.0066 |
| 3M | 0.0221 | 0.640 | 0.0030 |
| 8M | 0.0925 | 0.313 | 0.0131 |
| 28M | **0.1510** | **0.736** | **0.1440** |

The 28M transition therefore combines a larger local perturbation with substantially less downstream attenuation.

The baseline layer-0 attention-branch RMS relative to the incoming residual-stream RMS was approximately:

- 1M: 0.211
- 3M: 0.797
- 8M: 0.507
- 28M: **1.163**

In the 28M model the first attention branch is larger than the incoming residual RMS on these probes. This is a descriptive scaling signal, not proof of a general phase transition.

## QCO implication

A fixed per-layer codec policy is unsafe. The same architectural role can move into a different functional regime as model scale changes. Candidate generation should therefore treat layer identity and scale-conditioned functional sensitivity as first-class state, and exact replay remains required before committing a high-rate or low-rate representation.
