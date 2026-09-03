# Scaling and relocation of Q/K criticality in TinyStories GPT-Neo

This note records cross-scale fixed-probe evidence from local-container experiments on the user-provided `1M`, `3M`, `8M`, and `28M` GPT-Neo-family checkpoints. It is not a universal scaling theorem.

## Setup

For each 8-layer model, all Q/K matrices were compressed with K=2 block VQ using 32-scalar blocks (`1/32 = 0.03125` index bits/scalar). Individual layers were then restored exactly, one at a time, to measure marginal recovery of the all-Q/K damage. The same four held-out TinyStories-style passages were used throughout.

## Criticality does not stay on the same layer as scale changes

The positive single-layer restore gains are strongly redistributed across the scale family.

Approximate share of the sum of positive restore gains by layer:

| model | dominant layer(s) | dominant positive-gain share |
| --- | --- | ---: |
| 1M | layer 2 | ~57% |
| 3M | layer 1 | ~33% |
| 8M | layers 5 and 7 | ~25% each |
| 28M | layer 0 | ~61% |

For 28M, restoring layer 0 exactly recovers about 74% of the all-Q/K K2 damage; row-wise 2-bit layer-0 restoration recovers about 72%, so the effect is not merely an exact-weight artifact. But the smaller checkpoints place their criticality elsewhere. Therefore the correct conclusion is **criticality relocation**, not a universal early-layer rule.

Representative all-Q/K K2 base KL and single-layer exact-restore gains:

- 1M base KL ~0.512; layer-2 restore gain ~0.301
- 3M base KL ~0.423; layer-1 restore gain ~0.180
- 8M base KL ~0.251; layer-5 / layer-7 gains ~0.084 / ~0.083
- 28M base KL ~0.252; layer-0 restore gain ~0.188

## A simple attention-sharpness explanation fails for the 28M layer-0 result

Baseline layer-0 normalized attention entropy remained high and similar across sizes (~0.978–0.994), and QK-score standard deviation for 8M and 28M was nearly identical (~0.296). The 28M layer-0 result is therefore not explained simply by much sharper baseline attention probabilities.

## Perturbation transfer changes with scale

Zeroing only layer-0 Q/K gives approximate mean values:

| model | layer-0 post-attention-state relative delta | final/decoder transfer | KL |
| --- | ---: | ---: | ---: |
| 1M | 0.0328 | 0.523 | 0.0066 |
| 3M | 0.0221 | 0.640 | 0.0030 |
| 8M | 0.0925 | 0.313 | 0.0131 |
| 28M | **0.1510** | **0.736** | **0.1440** |

At 28M, layer-0 perturbation is both larger locally and less attenuated downstream than at the smaller sizes.

The baseline layer-0 attention-branch RMS relative to incoming residual-stream RMS was approximately:

- 1M: 0.211
- 3M: 0.797
- 8M: 0.507
- 28M: 1.163

This helps describe the 28M layer-0 regime but does not explain the full cross-scale relocation profile.

## QCO implication

A fixed layer-priority policy is unsafe. Layer identity, model scale, and functional perturbation transfer must be measured for the specific checkpoint. QCO candidate generation should therefore treat layer-specific functional sensitivity as a first-class state rather than assuming that the same depth or role stays important as models scale.
