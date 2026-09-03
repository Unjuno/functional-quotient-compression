# TinyStories 8M actual nonembedding K128 role map

This note records high-value **actual-model** results from container-side experiments on the uploaded TinyStories 8M GPT-Neo checkpoint. It is not a SmolLM2 result and does not establish model-wide 64x quality.

## Setup

- Model: uploaded TinyStories 8M GPT-Neo checkpoint.
- Manual PyTorch GPT-Neo replay with the checkpoint's GPT-2 BPE tokenizer files.
- Intervention: one weight tensor at a time.
- Shared codec primitive: 32-scalar blocks, K=128, 7-bit index (`0.21875` index bit/scalar), one shared codebook per tensor role across the eight layers.
- Codebook values are quantized; biases and other tensors remain unchanged.
- Evaluation: five fixed held-out short-text probes; metrics include KL and top-1 flips relative to the unmodified model.

## 1. Same nominal rate, radically different functional cost by role

Median one-tensor KL across eight layers:

| role | median KL | approximate range |
| --- | ---: | ---: |
| Q | **0.0089** | 0.0026–0.0249 |
| K | **0.0096** | 0.0030–0.0318 |
| V | 0.280 | 0.130–0.797 |
| O | 0.192 | 0.087–0.587 |
| MLP FC | 0.496 | 0.355–2.530 |
| MLP PROJ | 0.511 | 0.389–2.805 |

Weight NMSE is poor and relatively similar across these roles (roughly `0.72–0.81`), yet functional damage differs by orders of magnitude. Thus reconstruction error does not explain the role sensitivity.

The immediate architectural implication is that a single uniform `K128 everywhere` policy is unsuitable. Q/K can tolerate this primitive locally on many layers; V/O and especially MLP require more rate or a different representation.

## 2. Simple row normalization is only partly useful

A paid row-RMS normalization coordinate was tested before shared K128 VQ.

- MLP PROJ median KL improved from about `0.511` to `0.437`.
- O improved on several layers but the median effect was mixed.
- FC did not improve reliably.

Therefore simple weight normalization is role dependent and is not a general substitute for a functional metric.

## 3. Activation-metric normalization improves MLP function while worsening weight NMSE

A second experiment used actual calibration activation RMS as the normalization coordinate:

- FC uses the LayerNorm-output activation entering `c_fc`.
- PROJ uses the GELU activation entering `c_proj`.
- Decoder-visible scales are quantized and paid.

With the same K128 block-VQ index rate:

- FC median KL improved from approximately `0.496` to **`0.468`** (~5.8%).
- PROJ median KL improved from approximately `0.511` to **`0.470`** (~8.0%).

Ordinary weight NMSE became worse. This is direct actual-model evidence that aligning the distortion metric with observed activation geometry can improve function even when raw reconstruction quality degrades.

The improvement is not large enough to make MLP K128 safe by itself; absolute damage remains substantial.

## Boundaries

Do not infer that:

- Q/K can all be simultaneously compressed to K128 with additive task cost;
- V/O/MLP are impossible to compress at sub-bit rates under other representations;
- these short probes are benchmark-quality evaluation;
- the same role ordering transfers unchanged to SmolLM2.

The narrow conclusion is: **under one matched actual codec primitive, role and layer functional geometry dominate raw reconstruction error, and activation-aware metrics partially improve difficult MLP roles without solving them.**
